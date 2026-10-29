# ytauto — timestamped transcript → faceless YouTube video

YouTube-first B-roll automation. For every shot in your transcript it searches YouTube, has an AI judge watch
candidate videos *without downloading them*, downloads only the matching 5–12 s sections, cuts a 4–5 s clip, runs
local and AI quality checks, scores it, and assembles a 1080p30 video. Pexels → Pixabay → Coverr → Mixkit are used
**only for shots YouTube can't fill**.

The AI judge is a **fallback chain of providers**, not one key, so a 503/429 on any single model doesn't stop a run.

## Setup (Windows)

```powershell
cd E:\1-MARKETING\youtube-automation\yt-auto
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[test]" -e ..\mixkit-dl
copy .env.example .env      # then fill in keys
```
Requires `ffmpeg`/`ffprobe` on PATH (already installed).

`.env` keys: `PEXELS_API_KEY`, `PIXABAY_API_KEY`, `COVERR_API_KEY`, plus **at least one AI judge source**:

| `.env` setting | What it gives you |
| --- | --- |
| `HEART_CONFIG` | Path to a shared OpenAI-compatible provider config (`{"providers":{"<name>":{"baseUrl","apiKey"}}}`). **Read-only** — never modified or copied, so the owning project is untouched and a key rotated there is picked up here. Unlocks the multi-provider chain. |
| `GEMINI_API_KEY` | The native google-genai client, used as the `_direct` link in the chain. |

With neither, the tool runs in **offline heuristic mode**: no pre-download screening or AI verification, only the
local checks (resolution, blur, B&W, black/frozen, letterbox, faces, static watermark). Results are noticeably worse
— expect wrong-content and burnt-in-text clips to slip through.

### The judge chain

Set in `config/default.yaml` under `judge_chain` and walked top to bottom until a provider answers. `provider` is a
key in the `HEART_CONFIG` providers map, or `_direct` for the native Gemini client. Per link: `vision` (can read
frame grids), `youtube` (can analyse a public YouTube URL without downloading it), `max_tokens`.

A link that errors — HTTP 4xx/5xx, empty content, or a reply truncated before any answer — is **circuit-broken for
5 minutes** and the next link takes over. Reordering providers is a config edit, not a code change.

**Free-tier quota.** Free links are capped in requests per day. Each link can carry `rpm` / `rpd`; requests are
counted locally (persisted in `cache/`, per Pacific day, since Gemini resets at midnight Pacific), so a spent link is
skipped without being called. A "quota exceeded" 429 marks a link spent until the reset. When **every** link is
spent, `run` stops starting shots, saves what is finished and exits with code 3 — run it again after the reset, or
use `run --wait` to sleep until then and continue. `ytauto quota` shows today's use per link.

Local clips are judged as a **tiled frame grid** (`grid_cols`×`grid_rows` @ `grid_fps`, default 6 frames) rather than
uploaded video: every vision model accepts images, and it costs ~10× less than inline video.

> Verify a chain before relying on it — vendor "supports images" metadata is often wrong. Some models return HTTP 200
> with empty content, or spend the whole token budget on reasoning and never answer. Both are treated as failures.

## Usage

```powershell
.venv\Scripts\ytauto plan   samples\japan_5_destinations.txt --title "5 Best Destinations in Japan"   # briefs + queries only
.venv\Scripts\ytauto run    samples\japan_5_destinations.txt --title "5 Best Destinations in Japan"   # full pipeline
.venv\Scripts\ytauto run    script.txt --title "My Video" --vo narration.mp3                          # re-time to voice-over
.venv\Scripts\ytauto status script.txt --title "My Video"                                             # per-shot status
.venv\Scripts\ytauto redo   script.txt --title "My Video" --shot 7                                    # replace one clip
.venv\Scripts\ytauto search "kyoto temples 4k"                                                        # standalone YouTube search
.venv\Scripts\ytauto run    script.txt --title "My Video" --wait                                      # wait out the daily quota
.venv\Scripts\ytauto quota                                                                             # judge requests used today
```
`run` is resumable: Ctrl+C at any time, run it again and finished shots/downloads are reused. Every run writes
per-stage wall time to `projects/<title>/logs/timings.json`.

Transcript formats: `00:00-00:05 text`, `00:00–00:05: text`, `[00:05] text` (ends at next start), SRT, VTT, or JSON
`[{"start":0,"end":5,"text":"..."}]`.

## Output (`projects/<title>/`)

| path | content |
| --- | --- |
| `output/final.mp4` | finished video (1920×1080, 30 fps, H.264) |
| `output/sources.csv` | per-shot source URL, source timestamps, licence, score (paste into description credits) |
| `output/contact_sheet.jpg` | one thumbnail per shot |
| `report.html` / `report.json` | sources, scores, rejection stats, weak matches, QA issues, blockers |
| `ledger.jsonl` | every accept/reject decision with reasons (the benchmark history) |
| `sources/youtube`, `sources/stock` | downloaded sections / stock files |
| `clips/` | the cut, normalised clips used on the timeline |
| `state.json`, `logs/run.log` | resume state and logs |

Shared across projects: `cache/` (search results, metadata, Gemini answers — repeat runs are cheap).

## How selection works

1. The judge (cached) turns each shot into a brief: a required `subject`, an ideal `visual` (a preference only),
   entities and queries; a review pass strips details the narration does not support. Long segments are split
   into 3–6 s shots, and long videos are planned ~40 shots per call.
2. yt-dlp searches (nothing downloaded); pre-filters by duration/blacklist/height ≥ 1080; skips videos already
   known to carry a watermark or AI footage.
3. Local CLIP (CPU, `prerank`) ranks the candidates by how well their storyboard thumbnails match the subject.
4. The judge reads the top candidates' **storyboards** (seek-bar previews, `screen_batch` videos per call) and names
   matching thumbnails; the next candidates are only screened if those fail. Videos without a storyboard are
   skipped (`screen_mode: storyboard_only`) or URL-watched (`storyboard`).
5. Only those windows are downloaded; PySceneDetect picks a cut-free window; the clip is cut to spec.
6. Local checks + AI verification of a frame grid → **hard rejects** (never overridden by score) + weighted score
   vs `min_score`. While a judge is configured, a clip is never accepted without a verdict.
7. Failing shots retry with new queries, then fall back Pexels → Pixabay → Coverr → Mixkit, same benchmark.
8. Nothing passes → closest soft-failure is used and flagged **WEAK** in the report; no footage at all → blocker.
9. QA (spec, black/frozen detection, ledger audit, duplicate shots) re-selects failing shots automatically (max 2 loops).

Tune everything in `config/default.yaml` (thresholds, weights, shot length, crossfade, fallback order).

### Stock fallbacks

Pexels, Pixabay and Coverr use their REST APIs (keys in `.env`). **Mixkit has no API**, so it goes
through the `mixkit-dl` CLI (installed from `..\mixkit-dl`) using its documented JSON contract:
`mixkit-dl "<keywords>" --limit N --quality 1080 --out <dir> --json --quiet`, reading the exit code
first and then the JSON on stdout.

- Multi-word queries *are* real searches: the slug redirects to `/free-stock-video/discover/<slug>/`.
  Longer phrases match fewer clips, so each shot walks a **keyword ladder** from its own query down
  to a broad category (`mixkit_max_terms`, default 3).
- Exit codes are honoured: `2` = nothing found (broaden), `4` = rate-limited and `3` = disk guard,
  both of which drop Mixkit for the rest of the run rather than hammering it.
- Only items with status `downloaded`/`skipped` are used, and **portrait clips are discarded** —
  Mixkit mixes them into results and they cannot fill a 16:9 frame.
- Clips land in one shared folder (`mixkit_dir`), so repeat runs reuse them and never re-download.

## Modules (reusable on their own)

`transcript` (parse/split) · `sources.youtube` / `sources.stock` (search & fetch) · `llm.router` (provider chain) ·
`llm.providers` (read-only shared config) · `screen` + `inspect.verify` (AI) · `inspect.grid` (frame grids) ·
`inspect.*` (local checks) · `benchmark` (scoring) · `select` · `assemble` · `qa` · `report` · `align` (VO).

## Tests

`.venv\Scripts\python -m pytest` — offline (parsers, benchmark rules, ffmpeg assembly, QA catching black frames,
judge chain fallback/circuit breaker/JSON parsing, quota, frame-grid extraction). No network or API keys required.
GitHub Actions runs the same suite on every push (`.github/workflows/tests.yml`).

Measurement scripts (use cached data; the second spends a few judge calls):
`scripts\calibrate_prerank.py` (CLIP vs past judge verdicts) · `scripts\measure_batch_screen.py` (batched vs single
storyboard screening).

## Notes

- Licence info for each clip is recorded as reported by the source; YouTube clips are logged as their video's licence.
- Face detection uses OpenCV Haar cascades (no model download); it flags large faces, not every person in frame.
- Coverr's API response shape is parsed defensively; confirm with your key on first use.
