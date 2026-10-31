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
| `GEMINI_API_KEY` | Your own Gemini key, called straight at Google as the `_direct` links - the **primary** judge. Free quota is per model, so each model is its own link; when one model's day is spent the chain moves to the next, then to the `HEART_CONFIG` providers. |

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

## Speed settings (config/default.yaml)

| setting | what it does |
| --- | --- |
| `judge_hedge_after: 8` | A judge call slower than 8 s (or 2.5x its link's usual time) is also sent to the next link; first answer wins. `0` = off. |
| `download_ahead: true` | The next screened candidate's section downloads while the current clip is cut and judged. |
| `defer_ai_check: true` | Skips the extra "is it AI footage" call for clips already rejected on score; run later only if that clip becomes a weak match. |

Extra judge links verified live: `geminiopenai/models/gemini-3.1-flash-lite` (own quota) and `openrouter/nvidia/nemotron-3.5-super-vl:free` (slow, last resort).

### Local checks that save judge calls
- `local_overlay_check: true` - OCR (`inspect/overlay.py`) rejects text pinned in place over moving footage (watermarks, channel names) before any judge call. Optional dependency: `pip install --no-deps rapidocr-onnxruntime pyclipper shapely six` (plain `pip install rapidocr-onnxruntime` pulls `opencv-python`, which breaks `cv2.CascadeClassifier`). Without it the check is silently off. Measured on 126 past clips: 24 of 64 AI-flagged clips caught, 0 false rejects on 62 clean ones.
- Channel reputation (shared `cache/`): a channel with 2 watermarked/AI videos and no accepted clip is skipped; good channels rank higher.
- `shot_workers: 3` was measured: no gain over 2 (6.8 vs 7.0 min; CLIP and scene detection get slower), so 2 stays.

## ChatGPT as a judge (via Nibedita / HeartCode)

A `provider: _heart` link in `judge_chain` runs the `heart` CLI headless (`llm/heart.py`) and gets
ChatGPT's verdict on the same frame grid or preview sheet the other links see. No API key: it uses
HeartCode's ChatGPT login, a quota separate from every Gemini key. It is second in the default chain,
so it takes over when Gemini is slow (backup requests go to it) or out of quota.

- Measured on 126 labelled clips: 58/64 watermark/text clips caught, 3/62 clean ones flagged (2 of
  those carry an "ON RIDE" watermark Gemini missed), 0 errors, median 6.5 s.
- A full run with ChatGPT as the only judge (`GEMINI_API_KEY` empty) finished all 12 shots, QA passed.
- Not installed or login expired: the link is skipped / circuit-broken and the run carries on.
- It cannot watch YouTube URLs (`screen_mode: url`); that stays Gemini-only.

## New commands

```powershell
.venv\Scripts\ytauto doctor            # tools, opencv build, OCR, keys, judge links and their quota
.venv\Scripts\ytauto doctor --live     # + one tiny request per judge link, with its answer time
.venv\Scripts\ytauto batch scripts\    # every transcript in a folder, one after another, waits out quota
```

## Nibedita agent

`yt-video-reviewer` (Nibedita, Marketing): after a run it compares every thumbnail with its
narration, re-selects wrong or weak shots with `ytauto redo` (max 3 rounds) and reports.

## Neighbouring shots

`adjacent_source_gap: 30` - shot N and N+1 may not use footage of one video closer than 30 s
(two consecutive windows of one tripod shot played as a single frozen shot).

## Finishing pass (`ytauto finish`)

```powershell
.venv\Scripts\ytauto finish script.json --vo voice.mp3 --title "My Video" --spec finish.yaml --assets E:\video-assets --out final.mp4
```
Adds the voice-over, word-timed captions (spoken word highlighted; timing from faster-whisper, spelling
from the transcript), title / lower-third cards, a music bed ducked under the voice, SFX hits, film grain,
vignette, light grade, an optional particle overlay and fades. `finish.yaml` keys: `music`, `music_db`,
`grain`, `grain_opacity`, `particles`, `particles_from`, `tail`, `cards` (`text`, `start`, `end`, `kind:
title|lower`), `sfx` (`file`, `at`, `gain_db`), `captions`, `font`. Asset names are matched against
`--assets` sub-folders (`audio/music`, `audio/sfx`, `video/overlays`).

`brief_overrides` (project config): hand-fix one shot's brief when the planned subject has no footage,
e.g. `8: {subject: "...", youtube_queries: [...]}`, then `redo --shot 8`.

## Speed round 2 (measured on the 87-shot Philippines run)

What the time went to (`logs/timings.json`): section downloads 70 min of work, AI verify of
downloaded clips 44 min, storyboard screening 33 min; only ~1 in 5 downloads became the clip.

| Setting | Default | What it does |
|---|---|---|
| `screen_min_match` | 70 | Candidates the storyboard screen rated below this are tried last (below 70: 6 of 39 downloads usable; 80+: 55 of 154). Deferred, never dropped. |
| `sibling_extend` | 9.0 | The 2nd/3rd shot of a sentence first tries a *different scene* of the video its previous shot used; the first shot's section is downloaded 9 s longer for it. Shots of one sentence go to the same worker. Still checked and AI-verified. |
| `judge_lanes` | 2 | The first two judge links are used side by side (least expected wait first) instead of the second only taking overflow. |
| `library`, `library_min_sim`, `library_tries` | on, 0.28, 2 | Every accepted clip goes into `cache/library`; later shots with the same setting try the best CLIP matches first (no search/screen/download). Max 3 other videos per clip. Back-fill old projects: `ytauto library-add projects/<slug> [--skip N]`. |
| `remotion_export`, `remotion_bake`, `remotion_grain`, `remotion_dust` | off, on | Export each accepted clip for Remotion (1080p30, 0.6 s cloned tail, film look baked in) in the background while selection runs, plus `words.json`. Manual: `ytauto export-remotion <transcript> --out <dir> [--vo voice.mp3]`. |

Also automatic: videos YouTube reports as gone are remembered in the shared cache (104 repeat
metadata calls in one run), and a section download reuses the stream URL from the metadata call
(`ffmpeg -ss -t -i URL -c copy`, the same command yt-dlp runs) - about 2 s less per download,
identical output; it falls back to yt-dlp when the URL is stale. yt-dlp's
`concurrent_fragment_downloads` was measured and does nothing for sections (they go through ffmpeg).

Remotion: a composition fed baked clips rendered 900 frames in 69 s against 170 s for live CSS
grain/dust/filters (2.5x). Concurrency 10 (83 s) and `--gl=angle` (77 s) were slower than 6.

## Production round 3 (relevance, variety, fewer shots, stills, sound)

1. **No clip longer than 5 s.** `shot_target 4.5 / shot_min 3 / shot_max 5`. The maximum wins over the minimum: a
   5.1 s sentence becomes two 2.55 s shots (only sentences of 5-6 s or under 3 s give a shot below 3 s). Measured
   on the Philippines script: 4.5/3/5 = 93 shots, 4.5/3/6 = 87 (the 5 s cap costs shots, by design).
2. **Variety and reuse.** A YouTube video may give any number of different sections (`max_shots_per_source: 0`);
   sections never overlap and neighbouring shots stay 30 s apart in the source. Other sources (stock, stills) are
   picked once. The same footage may play a second time only as a last resort before a weak match (`reuse_max_uses 2`):
   at least `reuse_min_gap` 60 s apart on the timeline, never a still, and judged again against the new shot.
   `report.json` has `shares` (youtube / stock / still, % of running time), `reused_clips` and `reuse_problems`.
3. **Generic b-roll goes to stock first** (`stock_first`): the planner marks `generic: true` for ordinary
   footage (open sea, aerial coast, crowd) that names nothing specific; named places, people and events stay
   YouTube-first. This is what keeps YouTube near 65-70% of the footage.
4. **Stills** (`stills`, `sources/images.py`, `stills.py`): when YouTube and stock both fail, a licensed
   photograph (Pexels, Pixabay, Wikimedia Commons; at least `still_min_width` px) is judged like a clip (relevance,
   watermark/text, the dedicated AI question) and cut into a slow push / pull / pan clip, so QA, export and
   assembly need no change. Coverr and Mixkit have no photographs, so photos come from Pexels, Pixabay and Commons. A photo must score
   `still_min_relevance` 75 (clips: 60) and is judged with the narration before and after it, with the place in the
   search; a picture that only shares a keyword is rejected. CC-BY credits are written to
   `output/credits.txt`. `still_share_warn` (10 %) is a guide, not a limit.
5. **Corner logos** (`watermark_crop`): footage whose only defect is a small logo in one corner and whose
   source is tall enough (height / 1.15 >= 1080) is re-cut 1.15x closer and judged again, only after
   YouTube and stock failed. The source stays in the ledger; whether it may be reused is the user's call.
6. **`ytauto review-shots <project>`** (before rendering): one AI call per ~10 shots compares each shot's
   frame with its narration; lists shots under 60. Strict on abstract narration, so read the reason before
   redoing (`redo --shot N`).
7. **Sound:** `ytauto sfx-plan` picks the few cues and sets their level from measured files (a cue's peak 8-14 dB
   under the voice's peak, never closer than 8 dB to the voice in its own window; ambience 20 dB under the voice
   average); `ytauto sfx-audit` checks every cue (file, level under speech, sync with the cuts, spacing, per-minute
   cap), the voice+effects mix, and the rendered video's loudness and true peak.

Measured/known limits: GL Transitions were tried and not adopted (see `VIDEO-PRODUCTION-GUIDE.md`).
Everything above was checked with unit tests (270 pass) and one real check per feature; no before/after timing run.

## Round 4 (speed, one data-driven film, tidier code)

Measured on the Philippines run first (87 shots; 274 downloads, 336 judge calls, 77 accepted + 10 weak clips):
1. **No judge call for a clip a free check already rejects** (`benchmark.local_rejects`: resolution, B&W, black,
   frozen, blur, letterbox, face). 81 of 347 judge calls went to such clips. If one is later promoted to a weak match the
   AI-footage question is still asked then.
2. **`blur_threshold` 60 -> 40.** Of 18 clips rejected at 40-59, 9 were rated relevant (70-100) by the judge, "slightly soft"
   (underwater, drone). Below 40 the verdicts were wrong-subject or synthetic. `min_score` still guards quality.
3. **`max_downloads_per_shot: 8`.** 14 of 87 shots took 8+ downloads (up to 25) and 10 of those still ended weak. Past the
   budget a shot goes to stock, a still, a reuse, then a weak match. Counted per `select()`, so `redo` gets a fresh budget.
4. **`ytauto film`**: one `film.json` for the reusable `DocFilm` Remotion composition (`src/doc-kit/DocFilm.tsx`, fetched
   with `calculateMetadata`, the documented data-driven pattern): camera moves, transitions, chapters, stat cards, place chips,
   title card, caption words and the sound plan. One AI call picks chapters / stats / places / tagline; `film.check_facts` drops
   anything the narration does not say (a stat's number must be in its sentence, place names are re-spelled as the narration
   spells them), and the cards are timed to the spoken words. Checked against the hand-made Philippines film: same footage,
   grade, title, stat card and captions; 7,641 appeared at 28.36 s (hand-timed: 28.2 s). A new film needs no new Remotion code.
5. **Code:** the last-resort paths (logo crop, reuse, stills) moved from `select.py` to `lastresort.py` and share one
   `_judge_made_clip`; `select.py` 1,024 -> 846 lines. Removed `direct_verify_video` (measured 3x slower, off) and the unused
   `pydantic-settings` dependency.
6. **Nibedita:** workflow `yt-film` (script + voice-over -> finished film) with agents `yt-film-maker` and `yt-video-reviewer`
   (now driven by `review-shots`).
