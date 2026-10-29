# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`ytauto` turns a timestamped transcript into a finished faceless YouTube video. For each shot it
searches **YouTube first**, has an AI judge watch candidates *without downloading them*, downloads
only the matching section, cuts a ~4-5 s clip, scores it against a benchmark, and assembles a
1080p30 video. Stock sites are fallbacks only.

## Commands

```powershell
# Setup (Windows; ffmpeg/ffprobe must be on PATH)
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[test]" -e ..\mixkit-dl
copy .env.example .env

# Tests - all offline, no network or API keys needed
.venv\Scripts\python -m pytest
.venv\Scripts\python -m pytest tests/test_router.py            # one file
.venv\Scripts\python -m pytest tests/test_benchmark.py::test_large_face_is_rejected   # one test

# Pipeline
.venv\Scripts\ytauto run    samples\quick_test.txt --title "My Video"
.venv\Scripts\ytauto plan   script.txt --title "My Video"      # briefs + queries only
.venv\Scripts\ytauto status script.txt --title "My Video"
.venv\Scripts\ytauto redo   script.txt --title "My Video" --shot 7
.venv\Scripts\ytauto search "kyoto temples 4k"
.venv\Scripts\ytauto quota                                     # judge requests used today, per link
```

`run` is **resumable**: re-running a finished project takes ~3 s instead of minutes. Interrupt
freely; finished shots and downloads are reused via `state.json` + the shared `cache/`. When every
judge link is out of its daily quota, `run` pauses (exit code 3); `run --wait` sleeps until the reset.

**Set `PYTHONUTF8=1` when running anything that touches YouTube metadata.** The default Windows
encoding here is `cp1252`, so a non-ASCII video title (Japanese, emoji) raises `UnicodeEncodeError`
on any file write that does not pass `encoding=` explicitly.

## Architecture

Pipeline stages, each resumable and individually usable:

```
transcript -> planner -> select (per shot) -> assemble -> qa -> report
```

`pipeline.py` orchestrates; `select.py` holds the real logic. Per shot, `Selector`:
1. yt-dlp search (flat, cached) -> `youtube.details()` for metadata, nothing downloaded
2. `inspect/embed.py` - local CLIP ranks candidates by their best storyboard tile (no judge call)
3. `screen.py` - the judge reads the video's **storyboard** (seek-bar preview sprites, one numbered
   contact sheet via `sources/storyboard.py`), `screen_batch` videos per call, and names matching
   tiles; code maps tiles to times. Watching the URL only happens with `screen_mode: storyboard`
   (for a video with no storyboard) or `url`
4. downloads only that section, `inspect/scenes.py` picks a cut-free window, cuts the clip
5. `inspect/checks.py` local checks, then `inspect/verify.py` AI verification
6. `benchmark.py` -> hard rejects + weighted score
7. on failure: retry with new queries, then fall back to stock providers **per shot**

Shots run `shot_workers` at a time and a round's candidates are screened `screen_workers` at a time;
`judge_concurrency` caps judge calls in flight across all of it. Every run writes per-stage wall
time to `logs/timings.json` and prints it at the end - read it before guessing where time goes.

### Things that are easy to get wrong

**YouTube is the required primary source.** Stock (Pexels -> Pixabay -> Coverr -> Mixkit) is a
per-shot fallback only: if YouTube fills 11 of 12 shots, only the 12th may use stock. When YouTube
selection underperforms, fix the selection path - never widen stock usage.

**Briefs are per *shot*, not per segment** (`planner.py`). A 10 s segment splits into two shots; if
both carried the same brief the selector fetched two clips of the same subject and the cut read as
a stutter. `Shot.variant` is the discriminator - shots of one segment keep the subject and change
the framing (wide establishing -> closer detail -> aerial). `plan()` returns a dict keyed by
**shot id**, and `_differentiate()` repairs any repeat the model still emits.

**A brief must describe ONE continuous shot in ONE place.** `inspect/scenes.py` extracts a single
cut-free window, so a brief asking for a montage, quick cuts, or a move between two landmarks can
never be filled: measured, one such brief burned 24 candidates and dropped the shot to stock. The
prompt forbids these *and* `_reject_impossible()` rewrites them, because a live model produced "a
montage of quick, exciting clips ... ending with a question mark graphic" after being told not to.
Keep those regexes narrow - a moving camera ("a drone shot gliding over", "a slow pan across") is
still one shot and must not be flagged.

**Screening is a pre-filter, so storyboards are good enough for it.** Watching a YouTube URL cost
~50 s and ~240k tokens per candidate - the OpenAI-compatible proxy ignores `end`/`fps`, so it
watched whole videos - and spent the free daily quota within two 12-shot runs. A storyboard
judgement measured ~12 s and ~3k tokens, agreeing with URL screening on 11/15 real candidates; 3 of
the 4 disagreements were storyboards rejecting videos that failed after download anyway. It is
more lenient on fine detail (it passed a station *exterior* for an *inside* brief), which costs a
download, never quality: every clip still passes local checks, frame-grid verify and the dedicated
AI check. Never take the AI-generated flag from thumbnails - that is the dedicated check's job.

**Free judge quota is requests per day, so calls saved are the main lever** (`llm/quota.py`). A 25-min
video is ~300 shots; at ~11 judge calls per shot (measured, speed test) that was ~3,300 requests against
~1,000/day per free project. What brings it down, each measured before it was turned on:
- batched screening (`screen_batch: 2`): one call for two contact sheets agreed with single screens
  on `usable` 15/16 and on the best tile 12/16 (`scripts/measure_batch_screen.py`); waves are lazy, so
  the next pair is only screened when the first produced no accepted clip
- CLIP pre-ranking: best-tile similarity on the plain *subject* separated judge-usable videos at AUC
  0.74 vs 0.58 for the text score (`scripts/calibrate_prerank.py`). **CLIP must not choose which tiles
  the judge sees**: the judge's window was in CLIP's top 6 tiles only 12-20 of 43 times
- whole-video defects (watermark, AI footage; burnt-in text after 2 strikes) are cached as `bad_video`
  and never downloaded again; the dedicated AI check is skipped when verify already rejects the clip
- `QuotaExhausted` (all links spent) pauses the run; it is never recorded as a reject or "unavailable"

**A brief has a required `subject` and a preferred `visual`.** Screening and verify judge usability
and relevance on the subject; the framing only ranks. The planner's stacked or invented details (a
"snow groomer" the narration never mentions) made the judge reject good footage, so a review pass
(`_critic`, one call per ~40-shot batch) strips them. The planner also batches (`chunk_shots`):
one call for 300 shots would overflow a free model's output limit.

**A clip is never accepted unjudged while a judge is configured.** `verify_clip` raises on chain
exhaustion and a missing verdict is a reject; before, an outage during verify silently fell back to
local checks only, which cannot see watermarks, burnt-in text or AI footage.

**Parallel selection must reach the sequential decision.** Screens finish in any order, but the
decision loop walks candidates in ranked order. Two shots must never cut the same footage:
`Selector._claim()` reserves a source window under a lock against both the ledger and in-flight
claims, and ledger writes share a lock. Mixkit stays one call at a time (`_MIXKIT_LOCK`).

**Chain exhaustion is not a footage reject.** When every judge link is down, `screen()` raises
`JudgeError` instead of returning `None`, and the shot is recorded as `unavailable` and left
unfilled. Falling back to stock there would spend the per-shot stock fallback on *our* outage and
silently break the YouTube-first rule; the shot stays a blocker for `redo --shot N` instead.

**Hard rejects can never be overridden by a high score** (`benchmark.py`). A clip with 95 relevance
is still rejected for a presenter, watermark, burnt-in text, B&W, or AI-generated footage. Upload
date is only ever a *soft* penalty - never a hard reject.

**The AI judge is a provider chain, not one key** (`llm/router.py`). It mirrors the old `Gemini`
interface (`json`, `json_youtube`, `json_video_file`), so call sites do not know which provider
answered. Links come from `judge_chain` in `config/default.yaml`; `provider` is a key in the
provider config at `$HEART_CONFIG`, or `_direct` for native google-genai + `GEMINI_API_KEY`.
- `$HEART_CONFIG` belongs to **another project**. Read it, never write or copy it.
- A failing link is circuit-broken for 5 min. Failures include HTTP errors, empty `content`, and a
  reply truncated before any answer - all observed in practice. A 429 "exceeded your current
  quota" is a daily cap and trips the link for an hour. When every link fails, the chain waits
  `judge_patience` seconds once for a recoverable link before the shot is given up.
- OpenAI-compatible endpoints have no enforced schema, so `_schema_hint()` states the shape in the
  prompt and `_extract_json()` repairs fences, trailing commas, Python literals and truncation.
- Vendor "supports images" metadata is unreliable. **Probe a model live before adding it.**

**Local clips are judged as a tiled frame grid** (`inspect/grid.py`), not uploaded video - every
vision model accepts images, almost none accept video.

**`ai_or_animated` comes from a dedicated single-question check**, not the 12-field verdict. On a
5-clip benchmark the combined verdict erred in both directions (a real Tokyo skyline called
synthetic at 95% confidence); the dedicated check scored 4/4. See `Judge.confirm_ai_generated`.

**`checks.static_overlay_corners` is a soft hint and must not hard-reject.** Measured: a real
translucent watermark scored 0.004 (missed), a clean tripod shot of a lit street scored 1.000
(false positive). Real overlays are caught by the AI judge's `text_overlay` / `watermark_or_logo`.

**Mixkit has no API** - it runs through the `mixkit-dl` CLI's JSON contract (`--json --quiet`,
check the exit code first). Multi-word queries *are* real searches (the slug redirects to
`/discover/`), so each shot walks a keyword ladder from its own query to a broad term. Exit `4`
(rate-limited) and `3` (disk guard) drop Mixkit for the whole run. Run it **one call at a time**,
keep `--delay >= 1`, and discard portrait clips (it mixes them in; the JSON has no orientation
field, so probe with ffprobe).

### Configuration

`config/default.yaml` holds all thresholds, weights, shot length, fallback order and the judge
chain - no secrets. `.env` holds keys plus `HEART_CONFIG`. Changing providers or their order is a
config edit, not a code change.

### Output (`projects/<slug>/`)

`output/final.mp4`, `output/sources.csv` (credits with source timestamps and licence),
`report.html`/`report.json`, and `ledger.jsonl` - every accept/reject decision with its reasons.
The ledger is the audit trail; QA cross-checks the timeline against it.

## Testing

All tests are offline and deterministic; ffmpeg-dependent ones skip when it is absent. When
changing judge or benchmark behaviour, encode the *measured* rule in a test rather than the
intended one - several thresholds here were corrected after measuring real clips, and the tests
record those findings.
