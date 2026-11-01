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
.venv\Scripts\ytauto redo   script.txt --title "My Video" --shot 3,7,12   # all in one process
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

**Storyboard `overlay_tiles` only reorders, never rejects.** Thumbnails are ~160 px wide. Measured on
39 past downloads (`scripts/measure_overlay_screen.py`): it never flagged a clean clip (0/18), but made
only 3 of 21 videos that verify later rejected for a logo or captions `overlay_suspect`. So flagged
tiles are downloaded last and an all-flagged video waits until the round's clean candidates were
tried; the hard reject stays with verify on the full-resolution clip.

**Judge-free work is prefetched** (`Selector.warm`, `prefetch_ahead: 3`, started from `select_all`): while
workers wait on the judge, a background thread does upcoming shots' round-0 search, metadata,
storyboard fetch and CLIP tile embeddings, all cached on disk. It makes no judge calls and no
decisions, so a failure there is ignored. CLIP runs 6 threads under one lock (measured 37 -> 18
ms/tile versus 2 threads), not two slow sessions fighting for cores.

**Section downloads are stream-copied** (`download_section`, no `force_keyframes_at_cuts`). The
copied frames line up with a re-encoded download to the frame, so window maths is unchanged; the
re-encode only added a lossy generation. Download time (~12 s) is the transfer, not encoding.
Storyboard windows are capped at 8 s (11 s downloaded): on 130 past sections a cut-free 4.5 s window
still existed in 116/116 at 11 s, so the longer sections were only transfer time.

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
- The chain starts with ChatGPT through HeartCode (`_heart`): GPT Luna for every task, GPT Terra when Luna cannot give a valid answer, then the Gemini links (see Round 6; GPT Sol is refused by `JudgeLink`). `_direct` links (the user's own free-tier key, straight to Google) are the Gemini backup. Free quota is per
  project **per model**, so each Gemini model is its own link, and `Gemini(fallback=False)` calls
  exactly `link.model` - its old internal model-hopping spent quotas the chain was not counting.
  A daily-quota 429 ("requests per day") trips a link until midnight Pacific; a per-minute 429
  says "exceeded your current quota" too and only trips for its `retryDelay`.
  A 503 "high demand" / 504 (dozens per run on 2026-10-29) moves on to the next link with no sleep and
  trips the link for only 60 s, not 5 minutes.
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

## Production round 3 (30 Oct 2026)

- Shot order in `Selector.select`: sibling -> library -> (`generic` shots with `stock_first`: stock, YouTube; else YouTube, stock) -> `_crop_logo` (corner-logo-only footage, re-cut 1.15x) -> `_still` (licensed photograph, `stills.py` makes the push/pull/pan clip; relevance at least `still_min_relevance`) -> `_reuse` -> weak match. Clips are never longer than `shot_max` (5 s).
- `Selector._claim`: a YouTube video gives any number of non-overlapping sections (`max_shots_per_source` 0 = no limit), every other provider (stock, `still_*`) is picked once. A second use of the same footage is `Selector._reuse` (last resort after `_still`; `reuse_max_uses` 2, `reuse_min_gap` 60 s, never a still, judged again; notes carry `REUSED footage`, which `qa._declared_reuse` accepts as not a duplicate). Stills are never added to the library.
- New commands: `review-shots <project>` (AI compares each shot's frame with its narration), `sfx-plan`, `sfx-audit` (`sfx.py`: levels set from measured files; audit checks level under speech, cut sync, spacing, mix loudness/peak).
- `Decision.credit` holds a still's author; `report.py` writes `output/credits.txt` (CC-BY only) and `shares` / `reused_clips` / `reuse_problems` in `report.json`.
- The planner cache key is `plan-shots-v4` (the `generic` field); an old cached plan is re-planned.
- Tests: `tests/test_production3.py` (all offline; two need ffmpeg).

## Round 4 (31 Oct 2026)

- `benchmark.local_rejects` is the free-check half of `evaluate`; `_try_file` never sends a clip that fails it to the judge (note carries `[AI check deferred]`).
- Per-shot download budget: `Selector._downloads` / `_over_budget`, base taken at the start of each `select()`.
- `lastresort.py` is the `LastResort` mixin of `Selector` (`_keep_for_crop`, `_crop_logo`, `_reuse`, `_still`, `_judge_made_clip`). Tests patch a name on both modules (`_patch` in `tests/test_production3.py`).
- `film.py` + `ytauto film`: builds `film.json` (see the module docstring). `sfx.load` accepts a `film.json` as well as an `sfx.json`.
- Remotion side: `D:\remotion\my-video\src\doc-kit\DocFilm.tsx` (registered once in `Root.tsx` as `DocFilm`, `--props='{"slug":"<slug>"}'`).


## Round 5: speed (31 Oct 2026)

Measured baseline (5-guns film, 141 min): selection 76, 12 one-at-a-time redos 39, two Remotion renders 18.
- `redo --shot 3,7,12` runs every shot in ONE process (one `prepare`, `shot_workers` side by side, one export, one QA) and writes `logs/timings-redo-<n>.json` (one file per redo).
- `ffmpeg_render: auto` (default): with `remotion_export` set, `run`/`redo` skip the ffmpeg `final.mp4` and its QA, which the Remotion workflow never used. `qa.run_qa_clips` keeps the ledger, duplicate and contact-sheet checks on the clips. `ffmpeg_render: always` brings the old behaviour back (needed by `ytauto finish`).
- `judge_spill_after: 3`: a link whose per-minute bucket would block longer is skipped for the next live link (`Quota.acquire(max_wait=)` returns None); a call waits only when every link is busy. Blocking waits show as `judge wait` in `timings.json`. 0 = old behaviour.
- `storyboard_face_check` (default OFF): local Haar on the picked storyboard tiles; a face tile goes last and an all-face video is `face_suspect` (held back like `overlay_suspect`). Never a reject. Turn on only if `scripts/measure_face_tiles.py` says so on your ledgers.
- `planner._b_roll_queries` strips presenter words (review, unboxing, vs, tutorial...) from YouTube queries; the prompt asks for b-roll wording for objects/products. Cache key stays `plan-shots-v4` so projects in progress keep their briefs.
- `llm/codex.py` + `provider: _codex`: OpenAI's Codex CLI (`codex exec`, signed in with `codex login`, ChatGPT plan) as a judge link, written from `codex exec --help` of codex-cli 0.166.0. Prompt on stdin (never positional: `-i` swallows it), pictures as `--image` files, answer from `-o`. A plan-limit message becomes a quota 429, so the run pauses. OFF by default (a commented line in `default.yaml`): Plus allows few messages per few hours, so measure calls per film first. HeartCode (`_heart`) is unchanged. Tests: `tests/test_codex.py`.
- `transcript.join_cut_sentences` (run by `load_transcript`): a cue that stops mid-sentence (subtitle tools cut at a time limit, e.g. ElevenLabs 6 s) is joined to the next. Measured on a 7:33 voice-over: 99 cues / 130 shots (87 under 3 s, shortest 0.26 s) became 81 cues / 113 shots (shortest 1.58 s). Guards: only when 70% of cues already end in . ? !, and a join never passes 20 s, so scripts without end punctuation are untouched. Export subtitles with 90 chars / 2 lines / 6 s.
- Tests: `tests/test_redo_batch.py`, spill tests in `tests/test_quota.py`, face-tile tests in `tests/test_storyboard.py`.

## Round 6: video-4 fixes (1 Nov 2026)

Video-4 (Russian firearms, 127 shots) took 4 h 09 min: 1,636 first-pass attempts, half on stock and stills; the judge never waited.
Plan (A: stop wasted searching, B: footage -> photo -> category -> graphic ladder, C: GPT Luna first). Done so far:
- `Mixkit.keyword_ladder` no longer ends on an unconditional "nature": every gun shot fell through to it and one waterfall clip was judged for 55 shots (Mixkit 1 accepted / 227).
- `youtube.is_gone` treats "Sign in to confirm your age" as permanent (cached in `yt_gone`); the bot check ("not a bot") stays transient. One ID was looked up 62 times.
- `Decision.run` ("run" / "redo-<n>") on every ledger row; `logs/timings-redo-<n>.json` and `logs/judge-links*.json` (answers per judge link) per pass.
- Film-wide stock skip (`Selector._skip_irrelevant`, `stock_skip_after: 2`, `stock_skip_relevance: 25`, in `_stock` and `_still`): a stock/still item judged irrelevant on 2+ OTHER shots is not tried again unless its title shares a word with the shot's subject. Replayed on video-4's first pass: 206 of 827 stock/still attempts skipped, 0 accepted clips lost. Note the measured per-shot attempt counts: median 6, max 22 (judged attempts, screen-only rejects excluded), so a per-shot attempt cap is a weak lever (cap 16 saves 3%); the budget belongs in the ladder (footage -> photo -> category -> graphic).
- **ChatGPT first** (`config/default.yaml`): `chatgpt gpt-6-luna` -> `chatgpt gpt-6.1-terra` -> Gemini, `judge_lanes: 1` (every call starts on Luna; Terra is a fallback, not a lane), `heart_concurrency: 4`. Probed live 2026-11-01: `gpt-6-terra` is rejected on a ChatGPT account, `gpt-6.1-terra` answers. `JudgeLink` refuses any model named Sol.
- **Plan-wide usage limit** (`llm/limits.py`, `Judge._trip_plan`): Luna, Terra and Codex share ONE ChatGPT plan limit. A reply like "usage limit reached ... try again at 7:06 PM" pauses every ChatGPT link until that time (remembered across restarts) and the chain continues on Gemini; a hedge never goes to a second ChatGPT model. Before: each link was skipped for 5 min on its own (13 times in video-4). HeartCode's message names no reset time ("Usage limit reached. Review your plan or this app's limit in ChatGPT settings"), so the pause backs off 15 -> 30 -> 60 min and resets after the first successful ChatGPT answer; Codex's message does name the time and that time is used. Calls failing together are one event.
- **Reasoning effort per task** (`judge_effort: {task: low|medium|high}`, `JudgeLink.effort`): tasks are `plan, critic, screen, verify, aicheck, review, facts`. Heart gets `--effort`, Codex `-c model_reasoning_effort=`. Unset = each tool's default (nothing was ever set before). **Measured on 12 finished clips with Luna (`scripts/measure_effort.py`, medium vs low): no time gain (verify 8.1 s -> 8.1 s, aicheck 6.2 s -> 5.8 s; the time is per-call overhead, not reasoning), relevance within 15 points on only 9/12 clips, and the AI-made flag flipped on 4/12. So `judge_effort` stays EMPTY.** `--levels medium medium` measures the noise floor (the same setting twice); it also prints each flipped clip. **The noise floor itself is the finding: the SAME setting twice flipped the AI-made flag on 4 of 24 real clips (17%), so Luna does not answer the same way twice.** Hence `ai_check_votes: 2` (`Judge.confirm_ai_generated`, `verify_still`): a "real" answer stands after one call; an "AI" answer is asked again and the clip is rejected only if the second answer agrees (one extra call, only when the first says AI). Cache keys are `aicheck-v2` / `aicheck-still-v2`, so old single-answer verdicts are re-asked. Without this noise alone would reject about 1 real clip in 12, and `bad_video` would ban the whole video for the film.
- `scripts/measure_clip_gate.py <project>`: offline CLIP check of stock/still files against the shot subject vs the AI's relevance. **Measured on video-4 (822 stock/still attempts): AUC 0.72, and the largest gate that loses no relevant clip skips 1 of 822 AI calls. So there is NO CLIP reject gate before the AI** (same finding as for YouTube candidates: CLIP may order, never reject).
- Image generation from a script works on the ChatGPT plan with no API key: `codex exec -m gpt-6.1-terra "<prompt>"` uses Codex's built-in `image_gen` tool, saves under `~/.codex/generated_images/<session>/` and copies where asked. It cost about 17k tokens for one image: ration it (illustrations only, a small per-film budget).
- `scripts/measure_attempts.py <project> [--before ISO | --run run]`: yield per provider, stock items tried on many shots, what the film-wide stock skip and a per-shot attempt cap would save and lose. Thresholds for those fixes come from its output.

## Working in Claude Code (moved back from Nibedita, 31 Oct 2026)

**Two folders make up the system.** Run Claude Code from this folder; `.claude/settings.json` also opens `D:\remotion\my-video` (the Remotion project, own git repo, own `CLAUDE.md`) and `E:\1-MARKETING\yt-auto-test` (test films).

**Make a film:** `/yt-film <transcript> <voice-over> "<title>" ["<next video title>"]` (`.claude/commands/yt-film.md`). It uses two subagents, `yt-film-maker` (stages footage and render) and `yt-video-reviewer` (`review-shots` + `redo`), in `.claude/agents/`. They replace the Nibedita agents and the `yt-film` workflow, which still exist in Nibedita but are not used. Full steps: `D:\remotion\my-video\VIDEO-PRODUCTION-GUIDE.md`, "One data-driven film".

**Skills** are user-level (`~/.claude/skills`): `remotion-*`, `media-use`, `hyperframes-*`, `yt-script` and the other `yt-*` skills work here as they are. No skill was needed for ytauto itself: this file and the guide carry it.

**Keys and tools:** `.env` here (Gemini, Pexels, Pixabay, Coverr, `HEART_CONFIG`) and `D:\remotion\my-video\.env` (ElevenLabs) are git-ignored; `.venv` and `node_modules` are in place. `ytauto doctor` checks keys, tools and the AI judge links; the ChatGPT judge link needs the `heart` CLI logged in.

**Working style the owner asked for:** reply in simple words, keep answers short, put results first, never create automation scripts without asking, use agents (not scripts) for automation, do not run before/after timing tests unless asked, check a change yourself before saying it works, and say what failed.

**Known limits:** age-restricted YouTube videos are skipped (reading them needs browser cookies, `cookies_from_browser`, not enabled). `ytauto redo` took about 4 minutes per shot when run one shot at a time; pass every shot in one `--shot 3,7,12`. Gemini free limits (15 requests per minute) are the footage-stage ceiling.

**Asset library `E:\video-assets` (VFX, Lottie, SFX, music, overlays).** Read-only while making a film: look up an item in `E:\video-assets\index.json` (path, title, source, licence), then COPY the file into `D:\remotion\my-video\public\<slug>\` (or a shared `public\doc-kit-*` folder) and use it from there with `staticFile`. Never edit or move files in the library during a film. Its own rules are in `E:\video-assets\CLAUDE.md` (read it before adding anything: new downloads go to `_inbox\`, only free commercial licences, `metadata.json` is the source of truth, then `node build-index.mjs`). Light leaks come from `@remotion/light-leaks`, not from the library.
