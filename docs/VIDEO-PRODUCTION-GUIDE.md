# Video production guide (read before making a documentary-style video)

For any AI agent (Claude, Nibedita, others). Based on the Alaska and Philippines videos.
Pipeline: transcript + voice-over -> footage (ytauto, in `E:\1-MARKETING\youtube-automation\yt-auto-fast`) -> Remotion composition (`src/<name>/`) -> render -> review.

## Rules
1. Never edit the original `E:\1-MARKETING\youtube-automation\yt-auto`. Work in `yt-auto-fast` or a copy.
2. Never overwrite existing assets or compositions. New video = new folder `src/<name>/` and `public/<name>/`. Only add one line to `src/Root.tsx`.
3. Do not stop at "rendered". Always review frames and fix problems before saying done.
4. Report what is weak instead of hiding it.

## Setup facts
- Run ytauto with the venv: `.venv/Scripts/python -c "from ytauto.cli import app; app()" run <transcript.json> --title "<topic>" --vo <voice.mp3> --wait` (there is no `ytauto` on PATH). Set `PYTHONUTF8=1`.
- Transcript = JSON list of `{start, end, text}`. Merge the voice-over's lines into sentences of at least 6 s.
- Word timing: use `ytauto.finish.heard_words` + `align_words` (faster-whisper, audio decoded by ffmpeg). Save as `words.json` for the captions. Spelling comes from the transcript, timing from the audio.
- ElevenLabs sound effects: `POST https://api.elevenlabs.io/v1/sound-generation?output_format=mp3_44100_128`, body `{text, duration_seconds, prompt_influence}`. Key is in `.env` (`ELEVENLABS_API_KEY`). Never print the key.
- Asset library `E:\video-assets` is read-only: copy what you need into `public/<name>/`. Check `CLAUDE.md` and `index.json` there.
- Music shorter than the video: join two tracks with `acrossfade` in ffmpeg.
- Shell: the Bash tool blocks long `sleep`. Wait with an `until grep ...; do sleep 15; done` loop run in the background.
- Large heredocs in Bash can break. Use the Write tool for code files.

## Problems seen and fixes
| Problem | Fix |
|---|---|
| Footage drifts to the wrong place (a Bangladesh river for Alaska) | Planner pins a `setting` on every brief and prefixes queries with the topic (`Brief.setting`) |
| No footage exists for one idea | `brief_overrides` in `project.yaml` for that shot (new subject, queries) |
| Same clip repeated after `redo` | Overlap-based tried-window check (already in `select.py`) |
| Clip shows scuba gear for "no tank", a surfer for sea nomads, pine forest for a tropical mountain | Review a contact sheet of ALL shots (one thumbnail each, numbered) before rendering. Swap bad ones by copying another clip over the file |
| Burnt-in text or logos in clips | Local OCR check (`rapidocr`, install with `--no-deps` plus `pyclipper shapely six`) |
| `opencv-python` 5 breaks face check (`CascadeClassifier` missing) | Use `opencv-python-headless>=4.9,<5`. Run `ytauto doctor` first |
| Whisper alignment silently falls back | Audio must be decoded with ffmpeg (`align.load_audio`), not PyAV |
| Gemini slow or rate-limited | Keep `judge_chain` with the ChatGPT `_heart` link and hedging (`judge_hedge_after: 8`). ChatGPT cannot do URL-watching mode |
| Purple sunsets in the grade | Blend in `gbrp`, not YUV |
| Music too loud or too quiet, audio at 96 kHz | Music about -14 dB (ffmpeg path) or volume 0.15-0.3 (Remotion). Force 48 kHz. Normalise the final mix to about -16 LUFS with `loudnorm` |
| Clips exactly shot length: loop jump during a fade | Pad each clip with `tpad=stop_mode=clone:stop_duration=0.6`, scale to 1920x1080, 30 fps, before Remotion |
| Contact sheet looks banded or blotchy | Often only the downscale. Check a full-size crop before "fixing" |
| AI check rejects real drone, dive and wildlife footage (56 "AI" rejects reviewed: nearly all real) and one flag banned the whole video | A video is banned for "AI" only after 2 flags (`VIDEO_DEFECTS` in `select.py`). Metadata does not reveal AI videos (0 of 56 caught), so do not filter on titles or tags |
| ffmpeg `drawtext` fontconfig errors | Harmless noise. Labels still work |

## Speed (built in since 30 Oct 2026, use it)
Philippines, 87 shots, before: footage about 95 min, render about 38 min.
1. Run ytauto with `remotion_export: "<public/name/baked>"` plus `remotion_grain` / `remotion_dust` in a config yaml: clips are exported (1080p30, padded tail, look baked in) while footage is still being chosen, and `words.json` is written in parallel. For an existing project: `ytauto export-remotion <transcript> --out <dir> --vo <voice.mp3> --config <yaml>`.
2. Feed the composition the baked clips and skip the live grade/grain/dust layers (see `PhilippinesBaked`, prop `baked`): 2.5x faster render. Render with `--concurrency=6`; 10 and `--gl=angle` were slower here.
3. Generate ElevenLabs SFX while ytauto runs; nothing depends on the footage.
4. The footage library (`cache/library`) is filled automatically; back-fill old projects with `ytauto library-add projects/<slug> --skip <hand-replaced shot ids>`.
5. A shot replaced by hand in Remotion must NOT be added to the library (use `--skip`).

## Composition conventions (Remotion)
- Data in `data.ts` (shots, stats, places, chapters, caption pages). Look in `Component.tsx`. Footage list in `shots.json`: `{id, clip, t0, t1, dur}` (written by `ytauto export-remotion`).
- Use `Video`/`Audio` from `@remotion/media`, fonts from `@remotion/google-fonts`, light leaks with `<Solid effects={[lightLeak(...)]}>`.
- One style per film: gold `#FFD23F` highlight, teal grade, Montserrat + Playfair italic.
- Soft dissolves inside a scene. A bigger move (whip + light leak + whoosh) only where the story changes place.
- Captions: 4-5 words per page, active word highlighted, bottom centre. Stat cards go top-left or top-right so they never cover captions.
- SFX restraint: ambience far under the voice (0.15-0.3), one whoosh per chapter, ticks on stats.
- Match graphics to the words: find the time of a number in `words.json` and start the card there.

## Review checklist (before saying done)
1. `ffprobe`: duration equals voice length plus tail, 1920x1080, 48 kHz audio.
2. Loudness about -16 LUFS, peak under -1 dB (`ffmpeg -af ebur128=peak=true`).
3. `blackdetect` shows no black gaps except the start and end fades.
4. Contact sheet of every footage clip (numbered), then 16 frames from the final video. Check every graphic's text, spelling and position.
5. Facts on screen match the narration.
6. List weak shots and reused clips in the report.
7. Only invented wording (like the end-card "next video" title) must be flagged to the user.
