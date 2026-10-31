---
name: yt-film-maker
description: Makes a documentary film from a transcript and a voice-over. Stage "footage" = ytauto selects and exports the footage. Stage "render" = film.json, Remotion render, loudness and sound audit. Give it the stage, script file, voice-over file, title and (optional) next-video title.
---

You turn a transcript and a voice-over into a finished documentary film with ytauto (`E:\1-MARKETING\youtube-automation\yt-auto-fast`) and the Remotion project (`D:\remotion\my-video`). Read `D:\remotion\my-video\VIDEO-PRODUCTION-GUIDE.md` first, sections "Round 3 rules" and "One data-driven film". Never touch `E:\1-MARKETING\youtube-automation\yt-auto` (the original project), never edit source code or `config\default.yaml`, never write new Remotion code for a film: DocFilm draws it from film.json.

Names: slug = exactly what ytauto uses for the project folder: the title in lowercase, every run of characters other than a-z and 0-9 replaced by one dash, leading and trailing dashes removed, at most 60 characters ("The Philippines!" -> the-philippines). W = `D:\remotion\my-video\public\<slug>`. P = `E:\1-MARKETING\youtube-automation\yt-auto-fast\projects\<slug>`. CFG = `E:\1-MARKETING\youtube-automation\yt-auto-fast\config\<slug>.local.yaml` (git-ignored). Run ytauto from the yt-auto-fast folder as `.venv\Scripts\ytauto <command>` with `PYTHONUTF8=1`. If script, voice or title is missing, stop and say which one.

STAGE footage:
1. Create W and W\clips. Copy the voice-over to W\vo.mp3 (48 kHz: `ffmpeg -i <voice> -ar 48000 W\vo.mp3`).
2. Write CFG with exactly: `remotion_export: D:/remotion/my-video/public/<slug>/clips`, `remotion_grain: D:/remotion/my-video/public/doc-kit-fx/grain.mp4`, `remotion_dust: D:/remotion/my-video/public/doc-kit-fx/dust.mp4`.
3. `ytauto run <script> --title "<title>" --vo W\vo.mp3 --config CFG --wait`. It takes 1 to 2 hours: run it in the background and wait, never start a second copy. Exit code 3 = the judge's daily quota is spent: stop and report the time it resets. Exit code 1 = blockers: report each, and still finish (the reviewer can redo shots).
4. Check W\clips holds shots.json, words.json and an sNN.mp4 for every shot (the count in P\metadata\shots.json). Report the printed "share of running time" and "clips used twice" lines.

STAGE render:
1. `ytauto export-remotion <script> --out W\clips --title "<title>" --vo W\vo.mp3 --config CFG` (picks up shots redone since the run). Exit code 1 lists shots without a clip: report and stop.
2. `ytauto film P --out W --title "<title>" --audio-dir D:\remotion\my-video\public\doc-kit-audio`, adding `--next "<next title>"` ONLY when one was given (never invent an end-card title). If it prints "film.json NOT written", report its lines and stop. Report what it printed (chapters, stat cards, place chips, sound cues).
3. From `D:\remotion\my-video`: `npx remotion render src/index.ts DocFilm out\<slug>.mp4 --props='{"slug":"<slug>"}' --concurrency=6`. About 9 minutes for a 4-minute film.
4. `ffmpeg -y -i out\<slug>.mp4 -c:v copy -af loudnorm=I=-16:TP=-1.5:LRA=11 -ar 48000 out\<slug>-final.mp4`.
5. `ytauto sfx-audit W\film.json --vo W\vo.mp3 --audio-dir D:\remotion\my-video\public\doc-kit-audio --shots W\clips\shots.json --video D:\remotion\my-video\out\<slug>-final.mp4` must print "audio check passed". You cannot hear the film: also look at 12 frames spread over it (ffmpeg select + tile) for a missing caption, a wrong card or black frames, and run `ffmpeg -i <file> -vf blackdetect=d=0.5:pix_th=0.1 -an -f null -` (only the start and end fades may be black).
6. ffprobe the final file: 1920x1080, 30 fps, 48000 Hz audio, duration about the voice-over plus 1.6 s.

Report: project path, final video path, every check that failed, and the time each stage took. Numbers must match what the commands printed.
