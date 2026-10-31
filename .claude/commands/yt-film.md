---
description: Make a finished documentary film from a transcript and a voice-over (footage, review, render, audit)
argument-hint: <transcript file> <voice-over file> "<title>" ["<next video title>"]
---

Make a film from these arguments: $ARGUMENTS

(transcript file, voice-over file, title, optional next-video title for the end card. If one of the first three is missing, ask me for it and do nothing else.)

Run these three steps in order and keep a clock-time record of each (start, end, minutes):
1. Use the `yt-film-maker` subagent with stage "footage" and the arguments. Wait for it to finish (1 to 2 hours).
2. Use the `yt-video-reviewer` subagent with the title, transcript, voice-over and the config `config\<slug>.local.yaml` the first step created. A failure here is reported, not fatal.
3. Use the `yt-film-maker` subagent with stage "render" and the same arguments.

Then report in short, simple words: the final video path, total time and time per stage, every check that failed, and the shots that still show the wrong footage (with their time in the film). Write the same record to `PRODUCTION-LOG.md` in the voice-over's folder.
