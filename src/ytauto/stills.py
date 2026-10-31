"""A still picture becomes a slow-moving clip (push-in, pull-out or pan) in the project spec, so the
rest of the pipeline - QA, export to Remotion, assembly - treats it like any other clip.

The picture is cropped to 16:9 and enlarged 3x before `zoompan`: zoompan moves in whole pixels, and
on the output-sized picture a slow zoom visibly steps; on the 3x picture each step is a third of an
output pixel and the move reads as smooth."""
from __future__ import annotations

from pathlib import Path

from .assemble import run_ffmpeg
from .config import Settings

MOTIONS = ("push", "pull", "pan")
ZOOM = 0.12             # how far a push-in or pull-out travels (12%)


def motion_for(shot_id: int) -> str:
    """Rotate the three moves so neighbouring stills never move the same way."""
    return MOTIONS[shot_id % len(MOTIONS)]


def _zoompan(motion: str, frames: int, w: int, h: int, fps: int) -> str:
    n = max(frames - 1, 1)
    if motion == "pull":
        z = f"{1 + ZOOM}-{ZOOM}*on/{n}"
        x, y = "(iw-iw/zoom)/2", "(ih-ih/zoom)/2"
    elif motion == "pan":
        z = f"{1 + ZOOM}"
        x, y = f"(iw-iw/zoom)*on/{n}", "(ih-ih/zoom)/2"
    else:
        z = f"1+{ZOOM}*on/{n}"
        x, y = "(iw-iw/zoom)/2", "(ih-ih/zoom)/2"
    return f"zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={w}x{h}:fps={fps}"


def motion_clip(s: Settings, image: Path, out: Path, dur: float, motion: str) -> Path:
    """Write `dur` seconds (exact frame count) of `image` with `motion` to `out` (video only)."""
    frames = max(2, round(dur * s.fps))
    big_w, big_h = s.width * 3, s.height * 3
    vf = (f"scale={big_w}:{big_h}:force_original_aspect_ratio=increase:flags=lanczos,crop={big_w}:{big_h},"
          f"{_zoompan(motion, frames, s.width, s.height, s.fps)},setsar=1,format=yuv420p")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.mp4")
    run_ffmpeg(["-i", str(image), "-vf", vf, "-frames:v", str(frames), "-an", "-c:v", "libx264",
                "-preset", "medium", "-crf", str(s.crf), "-movflags", "+faststart", str(tmp)])
    tmp.replace(out)
    return out
