"""ffmpeg assembly: frame-exact normalised cuts, per-segment concat, optional crossfades, VO mux."""
from __future__ import annotations

import subprocess
from pathlib import Path

from .config import Settings
from .log import log
from .models import Shot
from .state import Project


class FFmpegError(RuntimeError):
    pass


def run_ffmpeg(args: list[str], timeout: int = 1800):
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise FFmpegError(r.stderr[-800:])


CORNERS = ("top-left", "top-right", "bottom-left", "bottom-right")


def corner_crop(corner: str, zoom: float) -> str:
    """ffmpeg crop that keeps the part of the frame away from `corner`, `zoom`x closer: a watermark
    that sits within the outer (1 - 1/zoom) of that corner ends up outside the picture."""
    x = "0" if corner.endswith("right") else f"iw-iw/{zoom}"
    y = "0" if corner.startswith("bottom") else f"ih-ih/{zoom}"
    return f"crop=iw/{zoom}:ih/{zoom}:{x}:{y}"


def vf_chain(s: Settings, stretch: float = 1.0, crop: str | None = None) -> str:
    f = []
    if crop:
        f.append(crop)
    if abs(stretch - 1.0) > 1e-3:
        f.append(f"setpts=PTS*{stretch:.5f}")
    f += [f"scale={s.width}:{s.height}:force_original_aspect_ratio=increase:flags=lanczos",
          f"crop={s.width}:{s.height}", f"fps={s.fps}", "setsar=1", "format=yuv420p"]
    return ",".join(f)


def cut_clip(s: Settings, src: Path, start: float, dur: float, out: Path, avail: float | None = None,
             crop: str | None = None) -> Path:
    """Cut `dur` seconds starting at `start` into the project spec (video only, exact frame count).
    If the source window is a little short (`avail` < dur, within 10%), slow it down to fit."""
    frames = max(1, round(dur * s.fps))
    stretch = 1.0
    read = dur
    if avail is not None and avail < dur:
        stretch = dur / avail
        read = avail
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.mp4")

    def encode(stretch: float, read: float):
        run_ffmpeg(["-ss", f"{start:.3f}", "-i", str(src), "-t", f"{read + 0.2:.3f}", "-an",
                    "-vf", vf_chain(s, stretch, crop), "-frames:v", str(frames),
                    "-c:v", "libx264", "-preset", "medium", "-crf", str(s.crf), "-movflags", "+faststart", str(tmp)])

    encode(stretch, read)
    # `-frames:v` caps the count but cannot invent frames: when the source section runs out early
    # (container duration overstated, or a keyframe-aligned section download ending short) the
    # clip silently comes out short and the whole timeline drifts. Measured: one 144-frame clip in
    # a 150-frame slot made a 60.0 s video 59.8 s and failed QA. Stretch to fit if it is close.
    got = count_frames(tmp)
    if 0 < got < frames:
        fix = stretch * frames / got
        if fix > MAX_STRETCH:
            tmp.unlink(missing_ok=True)
            raise FFmpegError(f"source ran out: {got}/{frames} frames, would need {fix:.2f}x slow-down")
        log.info("%s: source ran out at %d/%d frames; slowing it %.3fx to fill the slot",
                 out.name, got, frames, fix)
        encode(fix, read)               # needs dur/fix s of source, and dur/fix <= read
        got = count_frames(tmp)
        if got < frames:
            tmp.unlink(missing_ok=True)
            raise FFmpegError(f"clip still short after stretch: {got}/{frames} frames")
    tmp.replace(out)
    return out


MAX_STRETCH = 1.15      # beyond ~15% slow-motion reads as an effect, not as normal footage


def count_frames(path: Path) -> int:
    """Exact number of video frames (packets) in `path`; 0 if ffprobe cannot read it."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                        "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, timeout=120)
    try:
        return int(r.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return 0


def _concat(files: list[Path], out: Path):
    lst = out.with_suffix(".txt")
    lst.write_text("".join(f"file '{f.resolve().as_posix()}'\n" for f in files), encoding="utf-8")
    run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)])
    lst.unlink(missing_ok=True)


def tail_for(s: Settings, shots: list[Shot], i: int) -> float:
    """Extra tail a clip needs so a crossfade into the next segment keeps the timeline aligned."""
    if s.crossfade > 0 and i + 1 < len(shots) and shots[i + 1].segment != shots[i].segment:
        return s.crossfade
    return 0.0


def assemble(s: Settings, proj: Project, shots: list[Shot], clip_paths: dict[int, Path],
             vo: Path | None, out: Path) -> Path:
    """clip_paths[shot.id] must have been cut with duration shot.duration + tail_for(...)."""
    seg_files: list[tuple[Path, float]] = []
    work = proj.path("normalized")
    groups: dict[int, list[Shot]] = {}
    for sh in shots:
        groups.setdefault(sh.segment, []).append(sh)
    for seg, sl in groups.items():
        seg_out = work / f"seg_{seg:03d}.mp4"
        _concat([clip_paths[x.id] for x in sl], seg_out)
        tail = tail_for(s, shots, shots.index(sl[-1]))
        seg_files.append((seg_out, sum(x.duration for x in sl) + tail))

    video = work / "video_nocut.mp4"
    cf = s.crossfade
    if cf > 0 and len(seg_files) > 1:
        args, flt, cum, prev = [], [], 0.0, "[0:v]"
        for p, _ in seg_files:
            args += ["-i", str(p)]
        for k in range(1, len(seg_files)):
            cum += seg_files[k - 1][1] - cf
            lab = f"[v{k}]"
            flt.append(f"{prev}[{k}:v]xfade=transition=fade:duration={cf}:offset={cum:.3f}{lab}")
            prev = lab
        run_ffmpeg([*args, "-filter_complex", ";".join(flt), "-map", prev, "-c:v", "libx264", "-crf", str(s.crf),
                    "-pix_fmt", "yuv420p", "-r", str(s.fps), str(video)])
    else:
        _concat([p for p, _ in seg_files], video)

    total = shots[-1].end
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.mp4")
    if vo:
        run_ffmpeg(["-i", str(video), "-i", str(vo), "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k", "-af", "apad", "-t", f"{total:.3f}",
                    "-movflags", "+faststart", str(tmp)])
    else:
        run_ffmpeg(["-i", str(video), "-c:v", "copy", "-t", f"{total:.3f}", "-movflags", "+faststart", str(tmp)])
    tmp.replace(out)
    log.info("rendered %s", out)
    return out
