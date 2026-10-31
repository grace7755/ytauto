"""Export selected clips for a Remotion composition, as soon as each shot is accepted.

Measured on the Philippines video: after selection, the 87 clips were padded and re-encoded one by
one (minutes), and the Remotion render then spent ~38 min, much of it drawing two full-screen video
overlays (grain, dust) with CSS blend modes plus colour filters on every frame. Remotion's own
performance guide says to replace such GPU-bound effects with precomputed media. So each clip is
written once, in a background thread while selection goes on:

- 1920x1080, 30 fps, no audio, with a cloned tail (`pad` seconds) so a transition overlapping the
  next shot never shows the clip jump back to its start;
- optionally the film look baked in (`bake`): the same grade, vignette, grain and dust the
  composition used to draw live, so the composition only places footage and graphics.

`shots.json` lists `{id, clip, t0, t1, dur}` for the composition; `words.json` holds word timings.
"""
from __future__ import annotations

import json
import subprocess
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .log import log
from .models import Shot
from .timing import stage


@dataclass
class Look:
    """The baked film look. Defaults match the Philippines composition's CSS layers."""
    contrast: float = 1.07
    saturation: float = 1.18
    brightness: float = 0.02
    vignette: float = 0.42          # ffmpeg vignette angle (radians); larger = stronger
    grain: Path | None = None       # overlay video, blended "overlay"
    grain_opacity: float = 0.18
    dust: Path | None = None        # particles on black, blended "screen"
    dust_opacity: float = 0.22


def _filter(look: Look | None, pad: float, n_inputs: int) -> str:
    base = (f"[0:v]tpad=stop_mode=clone:stop_duration={pad},"
            "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,fps=30,setsar=1")
    if look is None:
        return base + ",format=yuv420p[out]"
    # teal shadows / warm highlights, then the vignette: the CSS soft-light tints of the composition
    v = (base + f",eq=contrast={look.contrast}:saturation={look.saturation}:brightness={look.brightness},"
         "colorbalance=rs=-0.03:gs=0.0:bs=0.04:rh=0.03:gh=0.01:bh=-0.03,"
         f"vignette=angle={look.vignette},format=gbrp[v0]")
    parts, cur, i = [v], "v0", 1
    for src, mode, op in ((look.grain, "overlay", look.grain_opacity), (look.dust, "screen", look.dust_opacity)):
        if src is None or i >= n_inputs:
            continue
        parts.append(f"[{i}:v]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,fps=30,"
                     f"setsar=1,format=gbrp[o{i}]")
        # gbrp: blending in YUV shifts hues (seen: a purple sunset), see finish.py
        parts.append(f"[{cur}][o{i}]blend=all_mode={mode}:all_opacity={op}:shortest=1[v{i}]")
        cur, i = f"v{i}", i + 1
    parts.append(f"[{cur}]format=yuv420p[out]")
    return ";".join(parts)


def export_clip(src: Path, dst: Path, *, pad: float = 0.6, look: Look | None = None, crf: int = 18) -> Path:
    """Write `src` as a composition-ready clip at `dst` (atomic: a half-written file never appears)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src)]
    overlays = [p for p in ((look.grain, look.dust) if look else ()) if p is not None]
    for p in overlays:
        cmd += ["-stream_loop", "-1", "-i", str(p)]
    # the overlays loop forever: the clip (plus its cloned tail) decides the length
    dur = _duration(src) + pad
    cmd += ["-filter_complex", _filter(look, pad, 1 + len(overlays)), "-map", "[out]", "-t", f"{dur:.3f}",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-movflags", "+faststart"]
    tmp = dst.with_name(dst.stem + ".part.mp4")
    subprocess.run(cmd + [str(tmp)], check=True, capture_output=True, text=True)
    tmp.replace(dst)
    return dst


def _duration(p: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                         capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def clip_name(shot_id: int) -> str:
    return f"s{shot_id:02d}.mp4"


class Exporter:
    """Background export of accepted clips while selection continues. `submit` returns at once;
    `finish` waits for the queue and writes `shots.json`. Failures are logged, never raised into
    the selection - the run's own output does not depend on the export."""

    def __init__(self, out_dir: Path, look: Look | None, pad: float = 0.6, workers: int = 2):
        self.dir, self.look, self.pad = out_dir, look, pad
        self._pool = ThreadPoolExecutor(workers, thread_name_prefix="export")
        self._jobs: dict[int, Future] = {}
        self._src: dict[int, str] = {}
        self._lock = threading.Lock()
        self.dir.mkdir(parents=True, exist_ok=True)
        self._stamp_path = self.dir / ".exported.json"
        try:
            self._stamps: dict[str, str] = json.loads(self._stamp_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._stamps = {}

    def _stamp(self, src: Path) -> str:
        st = src.stat()
        return f"{src.resolve()}|{st.st_size}|{int(st.st_mtime)}|{'baked' if self.look else 'plain'}"

    def submit(self, shot: Shot, clip_path: str | None):
        if not clip_path or not Path(clip_path).exists():
            return
        src, dst = Path(clip_path), self.dir / clip_name(shot.id)
        stamp = self._stamp(src)
        with self._lock:
            self._src[shot.id] = str(src)
            if dst.exists() and self._stamps.get(dst.name) == stamp:
                return                                  # already exported from this very clip
            self._jobs[shot.id] = self._pool.submit(self._one, shot.id, src, dst, stamp)

    def _one(self, sid: int, src: Path, dst: Path, stamp: str):
        try:
            # a private temp file per job: a QA re-selection can queue a newer clip for the same
            # shot while the older one is still encoding
            tmp = dst.with_name(f"{dst.stem}.{threading.get_ident()}.mp4")
            with stage("export (remotion)"):
                export_clip(src, tmp, pad=self.pad, look=self.look)
            with self._lock:
                if self._src.get(sid) != str(src):      # superseded meanwhile: keep the newer one
                    tmp.unlink(missing_ok=True)
                    return
                tmp.replace(dst)
                self._stamps[dst.name] = stamp
                self._stamp_path.write_text(json.dumps(self._stamps, indent=1), encoding="utf-8")
        except subprocess.CalledProcessError as e:
            log.warning("export of %s failed: %s", src.name, (e.stderr or "")[:200])

    def finish(self, shots: list[Shot]) -> Path:
        for f in list(self._jobs.values()):
            f.result()
        self._pool.shutdown(wait=True)
        rows = []
        for sh in shots:
            p = self.dir / clip_name(sh.id)
            if p.exists():                              # also shots exported by an earlier run
                row = {"id": f"s{sh.id:02d}", "clip": p.name, "t0": round(sh.start, 3),
                       "t1": round(sh.end, 3), "dur": round(_duration(p), 2)}
                if "_still_" in Path(self._src.get(sh.id, "")).name:     # made from a photograph (stills.py)
                    row["kind"] = "still"
                    row["label"] = (sh.brief.subject if sh.brief else "") or ""
                rows.append(row)
        out = self.dir / "shots.json"
        out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        log.info("remotion export: %d clips + shots.json in %s", len(rows), self.dir)
        return out


def write_words(vo: Path, segments, out: Path) -> Path:
    """Word timings for captions (faster-whisper times, transcript spelling): `[{text, start, end}]`."""
    from . import finish as F

    words = F.align_words(segments, F.heard_words(vo))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([{"text": w.text, "start": round(w.start, 2), "end": round(w.end, 2)}
                               for w in words]), encoding="utf-8")
    return out
