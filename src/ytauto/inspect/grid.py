"""Frame-grid extraction: turn a short clip into ONE tiled JPEG for vision models.

Every OpenAI-compatible vision model accepts images; almost none accept video. A 3x2 tile of
sequential frames costs ~10x less than inline video and was verified to catch both burnt-in text
("ON RIDE") and wrong content (relevance 0-10 vs 90-95 for a correct clip).
"""
from __future__ import annotations

import base64
import subprocess
from pathlib import Path

from ..log import log

# Keep the per-tile width high enough that burnt-in text stays readable after tiling.
TILE_W = 512


def extract_grid(clip: Path, cols: int = 3, rows: int = 2, fps: float = 1.2,
                 tile_width: int = TILE_W) -> bytes | None:
    """Return a tiled JPEG of `cols*rows` frames sampled at `fps`, or None if ffmpeg fails."""
    vf = f"fps={fps},scale={tile_width}:-1,tile={cols}x{rows}"
    cmd = ["ffmpeg", "-v", "error", "-i", str(clip), "-vf", vf,
           "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        log.warning("grid extraction failed for %s: %s", clip.name, str(e)[:160])
        return None
    # A too-small payload means ffmpeg produced no real tile (e.g. clip shorter than 1/fps).
    if len(r.stdout) < 2000:
        log.warning("grid extraction produced no frames for %s: %s",
                    clip.name, r.stderr[:160].decode("utf8", "replace"))
        return None
    return r.stdout


def extract_grid_data_url(clip: Path, **kw) -> str | None:
    """Frame grid as a `data:image/jpeg;base64,...` URL ready for an OpenAI-style image_url part."""
    raw = extract_grid(clip, **kw)
    if raw is None:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode()


GRID_HINT = ("These are {n} sequential frames from one short video clip, tiled {cols}x{rows} "
             "(left-to-right, then top-to-bottom). Judge the clip as a whole.")


def grid_hint(cols: int = 3, rows: int = 2) -> str:
    return GRID_HINT.format(n=cols * rows, cols=cols, rows=rows)
