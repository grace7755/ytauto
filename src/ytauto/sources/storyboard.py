"""YouTube storyboards: the seek-bar preview sprites, used to screen a video without watching it.

Why: screening by having the judge watch the YouTube URL cost ~50 s and ~240k tokens per
candidate (the OpenAI-compatible proxy ignores the start/end offsets, so it watches the whole
video). That exhausted the free Gemini quota within two 12-shot runs, and a 30-40 min video
(~450 shots) would need hundreds of hours of video screened against an 8 h/day cap. Storyboards
are a handful of small JPEGs per video: measured 2026-10-29, ~5 s to fetch and ~12 s to judge as
one numbered contact sheet, and every vision link can read them (not only the YouTube-capable
ones).

They are low resolution (160x90 or 320x180 per tile, one tile every ~10-25 s), so this is a
*pre-filter*: it finds where the subject is. Every clip is still cut from a real HD download and
must pass the local checks, the frame-grid verification and the dedicated AI check.
"""
from __future__ import annotations

import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import requests

from ..log import log

MAX_TILES = 96          # one contact sheet: 8 x 12 tiles; longer videos are subsampled evenly
SHEET_COLS = 8
MIN_TILE_W = 240        # upscale tiny 160x90 tiles a little so the judge can read them
SPEC_MAX_AGE = 2 * 3600  # sprite URLs are signed; refetch them rather than trust an old copy
MIN_AHEAD = 8.0         # a window must hold a ~5 s shot plus room for scene detection
FETCH_WORKERS = 6       # sprite sheets fetched at once; measured ~7.9 s/video when sequential


@dataclass
class Storyboard:
    video_id: str
    jpeg: bytes
    times: list[float]      # times[i] = source time (s) shown in tile i
    step: float             # seconds between consecutive tiles in the sheet
    duration: float

    def window(self, tile: int, max_len: float = 6.0) -> tuple[float, float] | None:
        """Source window around tile `tile`: from just before it to the next tile, but never less
        than MIN_AHEAD past it, capped at `max_len` past it. The cap (6 s, so an 8 s window and an
        11 s download) was measured on 130 downloaded sections: a cut-free 4.5 s window still existed
        in 116 of 116 at 11 s, and in 110 at 8 s - the extra seconds were pure transfer time.
        Measured: a 1-minute video has tiles ~1 s apart, so "to the next
        tile" gave ~3 s windows that screening dropped - losing five 85-95% matches in one run."""
        if not 0 <= tile < len(self.times):
            return None
        t = self.times[tile]
        return max(0.0, t - 2.0), min(self.duration, t + min(max(self.step, MIN_AHEAD), max_len))

    def tile(self, i: int):
        """Tile `i` cut back out of the contact sheet, as a BGR array (None if out of range)."""
        import cv2
        import numpy as np

        if not 0 <= i < len(self.times):
            return None
        sheet = self.__dict__.get("_sheet")
        if sheet is None:
            sheet = cv2.imdecode(np.frombuffer(self.jpeg, np.uint8), cv2.IMREAD_COLOR)
            self.__dict__["_sheet"] = sheet
        if sheet is None:
            return None
        rows = -(-len(self.times) // SHEET_COLS)
        w, h = sheet.shape[1] // SHEET_COLS, sheet.shape[0] // rows
        x, y = (i % SHEET_COLS) * w, (i // SHEET_COLS) * h
        return sheet[y:y + h, x:x + w]

    def nearest_tile(self, t: float) -> int:
        return min(range(len(self.times)), key=lambda i: abs(self.times[i] - t))


def spec_from_info(info: dict) -> dict | None:
    """The highest-resolution storyboard format from a yt-dlp info dict, in a cacheable form."""
    sbs = [f for f in info.get("formats") or [] if str(f.get("format_id", "")).startswith("sb")
           and f.get("fragments") and f.get("rows") and f.get("columns")]
    if not sbs:
        return None
    f = max(sbs, key=lambda f: f.get("width") or 0)
    return {"rows": f["rows"], "cols": f["columns"], "width": f.get("width"), "height": f.get("height"),
            "fragments": [{"url": fr["url"], "duration": fr.get("duration") or 0} for fr in f["fragments"]],
            "fetched": time.time()}


def _tiles(spec: dict, duration: float, timeout: float = 30) -> list[tuple[float, "Image.Image"]]:
    from PIL import Image

    rows, cols = spec["rows"], spec["cols"]
    per = rows * cols
    frags = spec["fragments"]
    # Every sheet but the last is full, so the first sheet gives the true tile interval; the last
    # sheet's `duration` is shorter and its unused slots are padding, not video.
    step = (frags[0]["duration"] or duration / max(1, len(frags))) / per
    def get(frag) -> bytes:
        r = requests.get(frag["url"], timeout=timeout)
        r.raise_for_status()
        return r.content

    # Sheets are independent requests; fetch them together, then slice them in time order.
    with ThreadPoolExecutor(max(1, min(FETCH_WORKERS, len(frags)))) as ex:
        blobs = list(ex.map(get, frags))
    out: list[tuple[float, Image.Image]] = []
    t0 = 0.0
    for blob in blobs:
        img = Image.open(io.BytesIO(blob)).convert("RGB")
        tw, th = img.width // cols, img.height // rows
        for k in range(per):
            ts = t0 + k * step
            if ts >= duration:
                break
            y, x = divmod(k, cols)
            out.append((ts, img.crop((x * tw, y * th, (x + 1) * tw, (y + 1) * th))))
        t0 += step * per
    return out


def _is_blank(tile) -> bool:
    """Padding slots and fades to black are uniform; they only waste the judge's attention."""
    from PIL import ImageStat

    return max(ImageStat.Stat(tile.convert("L")).stddev) < 4.0


def _contact_sheet(tiles, max_tiles: int = MAX_TILES) -> tuple[bytes, list[float]]:
    from PIL import Image, ImageDraw, ImageFont

    if len(tiles) > max_tiles:
        idx = [round(i * (len(tiles) - 1) / (max_tiles - 1)) for i in range(max_tiles)]
        tiles = [tiles[i] for i in idx]
    w = max(MIN_TILE_W, tiles[0][1].width)
    h = round(tiles[0][1].height * w / tiles[0][1].width)
    rows = -(-len(tiles) // SHEET_COLS)
    sheet = Image.new("RGB", (SHEET_COLS * w, rows * h), "black")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    for i, (_, im) in enumerate(tiles):
        x, y = (i % SHEET_COLS) * w, (i // SHEET_COLS) * h
        sheet.paste(im.resize((w, h)), (x, y))
        draw.rectangle((x, y, x + 36, y + 24), fill="black")
        draw.text((x + 3, y + 1), str(i), fill="yellow", font=font)
    buf = io.BytesIO()
    sheet.save(buf, "JPEG", quality=85)
    return buf.getvalue(), [t for t, _ in tiles]


def load(video_id: str, spec: dict | None, duration: float, cache_dir: Path,
         refetch_spec=None) -> Storyboard | None:
    """Contact sheet for `video_id`, cached on disk. `refetch_spec()` gets fresh sprite URLs when
    the cached ones have expired (they are signed and go stale after some hours)."""
    d = cache_dir / "storyboards"
    jpg, meta = d / f"{video_id}.jpg", d / f"{video_id}.json"
    if jpg.exists() and meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        return Storyboard(video_id, jpg.read_bytes(), m["times"], m["step"], m["duration"])
    if not duration:
        return None

    def attempt(sp: dict | None):
        if not sp:
            return None
        try:
            return [t for t in _tiles(sp, duration) if not _is_blank(t[1])]
        except Exception as e:          # expired signed URL (403) or a transient network error
            log.debug("storyboard fetch failed for %s: %s", video_id, str(e)[:160])
            return None

    fresh_enough = spec and time.time() - spec.get("fetched", 0) < SPEC_MAX_AGE
    tiles = attempt(spec) if fresh_enough else None
    if tiles is None and refetch_spec:
        tiles = attempt(refetch_spec())
    if not tiles or len(tiles) < 3:
        return None
    data, times = _contact_sheet(tiles)
    step = (times[-1] - times[0]) / max(1, len(times) - 1)
    d.mkdir(parents=True, exist_ok=True)
    jpg.write_bytes(data)
    meta.write_text(json.dumps({"times": times, "step": step, "duration": duration}), encoding="utf-8")
    return Storyboard(video_id, data, times, step, duration)
