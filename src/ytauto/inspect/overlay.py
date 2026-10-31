"""Local detector for burnt-in text that stays put while the footage moves (channel names, logos,
"ON RIDE"-style watermarks), so such a clip can be rejected without a judge call.

Free judge quota is the scarce resource and watermark / burnt-in text was the single biggest hard
reject measured over past runs (79 of 339 rejects). OCR (RapidOCR, ONNX, CPU) reads a few frames;
text is an overlay when the same string sits at the same place in nearly every frame **while the
scene behind it changes**. A tripod shot of a street sign also keeps its text in place, but there
the scene does not change either, so that case is left to the AI judge (it is never rejected here).
The result is conservative on purpose: it can skip a judge call, never add a false reject on a
static scene.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..log import log

FRAMES = 4                  # frames sampled evenly across the clip
WIDTH = 960                 # frames are scaled to this width before OCR (speed)
# Thresholds chosen by sweeping 4x2x3x3x3 settings over 126 clips of past runs (62 the AI judge
# accepted as clean, 64 it rejected for a watermark or burnt-in text; scripts/measure_overlay_ocr.py).
# Chosen: zero false rejects on real scenery - the single "clean" clip it flags carries an "ON RIDE"
# watermark the AI judge missed - and 24 of 64 AI-flagged clips caught. Looser settings caught more
# but flagged street signs and shop fronts seen from a moving camera.
MIN_CONF = 0.6              # OCR box confidence
MIN_CHARS = 4               # shorter strings are noise
POS_TOL = 0.01              # same place = box centres within this fraction of the frame width
SIZE_TOL = 0.3              # ... and the box keeps its size within this fraction
MOTION_MIN = 6.0            # mean abs grey difference (0-255) between sampled frames: "the scene moves"

_OCR = None
_LOCK = threading.Lock()
_FAILED = False


@dataclass
class OverlayResult:
    persistent: list[str] = field(default_factory=list)   # strings that stay in place
    motion: float = 0.0                                    # how much the scene behind changes
    frames: int = 0

    @property
    def burnt_in(self) -> bool:
        """Text pinned in place over footage that moves: a watermark or burnt-in caption."""
        return bool(self.persistent) and self.motion >= MOTION_MIN


def available() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return not _FAILED
    except Exception:
        return False


def _ocr():
    global _OCR, _FAILED
    with _LOCK:
        if _OCR is None and not _FAILED:
            try:
                from rapidocr_onnxruntime import RapidOCR

                _OCR = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)
            except Exception as e:
                log.warning("OCR unavailable (%s); burnt-in text is left to the AI judge", str(e)[:160])
                _FAILED = True
    return _OCR


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9぀-ヿ一-鿿]", "", s.lower())


def _frames(clip: Path, n: int, width: int):
    import cv2

    cap = cv2.VideoCapture(str(clip))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out = []
    if total >= n:
        for i in range(n):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((i + 0.5) * total / n))
            ok, fr = cap.read()
            if ok:
                h, w = fr.shape[:2]
                out.append(cv2.resize(fr, (width, max(1, int(h * width / w)))))
    cap.release()
    return out


def _motion(frames) -> float:
    import cv2

    g = [cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    return float(np.mean([np.abs(a - b).mean() for a, b in zip(g, g[1:])])) if len(g) > 1 else 0.0


def persistent_text(per_frame: list[list[tuple]]) -> list[str]:
    """Strings found at the same place, with the same size, in EVERY sampled frame.

    `per_frame[i]` holds `(text, cx, cy, w, h)` per OCR box, positions as a fraction of the frame
    width. Requiring all frames (not all but one) is what keeps scenery signs out."""
    out: list[str] = []
    if len(per_frame) < 2:
        return out
    for t, x, y, w, h in per_frame[0]:
        if t in out:
            continue
        if all(any(t == t2 and abs(x - x2) <= POS_TOL and abs(y - y2) <= POS_TOL
                   and abs(w - w2) <= SIZE_TOL * w and abs(h - h2) <= SIZE_TOL * h
                   for t2, x2, y2, w2, h2 in items) for items in per_frame[1:]):
            out.append(t)
    return out


def detect(clip: Path, n: int = FRAMES, width: int = WIDTH) -> OverlayResult | None:
    """OCR `n` frames of `clip`. None when OCR is unavailable or the clip cannot be read."""
    ocr = _ocr()
    if ocr is None:
        return None
    try:
        frames = _frames(Path(clip), n, width)
        if len(frames) < max(2, n - 1):
            return None
        per_frame: list[list[tuple]] = []
        for f in frames:
            res, _ = ocr(f)
            items = []
            for box, text, conf in res or []:
                t = _norm(text)
                if len(t) >= MIN_CHARS and float(conf) >= MIN_CONF:
                    xs = [p[0] for p in box]
                    ys = [p[1] for p in box]
                    items.append((t, (min(xs) + max(xs)) / 2 / width, (min(ys) + max(ys)) / 2 / width,
                                  (max(xs) - min(xs)) / width, (max(ys) - min(ys)) / width))
            per_frame.append(items)
        return OverlayResult(persistent=persistent_text(per_frame), motion=_motion(frames), frames=len(frames))
    except Exception as e:
        log.debug("overlay detection failed for %s: %s", Path(clip).name, str(e)[:160])
        return None
