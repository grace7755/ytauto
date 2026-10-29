"""Deterministic, free visual checks on a clip (OpenCV): blur, B&W, black/frozen, letterbox,
static overlay (watermark/logo), and large faces (talking head)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

_FACE = None


def sample_frames(path: Path, n: int = 9, width: int = 960) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frames = []
    if total <= 0:
        cap.release()
        return frames
    for i in np.linspace(total * 0.05, total * 0.95, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok and f is not None:
            h, w = f.shape[:2]
            frames.append(cv2.resize(f, (width, int(h * width / w)), interpolation=cv2.INTER_AREA))
    cap.release()
    return frames


def blur_score(frames) -> float:
    """Median Laplacian variance (higher = sharper) at a fixed 960px width."""
    vals = [cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() for f in frames]
    return float(np.median(vals)) if vals else 0.0


def mean_saturation(frames) -> float:
    vals = [cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[..., 1].mean() / 255.0 for f in frames]
    return float(np.mean(vals)) if vals else 0.0


def black_ratio(frames) -> float:
    if not frames:
        return 1.0
    return sum(1 for f in frames if cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean() < 14) / len(frames)


def frozen(frames) -> bool:
    """All sampled frames (spread over the clip) practically identical."""
    if len(frames) < 3:
        return False
    g = [cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    diffs = [np.abs(a - b).mean() for a, b in zip(g, g[1:])]
    return max(diffs) < 0.6


def letterbox_fraction(frames) -> float:
    """Fraction of height lost to black bars (rows near-black in every sample)."""
    if not frames:
        return 0.0
    rows = np.min([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).max(axis=1) for f in frames], axis=0)
    dark = rows < 18
    h = len(dark)
    top = 0
    while top < h and dark[top]:
        top += 1
    bot = 0
    while bot < h and dark[h - 1 - bot]:
        bot += 1
    return (top + bot) / h if top + bot < h else 1.0


def static_overlay_corners(frames) -> float:
    """Weak hint that a corner holds a static overlay (watermark/logo). Returns 0-1.

    Measured on real clips (2026-10-29) this signal is unreliable in BOTH directions, so it is a
    soft hint only and must not hard-reject on its own:
      * a real translucent "ON RIDE" watermark scored 0.004 - it is semi-transparent, so it
        inherits the motion of the footage underneath and never looks static;
      * a clean tripod shot of a Tokyo street at night scored 0.143 in three corners, because
        static shop signs and architecture are exactly what this looks for.
    Burnt-in text and logos are caught reliably by the AI judge's `text_overlay` /
    `watermark_or_logo` flags instead.

    Only a corner that is static while the frame as a whole clearly moves counts, and the score is
    damped when several corners trip at once (that means a static camera, not a logo).
    """
    if len(frames) < 4:
        return 0.0
    g = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames])
    h, w = g.shape[1:]
    ch, cw = int(h * 0.22), int(w * 0.25)
    corners = {"tl": (slice(0, ch), slice(0, cw)), "tr": (slice(0, ch), slice(w - cw, w)),
               "bl": (slice(h - ch, h), slice(0, cw)), "br": (slice(h - ch, h), slice(w - cw, w)),
               "bc": (slice(h - ch, h), slice(w // 2 - cw // 2, w // 2 + cw // 2)),
               "tc": (slice(0, ch), slice(w // 2 - cw // 2, w // 2 + cw // 2))}
    global_motion = float(g.std(axis=0).mean()) + 1e-3
    if global_motion < 3.0:
        return 0.0          # a near-static shot: nothing here distinguishes a logo from scenery
    fracs = []
    for sl in corners.values():
        reg = g[:, sl[0], sl[1]]
        temporal = reg.std(axis=0)
        static_mask = temporal < max(2.0, 0.25 * global_motion)
        edges = cv2.Canny(reg.mean(axis=0).astype(np.uint8), 80, 160) > 0
        fracs.append(float((static_mask & edges).mean()))
    hot = [f for f in fracs if f > 0.04]
    if len(hot) >= 3:
        return 0.0          # static scenery in many corners, not a watermark in one
    return min(1.0, max(fracs) * 6)


def largest_face_ratio(frames) -> float:
    global _FACE
    if _FACE is None:
        _FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    best = 0.0
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        faces = _FACE.detectMultiScale(g, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))
        for (_, _, w, h) in faces:
            best = max(best, (w * h) / (g.shape[0] * g.shape[1]))
    return float(best)


@dataclass
class CheckResult:
    blur: float = 0.0
    saturation: float = 0.0
    black: float = 0.0
    frozen: bool = False
    letterbox: float = 0.0
    overlay: float = 0.0
    face: float = 0.0
    n_frames: int = 0
    notes: list[str] = field(default_factory=list)


def run_checks(path: Path) -> CheckResult:
    fr = sample_frames(path)
    r = CheckResult(n_frames=len(fr))
    if not fr:
        r.notes.append("unreadable")
        return r
    r.blur, r.saturation, r.black = blur_score(fr), mean_saturation(fr), black_ratio(fr)
    r.frozen, r.letterbox = frozen(fr), letterbox_fraction(fr)
    r.overlay, r.face = static_overlay_corners(fr), largest_face_ratio(fr)
    return r
