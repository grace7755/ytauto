"""Pick a cut-free window of the wanted length inside a downloaded section."""
from __future__ import annotations

from pathlib import Path

from scenedetect import ContentDetector, detect


def scene_spans(path: Path, threshold: float = 27.0) -> list[tuple[float, float]]:
    try:
        scenes = detect(str(path), ContentDetector(threshold=threshold), show_progress=False)
    except Exception:
        scenes = []
    return [(a.get_seconds(), b.get_seconds()) for a, b in scenes]


def best_window(path: Path, clip_len: float, want: float, hint: tuple[float, float] | None = None,
                total: float | None = None) -> tuple[float, float, bool]:
    """Return (start, end, cut_free). Prefers the scene that overlaps `hint` (the Gemini window,
    expressed relative to the downloaded file) and is long enough; otherwise the longest scene."""
    spans = scene_spans(path) or [(0.0, total or clip_len)]
    ok = [s for s in spans if s[1] - s[0] >= want - 0.05]
    pool = ok or [max(spans, key=lambda s: s[1] - s[0])]

    def overlap(s):
        if not hint:
            return 0.0
        return max(0.0, min(s[1], hint[1]) - max(s[0], hint[0]))

    a, b = max(pool, key=lambda s: (overlap(s), s[1] - s[0]))
    length = b - a
    if length < want:
        return a, b, False
    # centre the window inside the scene, nudged toward the hint when available
    centre = (hint[0] + hint[1]) / 2 if hint else (a + b) / 2
    start = min(max(a, centre - want / 2), b - want)
    return start, start + want, True
