"""Relevance review before rendering: one AI call per ~10 shots looks at a numbered sheet of each shot's
picture next to its narration line and says whether the picture shows what is being said.

Relevance outranks everything else for this channel, and the per-clip verify only ever saw one clip at
a time. A weak shot is then redone (`ytauto redo <script> --shot N`), where a still can replace footage."""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .llm.router import Judge
from .models import Shot
from .state import Project

REVIEW_SCHEMA = {"type": "object", "properties": {"shots": {"type": "array", "items": {
    "type": "object", "properties": {"id": {"type": "integer"}, "match": {"type": "integer"},
                                     "why": {"type": "string"}}, "required": ["id", "match"]}}},
    "required": ["shots"]}

PROMPT = """These are {n} frames, each from the B-roll clip chosen for one line of a documentary's narration.
The number on a frame is the shot id. Narration per shot:
{lines}

For every shot, rate 0-100 how well the frame shows what its narration line is about (the named place,
person, object, action or idea - not just a similar mood or colour). 90+ = clearly shows it, 60-89 = plausible,
below 60 = unrelated or the wrong place/thing. Add a short `why` (under 15 words) for anything below 70.
Be strict: a pretty frame of the wrong subject is a low score."""


def frame_of(clip: Path):
    """The middle frame of a clip as a PIL image (what CLIP compares), or None."""
    f = _frame(clip)
    return None if f is None else Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))


def _frame(clip: Path) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(clip))
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(n // 2))
    ok, f = cap.read()
    cap.release()
    return cv2.resize(f, (480, 270)) if ok else None


def sheet(tiles: list[tuple[int, np.ndarray]], cols: int = 3) -> bytes:
    """Numbered tiles as one JPEG."""
    cells = []
    for sid, f in tiles:
        f = f.copy()
        cv2.rectangle(f, (0, 0), (70, 34), (0, 0, 0), -1)
        cv2.putText(f, str(sid), (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cells.append(f)
    while len(cells) % cols:
        cells.append(np.zeros_like(cells[0]))
    img = cv2.vconcat([cv2.hconcat(cells[i:i + cols]) for i in range(0, len(cells), cols)])
    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(buf, "JPEG", quality=85)
    return buf.getvalue()


def review(gem: Judge, proj: Project, shots: list[Shot], per: int = 10) -> list[dict]:
    """[{shot, match, why, narration}] for every shot that has a clip, in shot order."""
    items = []
    for sh in shots:
        d = proj.accepted(sh.id)
        f = _frame(Path(d.clip_path)) if d and d.clip_path and Path(d.clip_path).exists() else None
        if f is not None:
            items.append((sh, f))
    out = []
    for i in range(0, len(items), per):
        batch = items[i:i + per]
        lines = "\n".join(f"[{sh.id}] {sh.text}" for sh, _ in batch)
        res = gem.json_images(PROMPT.format(n=len(batch), lines=lines), [sheet([(sh.id, f) for sh, f in batch])],
                              REVIEW_SCHEMA, cache_key=("review-v1", lines, tuple(proj.accepted(sh.id).clip_path for sh, _ in batch)))
        got = {r["id"]: r for r in res.get("shots", []) if isinstance(r.get("id"), int)}
        for sh, _ in batch:
            r = got.get(sh.id, {})
            out.append({"shot": sh.id, "match": int(r.get("match", -1)), "why": r.get("why", ""), "narration": sh.text})
    return out
