"""Parse timestamped transcripts and split segments into 4-5 s shots."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .models import Segment, Shot

_TS = r"(?:(\d+):)?(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?"
_RANGE = re.compile(rf"^\s*[\[(]?\s*{_TS}\s*(?:-->|-|–|—|to)\s*{_TS}\s*[\])]?\s*[:\-–—|]?\s*(.*)$")
_SINGLE = re.compile(rf"^\s*[\[(]?\s*{_TS}\s*[\])]?\s*[:\-–—|]?\s*(.+)$")


def _sec(h, m, s, ms) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + (int((ms or "0").ljust(3, "0")) / 1000)


def parse_text(raw: str) -> list[Segment]:
    """Supports `00:00-00:05 text`, `00:00–00:05: text`, SRT/VTT blocks, and single `00:05 text` lines."""
    lines = [l.rstrip() for l in raw.replace("﻿", "").splitlines()]
    segs: list[tuple[float, float | None, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.upper().startswith("WEBVTT") or line.isdigit():
            i += 1
            continue
        m = _RANGE.match(line)
        if m:
            g = m.groups()
            start, end, text = _sec(*g[0:4]), _sec(*g[4:8]), g[8].strip()
            if not text:  # SRT/VTT: text on the following lines
                j, buf = i + 1, []
                while j < len(lines) and lines[j].strip():
                    buf.append(lines[j].strip())
                    j += 1
                text, i = " ".join(buf), j - 1
            segs.append((start, end, text))
            i += 1
            continue
        m = _SINGLE.match(line)
        if m:
            g = m.groups()
            segs.append((_sec(*g[0:4]), None, g[4].strip()))
        i += 1
    out: list[Segment] = []
    for n, (s, e, t) in enumerate(segs):
        if e is None:
            e = segs[n + 1][0] if n + 1 < len(segs) else s + 5
        if e > s and t:
            out.append(Segment(idx=len(out), start=s, end=e, text=t))
    return out


def load_transcript(path: Path) -> list[Segment]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        return [Segment(idx=i, start=float(d["start"]), end=float(d["end"]), text=d["text"]) for i, d in enumerate(data)]
    segs = parse_text(text)
    if not segs:
        raise ValueError(f"No timestamped segments found in {path}")
    return segs


def split_shots(segments: list[Segment], target: float = 4.5, smin: float = 3.0, smax: float = 6.0) -> list[Shot]:
    """Split each segment into near-equal shots between smin and smax (aim for `target`)."""
    shots: list[Shot] = []
    for seg in segments:
        d = seg.duration
        n = max(1, round(d / target))
        while n > 1 and d / n < smin:
            n -= 1
        while d / n > smax:
            n += 1
        step = d / n
        for k in range(n):
            a = seg.start + k * step
            b = seg.end if k == n - 1 else seg.start + (k + 1) * step
            shots.append(Shot(id=len(shots), segment=seg.idx, start=round(a, 3), end=round(b, 3),
                              text=seg.text, variant=k))
    return shots
