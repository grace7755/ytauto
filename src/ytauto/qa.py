"""Final-video QA: spec, black/frozen detection, timeline-vs-ledger audit, duplicate shots, contact sheet."""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

from .config import Settings
from .inspect.probe import probe
from .models import Shot
from .state import Project


@dataclass
class QAResult:
    passed: bool = True
    issues: list[str] = field(default_factory=list)
    bad_shots: set[int] = field(default_factory=set)
    contact_sheet: str = ""
    info: dict = field(default_factory=dict)

    def fail(self, msg: str, shot: int | None = None):
        self.passed = False
        self.issues.append(msg)
        if shot is not None:
            self.bad_shots.add(shot)


def _declared_reuse(proj: Project, a: int, b: int) -> bool:
    """Is one of these two shots a deliberate reuse (`Selector._reuse`: the same clip, far apart on the
    timeline) of the other? Anything else that looks the same is still a duplicate."""
    da, db = proj.accepted(a), proj.accepted(b)
    if not da or not db:
        return False
    same = (da.provider, da.source_id) == (db.provider, db.source_id) and abs(da.src_start - db.src_start) < 0.15
    return same and ("REUSED footage" in da.notes or "REUSED footage" in db.notes)


def _shot_at(shots: list[Shot], t: float) -> int | None:
    for sh in shots:
        if sh.start - 1e-3 <= t < sh.end:
            return sh.id
    return None


def _detect(video: Path, vf: str, pat: str) -> list[tuple[float, float]]:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(video), "-vf", vf, "-an", "-f", "null", "-"],
                       capture_output=True, text=True, timeout=1800)
    out = []
    for m in re.finditer(pat, r.stderr):
        out.append((float(m.group(1)), float(m.group(2))))
    return out


def contact_sheet(video: Path, shots: list[Shot], out: Path, cols: int = 6) -> Path:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    tiles = []
    for sh in shots:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((sh.start + sh.duration / 2) * fps))
        ok, f = cap.read()
        if ok:
            f = cv2.resize(f, (320, 180))
            cv2.putText(f, str(sh.id), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            tiles.append(f)
    cap.release()
    if not tiles:
        return out
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [cv2.hconcat(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(out), cv2.vconcat(rows))
    return out


def run_qa(s: Settings, proj: Project, shots: list[Shot], video: Path, expected_duration: float) -> QAResult:
    r = QAResult()
    info = probe(video)
    if not info:
        r.fail("final video is unreadable")
        return r
    r.info = info.__dict__.copy()
    if (info.width, info.height) != (s.width, s.height):
        r.fail(f"resolution {info.width}x{info.height} != {s.width}x{s.height}")
    if abs(info.fps - s.fps) > 0.05:
        r.fail(f"fps {info.fps:.2f} != {s.fps}")
    if abs(info.duration - expected_duration) > 1.5 / s.fps + 0.05:
        r.fail(f"duration {info.duration:.2f}s != expected {expected_duration:.2f}s")

    for a, _ in _detect(video, "blackdetect=d=0.3:pix_th=0.10", r"black_start:([\d.]+)\s+black_end:([\d.]+)"):
        r.fail(f"black frames at {a:.1f}s", _shot_at(shots, a))
    for a, _ in _detect(video, "freezedetect=n=0.001:d=1.0", r"freeze_start: ([\d.]+)[\s\S]*?freeze_end: ([\d.]+)"):
        r.fail(f"frozen frames at {a:.1f}s", _shot_at(shots, a))

    # every shot on the timeline must trace to an accepted/weak ledger entry whose clip exists
    for sh in shots:
        d = proj.accepted(sh.id)
        if d is None:
            r.fail(f"shot {sh.id} has no accepted clip in the ledger", sh.id)
        elif d.status == "weak":
            r.issues.append(f"shot {sh.id} is a WEAK match (score {d.score.total:.0f}) - review manually")

    # duplicates: perceptual hash of each shot's middle frame
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or s.fps
    hashes: list[tuple[int, imagehash.ImageHash]] = []
    for sh in shots:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((sh.start + sh.duration / 2) * fps))
        ok, f = cap.read()
        if ok:
            hashes.append((sh.id, imagehash.phash(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))))
    cap.release()
    for i, (ia, ha) in enumerate(hashes):
        for ib, hb in hashes[i + 1:]:
            if ha - hb <= 4 and not _declared_reuse(proj, ia, ib):
                r.fail(f"shots {ia} and {ib} look like duplicates", ib)

    r.contact_sheet = str(contact_sheet(video, shots, proj.path("output", "contact_sheet.jpg")))
    return r


def _middle_frame(clip: Path):
    cap = cv2.VideoCapture(str(clip))
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(n // 2))
    ok, f = cap.read()
    cap.release()
    return f if ok else None


def run_qa_clips(s: Settings, proj: Project, shots: list[Shot]) -> QAResult:
    """QA without a rendered video, for the Remotion workflow (the ffmpeg `final.mp4` is never used
    there). Same timeline-vs-ledger audit and duplicate check as `run_qa`, read from each shot's clip
    instead of the render. Black/frozen footage is already a hard reject per clip (`inspect/checks.py`),
    and resolution/fps/duration belong to the Remotion render, checked by the film workflow."""
    r = QAResult()
    hashes: list[tuple[int, imagehash.ImageHash]] = []
    tiles = []
    for sh in shots:
        d = proj.accepted(sh.id)
        if d is None:
            r.fail(f"shot {sh.id} has no accepted clip in the ledger", sh.id)
            continue
        if d.status == "weak":
            r.issues.append(f"shot {sh.id} is a WEAK match (score {d.score.total if d.score else 0:.0f}) - review manually")
        f = _middle_frame(Path(d.clip_path))
        if f is not None:
            hashes.append((sh.id, imagehash.phash(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))))
            t = cv2.resize(f, (320, 180))
            cv2.putText(t, str(sh.id), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            tiles.append(t)
    for i, (ia, ha) in enumerate(hashes):
        for ib, hb in hashes[i + 1:]:
            if ha - hb <= 4 and not _declared_reuse(proj, ia, ib):
                r.fail(f"shots {ia} and {ib} look like duplicates", ib)
    r.info = {"mode": "clips (no render)"}
    if tiles:                                  # the reviewer looks at it, same layout as `contact_sheet`
        cols = 6
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        out = proj.path("output", "contact_sheet.jpg")
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), cv2.vconcat([cv2.hconcat(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))
        r.contact_sheet = str(out)
    return r
