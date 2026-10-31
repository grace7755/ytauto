"""Shared library of approved footage, searched before YouTube.

Every clip a run accepts (it passed the local checks and the AI verify) is copied here with a CLIP
embedding of three of its frames, its source attribution and the video's setting. A later shot -
in any project - whose subject CLIP matches a library clip tries that clip first: no search, no
screening, no download. The clip still goes through the same local checks and the AI verify
against the new shot's own brief, so the library only saves time, never lowers the bar.

Kept deliberately simple: an append-only `index.jsonl` plus `emb.npy` (one row per entry), loaded
once per run. Hundreds of entries are a millisecond matrix product; no vector database needed.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import threading
from pathlib import Path

import numpy as np

from .log import log


def _frames(clip: Path, n: int = 3) -> list:
    """`n` frames (at 25/50/75%) as PIL images, decoded by ffmpeg."""
    from PIL import Image

    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(clip)], capture_output=True, text=True).stdout.strip() or 0)
    out = []
    for k in range(1, n + 1):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{dur * k / (n + 1):.2f}", "-i", str(clip),
                              "-frames:v", "1", "-vf", "scale=448:252", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             capture_output=True).stdout
        if len(raw) == 448 * 252 * 3:
            out.append(Image.frombytes("RGB", (448, 252), raw))
    return out


class Library:
    def __init__(self, root: Path, clip_model):
        self.root, self.clip = Path(root), clip_model
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.entries: list[dict] = []
        idx, emb = self.root / "index.jsonl", self.root / "emb.npy"
        if idx.exists():
            for line in idx.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.entries.append(json.loads(line))
        self.emb = np.load(emb).astype(np.float32) if emb.exists() else np.zeros((0, 512), np.float32)
        if len(self.emb) != len(self.entries):           # torn write: keep what is consistent
            n = min(len(self.emb), len(self.entries))
            self.entries, self.emb = self.entries[:n], self.emb[:n]
        self._keys = {e["key"] for e in self.entries}
        self._uses = self._load_uses()

    # -- bookkeeping ----------------------------------------------------------------
    def _load_uses(self) -> dict[str, list[str]]:
        f = self.root / "uses.json"
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def note_use(self, key: str, project: str):
        """Remember that `project` used library clip `key` (caps reuse across videos)."""
        with self._lock:
            users = self._uses.setdefault(key, [])
            if project not in users:
                users.append(project)
                (self.root / "uses.json").write_text(json.dumps(self._uses, indent=1), encoding="utf-8")

    def uses(self, key: str) -> list[str]:
        return list(self._uses.get(key, []))

    # -- add ---------------------------------------------------------------------------
    def add(self, *, clip: Path, provider: str, source_id: str, url: str, title: str, license: str,
            src_start: float, src_end: float, subject: str, setting: str, text: str, project: str) -> bool:
        key = f"{provider}:{source_id}:{src_start:.1f}"
        if key in self._keys or not clip.exists():
            return False
        frames = _frames(clip)
        if not frames:
            return False
        e = self.clip.images(frames).mean(axis=0)
        e = (e / max(float(np.linalg.norm(e)), 1e-8)).astype(np.float32)
        dst = self.root / "clips" / f"{provider}_{source_id}_{int(src_start * 10)}{clip.suffix}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(clip, dst)
        entry = {"key": key, "file": str(dst), "provider": provider, "source_id": source_id, "url": url,
                 "title": title, "license": license, "src_start": src_start, "src_end": src_end,
                 "dur": round(src_end - src_start, 2), "subject": subject, "setting": setting,
                 "text": text[:200], "project": project}
        with self._lock:
            if key in self._keys:
                return False
            self.entries.append(entry)
            self.emb = np.vstack([self.emb, e[None, :]])
            self._keys.add(key)
            with open(self.root / "index.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            np.save(self.root / "emb.npy", self.emb.astype(np.float16))
        return True

    # -- search ------------------------------------------------------------------------
    def search(self, query: np.ndarray, *, want: float, setting: str, project: str, min_sim: float,
               max_uses: int = 3, limit: int = 3) -> list[tuple[dict, float]]:
        """Library clips at least `want` seconds long, from the same setting, not first found by
        nor already used in `project`, used by fewer than `max_uses` other projects, best first."""
        with self._lock:
            if not self.entries:
                return []
            sims = self.emb @ query
            entries = list(self.entries)
        out = []
        for i in np.argsort(-sims):
            e, s = entries[i], float(sims[i])
            if s < min_sim:
                break
            if e["dur"] < want - 0.05 or (e.get("setting") or "").lower() != (setting or "").lower():
                continue
            if e["project"] == project or project in self.uses(e["key"]) or len(self.uses(e["key"])) >= max_uses:
                continue
            if not Path(e["file"]).exists():
                continue
            out.append((e, s))
            if len(out) >= limit:
                break
        return out


_shared: dict[str, Library] = {}
_shared_lock = threading.Lock()


def open_library(root: Path, clip_model) -> Library | None:
    """One Library per root per process (shots run on several threads)."""
    with _shared_lock:
        lib = _shared.get(str(root))
        if lib is None:
            try:
                lib = Library(root, clip_model)
            except Exception as e:                       # never fail a run over the library
                log.warning("footage library unavailable: %s", str(e)[:160])
                return None
            _shared[str(root)] = lib
        return lib
