"""Per-project state, ledger and atomic writes (enables resume)."""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

from .models import Decision

# Shots are selected on several threads; every ledger write goes through this lock.
_LEDGER_LOCK = threading.RLock()


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60] or "project"


def atomic_write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class Project:
    SUBDIRS = ["input", "sources/youtube", "sources/stock", "clips", "normalized", "frames", "metadata", "output", "logs"]

    def __init__(self, root: Path):
        self.root = root
        for d in self.SUBDIRS:
            (root / d).mkdir(parents=True, exist_ok=True)
        self.state_file = root / "state.json"
        self.ledger_file = root / "ledger.jsonl"
        self.state: dict = json.loads(self.state_file.read_text(encoding="utf-8")) if self.state_file.exists() else {}

    def path(self, *p) -> Path:
        return self.root.joinpath(*p)

    def done(self, stage: str) -> bool:
        return bool(self.state.get("stages", {}).get(stage))

    def mark(self, stage: str, value=True):
        self.state.setdefault("stages", {})[stage] = value
        self.save()

    def unmark(self, stage: str):
        self.state.get("stages", {}).pop(stage, None)
        self.save()

    def save(self):
        atomic_write(self.state_file, json.dumps(self.state, indent=2, default=str))

    # ---- ledger -------------------------------------------------------
    run_label = ""                     # set by the pipeline ("run" / "redo-<n>"), stamped on every row

    def log_decision(self, d: Decision):
        d.ts = d.ts or time.strftime("%Y-%m-%dT%H:%M:%S")
        d.run = d.run or self.run_label
        with _LEDGER_LOCK, self.ledger_file.open("a", encoding="utf-8") as f:   # shots run in parallel
            f.write(d.model_dump_json() + "\n")

    def ledger(self) -> list[Decision]:
        if not self.ledger_file.exists():
            return []
        return [Decision.model_validate_json(l) for l in self.ledger_file.read_text(encoding="utf-8").splitlines() if l.strip()]

    def accepted(self, shot: int) -> Decision | None:
        """Latest accepted (or weak) decision for a shot whose clip still exists on disk."""
        best = None
        for d in self.ledger():
            if d.shot == shot and d.status in ("accepted", "weak") and d.clip_path and Path(d.clip_path).exists():
                best = d
        return best

    def invalidate(self, shot: int, reason: str = "superseded"):
        """Mark a shot's current clip as superseded so it will be re-selected."""
        with _LEDGER_LOCK:
            rows = []
            for d in self.ledger():
                if d.shot == shot and d.status in ("accepted", "weak"):
                    d.status = "rejected"
                    d.notes += f" | {reason}"
                rows.append(d)
            atomic_write(self.ledger_file, "".join(r.model_dump_json() + "\n" for r in rows))
