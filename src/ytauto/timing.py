"""Wall-clock time per pipeline stage, so a slow run says *where* it was slow.

Thread-safe: shots and candidates are processed concurrently, so totals are summed CPU-independent
wall time per stage (they can add up to more than the run's elapsed time - that is the point of
running them in parallel).
"""
from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

_lock = threading.Lock()
_total: dict[str, float] = defaultdict(float)
_count: dict[str, int] = defaultdict(int)


@contextmanager
def stage(name: str):
    t = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t
        with _lock:
            _total[name] += dt
            _count[name] += 1


def reset():
    with _lock:
        _total.clear()
        _count.clear()


def snapshot() -> dict[str, dict]:
    with _lock:
        return {k: {"seconds": round(_total[k], 1), "calls": _count[k],
                    "avg": round(_total[k] / _count[k], 2) if _count[k] else 0.0}
                for k in sorted(_total, key=lambda k: -_total[k])}


def summary(elapsed: float) -> str:
    snap = snapshot()
    lines = [f"elapsed {elapsed / 60:.1f} min; stage time (summed across parallel workers):"]
    for k, v in snap.items():
        lines.append(f"  {k:18} {v['seconds'] / 60:7.1f} min  {v['calls']:5} calls  avg {v['avg']:6.1f}s")
    return "\n".join(lines)


def write(path: Path, elapsed: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"elapsed_s": round(elapsed, 1), "stages": snapshot()}, indent=2),
                    encoding="utf-8")
