r"""Should `storyboard_face_check` be turned on? Measures it on past ledgers, free (no judge calls).

For every downloaded YouTube candidate whose contact sheet is cached: positives were rejected after
download for a presenter / large face, negatives were accepted. The tile the window was cut from is
run through the same local face check screening would use. Turn the setting on only if it flags
(almost) no accepted clip AND a real share of the presenter rejects - it only reorders, so a miss
costs nothing, but a flagged good clip is tried later.

    .venv\Scripts\python scripts\measure_face_tiles.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ytauto.config import load_settings  # noqa: E402
from ytauto.screen import _tile_face  # noqa: E402
from calibrate_prerank import load_sb  # noqa: E402

PRESENTER = ("presenter", "large face")


def cases():
    seen = set()
    for proj in sorted((ROOT / "projects").iterdir()):
        lj = proj / "ledger.jsonl"
        if not lj.exists():
            continue
        for line in lj.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["provider"] != "youtube" or not r.get("clip_path"):
                continue
            rej = " ".join(r.get("hard_rejects") or []).lower()
            if r["status"] == "accepted":
                label = False
            elif any(k in rej for k in PRESENTER):
                label = True
            else:
                continue                      # rejected for something else
            key = (r["source_id"], round(r["src_start"]), label)
            if key in seen:
                continue
            seen.add(key)
            yield proj.name, r["source_id"], label, r["src_start"]


def main():
    limit = load_settings(None).max_face_ratio
    hit = {True: 0, False: 0}
    n = {True: 0, False: 0}
    flagged_good = []
    for proj, vid, label, start in cases():
        sb = load_sb(vid)
        if sb is None:
            continue
        tile = sb.nearest_tile(start + 2.0)       # Storyboard.window starts 2 s before its tile
        f = _tile_face(sb, tile, limit)
        n[label] += 1
        hit[label] += f
        if f and not label:
            flagged_good.append(f"{proj} {vid} @{start:.0f}s")
    print(f"presenter rejects flagged : {hit[True]}/{n[True]}")
    print(f"accepted clips flagged    : {hit[False]}/{n[False]}  (must be ~0)")
    for x in flagged_good[:20]:
        print("  flagged good:", x)
    if n[True] and n[False]:
        ok = hit[False] <= max(1, n[False] // 50) and hit[True] >= max(3, n[True] // 5)
        print("verdict: turn ON (storyboard_face_check: true)" if ok else "verdict: leave OFF")


if __name__ == "__main__":
    main()
