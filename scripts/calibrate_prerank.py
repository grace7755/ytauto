"""Measure the local CLIP pre-ranker against past judge verdicts before trusting it.

Data: every project's ledger (what the storyboard judge decided per candidate) and the contact
sheets cached in cache/storyboards/. No network, no judge calls (CLIP downloads once if missing).

    .venv\\Scripts\\python scripts\\calibrate_prerank.py

Reports:
- AUC of a candidate's best-tile similarity: judge "usable" vs "not usable" videos
- filter simulation: candidates dropped vs judge-usable candidates lost, per margin below the
  shot's best candidate
- tile recall: is the window the judge picked among CLIP's top-k tiles of that video?
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ytauto.inspect.embed import Clip, TileIndex, query_text  # noqa: E402
from ytauto.models import Brief  # noqa: E402
from ytauto.planner import primary_subject  # noqa: E402
from ytauto.sources import storyboard as SB  # noqa: E402

POS_PREFIX = ("screen: usable but no timestamped window",)


def label(row) -> int | None:
    if row["provider"] != "youtube" or row["status"] == "unavailable":
        return None
    rej = (row.get("hard_rejects") or [""])[0]
    if row["status"] in ("accepted", "weak"):
        return 1
    if rej.startswith(POS_PREFIX) or not rej.startswith("screen:"):
        return 1                         # passed screening (downloaded), whatever came after
    if "screening call failed" in rej:
        return None
    return 0


def load_sb(vid: str):
    d = ROOT / "cache" / "storyboards"
    jpg, meta = d / f"{vid}.jpg", d / f"{vid}.json"
    if not (jpg.exists() and meta.exists()):
        return None
    m = json.loads(meta.read_text(encoding="utf-8"))
    return SB.Storyboard(vid, jpg.read_bytes(), m["times"], m["step"], m["duration"])


def auc(pos, neg) -> float:
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main():
    idx = TileIndex(Clip(ROOT / "cache" / "models"), ROOT / "cache")
    rows = []                            # (project, shot, vid, label, window_start)
    briefs = {}
    for proj in sorted((ROOT / "projects").iterdir()):
        sj, lj = proj / "metadata" / "shots.json", proj / "ledger.jsonl"
        if not (sj.exists() and lj.exists()):
            continue
        for sh in json.loads(sj.read_text(encoding="utf-8")):
            if sh.get("brief"):
                briefs[(proj.name, sh["id"])] = Brief(**sh["brief"])
        for line in lj.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            y = label(r)
            if y is not None and (proj.name, r["shot"]) in briefs:
                rows.append((proj.name, r["shot"], r["source_id"], y, r.get("src_start") or 0.0))

    # one verdict per (shot, video): usable if any row says it passed screening
    per = defaultdict(lambda: [0, []])
    for p, s, v, y, st in rows:
        k = (p, s, v)
        per[k][0] = max(per[k][0], y)
        if y and st:
            per[k][1].append(st)

    qcache, best = {}, {}
    sims = {}
    for (p, s, v), (y, starts) in per.items():
        sb = load_sb(v)
        if sb is None:
            continue
        b = briefs[(p, s)]
        if not b.subject:
            b = b.model_copy(update={"subject": primary_subject(b)})
        q = qcache.setdefault(query_text(b), idx.clip.text(query_text(b)))
        sc = idx.scores(sb, q)
        sims[(p, s, v)] = (y, float(sc.max()), sc, sb, starts)
        best[(p, s)] = max(best.get((p, s), -1), float(sc.max()))

    pos = [x[1] for x in sims.values() if x[0]]
    neg = [x[1] for x in sims.values() if not x[0]]
    print(f"candidates with a cached storyboard: {len(sims)}  (judge-usable {len(pos)}, not usable {len(neg)})")
    print(f"AUC best-tile similarity, usable vs not: {auc(pos, neg):.3f}")

    print("\nfilter: drop a candidate if its best tile is more than MARGIN below the shot's best candidate")
    for margin in (0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10):
        dropped = [k for k, x in sims.items() if x[1] < best[k[:2]] - margin]
        lost = sum(1 for k in dropped if sims[k][0])
        print(f"  margin {margin:.2f}: drops {len(dropped):3d}/{len(sims)} candidates, "
              f"of which judge-usable {lost:2d}/{len(pos)}")

    print("\ntile recall: the judge's chosen window among CLIP's top-k tiles of that video")
    ranks = []
    for y, _, sc, sb, starts in sims.values():
        for st in starts:
            t = st + 2.0                  # windows start 2 s before their tile
            tile = int(np.argmin([abs(x - t) for x in sb.times]))
            order = list(np.argsort(-sc))
            # neighbouring tiles show the same scene; count a hit within one tile
            ranks.append(min(order.index(j) for j in (tile - 1, tile, tile + 1) if 0 <= j < len(sc)))
    for k in (3, 6, 10, 15):
        hit = sum(r < k for r in ranks)
        print(f"  top-{k:2d}: {hit}/{len(ranks)}")


if __name__ == "__main__":
    main()
