r"""Where did a film's footage search spend its attempts, and what would the planned budgets have skipped?

Replays one project's `ledger.jsonl` (no network, no AI calls) and prints:
  1. attempts and accepted clips per provider (yield)
  2. stock/still items tried on many different shots, and how relevant they were judged
  3. how many rows the film-wide stock skip (fix A2) would have skipped, and how many of those were accepted
     (an accepted row that would have been skipped is a real loss - it must stay near 0)
  4. per-shot attempt counts, the attempt index at which shots got their accepted clip, and what an attempt cap
     of N per shot (fix A3) would have saved and lost

    .venv\Scripts\python scripts\measure_attempts.py projects\<slug> [--before 2026-11-01T16:10:00]

`--before` keeps only rows logged before that time (e.g. the first pass, without the redo rounds). Ledgers written
after round 6 carry a `run` field ("run", "redo-1", ...); use `--run run` for the first pass instead.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

STOCK = {"pexels", "pixabay", "coverr", "mixkit", "still_pexels", "still_pixabay", "still_commons"}
SKIP_RELEVANCE = 25        # fix A2: a reject at or below this relevance counts against the item
SKIP_AFTER = 2             # ... on this many OTHER shots before it is skipped
CAPS = (4, 6, 8, 10, 12, 16, 20)
_WORD = re.compile(r"[a-z0-9]+")


def tokens(s: str) -> set[str]:
    return {w for w in _WORD.findall((s or "").lower()) if len(w) >= 4}


def load(project: Path, before: str | None, run: str | None) -> tuple[list[dict], dict[int, dict]]:
    rows = [json.loads(l) for l in (project / "ledger.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if before:
        rows = [r for r in rows if (r.get("ts") or "") < before]
    if run:
        rows = [r for r in rows if r.get("run") == run]
    shots = {}
    sj = project / "metadata" / "shots.json"
    if sj.exists():
        for d in json.loads(sj.read_text(encoding="utf-8")):
            shots[d["id"]] = d
    return rows, shots


def is_screen(r: dict) -> bool:
    return any(str(h).startswith("screen") for h in r.get("hard_rejects") or [])


def relevance(r: dict) -> float | None:
    sc = r.get("score") or {}
    v = sc.get("relevance")
    return float(v) if v is not None else None


def subject_words(shot: dict | None) -> set[str]:
    b = (shot or {}).get("brief") or {}
    return tokens(" ".join([b.get("subject") or "", " ".join(b.get("entities") or [])]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project", type=Path)
    ap.add_argument("--before", help="only rows with ts before this ISO time")
    ap.add_argument("--run", help='only rows of this pass ("run", "redo-1", ...)')
    a = ap.parse_args()
    rows, shots = load(a.project, a.before, a.run)
    if not rows:
        sys.exit("no ledger rows matched")
    judged = [r for r in rows if r["status"] in ("accepted", "rejected", "weak") and not is_screen(r)]
    print(f"ledger rows: {len(rows)}   judged attempts (no screen-only rejects): {len(judged)}")

    # 1. yield per provider
    print("\n1. attempts and accepted clips per provider")
    tried, acc = Counter(r["provider"] for r in rows), Counter(r["provider"] for r in rows if r["status"] == "accepted")
    for p, n in tried.most_common():
        print(f"   {p:14} {n:6} rows   {acc[p]:4} accepted   ({100 * acc[p] / n:.1f}%)")

    # 2. stock items tried on many shots
    print("\n2. stock/still items tried on the most different shots")
    per_item: dict[str, set] = defaultdict(set)
    rel_item: dict[str, list] = defaultdict(list)
    for r in judged:
        if r["provider"] in STOCK:
            k = f'{r["provider"]}:{r["source_id"]}'
            per_item[k].add(r["shot"])
            if relevance(r) is not None:
                rel_item[k].append(relevance(r))
    for k, ss in sorted(per_item.items(), key=lambda kv: -len(kv[1]))[:10]:
        rel = rel_item.get(k) or [0]
        print(f"   {k:24} {len(ss):3} shots   relevance max {max(rel):.0f}, median {sorted(rel)[len(rel) // 2]:.0f}")
    low = [r for r in judged if r["provider"] in STOCK and r["status"] == "rejected" and (relevance(r) or 0) <= 20]
    print(f"   stock/still rejects with relevance <= 20: {len(low)}")

    # 3. replay fix A2 in logged order
    print(f"\n3. film-wide stock skip (fix A2: skip after {SKIP_AFTER}+ rejects at relevance <= {SKIP_RELEVANCE} "
          "on other shots, unless the title shares a word with the subject)")
    bad: dict[str, set] = defaultdict(set)
    skipped = lost = 0
    for r in judged:
        if r["provider"] not in STOCK:
            continue
        k = f'{r["provider"]}:{r["source_id"]}'
        others = bad[k] - {r["shot"]}
        overlap = tokens(r.get("title", "")) & subject_words(shots.get(r["shot"]))
        if len(others) >= SKIP_AFTER and not overlap:
            skipped += 1
            lost += r["status"] == "accepted"
        if r["status"] == "rejected" and relevance(r) is not None and relevance(r) <= SKIP_RELEVANCE:
            bad[k].add(r["shot"])
    print(f"   would skip {skipped} of {sum(1 for r in judged if r['provider'] in STOCK)} stock/still attempts; "
          f"accepted clips among them: {lost}")

    # 4. attempts per shot and caps
    print("\n4. attempts per shot (judged attempts, all providers)")
    by_shot: dict[int, list] = defaultdict(list)
    for r in judged:
        by_shot[r["shot"]].append(r)
    counts = sorted(len(v) for v in by_shot.values())
    print(f"   shots {len(counts)}   median {counts[len(counts) // 2]}   max {counts[-1]}   "
          f"shots with 21+: {sum(c >= 21 for c in counts)}")
    hit = []
    for v in by_shot.values():
        idx = next((i + 1 for i, r in enumerate(v) if r["status"] == "accepted"), None)
        if idx:
            hit.append(idx)
    print(f"   shots that got an accepted clip: {len(hit)}; attempt index of the accept, median "
          f"{sorted(hit)[len(hit) // 2] if hit else '-'}")
    dead = {sid: v for sid, v in by_shot.items() if not any(r["status"] == "accepted" for r in v)}
    n_dead = sum(len(v) for v in dead.values())
    by_prov = Counter(r["provider"] for v in dead.values() for r in v)
    print(f"   shots with NO accepted clip: {len(dead)} of {len(by_shot)}; they used {n_dead} of {len(judged)} "
          f"attempts ({100 * n_dead / len(judged):.0f}%), median {sorted(len(v) for v in dead.values())[len(dead) // 2] if dead else 0} each")
    print("   those attempts by provider: " + ", ".join(f"{p} {n}" for p, n in by_prov.most_common()))
    print("   cap N per shot -> attempts saved / accepted clips lost")
    for n in CAPS:
        saved = sum(max(0, len(v) - n) for v in by_shot.values())
        lost_n = sum(1 for i in hit if i > n)
        print(f"     {n:3}: saved {saved:5} of {len(judged)} ({100 * saved / len(judged):4.1f}%)   lost {lost_n}")


if __name__ == "__main__":
    main()
