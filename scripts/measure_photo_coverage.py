r"""How often does a photo search find anything for a film's shots? Measure before trusting the photo step of the visual ladder.

Searches the photo providers (Pexels, Pixabay, Wikimedia Commons, in `still_order`) with the same queries the photo step uses
(`subject + setting`, `subject`, the first stock query) for every shot of a finished project. Search only: nothing is downloaded and
NO AI call is made, so it costs no judge quota (Pexels/Pixabay keys come from .env; Commons needs none).

It prints, per provider and in total, the share of shots with at least one usable hit (16:9-ish, wide enough), split by the
planner's `visual_kind` and by whether the shot's first pass ended weak or without a clip. A hit is not a relevant photo - the AI
still judges each one at relevance >= still_min_relevance - so this is an upper bound on what the photo step can fill.

    .venv\Scripts\python scripts\measure_photo_coverage.py projects\<slug> [--limit 60]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def queries_for(brief: dict) -> list[str]:
    """The photo step's queries (lastresort._still): the place goes into the first one."""
    subj, setting = (brief.get("subject") or "").strip(), (brief.get("setting") or "").strip()
    qs = [f"{subj} {setting}".strip(), subj, *(brief.get("stock_queries") or [])]
    return list(dict.fromkeys(q.strip() for q in qs if q and q.strip()))[:3]


def summarise(hits: dict[int, dict[str, int]], groups: dict[int, str]) -> dict:
    """hits[shot][provider] = usable results; groups[shot] = a label. Returns {group: {provider|"any": (covered, n)}}."""
    out: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for sid, per in hits.items():
        for g in (groups.get(sid, "?"), "all"):
            for prov, n in list(per.items()) + [("any", sum(per.values()))]:
                out[g][prov][0] += n > 0
                out[g][prov][1] += 1
    return {g: {p: tuple(v) for p, v in d.items()} for g, d in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project", type=Path)
    ap.add_argument("--limit", type=int, default=0, help="only the first N shots (0 = all)")
    a = ap.parse_args()

    from ytauto.config import load_settings
    from ytauto.sources.images import make_photo_providers

    s = load_settings(None)
    provs = {k: p for k, p in make_photo_providers(s).items() if p.available()}
    if not provs:
        sys.exit("no photo provider available (keys missing?)")
    shots = [d for d in json.loads((a.project / "metadata" / "shots.json").read_text(encoding="utf-8")) if d.get("brief")]
    if a.limit:
        shots = shots[: a.limit]
    status: dict[int, str] = {}
    led = a.project / "ledger.jsonl"
    if led.exists():
        for line in led.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r["status"] in ("accepted", "weak"):
                    status[r["shot"]] = r["status"]
    print(f"{len(shots)} shots; providers: {', '.join(provs)}; search only, no AI calls")

    hits: dict[int, dict[str, int]] = {}
    kind, outcome = {}, {}
    for d in shots:
        sid, b = d["id"], d["brief"]
        kind[sid] = b.get("visual_kind") or "footage"
        outcome[sid] = {"weak": "ended weak", None: "no clip"}.get(status.get(sid), "had a clip")
        hits[sid] = {}
        for name, p in provs.items():
            n = 0
            for q in queries_for(b):
                try:
                    n += len(p.search(q, 4))
                except Exception as e:              # noqa: BLE001 - one failed search must not end the measurement
                    print(f"  shot {sid} {name}: search failed ({str(e)[:60]})")
                if n:
                    break
            hits[sid][name] = n
        print(f"  shot {sid:3} [{kind[sid]:8}] " + "  ".join(f"{k} {v}" for k, v in hits[sid].items()) + f"   {b.get('subject', '')[:50]}")

    for title, groups in (("by visual_kind", kind), ("by first-pass outcome", outcome)):
        print(f"\nshots with at least one usable photo, {title}:")
        for g, per in sorted(summarise(hits, groups).items()):
            print(f"   {g:12} " + "   ".join(f"{p} {c}/{n}" for p, (c, n) in per.items()))


if __name__ == "__main__":
    main()
