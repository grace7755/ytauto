r"""How often does a photo search find anything for a film's shots? Measure before trusting the photo step of the visual ladder.

Searches the photo providers (Pexels, Pixabay, Wikimedia Commons, in `still_order`) with the same queries the photo step uses
(`subject + setting`, `subject`, the first stock query) for every shot of a finished project. Search only: nothing is downloaded and
NO AI call is made, so it costs no judge quota (Pexels/Pixabay keys come from .env; Commons needs none).

A "hit" (any usable picture) proves nothing: Pexels and Pixabay return SOME photo for any query (video-4: 127/127). So a hit only
counts as EXACT when its title / tags / alt text contains a distinctive word of the shot's subject or entities (a model name such as
"AK-12" or "Lobaev"; common words like rifle or russian do not count). Exact coverage is the number that says whether the photo step
can find the named thing. It is still a text match, not the AI's judgement. Commons is asked through the same polite client the
pipeline uses (gap between calls, 429 pause, disk cache in cache\), so a second run is fast.

    .venv\Scripts\python scripts\measure_photo_coverage.py projects\<slug> [--limit 60] [--providers commons]
"""
from __future__ import annotations

import argparse
import json
import re
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


_WORD = re.compile(r"[a-z0-9][a-z0-9\-]*")
# words that name a kind of thing, not the thing: matching them proves nothing (every gun photo says "rifle")
COMMON = {"rifle", "rifles", "pistol", "gun", "guns", "firearm", "firearms", "russian", "russia", "weapon", "weapons", "photo",
          "close", "slow", "motion", "shot", "the", "and", "with", "for", "from", "that", "this", "barrel", "sniper", "military",
          "tactical", "modern", "assault", "carbine", "machine", "compact", "firing", "folding", "stock", "magazine", "sight"}


def distinctive(brief: dict) -> set[str]:
    """The words of the subject and entities that identify THIS thing: a digit or hyphen in them (AK-12, PPK-20, 9x39mm), a
    capitalised proper name (Lobaev, Orsis, Kalashnikov), or any word that is not a common category word."""
    out = set()
    for text in [brief.get("subject") or "", *(brief.get("entities") or [])]:
        for raw in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", text):
            w = raw.lower()
            if w in COMMON or len(w) < 3:
                continue
            out.add(w)
    return out


def is_exact(candidate_text: str, brief: dict) -> bool:
    """Does a photo's own text (title, tags, alt) name a distinctive word of the shot's subject?"""
    words = distinctive(brief)
    have = set(_WORD.findall((candidate_text or "").lower()))
    have |= {w for h in have for w in h.split("-") if len(w) >= 3}
    return bool(words & have)


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
    ap.add_argument("--providers", nargs="+", help="only these (pexels pixabay commons); default every available one")
    a = ap.parse_args()

    from ytauto.cache import Cache
    from ytauto.config import load_settings
    from ytauto.sources.images import make_photo_providers

    s = load_settings(None)
    provs = {k: p for k, p in make_photo_providers(s, Cache(s.cache_dir)).items()
             if p.available() and (not a.providers or k in a.providers)}
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

    anyhit: dict[int, dict[str, int]] = {}
    exact: dict[int, dict[str, int]] = {}
    kind, outcome = {}, {}
    for d in shots:
        sid, b = d["id"], d["brief"]
        kind[sid] = b.get("visual_kind") or "footage"
        outcome[sid] = {"weak": "ended weak", None: "no clip"}.get(status.get(sid), "had a clip")
        anyhit[sid], exact[sid] = {}, {}
        for name, p in provs.items():
            n_any = n_exact = 0
            for q in queries_for(b):
                try:
                    found = p.search(q, 4)
                except Exception as e:              # noqa: BLE001 - one failed search must not end the measurement
                    print(f"  shot {sid} {name}: search failed ({str(e)[:60]})")
                    continue
                n_any += len(found)
                n_exact += sum(is_exact(f"{c.title} {' '.join(c.tags or [])}", b) for c in found)
                if n_exact:
                    break
            anyhit[sid][name], exact[sid][name] = n_any, n_exact
        print(f"  shot {sid:3} [{kind[sid]:8}] " + "  ".join(f"{k} {exact[sid][k]}/{anyhit[sid][k]}" for k in provs)
              + f"   {b.get('subject', '')[:40]}   key words: {', '.join(sorted(distinctive(b))[:4])}")

    print("\n'exact' = the photo's own text names a distinctive word of the subject; 'any' = some photo came back (proves nothing).")
    for title, groups in (("by visual_kind", kind), ("by first-pass outcome", outcome)):
        for label, hits in (("EXACT", exact), ("any", anyhit)):
            print(f"\nshots with at least one {label} photo, {title}:")
            for g, per in sorted(summarise(hits, groups).items()):
                print(f"   {g:12} " + "   ".join(f"{p} {c}/{n}" for p, (c, n) in per.items()))


if __name__ == "__main__":
    main()
