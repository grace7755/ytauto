"""Does screening k storyboards in one judge call agree with screening them one at a time?

Picks shots from past projects that have >= 2 candidates with a cached contact sheet, screens each
candidate alone and then the group in one call (both with today's prompt, fresh - no judge cache),
and reports agreement on `usable` and on the chosen tiles. Costs (k + 1) judge calls per group.

    .venv\\Scripts\\python scripts\\measure_batch_screen.py [groups=8] [k=2]
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ytauto.cache import Cache  # noqa: E402
from ytauto.config import load_settings  # noqa: E402
from ytauto.llm.router import Judge  # noqa: E402
from ytauto.models import Brief, Candidate, Shot  # noqa: E402
from ytauto.planner import primary_subject  # noqa: E402
from ytauto.screen import screen_storyboard, screen_storyboards  # noqa: E402
from calibrate_prerank import load_sb  # noqa: E402


def groups(k: int):
    for proj in sorted((ROOT / "projects").iterdir()):
        sj, lj = proj / "metadata" / "shots.json", proj / "ledger.jsonl"
        if not (sj.exists() and lj.exists()):
            continue
        shots = {d["id"]: Shot(**d) for d in json.loads(sj.read_text(encoding="utf-8")) if d.get("brief")}
        vids: dict[int, list[str]] = {}
        for line in lj.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["provider"] == "youtube" and r["shot"] in shots and load_sb(r["source_id"]):
                lst = vids.setdefault(r["shot"], [])
                if r["source_id"] not in lst:
                    lst.append(r["source_id"])
        for sid, vs in vids.items():
            if len(vs) >= k:
                sh = shots[sid]
                b: Brief = sh.brief
                if not b.subject:
                    sh.brief = b.model_copy(update={"subject": primary_subject(b)})
                yield proj.name, sh, vs[:k]


def main():
    n_groups = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    s = load_settings(None)
    gem = Judge(s, Cache(Path(tempfile.mkdtemp())))       # fresh cache: no stale verdicts
    same_usable = same_tile = total = calls = 0
    for g, (proj, sh, vs) in enumerate(groups(k)):
        if g >= n_groups:
            break
        cands = [Candidate(provider="youtube", source_id=v, url=f"https://youtu.be/{v}") for v in vs]
        sbs = [load_sb(v) for v in vs]
        single = [screen_storyboard(gem, c, sh, sb) for c, sb in zip(cands, sbs)]
        gem.cache = Cache(Path(tempfile.mkdtemp()))      # the batch must not reuse the singles
        batch = screen_storyboards(gem, cands, sh, sbs)
        calls += k + 1
        for v, a, b in zip(vs, single, batch):
            if a is None or b is None:
                continue
            total += 1
            same_usable += a["usable"] == b["usable"]
            ta = {int(w["description"].split()[-1]) for w in a["windows"][:1]}
            tb = {int(w["description"].split()[-1]) for w in b["windows"]}
            near = any(abs(x - y) <= 1 for x in ta for y in tb) if ta and tb else (not ta and not tb)
            same_tile += near
            print(f"{proj[:28]:28} shot {sh.id:2} {v}: single usable={a['usable']!s:5} "
                  f"batch usable={b['usable']!s:5} single best tile={sorted(ta)} batch tiles="
                  f"{[int(w['description'].split()[-1]) for w in b['windows']]}")
    print(f"\n{total} verdicts in groups of {k}: usable agrees {same_usable}/{total}, "
          f"single's best tile within the batch's tiles (+-1) {same_tile}/{total}; ~{calls} calls")


if __name__ == "__main__":
    main()
