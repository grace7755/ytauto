r"""Does storyboard screening's `overlay_tiles` predict what verify later found, and does a given
judge model agree with past screening verdicts?

Takes every downloaded YouTube candidate in past ledgers whose contact sheet is cached: positives
were rejected after download for a watermark/logo or burnt-in text, negatives were accepted. Each
is screened again, fresh (no judge cache), on ONE chosen chain link. Reports how often the video
came out `overlay_suspect` / had its proposed window flagged, per class, plus agreement on
`usable` with the verdict cached from earlier runs (made by whichever link answered then).
Costs one judge call per candidate.

    .venv\Scripts\python scripts\measure_overlay_screen.py [link_index=0] [limit=30]
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
from ytauto.models import Candidate, Shot  # noqa: E402
from ytauto.planner import primary_subject  # noqa: E402
from ytauto.screen import _sb_cache_key, screen_storyboard  # noqa: E402
from calibrate_prerank import load_sb  # noqa: E402

OVERLAY = ("watermark", "text overlay", "burnt-in", "burned-in", "caption", "subtitle")


def cases():
    seen = set()
    for proj in sorted((ROOT / "projects").iterdir()):
        sj, lj = proj / "metadata" / "shots.json", proj / "ledger.jsonl"
        if not (sj.exists() and lj.exists()):
            continue
        shots = {d["id"]: Shot(**d) for d in json.loads(sj.read_text(encoding="utf-8")) if d.get("brief")}
        for line in lj.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["provider"] != "youtube" or r["shot"] not in shots or not r.get("clip_path"):
                continue
            rej = " ".join(r.get("hard_rejects") or []).lower()
            if r["status"] == "accepted":
                label = False
            elif any(k in rej for k in OVERLAY):
                label = True
            else:
                continue                      # rejected for something screening cannot see
            key = (r["source_id"], label)
            if key in seen or load_sb(r["source_id"]) is None:
                continue
            seen.add(key)
            sh = shots[r["shot"]]
            if not sh.brief.subject:
                sh.brief = sh.brief.model_copy(update={"subject": primary_subject(sh.brief)})
            yield sh, r["source_id"], label, r["src_start"]


def main():
    link_i = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    s = load_settings(None)
    link = s.judge_chain[link_i]
    s = s.model_copy(update={"judge_chain": [link], "judge_patience": 0})
    old = Cache(s.cache_dir)
    gem = Judge(s, Cache(Path(tempfile.mkdtemp())))
    print(f"link: {link.provider}/{link.model}")
    stats = {True: [0, 0, 0], False: [0, 0, 0]}        # n, suspect, chosen window flagged
    agree = compared = 0
    for n, (sh, vid, label, at) in enumerate(cases()):
        if n >= limit:
            break
        c = Candidate(provider="youtube", source_id=vid, url=f"https://youtu.be/{vid}")
        sb = load_sb(vid)
        try:
            r = screen_storyboard(gem, c, sh, sb)
        except Exception as e:
            print(f"{vid}: judge failed ({str(e)[:100]})")
            continue
        if r is None:
            continue
        near = [w for w in r["windows"] if w["start"] - 3 <= at <= w["end"] + 3]
        st = stats[label]
        st[0] += 1
        st[1] += bool(r.get("overlay_suspect"))
        st[2] += bool(near and near[0]["overlay"])
        prev = old.get("judge", *_sb_cache_key(c, sh))
        if isinstance(prev, dict):
            compared += 1
            agree += bool(prev.get("usable")) == bool(r["usable"])
        print(f"{'OVERLAY' if label else 'clean  '} {vid} usable={r['usable']!s:5} "
              f"suspect={r.get('overlay_suspect')!s:5} flagged={[w['description'].split()[-1] for w in r['windows'] if w['overlay']]}"
              f" was_usable={prev.get('usable') if isinstance(prev, dict) else '-'}", flush=True)
    for label, (k, sus, fl) in stats.items():
        name = "rejected for overlay" if label else "accepted (clean)   "
        print(f"{name}: {k} videos, overlay_suspect {sus}/{k}, downloaded window flagged {fl}/{k}")
    print(f"usable agrees with the earlier cached verdict {agree}/{compared}; {gem.calls} calls")


if __name__ == "__main__":
    main()
