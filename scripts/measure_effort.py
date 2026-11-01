r"""Does a lower reasoning effort give the same verdicts, faster? Measure before setting `judge_effort` in the config.

Re-judges N of a finished project's accepted clips (and, with --stills, accepted photos) twice with the verify prompt - once at
each effort level - through the normal judge chain (GPT Luna first), then compares: time per call, the AI's relevance score, and the
verdict flags (presenter, watermark, text overlay, black-and-white, AI-generated). The first level is the reference.

THIS USES YOUR CHATGPT PLAN: 2 calls per clip. Start with 12.

    .venv\Scripts\python scripts\measure_effort.py projects\<slug> --n 12 --levels medium low
    .venv\Scripts\python scripts\measure_effort.py projects\<slug> --n 12 --levels medium low --task aicheck
    .venv\Scripts\python scripts\measure_effort.py projects\<slug> --n 24 --levels medium medium --task aicheck   # noise floor

--task verify (default) uses the 12-field verdict on a frame grid; --task aicheck asks the single question "real footage or AI?".
Read the result per task, then set e.g.  judge_effort: {aicheck: low}  in a config file. Keep medium wherever the flags disagree.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FLAGS = ("presenter_on_camera", "watermark_or_logo", "text_overlay", "black_and_white", "ai_or_animated")


def compare(ref: list[dict], other: list[dict], tol: float = 15.0) -> dict:
    """ref/other = one verdict per clip, in the same order: {"relevance": x, <flags>, "secs": s}. Returns the agreement summary."""
    n = len(ref)
    rel = [abs(float(a.get("relevance", 0)) - float(b.get("relevance", 0))) for a, b in zip(ref, other)]
    flag_diffs = {f: sum(bool(a.get(f)) != bool(b.get(f)) for a, b in zip(ref, other)) for f in FLAGS}
    return {
        "n": n,
        "secs_ref": sum(a["secs"] for a in ref) / max(n, 1),
        "secs_other": sum(b["secs"] for b in other) / max(n, 1),
        "relevance_within_tol": sum(d <= tol for d in rel),
        "relevance_mean_diff": sum(rel) / max(n, 1),
        "flag_disagreements": {f: c for f, c in flag_diffs.items() if c},
        "clips_with_any_flag_disagreement": sum(any(bool(a.get(f)) != bool(b.get(f)) for f in FLAGS) for a, b in zip(ref, other)),
    }


def disagreements(ref: list[dict], other: list[dict], ids: list) -> list[str]:
    """One line per clip whose flags differ: which flag, and what each run said (reference -> test)."""
    out = []
    for i, a, b in zip(ids, ref, other):
        diff = [f"{f} {bool(a.get(f))} -> {bool(b.get(f))}" for f in FLAGS if bool(a.get(f)) != bool(b.get(f))]
        if diff:
            out.append(f"shot {i}: " + "; ".join(diff))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project", type=Path)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--levels", nargs=2, default=["medium", "low"], metavar=("REFERENCE", "TEST"),
                    help="two effort levels; `medium medium` measures the noise (the same setting twice)")
    ap.add_argument("--task", choices=["verify", "aicheck"], default="verify")
    a = ap.parse_args()

    from ytauto.cache import Cache
    from ytauto.config import load_settings
    from ytauto.inspect.verify import PROMPT, VERIFY_SCHEMA
    from ytauto.llm.router import make_judge
    from ytauto.models import Shot

    s = load_settings(None)
    gem = make_judge(s, Cache(s.cache_dir))
    if gem is None:
        sys.exit("no judge configured")
    shots = {d["id"]: Shot(**d) for d in json.loads((a.project / "metadata" / "shots.json").read_text(encoding="utf-8"))
             if d.get("brief")}
    rows = [json.loads(l) for l in (a.project / "ledger.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    acc = {}
    for r in rows:
        if r["status"] == "accepted" and r.get("clip_path") and Path(r["clip_path"]).exists() and r["shot"] in shots:
            acc[r["shot"]] = r
    picks = list(acc.values())[:: max(1, len(acc) // a.n)][: a.n]
    print(f"{len(picks)} accepted clips; task {a.task}; levels {a.levels}; {2 * len(picks)} calls on your plan")

    same = a.levels[0] == a.levels[1]       # `--levels medium medium` = the noise floor: the SAME setting twice
    labels = [f"{lv}#{k + 1}" if same else lv for k, lv in enumerate(a.levels)]
    runs: list[list[dict]] = [[], []]       # one list per run, so two runs at the same level stay apart
    for r in picks:
        sh, clip = shots[r["shot"]], Path(r["clip_path"])
        for k, lv in enumerate(a.levels):
            s.judge_effort = {a.task: lv}
            gem.s = s
            t0 = time.perf_counter()
            try:
                if a.task == "verify":
                    b = sh.brief
                    prompt = PROMPT.format(text=sh.text, subject=b.subject or b.visual, visual=b.visual, entities=b.names())
                    v = gem.json_video_file(clip, prompt, VERIFY_SCHEMA, fps=2.0, task="verify")   # no cache_key: a fresh call
                else:
                    v = {"ai_or_animated": bool(gem.confirm_ai_generated(clip, cache_key=None)), "relevance": 0}
            except Exception as e:                       # noqa: BLE001 - one failed call must not end the measurement
                print(f"  shot {r['shot']} {lv}: failed ({str(e)[:80]})")
                v = {}
            v["secs"] = time.perf_counter() - t0
            runs[k].append(v)
        print(f"  shot {r['shot']:3}: " + "   ".join(f"{labels[k]} {runs[k][-1]['secs']:.1f}s" for k in (0, 1)))

    ok = [i for i in range(len(picks)) if all(len(runs[k][i]) > 1 for k in (0, 1))]
    ref, other = [runs[0][i] for i in ok], [runs[1][i] for i in ok]
    res = compare(ref, other)
    print(f"\n{res['n']} clips compared (reference {labels[0]}, test {labels[1]})")
    print(f"time per call: {res['secs_ref']:.1f}s -> {res['secs_other']:.1f}s")
    print(f"relevance within 15 points: {res['relevance_within_tol']} of {res['n']} (mean difference {res['relevance_mean_diff']:.1f})")
    print(f"flag disagreements: {res['flag_disagreements'] or 'none'}; clips with any: {res['clips_with_any_flag_disagreement']}")
    for line in disagreements(ref, other, [picks[i]["shot"] for i in ok]):
        print("   " + line)
    if same:
        print("\nSame setting twice: any disagreement above is the model not giving the same answer twice (noise), not the effort.")


if __name__ == "__main__":
    main()
