r"""Could a free local CLIP check have rejected the obvious stock/still misses before the AI was asked?

On video-4 (Russian firearms, 127 shots) 442 stock/still attempts were judged by the AI at relevance <= 20 (a waterfall for a gun line),
and verification was 29% of the footage stage. This script measures, offline and with no AI calls, whether CLIP similarity between a
frame of the downloaded stock clip (or photo) and the shot's subject separates those misses from the clips the AI rated relevant.

It prints the AUC (0.5 = useless, 1.0 = perfect) and, for a range of thresholds, how many AI verifications a "reject below this
similarity" gate would have skipped and how many clips the AI rated relevant it would have lost. A gate is only worth turning on if
it skips a lot and loses none (or almost none).

    .venv\Scripts\python scripts\measure_clip_gate.py projects\<slug> [--before ISO] [--run run]

Needs the downloaded files still on disk: projects\<slug>\sources\stock\ and \stills\, and the Mixkit folder (mixkit_dir).
CLIP is the local ONNX model in cache\models (downloads once if missing).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REL_POS = 60.0             # the AI rated it relevant (the pipeline's min_relevance)
REL_NEG = 20.0             # the AI rated it an obvious miss
VIDEO_PROVIDERS = {"pexels", "pixabay", "coverr", "mixkit"}
STILL_PREFIX = "still_"


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def summarise(items: list[tuple[float, float]], rel_pos: float = REL_POS, rel_neg: float = REL_NEG) -> dict:
    """items = (clip similarity, AI relevance). Returns AUC and a threshold table.

    A threshold row means: reject every clip whose similarity is below `t`. `skipped` counts all AI verifications that gate would
    have saved; `lost_pos` the clips the AI rated relevant that it would have thrown away; `caught_neg` the obvious misses it caught."""
    pos = [s for s, r in items if r >= rel_pos]
    neg = [s for s, r in items if r <= rel_neg]
    out = {"n": len(items), "pos": len(pos), "neg": len(neg), "auc": auc(pos, neg), "table": []}
    sims = sorted({round(s, 4) for s, _ in items})
    for t in sims[:: max(1, len(sims) // 24)]:
        out["table"].append({"t": t, "skipped": sum(s < t for s, _ in items), "lost_pos": sum(s < t for s in pos),
                             "caught_neg": sum(s < t for s in neg)})
    safe = [row for row in out["table"] if row["lost_pos"] == 0]
    out["best_safe"] = max(safe, key=lambda r: r["skipped"]) if safe else None
    return out


def find_source(project: Path, row: dict, mixkit_dir: Path) -> Path | None:
    prov, sid = row["provider"], str(row["source_id"])
    if prov.startswith(STILL_PREFIX):
        p = project / "sources" / "stills" / f"{prov}_{sid}.jpg"
        return p if p.exists() else None
    if prov == "mixkit":
        hits = list(mixkit_dir.glob(f"**/*{sid}*.mp4")) if mixkit_dir.exists() else []
        return hits[0] if hits else None
    p = project / "sources" / "stock" / f"{prov}_{sid}.mp4"
    return p if p.exists() else None


def frame_of(path: Path):
    """The middle frame of a video, or the picture itself, as a PIL image (None if unreadable)."""
    from PIL import Image
    if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None
    import cv2
    cap = cv2.VideoCapture(str(path))
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(n // 2))
    ok, f = cap.read()
    cap.release()
    return Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project", type=Path)
    ap.add_argument("--before", help="only rows with ts before this ISO time")
    ap.add_argument("--run", help='only rows of this pass ("run", "redo-1", ...)')
    a = ap.parse_args()

    from ytauto.config import load_settings
    from ytauto.inspect.embed import Clip, query_text
    from ytauto.models import Brief

    s = load_settings(None)
    rows = [json.loads(l) for l in (a.project / "ledger.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.before:
        rows = [r for r in rows if (r.get("ts") or "") < a.before]
    if a.run:
        rows = [r for r in rows if r.get("run") == a.run]
    shots = {d["id"]: d for d in json.loads((a.project / "metadata" / "shots.json").read_text(encoding="utf-8"))}
    stock = [r for r in rows if r["provider"] != "youtube" and r["status"] in ("accepted", "rejected", "weak")
             and (r.get("score") or {}).get("relevance") is not None and shots.get(r["shot"], {}).get("brief")]
    print(f"stock/still attempts with an AI relevance: {len(stock)}")

    clip = Clip(s.cache_dir / "models")
    texts: dict[int, np.ndarray] = {}
    frames, meta, missing = [], [], 0
    for r in stock:
        path = find_source(a.project, r, s.mixkit_dir)
        im = frame_of(path) if path else None
        if im is None:
            missing += 1
            continue
        frames.append(im)
        meta.append(r)
    print(f"source file found for {len(frames)} of them ({missing} missing on disk)")
    if not frames:
        sys.exit("nothing to measure: the downloaded stock/still files are not on disk")

    emb = clip.images(frames)
    items = []
    for e, r in zip(emb, meta):
        sid = r["shot"]
        if sid not in texts:
            texts[sid] = clip.text(query_text(Brief.model_validate(shots[sid]["brief"])))
        items.append((float(e @ texts[sid]), float(r["score"]["relevance"])))

    res = summarise(items)
    print(f"\nAI relevant (>= {REL_POS:.0f}): {res['pos']}   obvious misses (<= {REL_NEG:.0f}): {res['neg']}   AUC {res['auc']:.2f}")
    print("\nreject below similarity t -> AI verifications skipped / relevant clips lost / obvious misses caught")
    for row in res["table"]:
        print(f"   t={row['t']:.3f}   skipped {row['skipped']:4} of {res['n']}   lost {row['lost_pos']:3}   caught {row['caught_neg']:4} of {res['neg']}")
    b = res["best_safe"]
    print("\nlargest gate that loses no relevant clip: " +
          (f"t={b['t']:.3f} skips {b['skipped']} of {res['n']} AI verifications ({100 * b['skipped'] / res['n']:.0f}%)"
           if b else "none (CLIP does not separate them here)"))


if __name__ == "__main__":
    main()
