"""report.json / report.html / sources.csv — sources, timestamps, scores, licences, unresolved issues."""
from __future__ import annotations

import csv
import html
import json
from collections import Counter

from .models import Shot
from .qa import QAResult
from .state import Project, atomic_write


def kind(provider: str) -> str:
    return "still" if provider.startswith("still_") else "youtube" if provider == "youtube" else "stock" if provider else ""


def substitutions(rows: list[dict]) -> dict:
    """Shots that do not show the exact subject as video, by kind, so no substitution is silent: a photo, a category clip
    (CATEGORY FALLBACK), reused footage, or a weak match."""
    out: dict[str, list[int]] = {"photo": [], "category": [], "reuse": [], "weak": []}
    for r in rows:
        notes = r.get("notes") or ""
        if r["status"] == "weak":
            out["weak"].append(r["shot"])
        elif "CATEGORY FALLBACK:" in notes:
            out["category"].append(r["shot"])
        elif "REUSED footage" in notes:
            out["reuse"].append(r["shot"])
        elif str(r.get("provider", "")).startswith("still_"):
            out["photo"].append(r["shot"])
    return out


def shares(rows: list[dict], shots: list[Shot]) -> dict:
    """Share of the video's running time per kind of visual (youtube / stock / still), in percent."""
    secs: Counter = Counter()
    by_id = {s.id: s for s in shots}
    for r in rows:
        if r["provider"]:
            secs[kind(r["provider"])] += by_id[r["shot"]].duration
    total = sum(secs.values()) or 1.0
    return {k: round(100 * v / total, 1) for k, v in secs.items()}


def reused(rows: list[dict]) -> dict[str, list[int]]:
    """Footage that plays in more than one shot: `url @ source window` -> shot ids. Several different
    sections of one YouTube video are fine and are not listed."""
    by_clip: dict[str, list[int]] = {}
    for r in rows:
        if r["url"]:
            by_clip.setdefault(f"{r['url']} @ {r['source_window']}", []).append(r["shot"])
    return {u: ids for u, ids in by_clip.items() if len(ids) > 1}


def reuse_problems(clips: dict[str, list[int]], shots: list[Shot], max_uses: int = 2, min_gap: float = 60.0) -> list[str]:
    """Breaches of the reuse rules: a clip more than `max_uses` times, or two appearances closer than `min_gap` s."""
    by_id = {s.id: s for s in shots}
    out = []
    for clip, ids in clips.items():
        if len(ids) > max_uses:
            out.append(f"{clip} plays in {len(ids)} shots (max {max_uses}): {ids}")
        ids = sorted(ids, key=lambda i: by_id[i].start)
        for a, b in zip(ids, ids[1:]):
            gap = by_id[b].start - by_id[a].end
            if gap < min_gap:
                out.append(f"{clip}: shots {a} and {b} are only {gap:.0f} s apart (min {min_gap:.0f})")
    return out


def build(proj: Project, title: str, shots: list[Shot], qa: QAResult | None, blockers: list[str]):
    ledger = proj.ledger()
    rows = []
    for sh in shots:
        d = proj.accepted(sh.id)
        rej = [x for x in ledger if x.shot == sh.id and x.status == "rejected"]
        reasons = Counter(r.split(" (")[0].split(":")[0][:50] for x in rej for r in x.hard_rejects)
        rows.append({
            "shot": sh.id, "segment": sh.segment, "timeline": f"{sh.start:.1f}-{sh.end:.1f}", "narration": sh.text,
            "status": d.status if d else "MISSING", "provider": d.provider if d else "", "url": d.url if d else "",
            "title": d.title if d else "", "source_window": f"{d.src_start:.1f}-{d.src_end:.1f}" if d else "",
            "license": d.license if d else "", "credit": d.credit if d else "", "score": d.score.total if d and d.score else None,
            "scores": d.score.model_dump() if d and d.score else {}, "rejected_candidates": len(rej),
            "top_reject_reasons": dict(reasons.most_common(4)), "notes": d.notes if d else "",
        })
    prov = Counter(r["provider"] for r in rows if r["provider"])
    clips = reused(rows)
    data = {"title": title, "shots": rows, "provider_counts": dict(prov), "shares": shares(rows, shots),
            "reused_clips": clips, "reuse_problems": reuse_problems(clips, shots), "substitutions": substitutions(rows),
            "qa": {"passed": qa.passed, "issues": qa.issues, "info": qa.info} if qa else None, "blockers": blockers}
    atomic_write(proj.path("report.json"), json.dumps(data, indent=2, default=str))

    with proj.path("output", "sources.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["shot", "timeline", "provider", "title", "url", "source_window", "license", "score", "status"])
        for r in rows:
            w.writerow([r["shot"], r["timeline"], r["provider"], r["title"], r["url"], r["source_window"],
                        r["license"], r["score"], r["status"]])

    credits = [f"{r['title']} - {r['credit']} - {r['license']} - {r['url']}" for r in rows
               if r["provider"] == "still_commons" and r["license"].lower().startswith("cc by")]
    proj.path("output", "credits.txt").write_text("\n".join(credits) + ("\n" if credits else ""), encoding="utf-8")

    e = html.escape
    body = "".join(
        f"<tr class='{r['status']}'><td>{r['shot']}</td><td>{r['timeline']}</td><td>{e(r['narration'][:70])}</td>"
        f"<td>{r['provider']}</td><td><a href='{e(r['url'])}'>{e(r['title'][:50])}</a><br><small>{r['source_window']}</small></td>"
        f"<td>{r['score'] if r['score'] is not None else ''}</td><td>{e(r['license'])}</td>"
        f"<td>{r['status']}<br><small>{r['rejected_candidates']} rejected</small></td></tr>" for r in rows)
    issues = "".join(f"<li>{e(i)}</li>" for i in (qa.issues if qa else [])) or "<li>none</li>"
    blk = "".join(f"<li>{e(b)}</li>" for b in blockers) or "<li>none</li>"
    page = f"""<!doctype html><meta charset=utf-8><title>{e(title)} report</title>
<style>body{{font:14px system-ui;margin:24px}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:4px 8px;vertical-align:top}}
tr.weak{{background:#fff3cd}}tr.MISSING{{background:#f8d7da}}small{{color:#666}}</style>
<h1>{e(title)}</h1><p>QA: <b>{'PASSED' if qa and qa.passed else 'ISSUES'}</b> &middot; sources: {dict(prov)} &middot; shares: {data['shares']} &middot; clips used twice: {len(data['reused_clips'])} &middot; substitutions: {e(str({k: v for k, v in data['substitutions'].items() if v}))} &middot; credits to add to the description: {len(credits)}</p>
<h3>Blockers</h3><ul>{blk}</ul><h3>QA notes</h3><ul>{issues}</ul>
<img src="output/contact_sheet.jpg" style="max-width:100%">
<table><tr><th>#</th><th>Timeline</th><th>Narration</th><th>Provider</th><th>Source</th><th>Score</th><th>Licence</th><th>Status</th></tr>{body}</table>"""
    atomic_write(proj.path("report.html"), page)
    return data
