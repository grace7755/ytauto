"""report.json / report.html / sources.csv — sources, timestamps, scores, licences, unresolved issues."""
from __future__ import annotations

import csv
import html
import json
from collections import Counter

from .models import Shot
from .qa import QAResult
from .state import Project, atomic_write


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
            "license": d.license if d else "", "score": d.score.total if d and d.score else None,
            "scores": d.score.model_dump() if d and d.score else {}, "rejected_candidates": len(rej),
            "top_reject_reasons": dict(reasons.most_common(4)), "notes": d.notes if d else "",
        })
    prov = Counter(r["provider"] for r in rows if r["provider"])
    data = {"title": title, "shots": rows, "provider_counts": dict(prov),
            "qa": {"passed": qa.passed, "issues": qa.issues, "info": qa.info} if qa else None, "blockers": blockers}
    atomic_write(proj.path("report.json"), json.dumps(data, indent=2, default=str))

    with proj.path("output", "sources.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["shot", "timeline", "provider", "title", "url", "source_window", "license", "score", "status"])
        for r in rows:
            w.writerow([r["shot"], r["timeline"], r["provider"], r["title"], r["url"], r["source_window"],
                        r["license"], r["score"], r["status"]])

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
<h1>{e(title)}</h1><p>QA: <b>{'PASSED' if qa and qa.passed else 'ISSUES'}</b> &middot; sources: {dict(prov)}</p>
<h3>Blockers</h3><ul>{blk}</ul><h3>QA notes</h3><ul>{issues}</ul>
<img src="output/contact_sheet.jpg" style="max-width:100%">
<table><tr><th>#</th><th>Timeline</th><th>Narration</th><th>Provider</th><th>Source</th><th>Score</th><th>Licence</th><th>Status</th></tr>{body}</table>"""
    atomic_write(proj.path("report.html"), page)
    return data
