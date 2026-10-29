"""ytauto CLI."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import typer

from . import log as logmod
from .cache import Cache
from .config import load_settings
from .llm.router import QuotaExhausted, make_judge
from .pipeline import Pipeline
from .sources.youtube import YouTube, text_score

app = typer.Typer(add_completion=False, help="Transcript -> faceless YouTube video (YouTube-first B-roll).")


def _pipe(transcript: Path, title, vo, config, verbose) -> Pipeline:
    s = load_settings(config)
    p = Pipeline(s, transcript, title, vo)
    logmod.setup(p.proj.root, verbose)
    return p


@app.command()
def run(transcript: Path, title: Optional[str] = typer.Option(None, help="Project/video title (also guides search)"),
        vo: Optional[Path] = typer.Option(None, help="Voice-over audio; segments are re-timed to it"),
        config: Optional[Path] = typer.Option(None, help="YAML overrides"), verbose: bool = False,
        wait: bool = typer.Option(False, help="When the judge's daily quota runs out, wait for the "
                                              "reset and continue instead of stopping")):
    """Full pipeline; resumes automatically if interrupted."""
    p = _pipe(transcript, title, vo, config, verbose)
    try:
        data = p.run(wait=wait)
    except QuotaExhausted as e:
        typer.echo(f"\nPAUSED before selection: {e}.\nRun the same command again after the reset, "
                   "or add --wait.")
        raise typer.Exit(3)
    if data.get("quota_paused_until"):
        typer.echo(f"\nPAUSED: the judge's daily quota is spent. Finished shots are saved; run the "
                   f"same command again after {time.strftime('%Y-%m-%d %H:%M', time.localtime(data['quota_paused_until']))}"
                   " (or add --wait) to continue.")
        raise typer.Exit(3)
    bad = [r for r in data["shots"] if r["status"] in ("weak", "MISSING")]
    typer.echo(f"\nvideo : {p.proj.path('output', 'final.mp4')}\nreport: {p.proj.path('report.html')}")
    typer.echo(f"sources: {data['provider_counts']}   weak/missing shots: {[r['shot'] for r in bad]}")
    if data["blockers"]:
        typer.echo("BLOCKERS:\n  " + "\n  ".join(data["blockers"]))
        raise typer.Exit(1)


@app.command()
def plan(transcript: Path, title: Optional[str] = None, config: Optional[Path] = None):
    """Show shots, visual briefs and search queries without searching or downloading."""
    p = _pipe(transcript, title, None, config, False)
    p.prepare()
    for sh in p.shots:
        b = sh.brief
        typer.echo(f"[{sh.id}] {sh.start:.1f}-{sh.end:.1f}s seg{sh.segment}: {b.visual}\n     queries: {b.youtube_queries}")


@app.command()
def search(query: str, n: int = 10, config: Optional[Path] = None):
    """Search YouTube and show metadata (nothing is downloaded)."""
    s = load_settings(config)
    yt = YouTube(s, Cache(s.cache_dir))
    for c in yt.search(query, n):
        why = yt.prefilter(c)
        yt.details(c) if not why else None
        typer.echo(f"{c.source_id} {c.duration:>6.0f}s {c.height:>4}p {c.upload_date} {c.license[:10]:10} "
                   f"{c.title[:60]}{'   [skip: ' + why + ']' if why else ''}")


@app.command()
def redo(transcript: Path, shot: Optional[int] = typer.Argument(None),
         shot_opt: Optional[int] = typer.Option(None, "--shot", help="Shot id (same as the positional SHOT)."),
         title: Optional[str] = None, vo: Optional[Path] = None, config: Optional[Path] = None):
    """Replace one shot's clip and re-render. `redo script.txt 7` and `redo script.txt --shot 7`
    are the same; the docs and blocker messages use the --shot form."""
    sid = shot if shot is not None else shot_opt
    if sid is None:
        raise typer.BadParameter("give a shot id: `redo script.txt 7` or `--shot 7`")
    p = _pipe(transcript, title, vo, config, False)
    p.redo(sid)
    typer.echo(f"re-rendered {p.proj.path('output', 'final.mp4')}")


@app.command()
def status(transcript: Path, title: Optional[str] = None, config: Optional[Path] = None):
    """Show per-shot status from the ledger."""
    p = _pipe(transcript, title, None, config, False)
    p.prepare()
    for sh in p.shots:
        d = p.proj.accepted(sh.id)
        typer.echo(f"[{sh.id:>2}] {sh.start:>5.1f}-{sh.end:>5.1f}  "
                   + (f"{d.status:8} {d.provider:8} score={d.score.total if d.score else '-'} {d.title[:45]}" if d else "pending"))


@app.command()
def quota(config: Optional[Path] = None):
    """Judge requests used today per link, against its configured daily cap."""
    s = load_settings(config)
    j = make_judge(s, Cache(s.cache_dir))
    if j is None:
        typer.echo("no judge configured (offline mode)")
        return
    for r in j.quota.status(j.chain):
        cap = r["rpd"] if r["rpd"] is not None else "?"
        out = (" OUT until " + time.strftime("%Y-%m-%d %H:%M", time.localtime(r["out_until"]))
               if r["out_until"] else "")
        typer.echo(f"{r['link']:50} {r['used_today']:>5} / {cap:<5} rpm={r['rpm'] or '-'}{out}")


if __name__ == "__main__":
    app()
