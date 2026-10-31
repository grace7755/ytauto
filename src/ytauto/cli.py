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
    if p.wants_render():
        typer.echo(f"\nvideo : {p.proj.path('output', 'final.mp4')}")
    typer.echo(f"report: {p.proj.path('report.html')}")
    typer.echo(f"share of running time: {data['shares']}   clips used twice: {len(data['reused_clips'])}")
    for prob in data["reuse_problems"]:
        typer.echo(f"REUSE RULE BROKEN: {prob}")
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


def _shot_ids(text: str) -> list[int]:
    """`7`, `3,7,12` or `3 7 12` -> [3, 7, 12]."""
    try:
        return [int(x) for x in text.replace(",", " ").split()]
    except ValueError:
        raise typer.BadParameter(f"shot ids must be numbers, e.g. `--shot 3,7,12` (got {text!r})")


@app.command()
def redo(transcript: Path, shot: Optional[str] = typer.Argument(None),
         shot_opt: Optional[str] = typer.Option(None, "--shot", help="Shot id(s): 7 or 3,7,12 (same as the positional SHOT)."),
         title: Optional[str] = None, vo: Optional[Path] = None, config: Optional[Path] = None,
         render: Optional[bool] = typer.Option(None, "--render/--no-render",
                                               help="Build the ffmpeg final.mp4 (default: only when not exporting to Remotion)")):
    """Replace one or more shots' clips in one go: `redo script.txt --shot 3,7,12`. All shots run in one
    process, side by side (`shot_workers`), with one export and one QA at the end."""
    text = shot if shot is not None else shot_opt
    if not text:
        raise typer.BadParameter("give shot id(s): `redo script.txt 7` or `--shot 3,7,12`")
    p = _pipe(transcript, title, vo, config, False)
    if render is not None:
        p.s.ffmpeg_render = "always" if render else "never"
    data = p.redo(_shot_ids(text))
    if p.quota_until:
        typer.echo(f"PAUSED: the judge's daily quota is spent; run the same redo again after "
                   f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(p.quota_until))}.")
        raise typer.Exit(3)
    if p.wants_render():
        typer.echo(f"re-rendered {p.proj.path('output', 'final.mp4')}")
    else:
        typer.echo("clips re-selected and exported (no ffmpeg render: the film is rendered in Remotion)")
    for b in data["blockers"]:
        typer.echo(f"BLOCKER: {b}")
    if data["qa"] and data["qa"]["issues"]:
        typer.echo("QA:\n  " + "\n  ".join(data["qa"]["issues"]))


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


@app.command()
def finish(transcript: Path, vo: Path = typer.Option(..., help="Voice-over audio"),
           title: Optional[str] = None, config: Optional[Path] = None,
           spec: Optional[Path] = typer.Option(None, help="YAML: music, cards, sfx, grain, particles"),
           assets: Optional[Path] = typer.Option(None, help="Asset library (music/sfx/overlays) for names in the spec"),
           out: Optional[Path] = typer.Option(None, help="Output file (default output/final_finished.mp4)")):
    """Finish a selected project: voice-over, word-timed captions, titles, ducked music, SFX, grain."""
    import yaml

    from . import finish as F
    from .transcript import load_transcript

    p = _pipe(transcript, title, None, config, False)
    video = p.proj.path("output", "final.mp4")
    if not video.exists():
        raise typer.BadParameter(f"run the project first: {video} is missing (with `remotion_export` set, "
                                 "add `ffmpeg_render: always` to the config to build it)")
    data = yaml.safe_load(spec.read_text(encoding="utf-8")) if spec else {}

    def asset(v, sub):
        if not v:
            return None
        f = Path(v)
        if f.exists():
            return f
        hit = F.pick(assets, sub, str(v).lower()) if assets else None
        if hit is None:
            raise typer.BadParameter(f"asset not found: {v}")
        return hit

    fs = F.FinishSpec(
        vo=vo, music=asset(data.get("music"), "audio/music"), music_db=float(data.get("music_db", -14)),
        sfx=[F.Sfx(asset(x["file"], "audio/sfx"), float(x["at"]), float(x.get("gain_db", -14)))
             for x in data.get("sfx", [])],
        cards=[F.TextCard(c["text"], float(c["start"]), float(c["end"]), c.get("kind", "lower"))
               for c in data.get("cards", [])],
        grain=asset(data.get("grain"), "video/overlays"), grain_opacity=float(data.get("grain_opacity", 0.10)),
        particles=asset(data.get("particles"), "video/overlays"), particles_from=data.get("particles_from"),
        tail=float(data.get("tail", 1.5)), captions=bool(data.get("captions", True)),
        font=data.get("font", "Arial Black"))
    target = out or p.proj.path("output", "final_finished.mp4")
    F.render(video, fs, load_transcript(transcript), target, p.proj.path("finish"))
    typer.echo(f"finished: {target}")


@app.command("export-remotion")
def export_remotion(transcript: Path, out: Path = typer.Option(..., help="Folder for sNN.mp4 + shots.json + words.json"),
                    title: Optional[str] = None, vo: Optional[Path] = typer.Option(None, help="Voice-over: also write words.json"),
                    config: Optional[Path] = None,
                    bake: Optional[bool] = typer.Option(None, help="Bake the film look in (default: remotion_bake)")):
    """Export a selected project's clips for a Remotion composition (1080p30, padded tail, optional
    baked look). `run` does this during selection when `remotion_export` is set; this is for an
    existing project or after a `redo`. Clips already exported from the same source are skipped."""
    from .export import write_words

    p = _pipe(transcript, title, vo, config, False)
    if vo is None and p.proj.path("metadata", "aligned_segments.json").exists():
        # without the voice-over, prepare() would re-split the transcript on its own timestamps:
        # different shots than the project's (and it rewrites metadata/shots.json)
        raise typer.BadParameter("this project was timed to a voice-over: pass the same --vo")
    if bake is not None:
        p.s = p.s.model_copy(update={"remotion_bake": bake})
    p.prepare()
    ex = p.make_exporter(out)
    missing = []
    for sh in p.shots:
        d = p.proj.accepted(sh.id)
        if d is None:
            missing.append(sh.id)
            continue
        ex.submit(sh, d.clip_path)
    shots_json = ex.finish(p.shots)
    typer.echo(f"clips + {shots_json}")
    if vo:
        typer.echo(f"words: {write_words(vo, p.segments, out / 'words.json')}")
    if missing:
        typer.echo(f"shots without a clip (not exported): {missing}")
        raise typer.Exit(1)


@app.command("library-add")
def library_add(project: Path = typer.Argument(..., help="A project folder (projects/<slug>)"),
                skip: list[int] = typer.Option([], help="Shot ids NOT to add (clips you replaced by hand)"),
                config: Optional[Path] = None):
    """Add a finished project's accepted clips to the shared footage library (`run` does this for
    every clip it accepts; this back-fills projects made before the library existed)."""
    from .inspect.embed import Clip
    from .library import open_library
    from .models import Shot
    from .state import Project

    s = load_settings(config)
    lib = open_library(s.cache_dir / "library", Clip(s.cache_dir / "models"))
    shots = {d["id"]: Shot(**d) for d in json.loads((project / "metadata" / "shots.json").read_text(encoding="utf-8"))}
    added, proj = 0, Project(project)
    for sid, sh in shots.items():
        d = proj.accepted(sid)                   # the current pick only: not clips later replaced
        if d is None or d.status != "accepted" or sid in skip or d.provider.startswith("still_"):
            continue
        b = sh.brief
        added += lib.add(clip=Path(d.clip_path), provider=d.provider, source_id=d.source_id, url=d.url,
                         title=d.title, license=d.license, src_start=d.src_start, src_end=d.src_end,
                         subject=(b.subject or b.visual) if b else "", setting=b.setting if b else "",
                         text=sh.text, project=project.name)
    typer.echo(f"added {added} clips; library has {len(lib.entries)}")


@app.command()
def doctor(live: bool = typer.Option(False, help="Also send one tiny request to every judge link "
                                                 "(costs one request of each link's quota)"),
           config: Optional[Path] = None):
    """Check tools, optional features and the judge chain before a long run."""
    from .doctor import run_all

    s = load_settings(config)
    checks = run_all(s, Cache(s.cache_dir), live)
    for name, ok, detail in checks:
        typer.echo(f"{'OK ' if ok else 'NO '} {name:42} {detail}")
    hard = [c for c in checks if not c[1] and c[0] in ("ffmpeg", "ffprobe", "yt-dlp", "opencv", "judge chain")]
    if hard:
        raise typer.Exit(1)


@app.command()
def batch(scripts: list[Path] = typer.Argument(..., help="Transcript files, or folders of .txt/.srt/.vtt/.json"),
          config: Optional[Path] = None, verbose: bool = False,
          wait: bool = typer.Option(True, help="Wait out the judge's daily quota instead of stopping")):
    """Run several transcripts one after another (title = file name). A failing video is
    reported and skipped; the others still run. Each one is resumable like `run`."""
    files: list[Path] = []
    for p in scripts:
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in (".txt", ".srt", ".vtt", ".json"))
        else:
            files.append(p)
    results = []
    for i, f in enumerate(files, 1):
        typer.echo(f"\n=== [{i}/{len(files)}] {f.name}")
        t0 = time.time()
        try:
            pipe = _pipe(f, None, None, config, verbose)
            data = pipe.run(wait=wait)
            bad = [r["shot"] for r in data["shots"] if r["status"] in ("weak", "MISSING")]
            state = "BLOCKED" if data["blockers"] else ("PAUSED" if data.get("quota_paused_until") else "done")
            results.append((f.name, state, f"{(time.time() - t0) / 60:.1f} min, weak/missing {bad}",
                            str(pipe.proj.path("output", "final.mp4"))))
        except Exception as e:                     # noqa: BLE001 - one bad script must not stop the queue
            results.append((f.name, "FAILED", str(e)[:120], ""))
    typer.echo("\n=== batch summary")
    for name, state, detail, out in results:
        typer.echo(f"{state:8} {name:40} {detail}  {out}")
    if any(r[1] != "done" for r in results):
        raise typer.Exit(1)


if __name__ == "__main__":
    app()


@app.command("sfx-plan")
def sfx_plan(events: Path = typer.Argument(..., help='JSON list of {"kind": hook|chapter|zoom|stat|statbig|impact, "at": seconds}'),
             vo: Path = typer.Option(..., help="The voice-over file (levels are set against it)"),
             audio_dir: Path = typer.Option(..., help="Folder of <name>.mp3 sound files"),
             total: float = typer.Option(..., help="Video length in seconds"),
             out: Path = typer.Option(..., help="Where to write sfx.json"),
             beds: Optional[Path] = typer.Option(None, help='JSON list of {"name", "from", "to"} ambience beds'),
             shots: Optional[Path] = typer.Option(None, help="shots.json: chapter and zoom effects snap onto its cuts")):
    """Choose the few sound effects a film needs and set their levels under the voice."""
    from . import sfx

    p = sfx.plan(json.loads(events.read_text(encoding="utf-8")), vo, audio_dir, total,
                 beds=json.loads(beds.read_text(encoding="utf-8")) if beds else None,
                 cuts=[float(s["t0"]) for s in json.loads(shots.read_text(encoding="utf-8"))] if shots else None)
    out.write_text(json.dumps(p, indent=1), encoding="utf-8")
    typer.echo(f"{len(p['cues'])} cues, {len(p['beds'])} beds, {len(p['dropped'])} events dropped (too close or too many) -> {out}")


@app.command("sfx-audit")
def sfx_audit(plan: Path = typer.Argument(..., help="sfx.json from `sfx-plan`"),
              vo: Path = typer.Option(..., help="The voice-over file"),
              audio_dir: Path = typer.Option(..., help="Folder of <name>.mp3 sound files"),
              shots: Path = typer.Option(..., help="shots.json (cut times t0), as written by export-remotion"),
              video: Optional[Path] = typer.Option(None, help="The rendered video: its loudness and peak are checked too")):
    """Check every sound effect (level under the voice, sync with the cuts, spacing, files) and the whole mix."""
    from . import sfx

    r = sfx.audit(sfx.load(plan), vo, audio_dir, json.loads(shots.read_text(encoding="utf-8")), video)
    for row in r["rows"]:
        typer.echo(f"{row['at']:7.2f}s {row['kind']:8} {row['name']:14} margin={row.get('margin', '-')} cut_offset={row.get('cut_offset', '-')}")
    typer.echo(f"mix: {r['mix_lufs']:.1f} LUFS, peak {r['mix_peak']:.1f} dBTP"
               + (f"; video: {r['video_lufs']:.1f} LUFS, peak {r['video_peak']:.1f} dBTP" if "video_lufs" in r else ""))
    if r["issues"]:
        typer.echo("ISSUES:\n  " + "\n  ".join(r["issues"]))
        raise typer.Exit(1)
    typer.echo("audio check passed")


@app.command("review-shots")
def review_shots(project: Path = typer.Argument(..., help="A project folder (projects/<slug>)"),
                 below: int = typer.Option(60, help="List shots the review rates under this (0-100)"),
                 config: Optional[Path] = None):
    """Before rendering: have the AI compare every shot's picture with its narration (one call per ~10
    shots) and list the ones that do not show what is being said. Fix them with `redo --shot N`."""
    from .models import Shot
    from .state import Project

    s = load_settings(config)
    gem = make_judge(s, Cache(s.cache_dir))
    if gem is None:
        typer.echo("no AI judge is configured; nothing to review")
        raise typer.Exit(2)
    from .review import review

    shots = [Shot(**d) for d in json.loads((project / "metadata" / "shots.json").read_text(encoding="utf-8"))]
    rows = review(gem, Project(project), shots)
    (project / "output").mkdir(exist_ok=True)
    (project / "output" / "review.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    weak = [r for r in rows if 0 <= r["match"] < below]
    unrated = [r for r in rows if r["match"] < 0]
    typer.echo(f"reviewed {len(rows)} shots: {len(weak)} below {below}, {len(unrated)} not rated -> {project / 'output' / 'review.json'}")
    for r in weak:
        typer.echo(f"  shot {r['shot']:3} match={r['match']:3}  {r['narration'][:60]!r}: {r['why']}")
    if weak or unrated:
        raise typer.Exit(1)


@app.command("film")
def film_cmd(project: Path = typer.Argument(..., help="A finished project folder (projects/<slug>) made with --vo"),
             out: Path = typer.Option(..., help="public/<slug> of the Remotion project: clips/ (from export-remotion --out <this>/clips) "
                                                "and vo.mp3 are read here, film.json is written here"),
             title: str = typer.Option(..., help="Film title (title card, and the context for the AI)"),
             audio_dir: Optional[Path] = typer.Option(None, help="Shared sound-effect folder inside public/ (<name>.mp3 files): "
                                                                 "plans the sound and writes it into film.json"),
             next_video: Optional[str] = typer.Option(None, "--next", help="Title of the next video for the end card; no end card without it"),
             no_title: bool = typer.Option(False, help="No title card"),
             music: Optional[str] = typer.Option(None, help="Music file name inside <out> (optional)"),
             config: Optional[Path] = None):
    """Write `film.json` for the reusable DocFilm composition: camera moves, transitions, chapters, stat cards,
    place chips, captions' words and the sound plan, so a new film needs no new Remotion code."""
    from . import film as F
    from . import sfx
    from .inspect.probe import probe
    from .models import Segment

    s = load_settings(config)
    segs = [Segment(**d) for d in json.loads((project / "metadata" / "aligned_segments.json").read_text(encoding="utf-8"))]
    clips = out / "clips"
    shots = json.loads((clips / "shots.json").read_text(encoding="utf-8"))
    words = json.loads((clips / "words.json").read_text(encoding="utf-8"))
    n_shots = len(json.loads((project / "metadata" / "shots.json").read_text(encoding="utf-8")))
    broken = F.check_complete(n_shots, shots, clips)
    if broken:
        typer.echo("film.json NOT written, the clips do not make a whole film:\n  " + "\n  ".join(broken))
        raise typer.Exit(1)
    vo = out / "vo.mp3"
    info = probe(vo)
    seconds = info.duration if info else segs[-1].end
    gem = make_judge(s, Cache(s.cache_dir))
    facts = F.extract_facts(gem, segs, title, seconds)
    film = F.build(slug=out.name, title=title, segments=segs, shots=shots, words=words, seconds=seconds, facts=facts,
                   fps=s.fps, next_video=next_video, show_title=not no_title, music=music)
    if audio_dir is not None:
        plan = sfx.plan(film["events"], vo, audio_dir, film["seconds"] + film["tail"], cuts=[x["t0"] for x in film["shots"]])
        film["sfx"], film["audioBase"] = plan, audio_dir.name
    path = F.write(film, out / "film.json")
    typer.echo(f"{path}: {len(film['shots'])} shots, {len(film['chapters'])} chapters, {len(film['stats'])} stat cards, "
               f"{len(film['places'])} place chips, {len(film['sfx']['cues']) if film['sfx'] else 0} sound cues")
