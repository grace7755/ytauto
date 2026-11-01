"""End-to-end orchestration with resume: parse -> plan -> select -> assemble -> QA (loop) -> report."""
from __future__ import annotations

import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import timing
from .align import align_segments
from .assemble import assemble, cut_clip, tail_for
from .cache import Cache
from .config import Settings
from .llm.router import Judge, JudgeError, QuotaExhausted, make_judge
from .log import log
from .models import Brief, Segment, Shot
from .planner import plan
from .qa import QAResult, run_qa, run_qa_clips
from .report import build
from .select import Selector
from .sources.stock import make_providers
from .sources.youtube import YouTube
from .state import Project, atomic_write, slugify
from .transcript import load_transcript, split_shots


class Pipeline:
    def __init__(self, s: Settings, transcript: Path, title: str | None = None, vo: Path | None = None):
        self.s, self.transcript, self.vo = s, transcript, vo
        self.title = title or transcript.stem.replace("_", " ").replace("-", " ").title()
        self.proj = Project(s.projects_dir / slugify(self.title))
        self.cache = Cache(s.cache_dir)
        self.gem: Judge | None = make_judge(s, self.cache)
        self.yt = YouTube(s, self.cache)
        self.providers = make_providers(s)
        self.shots: list[Shot] = []
        self.segments: list[Segment] = []
        self.quota_until: float | None = None    # set when the judge's daily quota paused the run
        self.exporter = None                     # export.Exporter while a run exports for Remotion

    # ------------------------------------------------------------------
    def prepare(self):
        inp = self.proj.path("input")
        shutil.copy2(self.transcript, inp / self.transcript.name)
        segs = load_transcript(self.transcript)
        if self.vo:
            shutil.copy2(self.vo, inp / self.vo.name)
            cached = self.proj.path("metadata", "aligned_segments.json")
            if cached.exists():
                segs = [Segment(**d) for d in json.loads(cached.read_text(encoding="utf-8"))]
            else:
                segs = align_segments(segs, self.vo)
                atomic_write(cached, json.dumps([x.model_dump() for x in segs], indent=2))
        self.segments = segs
        self.shots = split_shots(segs, self.s.shot_target, self.s.shot_min, self.s.shot_max)
        briefs = plan(segs, self.shots, self.title, self.s, self.cache, self.gem)
        for sid, upd in self.s.brief_overrides.items():
            if sid in briefs:
                briefs[sid] = briefs[sid].model_copy(update=upd)
                log.info("shot %d: brief overridden by config (%s)", sid, ", ".join(upd))
        for sh in self.shots:
            sh.brief = briefs[sh.id]
        atomic_write(self.proj.path("metadata", "shots.json"), json.dumps([x.model_dump() for x in self.shots], indent=2))
        log.info("%d segments -> %d shots (%.1fs total)", len(segs), len(self.shots), self.shots[-1].end)

    def select_all(self, only: set[int] | None = None) -> list[str]:
        """Select every shot (or `only` these), `shot_workers` at a time. Shots are independent
        apart from source reuse, which `Selector` guards with a lock; blockers keep shot order."""
        sel = Selector(self.s, self.proj, self.cache, self.yt, self.gem, self.providers, self.shots)
        todo = [sh for sh in self.shots if only is None or sh.id in only]
        self.quota_until = None
        paused = threading.Event()
        lock = threading.Lock()

        started = [0]                      # shots a worker has picked up, for the prefetcher
        cond = threading.Condition()

        def prefetch():
            """Prepare the judge-free work of upcoming shots while the workers wait on the judge.
            Stays `prefetch_ahead` shots in front of the workers, so it never runs far ahead of
            what a pause or an interrupt would leave undone."""
            for i, sh in enumerate(todo):
                if i < workers:
                    continue                      # the workers start these at once
                with cond:
                    cond.wait_for(lambda: paused.is_set() or i < started[0] + self.s.prefetch_ahead)
                if paused.is_set():
                    return                        # quota pause, or the run has ended
                if i < started[0]:
                    continue                      # a worker already has it
                try:
                    sel.warm(sh)
                except Exception as e:            # purely an optimisation: never fail the run
                    log.debug("prefetch for shot %d failed: %s", sh.id, str(e)[:160])

        def one(sh: Shot) -> str | None:
            with cond:
                started[0] += 1
                cond.notify_all()
            if paused.is_set():
                return f"shot {sh.id}: not started - judge quota spent; `run` again to continue"
            log.info("== shot %d/%d [%.1f-%.1fs] %s", sh.id + 1, len(self.shots), sh.start, sh.end, sh.text[:60])
            try:
                d = sel.select(sh)
            except QuotaExhausted as e:
                # Every link is out of its daily quota. Starting more shots would only turn each
                # into an "unavailable" blocker, so stop here; `run` resumes from this point.
                paused.set()
                with lock:
                    self.quota_until = min(self.quota_until or e.until, e.until)
                return f"shot {sh.id}: paused - judge quota spent ({e}); `run` again (or `run --wait`) to continue"
            except JudgeError as e:
                # The judge chain is down, not the footage. Record it as its own blocker and keep
                # going: later shots may succeed once a link's circuit breaker reopens, and the
                # shot can be retried with `redo --shot N` without re-doing the finished ones.
                return (f"shot {sh.id} ({sh.start:.0f}-{sh.end:.0f}s): judge unavailable, "
                        f"not a footage problem; retry with `redo --shot {sh.id}` ({str(e)[:120]})")
            if d is None:
                return f"shot {sh.id} ({sh.start:.0f}-{sh.end:.0f}s): no footage found; manual footage needed"
            exporter = getattr(self, "exporter", None)
            if exporter is not None:
                exporter.submit(sh, d.clip_path)     # background: selection carries on
            return None

        workers = max(1, min(self.s.shot_workers, len(todo)))
        pre = None
        if self.gem and self.s.prefetch_ahead > 0 and len(todo) > workers:
            pre = threading.Thread(target=prefetch, name="prefetch", daemon=True)
            pre.start()
        try:
            if workers == 1:
                results = [one(sh) for sh in todo]
            else:
                # The shots of one sentence go to the same worker, in order, so a 2nd shot can
                # reuse its sibling's download (Selector._sibling); different sentences still run
                # side by side.
                groups: list[list[Shot]] = []
                for sh in todo:
                    if self.s.sibling_extend > 0 and groups and groups[-1][-1].segment == sh.segment:
                        groups[-1].append(sh)
                    else:
                        groups.append([sh])
                with ThreadPoolExecutor(workers) as ex:
                    results = [r for g in ex.map(lambda g: [one(sh) for sh in g], groups) for r in g]
        finally:
            paused.set()                          # releases the prefetcher
            with cond:
                cond.notify_all()
        return [b for b in results if b]

    def render(self) -> Path:
        clips = {}
        for sh in self.shots:
            d = self.proj.accepted(sh.id)
            if d is None:
                raise RuntimeError(f"shot {sh.id} has no clip")
            clips[sh.id] = Path(d.clip_path)
        out = self.proj.path("output", "final.mp4")
        return assemble(self.s, self.proj, self.shots, clips, self.vo, out)

    def _finish_timing(self, t0: float, name: str = "timings.json"):
        elapsed = time.time() - t0
        timing.write(self.proj.path("logs", name), elapsed)
        log.info(timing.summary(elapsed))
        gem = getattr(self, "gem", None)
        if gem is not None and hasattr(gem, "link_stats"):
            stem = name.rsplit(".", 1)[0].replace("timings", "judge-links")
            atomic_write(self.proj.path("logs", f"{stem}.json"), json.dumps(gem.link_stats(), indent=2))

    def _next_redo(self) -> int:
        """1 for the first redo of a project, then 2, 3... (each keeps its own timings file)."""
        n = 1
        while self.proj.path("logs", f"timings-redo-{n}.json").exists():
            n += 1
        return n

    def run(self, wait: bool = False) -> dict:
        """`wait`: when the judge's daily quota runs out, sleep until it resets and carry on,
        instead of returning with the remaining shots paused."""
        t0 = time.time()
        timing.reset()
        self.proj.run_label = "run"
        try:
            return self._run(wait)
        finally:
            self._finish_timing(t0)

    def _sleep_until(self, until: float):
        until += 60                       # the reset is not instantaneous on the provider side
        log.warning("judge quota spent; waiting until %s to continue",
                    time.strftime("%Y-%m-%d %H:%M", time.localtime(until)))
        while (left := until - time.time()) > 0:
            time.sleep(min(left, 600))

    def make_exporter(self, out_dir: Path | None = None):
        """An `export.Exporter` for `remotion_export` (or `out_dir`), with the configured look."""
        from .export import Exporter, Look

        target = out_dir or (Path(self.s.remotion_export) if self.s.remotion_export else None)
        if target is None:
            return None
        look = None
        if self.s.remotion_bake:
            look = Look(grain=Path(self.s.remotion_grain) if self.s.remotion_grain else None,
                        dust=Path(self.s.remotion_dust) if self.s.remotion_dust else None)
        return Exporter(target, look)

    def _words_async(self, out_dir: Path) -> threading.Thread | None:
        """Word timings for the captions, computed while the footage is being selected."""
        if not self.vo:
            return None
        from .export import write_words

        def work():
            try:
                with timing.stage("word timing"):
                    write_words(self.vo, self.segments, out_dir / "words.json")
            except Exception as e:                       # captions can be redone with export-remotion
                log.warning("word timing failed: %s", str(e)[:200])

        t = threading.Thread(target=work, name="words", daemon=True)
        t.start()
        return t

    def _run(self, wait: bool = False) -> dict:
        while True:
            try:
                self.prepare()
                break
            except QuotaExhausted as e:
                if not wait:
                    raise
                self._sleep_until(e.until)
        self.exporter = self.make_exporter()
        words = self._words_async(self.exporter.dir) if self.exporter else None
        with timing.stage("select (wall)"):
            blockers = self.select_all()
            while self.quota_until and wait:
                self._sleep_until(self.quota_until)
                blockers = self.select_all()
        qa: QAResult | None = None
        if not blockers:
            for loop in range(self.s.qa_max_loops + 1):
                qa = self._render_and_qa()
                for i in qa.issues:
                    log.info("QA: %s", i)
                if qa.passed or loop == self.s.qa_max_loops or not qa.bad_shots:
                    break
                log.info("QA failed on shots %s -> re-selecting (loop %d)", sorted(qa.bad_shots), loop + 1)
                for sid in qa.bad_shots:
                    self.proj.invalidate(sid, "failed QA")
                blockers = self.select_all(only=set(qa.bad_shots))
                if blockers:
                    break
            if qa and not qa.passed:
                blockers.append("QA still failing: " + "; ".join(qa.issues[:5]))
        if self.exporter is not None:             # after QA: re-selected shots are exported too
            with timing.stage("export (wait)"):
                self.exporter.finish(self.shots)
                if words:
                    words.join()
            self.exporter = None
        data = build(self.proj, self.title, self.shots, qa, blockers)
        data["quota_paused_until"] = self.quota_until
        log.info("report: %s", self.proj.path("report.html"))
        return data

    def wants_render(self) -> bool:
        """Build the ffmpeg `final.mp4`? `ffmpeg_render: auto` skips it when clips go to Remotion
        (`remotion_export`): that workflow renders the film in Remotion and never used this file,
        which cost a full render on every `run` and every `redo`."""
        mode = str(self.s.ffmpeg_render).lower()
        if mode in ("always", "true", "on"):
            return True
        if mode in ("never", "false", "off"):
            return False
        return not self.s.remotion_export

    def _render_and_qa(self) -> QAResult:
        """Render + full QA, or (no render) the same ledger and duplicate checks on the clips."""
        if not self.wants_render():
            with timing.stage("qa (clips)"):
                return run_qa_clips(self.s, self.proj, self.shots)
        with timing.stage("render"):
            video = self.render()
        with timing.stage("qa"):
            return run_qa(self.s, self.proj, self.shots, video, self.shots[-1].end)

    def redo(self, shot_ids: int | list[int]) -> dict:
        """Re-select one or more shots in ONE process: one `prepare()`, the shots run `shot_workers`
        at a time with prefetch, then one export and one QA. Before, each shot was its own process
        with its own full render and QA (~4 min a shot on the 5-guns film)."""
        ids = [shot_ids] if isinstance(shot_ids, int) else list(dict.fromkeys(shot_ids))
        t0 = time.time()
        timing.reset()
        n = self._next_redo()
        self.proj.run_label = f"redo-{n}"
        try:
            self.prepare()
            bad = [i for i in ids if i not in {sh.id for sh in self.shots}]
            if bad:
                raise ValueError(f"no such shot id(s): {bad} (shots are 0-{len(self.shots) - 1})")
            for sid in ids:
                self.proj.invalidate(sid, "manual redo")
            self.exporter = self.make_exporter()      # keeps an exported composition in step
            try:
                with timing.stage("select (wall)"):
                    blockers = self.select_all(only=set(ids))
            finally:
                if self.exporter is not None:
                    with timing.stage("export (wait)"):
                        self.exporter.finish(self.shots)
                self.exporter = None
            qa = self._render_and_qa()
            return build(self.proj, self.title, self.shots, qa, blockers)
        finally:
            # one file per redo: video-4 lost the timings of redo 1 and 2 when each redo overwrote one file
            self._finish_timing(t0, f"timings-redo-{n}.json")
