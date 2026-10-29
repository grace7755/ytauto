"""Per-shot selection: YouTube first (search -> screen -> section download -> inspect -> score),
then stock fallbacks (Pexels -> Pixabay -> Coverr -> Mixkit) for that shot only."""
from __future__ import annotations

import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .assemble import cut_clip, tail_for
from .benchmark import evaluate
from .cache import Cache
from .config import Settings
from .inspect.checks import run_checks
from .inspect.probe import probe
from .inspect.scenes import best_window
from .inspect.verify import verify_clip
from .llm.router import Judge, JudgeError
from .log import log
from .models import Candidate, ClipScore, Decision, Shot
from .screen import screen
from .sources.stock import Provider
from .sources.youtube import YouTube, text_score
from .state import Project
from .timing import stage

BAD_FLAGS = {"presenter_on_camera": "creator/presenter on camera", "ai_or_animated": "AI/animated/CGI",
             "black_and_white": "black & white"}


class Selector:
    def __init__(self, s: Settings, proj: Project, cache: Cache, yt: YouTube, gem: Judge | None,
                 providers: dict[str, Provider], shots: list[Shot]):
        self.s, self.proj, self.cache, self.yt, self.gem = s, proj, cache, yt, gem
        self.providers, self.shots = providers, shots
        self.sid = {sh.id: i for i, sh in enumerate(shots)}
        self._lock = threading.Lock()
        self._claims: dict[str, list[tuple[float, float]]] = defaultdict(list)   # in-flight windows

    # ------------------------------------------------------------------
    def _decide(self, shot: Shot, c: Candidate, status: str, *, clip: Path | None = None, win=(0.0, 0.0),
                score: ClipScore | None = None, rejects: list[str] | None = None, notes: str = "") -> Decision:
        d = Decision(shot=shot.id, status=status, provider=c.provider, source_id=c.source_id, url=c.url,
                     title=c.title, license=c.license, src_start=round(win[0], 2), src_end=round(win[1], 2),
                     clip_path=str(clip) if clip else "", score=score, hard_rejects=rejects or [], notes=notes)
        self.proj.log_decision(d)
        return d

    def _clip_len(self, shot: Shot) -> float:
        i = self.sid[shot.id]
        return shot.duration + tail_for(self.s, self.shots, i)

    def _used(self) -> tuple[Counter, dict[str, list[tuple[float, float]]]]:
        n: Counter = Counter()
        wins: dict[str, list[tuple[float, float]]] = {}
        for d in self.proj.ledger():
            if d.status in ("accepted", "weak"):
                n[f"{d.provider}:{d.source_id}"] += 1
                wins.setdefault(f"{d.provider}:{d.source_id}", []).append((d.src_start, d.src_end))
        return n, wins

    def _tried(self, shot: Shot) -> set[str]:
        return {f"{d.provider}:{d.source_id}:{d.src_start:.0f}" for d in self.proj.ledger() if d.shot == shot.id}

    # ------------------------------------------------------------------
    def _try_file(self, shot: Shot, c: Candidate, src: Path, hint: tuple[float, float] | None,
                  prior: float, offset: float) -> tuple[Decision, Path | None]:
        """Cut a clip out of `src`, inspect it, score it. `offset` converts file time -> source video time."""
        want = self._clip_len(shot)
        info = probe(src)
        if not info or info.height > info.width:
            return self._decide(shot, c, "rejected", rejects=["unreadable or portrait source"]), None
        with stage("scene detect"):
            a, b, cut_free = best_window(src, info.duration, want, hint, info.duration)
        avail = b - a
        if not cut_free and avail < want * 0.9:
            return self._decide(shot, c, "rejected", win=(a + offset, b + offset),
                                rejects=[f"no cut-free {want:.1f}s window (longest {avail:.1f}s)"]), None
        out = self.proj.path("clips") / f"shot{shot.id:03d}_{c.provider}_{c.source_id}_{int((a + offset) * 10)}.mp4"
        try:
            with stage("cut"):
                cut_clip(self.s, src, a, want, out, avail=None if cut_free else avail)
        except Exception as e:
            return self._decide(shot, c, "rejected", rejects=[f"ffmpeg cut failed: {str(e)[:100]}"]), None
        win = (a + offset, a + offset + want)
        with stage("local checks"):
            out_info, chk = probe(out), run_checks(out)
        # source resolution matters (the output is always 1080p): judge the source, not the re-encode
        if out_info and info:
            out_info.height = min(info.height, out_info.height)
        verdict = None
        if self.gem and self.s.use_gemini_verify and not chk.frozen:
            with stage("verify (AI)"):
                verdict = verify_clip(self.gem, out, shot, f"{c.provider}:{c.source_id}:{int(win[0])}")
        sc, rejects = evaluate(self.s, c, out_info, chk, verdict, prior)
        note = f"blur={chk.blur:.0f} sat={chk.saturation:.2f} face={chk.face:.2f} overlay={chk.overlay:.2f}"
        if verdict:
            note += " | " + (verdict.get("notes") or "")[:200]
        if rejects:
            d = self._decide(shot, c, "rejected", clip=out, win=win, score=sc, rejects=rejects, notes=note)
            return d, out
        return self._decide(shot, c, "accepted", clip=out, win=win, score=sc, notes=note), out

    # ------------------------------------------------------------------
    def _youtube_candidates(self, shot: Shot, rnd: int) -> list[Candidate]:
        b = shot.brief
        qs = b.youtube_queries
        extra = [f"{e} 4k b-roll" for e in b.entities[:2]] + [f"{e} cinematic aerial" for e in b.entities[:1]]
        pool = (qs[rnd * 2: rnd * 2 + 2] or qs[:2]) + extra[rnd: rnd + 1] if rnd < 2 else qs[rnd:] + extra
        seen, cands = set(), []
        for q in pool:
            for c in self.yt.search(q, self.s.results_per_query):
                if c.source_id in seen:
                    continue
                seen.add(c.source_id)
                why = self.yt.prefilter(c)
                if why:
                    continue
                c.text_score = text_score(c, b.youtube_queries, b.entities)
                cands.append(c)
        cands.sort(key=lambda c: -c.text_score)
        return cands

    def _pmap(self, fn, items: list, workers: int) -> list:
        """`fn` over `items` on up to `workers` threads, results in input order."""
        workers = max(1, min(workers, len(items)))
        if workers == 1:
            return [fn(x) for x in items]
        with ThreadPoolExecutor(workers) as ex:
            return list(ex.map(fn, items))

    def _claim(self, key: str, window: tuple[float, float]) -> bool:
        """Reserve `window` of source `key` for this shot, or refuse if another shot has it.

        Shots run concurrently, so "is this window free?" must count both the ledger (finished
        shots) and windows other shots are working on right now, atomically. Without it two shots
        could cut the same footage from the same video at the same moment.
        """
        with self._lock:
            used_n, used_w = self._used()
            busy = self._claims[key]
            if used_n[key] + len(busy) >= self.s.max_shots_per_source:
                return False
            if any(not (window[1] <= a or window[0] >= z) for a, z in used_w.get(key, []) + busy):
                return False
            busy.append(window)
            return True

    def _release(self, key: str, window: tuple[float, float]):
        with self._lock:
            if window in self._claims[key]:
                self._claims[key].remove(window)

    def _screen_one(self, shot: Shot, c: Candidate):
        """Screening result for one candidate; a JudgeError is returned, not raised, so a pool of
        concurrent screens hands it back to the ordered decision loop."""
        try:
            return screen(self.gem, c, shot, yt=self.yt, mode=self.s.screen_mode)
        except JudgeError as e:
            return e

    def _youtube(self, shot: Shot, best: list) -> Decision | None:
        tried = self._tried(shot)
        for rnd in range(self.s.rounds_per_shot):
            cands = self._youtube_candidates(shot, rnd)[: self.s.screen_top_k * 3]
            # Metadata is a ~2 s network call per video, independent of the others.
            with stage("metadata"):
                self._pmap(self.yt.details, cands[: self.s.screen_top_k * 2], self.s.screen_workers)
            ready = []
            for c in cands:
                self.yt.details(c)                       # cached unless it was past the first batch
                if c.screen_flags.get("live") or (c.height and c.height < self.s.min_source_height):
                    continue
                ready.append(c)
                if len(ready) >= self.s.screen_top_k:
                    break
            used_n, _ = self._used()
            ready = [c for c in ready if used_n[f"youtube:{c.source_id}"] < self.s.max_shots_per_source]
            log.info("shot %d round %d: %d YouTube candidates", shot.id, rnd, len(ready))
            # Screen concurrently, then decide in the original ranked order - so the outcome is the
            # one a sequential run would reach, just without waiting on each call in turn.
            screened = (self._pmap(lambda c: self._screen_one(shot, c), ready, self.s.screen_workers)
                        if self.gem else [None] * len(ready))
            for c, r in zip(ready, screened):
                key = f"youtube:{c.source_id}"
                prior = 40 + 0.6 * c.text_score
                if isinstance(r, JudgeError):
                    # Every judge link is down. This says nothing about the footage, so it must
                    # not be logged as a reject and must not push the shot to stock - that would
                    # spend the stock fallback on our outage. Abandon the shot; it stays a
                    # blocker and `redo --shot N` picks it up once a provider recovers.
                    self._decide(shot, c, "unavailable", rejects=[f"judge chain exhausted: {str(r)[:160]}"])
                    log.error("shot %d: judge chain exhausted; leaving the shot unfilled", shot.id)
                    raise r
                if self.gem:
                    # Distinguish the three reasons: a failed call, the judge saying "not usable",
                    # and "usable but no window survived parsing" - they are not the same problem.
                    if not r:
                        self._decide(shot, c, "rejected", rejects=["screen: screening call failed"])
                        continue
                    if not r.get("usable"):
                        self._decide(shot, c, "rejected",
                                     rejects=[f"screen: not usable - {r.get('reason', 'no reason given')}"])
                        continue
                    if not r["windows"]:
                        self._decide(shot, c, "rejected",
                                     rejects=["screen: usable but no timestamped window was returned"])
                        continue
                    flagged = [msg for k, msg in BAD_FLAGS.items() if r.get(k)]
                    if flagged:
                        self._decide(shot, c, "rejected", rejects=["screen: " + ", ".join(flagged)])
                        continue
                    options = [((w["start"], w["end"]), float(w["match"])) for w in r["windows"]]
                else:   # offline mode: spread windows over the video by shot variant
                    d = c.duration or 120
                    _, used_w = self._used()
                    st = min(max(5.0, d * (0.15 + 0.2 * ((shot.variant + len(used_w.get(key, []))) % 4))), max(5.0, d - 12))
                    options = [((st, st + 8), prior)]
                options = [o for o in options if f"youtube:{c.source_id}:{o[0][0]:.0f}" not in tried]
                window = next((o for o in options if self._claim(key, o[0])), None)
                if window is None:
                    continue                             # every window is taken or already tried
                (window, prior) = window
                try:
                    with stage("download"):
                        path = self.yt.download_section(c, window[0], window[1], self.proj.path("sources", "youtube"))
                    if not path:
                        continue
                    pad = max(0.0, window[0] - 1.5)
                    d, clip = self._try_file(shot, c, path, (window[0] - pad, window[1] - pad), prior, pad)
                finally:
                    self._release(key, window)           # an accepted clip is now in the ledger
                if d.status == "accepted":
                    return d
                if clip and d.score and not any(("black" in r or "resolution" in r or "AI" in r or "presenter" in r
                                                 or "watermark" in r or "overlay" in r or "face" in r or "frozen" in r)
                                                for r in d.hard_rejects):
                    best.append(d)       # only soft failures (score/relevance) can become a weak fallback
        return None

    def _stock(self, shot: Shot, best: list) -> Decision | None:
        b = shot.brief
        queries = (b.stock_queries or [shot.text])[:2]
        limit = self.s.stock_tries_per_provider
        for name, prov in self.providers.items():
            if not prov.available():
                log.info("shot %d: %s skipped (no key/tool)", shot.id, name)
                continue
            tried = 0
            for q in queries:
                for c in prov.search(q, 6):
                    if tried >= limit:
                        break
                    if (c.height or 1080) < self.s.min_clip_height:
                        continue
                    key = f"{c.provider}:{c.source_id}"
                    if any(t.startswith(key + ":") for t in self._tried(shot)):
                        continue
                    with stage("stock fetch"):
                        path = prov.fetch(c, self.proj.path("sources", "stock"))
                    if not path:
                        continue
                    tried += 1
                    d, clip = self._try_file(shot, c, path, None, 62.0, 0.0)
                    if d.status == "accepted":
                        return d
                    if clip and d.score:
                        best.append(d)
                if tried >= limit:
                    break
        return None

    # ------------------------------------------------------------------
    def select(self, shot: Shot) -> Decision | None:
        have = self.proj.accepted(shot.id)
        if have:
            return have
        best: list[Decision] = []
        d = self._youtube(shot, best) or self._stock(shot, best)
        if d:
            self._cleanup(shot, keep=d, best=best)
            return d
        # nothing passed: promote the best soft-failure as a clearly flagged weak match
        best = [x for x in best if x.clip_path and Path(x.clip_path).exists()]
        if best:
            top = max(best, key=lambda x: x.score.total if x.score else 0)
            self._cleanup(shot, keep=top, best=best)
            weak = top.model_copy(update={"status": "weak", "notes": top.notes + " | WEAK MATCH: closest available"})
            self.proj.log_decision(weak)
            log.warning("shot %d: using weak match (score %.0f)", shot.id, weak.score.total)
            return weak
        log.error("shot %d: no footage found anywhere", shot.id)
        return None

    def _cleanup(self, shot: Shot, keep: Decision, best: list[Decision]):
        for x in best:
            if x.clip_path and x.clip_path != keep.clip_path:
                Path(x.clip_path).unlink(missing_ok=True)
        for f in self.proj.path("clips").glob(f"shot{shot.id:03d}_*"):
            if str(f) != keep.clip_path:
                f.unlink(missing_ok=True)
