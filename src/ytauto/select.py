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
from .llm.router import Judge, JudgeError, QuotaExhausted
from .log import log
from .models import Candidate, ClipScore, Decision, Shot
from .screen import screen, screen_storyboards
from .sources.stock import Provider
from .sources.youtube import YouTube, text_score
from .state import Project
from .timing import stage

BAD_FLAGS = {"presenter_on_camera": "creator/presenter on camera", "ai_or_animated": "AI/animated/CGI",
             "black_and_white": "black & white"}

# Hard rejects that describe the whole *video*, not the window: a channel logo stays on screen for
# the whole upload and AI footage comes from AI channels. Measured on the speed test, 14 of ~35
# downloads failed on overlays. Burnt-in text needs two strikes - travel videos often caption only
# some scenes with a place name, and another window may be clean.
VIDEO_DEFECTS = {"watermark / channel logo": 1, "AI-generated / animated / CGI": 1, "burnt-in text": 2}


class Selector:
    def __init__(self, s: Settings, proj: Project, cache: Cache, yt: YouTube, gem: Judge | None,
                 providers: dict[str, Provider], shots: list[Shot]):
        self.s, self.proj, self.cache, self.yt, self.gem = s, proj, cache, yt, gem
        self.providers, self.shots = providers, shots
        self.sid = {sh.id: i for i, sh in enumerate(shots)}
        self._lock = threading.Lock()
        self._claims: dict[str, list[tuple[float, float]]] = defaultdict(list)   # in-flight windows
        self._index = None                       # CLIP tile index, loaded on first use
        self._index_failed = False

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

    def _note_defects(self, c: Candidate, rejects: list[str]):
        """Remember whole-video defects in the shared cache, so no later shot - in this project or
        another - downloads this video again."""
        if self.cache is None:
            return
        for defect, strikes in VIDEO_DEFECTS.items():
            if not any(r.startswith(defect) for r in rejects):
                continue
            seen = (self.cache.get("video_strikes", c.source_id, defect) or 0) + 1
            self.cache.set("video_strikes", seen, c.source_id, defect)
            if seen >= strikes:
                self.cache.set("bad_video", defect, c.source_id)
                log.info("youtube:%s marked bad for every shot (%s)", c.source_id, defect)

    def bad_video(self, c: Candidate) -> str | None:
        return self.cache.get("bad_video", c.source_id) if self.cache is not None else None

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
        judged = self.gem and self.s.use_gemini_verify and not chk.frozen
        if judged:
            try:
                with stage("verify (AI)"):
                    verdict = verify_clip(self.gem, out, shot, f"{c.provider}:{c.source_id}:{int(win[0])}",
                                          self.s.min_relevance)
            except JudgeError as e:
                if not isinstance(e, QuotaExhausted):
                    self._decide(shot, c, "unavailable", win=win,
                                 rejects=[f"judge chain exhausted during verify: {str(e)[:160]}"])
                raise
        sc, rejects = evaluate(self.s, c, out_info, chk, verdict, prior)
        if judged and verdict is None:
            # Without a verdict nothing has checked for watermarks, burnt-in text or AI footage;
            # local checks alone must never accept a clip while a judge is configured.
            rejects.append("AI verify gave no verdict")
        note = f"blur={chk.blur:.0f} sat={chk.saturation:.2f} face={chk.face:.2f} overlay={chk.overlay:.2f}"
        if verdict:
            note += " | " + (verdict.get("notes") or "")[:200]
        if rejects:
            d = self._decide(shot, c, "rejected", clip=out, win=win, score=sc, rejects=rejects, notes=note)
            if c.provider == "youtube":
                self._note_defects(c, rejects)
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

    # -- local pre-ranking -------------------------------------------------------
    def _tile_index(self):
        if self._index is None and not self._index_failed:
            with self._lock:
                if self._index is None and not self._index_failed:
                    try:
                        from .inspect.embed import Clip, TileIndex

                        self._index = TileIndex(Clip(self.s.cache_dir / "models"), self.s.cache_dir)
                    except Exception as e:          # missing package / first download offline
                        log.warning("CLIP pre-ranking unavailable (%s); using text ranking", str(e)[:160])
                        self._index_failed = True
        return self._index

    def _prerank(self, shot: Shot, pool: list[Candidate]) -> list[Candidate]:
        """Order `pool` by how well the best storyboard tile matches the shot's subject (local
        CLIP, no judge call) and keep `screen_top_k`.

        Measured on 77 past candidates (scripts/calibrate_prerank.py): CLIP's best-tile similarity
        separates judge-usable from not-usable videos with AUC 0.74, the text score only 0.58, and
        mixing them added nothing. CLIP does NOT pick tiles well (the judge's window was in CLIP's
        top 6 tiles only 12-20 of 43 times), so the judge still sees every tile. `filter` also drops
        candidates more than `prerank_margin` below the best: at 0.04 that removed 12% of
        candidates and lost 1 of 46 usable ones.
        """
        k = self.s.screen_top_k
        idx = self._tile_index() if self.s.prerank != "off" else None
        if idx is None:
            return pool[:k]
        with stage("storyboard fetch"):
            sbs = self._pmap(self.yt.storyboard, pool, self.s.screen_workers)
        has = [(c, sb) for c, sb in zip(pool, sbs) if sb is not None]
        without = [c for c, sb in zip(pool, sbs) if sb is None]
        if not has:
            return pool[:k]
        from .inspect.embed import query_text

        try:
            with stage("prerank (CLIP)"):
                q = idx.clip.text(query_text(shot.brief))
                sims = {c.source_id: float(idx.scores(sb, q).max()) for c, sb in has}
        except Exception as e:
            log.warning("CLIP pre-ranking failed (%s); using text ranking", str(e)[:160])
            self._index_failed = True
            return pool[:k]
        ranked = sorted((c for c, _ in has), key=lambda c: (-round(sims[c.source_id], 3), -c.text_score))
        if self.s.prerank == "filter":
            top = sims[ranked[0].source_id]
            keep = [c for c in ranked if sims[c.source_id] >= top - self.s.prerank_margin]
            if len(keep) < len(ranked):
                log.info("shot %d: CLIP set aside %d low-similarity candidate(s)", shot.id, len(ranked) - len(keep))
            ranked = keep
        for c in ranked:
            c.screen_flags["clip"] = round(sims[c.source_id], 3)
        # videos without a storyboard can only be URL-screened, the expensive path: last
        return (ranked + (without if self.s.screen_mode == "storyboard" else []))[:k]

    def _screen_waves(self, shot: Shot, ready: list[Candidate]):
        """`_screened` in ranked order, except that a usable candidate whose every proposed tile
        shows text or a logo (`overlay_suspect`) is held back until the clean ones were tried. It
        is a deferral, never a reject: thumbnails are too small to rule on a watermark."""
        held = []
        for c, r in self._screened(shot, ready):
            if isinstance(r, dict) and r.get("usable") and r.get("overlay_suspect"):
                held.append((c, r))
                continue
            yield c, r
        if held:
            log.info("shot %d: trying %d candidate(s) whose thumbnails all show text or a logo",
                     shot.id, len(held))
        yield from held

    def _screened(self, shot: Shot, ready: list[Candidate]):
        """Yield (candidate, screening result) in ranked order, screening lazily.

        `screen_batch` > 1: candidates are screened `screen_batch` at a time with their contact
        sheets in ONE judge call, and the next group is only screened if this one produced no
        accepted clip - the consumer stops iterating on an accept. Measured live on 16 past
        verdicts: batches of 2 agreed with single screens on `usable` 15/16 and on the best tile
        (+-1) 12/16, for half the requests. `screen_batch: 1` keeps the old behaviour: every
        candidate of the round screened concurrently, one call each.
        """
        if not self.gem:
            for c in ready:
                yield c, None
            return
        size = max(1, self.s.screen_batch)
        if size == 1 or self.s.screen_mode == "url":
            yield from zip(ready, self._pmap(lambda c: self._screen_one(shot, c), ready, self.s.screen_workers))
            return
        for i in range(0, len(ready), size):
            wave = ready[i:i + size]
            with stage("storyboard fetch"):
                sbs = [self.yt.storyboard(c) for c in wave]           # cached on disk by now
            boards = [(c, sb) for c, sb in zip(wave, sbs) if sb is not None]
            results: dict[str, object] = {}
            if boards:
                try:
                    with stage("screen (storyboard)"):
                        got = screen_storyboards(self.gem, [c for c, _ in boards], shot, [sb for _, sb in boards])
                    results.update({c.source_id: r for (c, _), r in zip(boards, got)})
                except JudgeError as e:
                    results.update({c.source_id: e for c, _ in boards})
            for c in wave:
                if c.source_id not in results:                  # no storyboard: single path
                    results[c.source_id] = self._screen_one(shot, c)
            for c in wave:
                yield c, results[c.source_id]

    def _screen_one(self, shot: Shot, c: Candidate):
        """Screening result for one candidate; a JudgeError is returned, not raised, so a pool of
        concurrent screens hands it back to the ordered decision loop."""
        try:
            return screen(self.gem, c, shot, yt=self.yt, mode=self.s.screen_mode)
        except JudgeError as e:
            return e

    def _ready(self, cands: list[Candidate]) -> list[Candidate]:
        """Candidates worth screening: not live, big enough, not a known-bad video."""
        pool_n = self.s.screen_top_k * (2 if self.s.prerank != "off" and self.gem else 1)
        ready = []
        for c in cands:
            self.yt.details(c)                           # cached unless it was past the first batch
            if c.screen_flags.get("live") or (c.height and c.height < self.s.min_source_height):
                continue
            if self.bad_video(c):
                continue                                 # known watermark / AI channel
            ready.append(c)
            if len(ready) >= pool_n:
                break
        return ready

    def warm(self, shot: Shot):
        """Do a shot's judge-free round-0 work ahead of time: search, metadata, storyboards and
        CLIP tile embeddings, all cached on disk, so that when a worker reaches the shot it goes
        straight to the judge. No judge calls and no decisions: a failure here costs nothing, the
        shot just does the work itself. Measured per shot in a live run: ~6 s metadata, ~5 s
        storyboard fetch and ~18 s of CLIP, all while the judge was the only thing that could
        not be sped up."""
        if self.proj.accepted(shot.id):
            return
        k = self.s.screen_top_k
        with stage("prefetch (background)"):
            cands = self._youtube_candidates(shot, 0)[: k * 3]
            self._pmap(self.yt.details, cands[: k * 2], self.s.screen_workers)
            ready = self._ready(cands)
            sbs = self._pmap(self.yt.storyboard, ready, self.s.screen_workers)
            idx = self._tile_index() if self.s.prerank != "off" and self.gem else None
            if idx is not None:
                for sb in sbs:
                    if sb is not None:
                        idx.tiles(sb)

    def _youtube(self, shot: Shot, best: list) -> Decision | None:
        tried = self._tried(shot)
        for rnd in range(self.s.rounds_per_shot):
            cands = self._youtube_candidates(shot, rnd)[: self.s.screen_top_k * 3]
            # Metadata is a ~2 s network call per video, independent of the others.
            with stage("metadata"):
                self._pmap(self.yt.details, cands[: self.s.screen_top_k * 2], self.s.screen_workers)
            ready = self._ready(cands)
            used_n, _ = self._used()
            ready = [c for c in ready if used_n[f"youtube:{c.source_id}"] < self.s.max_shots_per_source]
            ready = self._prerank(shot, ready) if self.gem else ready[: self.s.screen_top_k]
            log.info("shot %d round %d: %d YouTube candidates", shot.id, rnd, len(ready))
            # Screen (in parallel or in batches), then decide in the ranked order - so the outcome
            # is the one a sequential run would reach, just without waiting on each call in turn.
            for c, r in self._screen_waves(shot, ready):
                key = f"youtube:{c.source_id}"
                prior = 40 + 0.6 * c.text_score
                if isinstance(r, QuotaExhausted):
                    raise r                              # paused, not failed: nothing to record
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
