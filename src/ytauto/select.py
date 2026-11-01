"""Per-shot selection: YouTube first (search -> screen -> section download -> inspect -> score),
then stock fallbacks (Pexels -> Pixabay -> Coverr -> Mixkit) for that shot only."""
from __future__ import annotations

import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .assemble import cut_clip, tail_for
from .benchmark import evaluate, local_rejects
from .cache import Cache
from .config import Settings
from .inspect import overlay
from .inspect.checks import run_checks
from .inspect.probe import probe
from .inspect.scenes import best_window
from .inspect.verify import verify_clip
from .lastresort import REUSED, LastResort  # noqa: F401  (REUSED: re-exported for qa and tests)
from .llm.router import Judge, JudgeError, QuotaExhausted
from .log import log
from .models import Candidate, ClipScore, Decision, Shot
from .planner import generalise_brief
from .screen import screen, screen_storyboards
from .sources.stock import Provider
from .sources.youtube import YouTube, text_score
from .state import Project
from .timing import stage

AI_DEFERRED = " [AI check deferred]"   # marker in a decision's notes: see Selector._ai_flagged

BAD_FLAGS = {"presenter_on_camera": "creator/presenter on camera", "ai_or_animated": "AI/animated/CGI",
             "black_and_white": "black & white"}

# Hard rejects that describe the whole *video*, not the window: a channel logo stays on screen for
# the whole upload and AI footage comes from AI channels. Measured on the speed test, 14 of ~35
# downloads failed on overlays. Burnt-in text needs two strikes - travel videos often caption only
# some scenes with a place name, and another window may be clean. The AI flag needs two as well:
# reviewed on 56 videos it rejected (Philippines, Alaska, Japan, Iceland), nearly all were real
# drone, dive and wildlife footage (a Nat Geo eagle film among them), and one video was accepted
# for one shot and flagged for the next - one strike banned good videos for every later shot.
VIDEO_DEFECTS = {"watermark / channel logo": 1, "AI-generated / animated / CGI": 2, "burnt-in text": 2}


_WORD = re.compile(r"[a-z0-9]+")
CATEGORY = "CATEGORY FALLBACK:"          # in a decision's notes: the clip shows the category, not the exact subject


def _words(text: str) -> set[str]:
    """Significant lower-case words (4+ letters) for the film-wide stock skip."""
    return {w for w in _WORD.findall((text or "").lower()) if len(w) >= 4}


class Selector(LastResort):
    def __init__(self, s: Settings, proj: Project, cache: Cache, yt: YouTube, gem: Judge | None,
                 providers: dict[str, Provider], shots: list[Shot]):
        self.s, self.proj, self.cache, self.yt, self.gem = s, proj, cache, yt, gem
        self.providers, self.shots = providers, shots
        self.sid = {sh.id: i for i, sh in enumerate(shots)}
        self._lock = threading.Lock()
        self._claims: dict[str, list[tuple[float, float]]] = defaultdict(list)   # in-flight windows
        self._claim_shots: dict[str, list] = defaultdict(list)   # in-flight (window, shot id)
        self._index = None                       # CLIP tile index, loaded on first use
        self._index_failed = False
        self._wm: dict[int, list[dict]] = defaultdict(list)   # per shot: footage whose only defect is a corner logo
        self._dl_base: dict[int, int] = {}      # downloads already on a shot's ledger when this select() began
        self._reusing: set[tuple] = set()      # clips being reused right now (never more than reuse_max_uses)
        self._photos = None                     # still-picture providers, built on first use
        self._dl_pool: ThreadPoolExecutor | None = None   # background section downloads (download_ahead)
        self._low_rel: dict[str, set[int]] | None = None   # stock/still item -> shots it was judged irrelevant for

    # ------------------------------------------------------------------
    # -- channel reputation ------------------------------------------------------
    # Whole-video defects cluster by channel: one that stamps a logo on one upload stamps it on all
    # of them, and AI-footage channels upload nothing else. The cache is shared across projects, so
    # every run teaches the next. Only counts; a channel is skipped after two bad videos and no
    # accepted clip, and merely ranked down or up before that.
    def _note_channel(self, c: Candidate, status: str, rejects: list[str]):
        if self.cache is None or c.provider != "youtube" or not c.channel:
            return
        bad = status == "rejected" and any(r.startswith(d) for r in rejects for d in VIDEO_DEFECTS)
        if status != "accepted" and not bad:
            return
        with self._lock:
            st = self.cache.get("channel_stats", c.channel) or {"ok": 0, "bad": []}
            if status == "accepted":
                st["ok"] += 1
            elif c.source_id not in st["bad"]:
                st["bad"].append(c.source_id)
            self.cache.set("channel_stats", st, c.channel)

    def channel_bias(self, c: Candidate) -> int:
        """-2..+2: how this channel's past clips fared. 0 when unknown."""
        st = self.cache.get("channel_stats", c.channel) if self.cache is not None and c.channel else None
        if not st:
            return 0
        return max(-2, min(2, min(st["ok"], 2) - min(len(st["bad"]), 2)))

    def bad_channel(self, c: Candidate) -> bool:
        st = self.cache.get("channel_stats", c.channel) if self.cache is not None and c.channel else None
        return bool(st) and len(st["bad"]) >= 2 and st["ok"] == 0

    def _decide(self, shot: Shot, c: Candidate, status: str, *, clip: Path | None = None, win=(0.0, 0.0),
                score: ClipScore | None = None, rejects: list[str] | None = None, notes: str = "") -> Decision:
        self._note_channel(c, status, rejects or [])
        if shot.brief is not None and shot.brief.category_of:
            # disclosed, never silent: the report, the reviewer and qa can see this clip shows the category
            mark = f"{CATEGORY} {shot.brief.subject} (for {shot.brief.category_of})"
            notes = f"{notes} | {mark}" if notes else mark
        d = Decision(shot=shot.id, status=status, provider=c.provider, source_id=c.source_id, url=c.url,
                     title=c.title, license=c.license, src_start=round(win[0], 2), src_end=round(win[1], 2),
                     clip_path=str(clip) if clip else "", score=score, hard_rejects=rejects or [], notes=notes,
                     credit=c.channel if c.provider.startswith("still_") else "")
        self.proj.log_decision(d)
        if status == "rejected" and c.provider != "youtube":
            self._remember_low(d)
        if status == "accepted" and clip is not None:
            self._to_library(shot, c, d, clip)
        return d

    # -- film-wide memory of irrelevant stock ---------------------------------------------------
    # `_tried(shot)` only knows what was tried for THIS shot, so a stock clip judged irrelevant for one shot was
    # fetched, cut and AI-judged again for the next. Video-4 (Russian firearms, 127 shots): one Mixkit waterfall
    # was tried for 53 shots, 442 stock/still rejects had relevance <= 20. Replayed on that ledger
    # (scripts/measure_attempts.py), skipping an item after 2 low-relevance rejects on other shots would have
    # skipped 206 of 827 stock/still attempts and lost 0 accepted clips.
    def _low_relevance(self) -> dict[str, set[int]]:
        with self._lock:
            if self._low_rel is None:
                self._low_rel = defaultdict(set)
                for d in self.proj.ledger():
                    self._add_low(d)
            return self._low_rel

    def _add_low(self, d: Decision):
        if d.status == "rejected" and d.provider != "youtube" and d.score is not None \
                and d.score.relevance <= self.s.stock_skip_relevance:
            self._low_rel[f"{d.provider}:{d.source_id}"].add(d.shot)

    def _remember_low(self, d: Decision):
        self._low_relevance()                          # built from the ledger first
        with self._lock:
            self._add_low(d)

    def _skip_irrelevant(self, c: Candidate, shot: Shot) -> bool:
        """Was this stock/still item already judged irrelevant on `stock_skip_after` other shots? It is still
        allowed when its title or tags share a word with this shot's subject: the same clip can be right for
        a later shot."""
        if self.s.stock_skip_after <= 0:
            return False
        others = self._low_relevance().get(f"{c.provider}:{c.source_id}", set()) - {shot.id}
        if len(others) < self.s.stock_skip_after:
            return False
        b = shot.brief
        subject = _words(" ".join([b.subject, *b.entities])) if b else set()
        if subject & _words(" ".join([c.title, *c.tags])):
            return False
        log.debug("shot %d: skipping %s:%s (judged irrelevant on %d other shots)", shot.id, c.provider,
                  c.source_id, len(others))
        return True

    # -- shared footage library (library.py) ------------------------------------------
    def _library(self):
        if not self.s.library or not self.gem:
            return None                          # offline mode never judged a clip: nothing to share
        idx = self._tile_index()
        if idx is None:
            return None
        from .library import open_library

        return open_library(self.s.cache_dir / "library", idx.clip)

    def _to_library(self, shot: Shot, c: Candidate, d: Decision, clip: Path):
        lib = self._library()
        if lib is None or c.provider.startswith("still_"):      # a still is a last resort, never a library clip
            return
        if c.screen_flags.get("library_key"):
            lib.note_use(c.screen_flags["library_key"], self.proj.root.name)
            return
        b = shot.brief
        try:
            lib.add(clip=clip, provider=c.provider, source_id=c.source_id, url=c.url, title=c.title,
                    license=c.license, src_start=d.src_start, src_end=d.src_end,
                    subject=(b.subject or b.visual) if b else "", setting=b.setting if b else "",
                    text=shot.text, project=self.proj.root.name)
        except Exception as e:                   # the library is an optimisation only
            log.debug("library add failed for %s: %s", c.source_id, str(e)[:160])

    def _from_library(self, shot: Shot) -> Decision | None:
        """Try approved clips from earlier runs before searching (see library.py). Each one still
        passes the local checks and the AI verify against this shot's brief."""
        if self.s.library_tries <= 0 or shot.brief is None:
            return None
        lib = self._library()
        if lib is None or not lib.entries:
            return None
        from .inspect.embed import query_text

        try:
            q = lib.clip.text(query_text(shot.brief))
        except Exception:
            return None
        tried = self._tried(shot)
        hits = lib.search(q, want=self._clip_len(shot), setting=shot.brief.setting,
                          project=self.proj.root.name, min_sim=self.s.library_min_sim, limit=self.s.library_tries)
        for e, sim in hits:
            key = f"{e['provider']}:{e['source_id']}"
            win = (e["src_start"], e["src_end"])
            if f"{key}:{e['src_start']:.0f}" in tried or not self._claim(key, win, shot.id):
                continue
            c = Candidate(provider=e["provider"], source_id=e["source_id"], url=e["url"], title=e["title"],
                          license=e["license"])
            c.screen_flags["library_key"] = e["key"]
            log.info("shot %d: trying library clip %s (CLIP %.2f, first used for %r)", shot.id, e["key"], sim,
                     e["subject"])
            try:
                with stage("library"):
                    d, _ = self._try_file(shot, c, Path(e["file"]), None, 70.0, e["src_start"])
            finally:
                self._release(key, win)
            if d.status == "accepted":
                return d
        return None

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

    def _tried_windows(self, shot: Shot) -> dict[str, list[tuple[float, float]]]:
        """Source footage this shot already used or rejected, by `provider:source_id`."""
        out: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for d in self.proj.ledger():
            if d.shot == shot.id and d.src_end > d.src_start:
                out[f"{d.provider}:{d.source_id}"].append((d.src_start, d.src_end))
        return out

    def _was_tried(self, shot_windows: dict, key: str, window: tuple[float, float]) -> bool:
        """Does a screened `window` overlap footage this shot already used or rejected?

        The old check compared the screen window's *start* with the ledger's *clip* start; the
        clip is cut from inside the window, so they never matched and `redo --shot N` cut the
        same clip out of the same window again (seen: 5 of 7 redos returned the identical clip).
        """
        return any(window[0] < z and a < window[1] for a, z in shot_windows.get(key, []))

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
        if c.provider == "youtube" and self.s.local_overlay_check and overlay.available():
            # Text pinned in place over moving footage is a watermark / channel mark. Reading it
            # locally costs ~1 s and saves the judge call (and quota) a verify would spend finding
            # the same thing: the biggest single hard reject measured over past runs.
            with stage("overlay (OCR)"):
                ov = overlay.detect(out)
            if ov is not None and ov.burnt_in:
                why = f"burnt-in text / branding overlay (local: {', '.join(ov.persistent[:3])})"
                d = self._decide(shot, c, "rejected", clip=out, win=win, rejects=[why],
                                 notes=f"overlay motion={ov.motion:.0f}")
                self._note_defects(c, [why])
                return d, out
        verdict = None
        # A clip the free local checks already reject is rejected whatever the judge says (measured on the
        # Philippines run: 81 of 347 judge calls went to clips that had already failed blur, letterbox,
        # resolution or face), so it never reaches the judge. If it is later promoted to a weak match the
        # AI-footage question is still asked then (AI_DEFERRED).
        local = local_rejects(self.s, out_info, chk)
        judged = self.gem and self.s.use_gemini_verify and not local
        if judged:
            # A clip already rejected on score gains nothing from the dedicated AI check (it can
            # only lower the score); measured, ~1 in 5 downloaded clips failed exactly that way.
            defer = ((lambda v: bool(evaluate(self.s, c, out_info, chk, v, prior)[1]))
                     if self.s.defer_ai_check else None)
            try:
                with stage("verify (AI)"):
                    verdict = verify_clip(self.gem, out, shot, f"{c.provider}:{c.source_id}:{int(win[0])}",
                                          self.s.min_relevance, rejected_anyway=defer)
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
        if local and self.gem and self.s.use_gemini_verify:
            note += " | not sent to the judge: failed a local check" + AI_DEFERRED
        if verdict:
            note += " | " + (verdict.get("notes") or "")[:200]
            if verdict.get("ai_check_deferred"):
                note += AI_DEFERRED
        if rejects:
            d = self._decide(shot, c, "rejected", clip=out, win=win, score=sc, rejects=rejects, notes=note)
            if c.provider == "youtube":
                self._note_defects(c, rejects)
                self._keep_for_crop(shot, c, src, a, want, offset, cut_free, avail, info, rejects, verdict, sc)
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
                if self.bad_channel(c):
                    continue                            # a channel that only gave watermarks / AI footage
                cands.append(c)
        cands.sort(key=lambda c: -(c.text_score + 4 * self.channel_bias(c)))
        return cands

    def _pmap(self, fn, items: list, workers: int) -> list:
        """`fn` over `items` on up to `workers` threads, results in input order."""
        workers = max(1, min(workers, len(items)))
        if workers == 1:
            return [fn(x) for x in items]
        with ThreadPoolExecutor(workers) as ex:
            return list(ex.map(fn, items))

    def _near_neighbour(self, key: str, window: tuple[float, float], shot: int) -> bool:
        """Does the shot before or after this one use source `key` within `adjacent_source_gap`
        seconds of `window`? Measured: a ChatGPT-judged run put 108-113 s and 115-120 s of one
        tripod shot of Fuji on shots 1 and 2 - two clean, relevant clips that played as one
        frozen shot, and the duplicate-frame QA hash did not catch it. Caller holds the lock."""
        gap = self.s.adjacent_source_gap
        if gap <= 0:
            return False
        near = [(d.src_start, d.src_end) for d in self.proj.ledger()
                if d.status in ("accepted", "weak") and f"{d.provider}:{d.source_id}" == key
                and abs(d.shot - shot) == 1]
        near += [w for w, sid in self._claim_shots[key] if sid is not None and abs(sid - shot) == 1]
        return any(max(window[0], a) - min(window[1], z) < gap for a, z in near)

    def _claim(self, key: str, window: tuple[float, float], shot: int | None = None) -> bool:
        """Reserve `window` of source `key` for this shot, or refuse if another shot has it.

        Shots run concurrently, so "is this window free?" must count both the ledger (finished
        shots) and windows other shots are working on right now, atomically. Without it two shots
        could cut the same footage from the same video at the same moment.
        """
        with self._lock:
            used_n, used_w = self._used()
            busy = self._claims[key]
            # a YouTube video may give any number of different sections (they never overlap, below); every
            # other source is picked once here - a second use goes through `_reuse`, with its gap rules
            cap = (self.s.max_shots_per_source or 10 ** 6) if key.startswith("youtube:") else 1
            if used_n[key] + len(busy) >= cap:
                return False
            if any(not (window[1] <= a or window[0] >= z) for a, z in used_w.get(key, []) + busy):
                return False
            if shot is not None and self._near_neighbour(key, window, shot):
                return False
            busy.append(window)
            self._claim_shots[key].append((window, shot))
            return True

    def _release(self, key: str, window: tuple[float, float]):
        with self._lock:
            if window in self._claims[key]:
                self._claims[key].remove(window)
            for item in self._claim_shots[key]:
                if item[0] == window:
                    self._claim_shots[key].remove(item)
                    break

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
        ranked = sorted((c for c, _ in has),
                        key=lambda c: (-(round(sims[c.source_id], 3) + 0.004 * self.channel_bias(c)), -c.text_score))
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
        held, low = [], []
        for c, r in self._screened(shot, ready):
            if isinstance(r, dict) and r.get("usable") and (r.get("overlay_suspect") or r.get("face_suspect")):
                held.append((c, r))
                continue
            if isinstance(r, dict) and r.get("usable") and self._low_match(r):
                low.append((c, r))
                continue
            yield c, r
        if held:
            log.info("shot %d: trying %d candidate(s) whose thumbnails all show text, a logo or a large face",
                     shot.id, len(held))
        yield from held
        if low:
            log.info("shot %d: trying %d candidate(s) the screen rated below %d", shot.id, len(low),
                     self.s.screen_min_match)
        yield from low

    def _low_match(self, r: dict) -> bool:
        """Screened usable, but no window reaches `screen_min_match`: tried last (see config)."""
        floor = self.s.screen_min_match
        return floor > 0 and bool(r.get("windows")) and max(w.get("match", 0) for w in r["windows"]) < floor

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
            self._download_ahead(shot, wave, results)
            for c in wave:
                yield c, results[c.source_id]

    def _download_ahead(self, shot: Shot, wave: list[Candidate], results: dict):
        """Start fetching the sections of a screened wave's later candidates now.

        The decision loop handles candidates one at a time: download (~10 s of pure transfer),
        cut, local checks, then a judge call that can take 20 s or more. While that runs the
        network is idle, so the next candidate's best window is fetched in the background; if the
        first candidate is accepted the extra section is simply never used. `download_section`
        holds a per-file lock, so the loop and this thread can never write the same file. It
        spends no judge calls and makes no decisions; any failure is ignored.
        """
        if not self.s.download_ahead or not self.gem:
            return
        tried = self._tried(shot)
        used = self._tried_windows(shot)
        for c in wave[1:]:                    # the first one is fetched by the loop right away
            r = results.get(c.source_id)
            if not isinstance(r, dict) or not r.get("usable") or not r.get("windows"):
                continue
            if r.get("overlay_suspect") or r.get("face_suspect") or any(r.get(k) for k in BAD_FLAGS) or self.bad_video(c) or self._low_match(r):
                continue
            w = next(((x["start"], x["end"]) for x in r["windows"]
                      if f"youtube:{c.source_id}:{x['start']:.0f}" not in tried
                      and not self._was_tried(used, f"youtube:{c.source_id}", (x["start"], x["end"]))), None)
            if w is None:
                continue
            with self._lock:
                if self._dl_pool is None:
                    self._dl_pool = ThreadPoolExecutor(2, thread_name_prefix="dl-ahead")
                pool = self._dl_pool
            pool.submit(self._bg_download, c, (w[0], w[1] + self._extend(shot)))

    def _bg_download(self, c: Candidate, w: tuple[float, float]):
        try:
            with stage("download (ahead)"):
                self.yt.download_section(c, w[0], w[1], self.proj.path("sources", "youtube"))
        except Exception as e:                # purely an optimisation
            log.debug("download-ahead for %s failed: %s", c.source_id, str(e)[:160])

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
            if c.screen_flags.get("meta_error"):
                continue                                 # gone / private: nothing to screen
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

    def _downloads(self, shot: Shot) -> int:
        """Clips downloaded and judged for this shot so far (screen-only rejects cost no download)."""
        return sum(1 for d in self.proj.ledger() if d.shot == shot.id and d.status in ("accepted", "rejected", "weak")
                   and not any(r.startswith("screen") for r in d.hard_rejects))

    def _over_budget(self, shot: Shot, cap: int | None = None) -> bool:
        """Has this select() already spent `max_downloads_per_shot` downloads on the shot? Measured on the
        Philippines run: 14 of 87 shots took more than 8 downloads (up to 25), and 10 of those 14 still ended as
        weak matches - the extra downloads bought nothing. Past the budget the shot moves on to stock, a still,
        a reuse or a weak match. 0 = no budget."""
        cap = self.s.max_downloads_per_shot if cap is None else cap
        return cap > 0 and self._downloads(shot) - self._dl_base.get(shot.id, 0) >= cap

    def _over_attempts(self, shot: Shot, cap: int | None = None) -> bool:
        """Visual ladder: has this shot spent its judged-attempt budget (YouTube + stock)? Then it moves on to a photo, the
        category rewrite or a reuse instead of trying more of the same (video-4 replay: median 6 judged attempts per shot,
        max 22; shots with no accepted clip spent the most). Only stock checks it: YouTube keeps `max_downloads_per_shot`."""
        if not self.s.visual_ladder:
            return False
        cap = self.s.shot_attempts_budget if cap is None else cap
        return cap > 0 and self._downloads(shot) - self._dl_base.get(shot.id, 0) >= cap

    def _youtube(self, shot: Shot, best: list, rounds: int | None = None, cap: int | None = None) -> Decision | None:
        tried = self._tried(shot)
        for rnd in range(rounds or self.s.rounds_per_shot):
            if self._over_budget(shot, cap):
                log.info("shot %d: %d downloads without a usable clip; leaving YouTube", shot.id, self.s.max_downloads_per_shot)
                break
            cands = self._youtube_candidates(shot, rnd)[: self.s.screen_top_k * 3]
            # Metadata is a ~2 s network call per video, independent of the others.
            with stage("metadata"):
                self._pmap(self.yt.details, cands[: self.s.screen_top_k * 2], self.s.screen_workers)
            ready = self._ready(cands)
            used_n, _ = self._used()
            ready = [c for c in ready if not self.s.max_shots_per_source
                     or used_n[f"youtube:{c.source_id}"] < self.s.max_shots_per_source]
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
                used = self._tried_windows(shot)
                options = [o for o in options if f"youtube:{c.source_id}:{o[0][0]:.0f}" not in tried
                           and not self._was_tried(used, key, o[0])]
                window = next((o for o in options if self._claim(key, o[0], shot.id)), None)
                if window is None:
                    continue                             # every window is taken or already tried
                (window, prior) = window
                if self._over_budget(shot, cap):
                    self._release(key, window)
                    log.info("shot %d: download budget spent; leaving YouTube", shot.id)
                    return None
                try:
                    with stage("download"):
                        path = self.yt.download_section(c, window[0], window[1] + self._extend(shot),
                                                        self.proj.path("sources", "youtube"))
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

    def _stock(self, shot: Shot, best: list, cap: int | None = None) -> Decision | None:
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
                    if self._over_attempts(shot, cap):
                        log.info("shot %d: attempt budget spent; leaving stock", shot.id)
                        return None
                    if (c.height or 1080) < self.s.min_clip_height:
                        continue
                    key = f"{c.provider}:{c.source_id}"
                    if any(t.startswith(key + ":") for t in self._tried(shot)):
                        continue
                    if self._skip_irrelevant(c, shot):
                        continue
                    win = (0.0, 1e6)                      # the whole file: a stock clip is used once per project
                    if not self._claim(key, win, shot.id):
                        continue
                    try:
                        with stage("stock fetch"):
                            path = prov.fetch(c, self.proj.path("sources", "stock"))
                        if not path:
                            continue
                        tried += 1
                        d, clip = self._try_file(shot, c, path, None, 62.0, 0.0)
                    finally:
                        self._release(key, win)
                    if d.status == "accepted":
                        return d
                    if clip and d.score:
                        best.append(d)
                if tried >= limit:
                    break
        return None

    # -- siblings: two shots of one sentence from one download ------------------------
    def _next_is_sibling(self, shot: Shot) -> bool:
        i = self.sid[shot.id]
        return i + 1 < len(self.shots) and self.shots[i + 1].segment == shot.segment

    def _extend(self, shot: Shot) -> float:
        """Extra seconds to download after the screened window, so the next shot of the same
        sentence can be cut from this file (`_sibling`) instead of searching from zero."""
        return self.s.sibling_extend if self.s.sibling_extend > 0 and self._next_is_sibling(shot) else 0.0

    def _sibling(self, shot: Shot) -> Decision | None:
        """Try a different scene of the YouTube section the previous shot of this sentence used.

        Measured on the Philippines run: 87 shots from 42 sentences, so about half the shots are
        a 2nd or 3rd shot of a sentence, and each searched, screened and downloaded from zero
        (~65 s a shot). Two scenes from one well-chosen video is how a documentary cuts anyway.
        The clip still goes through every local check and the AI verify against its own brief;
        it must be a different scene (a detected cut between them), so it never plays as one
        frozen shot - the reason `adjacent_source_gap` exists."""
        if self.s.sibling_extend <= 0 or shot.variant == 0:
            return None
        i = self.sid[shot.id]
        if i == 0 or self.shots[i - 1].segment != shot.segment:
            return None
        prev = self.proj.accepted(self.shots[i - 1].id)
        if prev is None or prev.provider != "youtube" or prev.status != "accepted":
            return None
        key = f"youtube:{prev.source_id}"
        if any(t.startswith(key + ":") for t in self._tried(shot)):
            return None                                   # already tried for this shot (resume)
        src = self._section_with(prev)
        if src is None:
            return None
        path, offset = src
        want = self._clip_len(shot)
        from .inspect.scenes import scene_spans
        used = (prev.src_start - offset, prev.src_end - offset)
        info = probe(path)
        if not info:
            return None
        spans = scene_spans(path) or [(0.0, info.duration)]
        # a scene that does not contain any of the used window, long enough for this shot
        free = [(a, z) for a, z in spans if (z <= used[0] or a >= used[1]) and z - a >= want - 0.05]
        if not free:
            return None
        a, z = max(free, key=lambda sp: sp[1] - sp[0])
        win = (a + offset, z + offset)
        if not self._claim(key, win):                     # no neighbour check: different scene
            return None
        c = Candidate(provider="youtube", source_id=prev.source_id, url=prev.url, title=prev.title,
                      license=prev.license)
        try:
            self.yt.details(c)                            # cached: recency for the benchmark
        except Exception:
            pass
        try:
            log.info("shot %d: trying the next scene of shot %d's video %s", shot.id, shot.id - 1, prev.source_id)
            with stage("sibling"):
                d, _ = self._try_file(shot, c, path, (a, z), 75.0, offset)
        finally:
            self._release(key, win)
        return d if d.status == "accepted" else None

    def _section_with(self, prev: Decision) -> tuple[Path, float] | None:
        """The downloaded section of `prev`'s video that covers its window, as (file, start in
        source seconds). Section files are named `<id>_<start*10>_<end*10>.mp4`."""
        best = None
        for f in self.proj.path("sources", "youtube").glob(f"{prev.source_id}_*"):
            parts = f.stem[len(prev.source_id) + 1:].split("_")
            if len(parts) != 2 or not all(p.isdigit() for p in parts):
                continue
            a, z = int(parts[0]) / 10, int(parts[1]) / 10
            if a <= prev.src_start + 0.05 and z >= prev.src_end - 0.05 and (best is None or z - a > best[2]):
                best = (f, a, z - a)
        return (best[0], best[1]) if best else None

    # ------------------------------------------------------------------
    def select(self, shot: Shot) -> Decision | None:
        have = self.proj.accepted(shot.id)
        if have:
            return have
        best: list[Decision] = []
        self._dl_base[shot.id] = self._downloads(shot)
        d = self._sibling(shot) or self._from_library(shot)
        photo_first = self.s.visual_ladder and shot.brief is not None and shot.brief.visual_kind == "photo"
        if not d:
            if photo_first:
                # a named product rarely exists as clean footage: its photo first, then one quick YouTube round, then stock
                d = (self._still(shot, only=self.s.photo_first_providers) or self._youtube(shot, best, rounds=1)
                     or self._stock(shot, best))
            elif shot.brief is not None and shot.brief.generic and self.s.stock_first:
                d = self._stock(shot, best) or self._youtube(shot, best)
            else:
                d = self._youtube(shot, best) or self._stock(shot, best)
        if d:
            self._cleanup(shot, keep=d, best=best)
            return d
        if self.s.visual_ladder:
            # every step beats a weak match; a photo of the exact thing beats a clip of its category
            d = (self._crop_logo(shot) or (None if photo_first else self._still(shot))
                 or self._category(shot, best) or self._reuse(shot))
        else:
            d = self._crop_logo(shot) or self._still(shot) or self._reuse(shot)   # all of them beat a weak match
        if d:
            self._cleanup(shot, keep=d, best=best)
            return d
        # nothing passed: promote the best soft-failure as a clearly flagged weak match
        best = [x for x in best if x.clip_path and Path(x.clip_path).exists()]
        while best:
            # The most RELEVANT soft-failure, then the best overall: ranking on the total alone picked a lovely
            # mountain road (relevance 0, quality 95) for a sentence about a pistol on the 5 Guns film
            top = max(best, key=lambda x: (x.score.relevance, x.score.total) if x.score else (0, 0))
            if self._ai_flagged(shot, top):
                best.remove(top)
                continue
            self._cleanup(shot, keep=top, best=best)
            weak = top.model_copy(update={"status": "weak", "notes": top.notes + " | WEAK MATCH: closest available"})
            self.proj.log_decision(weak)
            log.warning("shot %d: using weak match (score %.0f)", shot.id, weak.score.total)
            return weak
        log.error("shot %d: no footage found anywhere", shot.id)
        return None

    def _category(self, shot: Shot, best: list) -> Decision | None:
        """Query relaxation: the exact subject found nothing close, so search once for its category ("PPK-20 blowback action" ->
        "submachine gun bolt cycling"). Only when nothing relevant turned up (best relevance below the trigger): a near miss of
        the real thing still wins as a weak match. A small budget of its own; every clip is marked CATEGORY FALLBACK."""
        b = shot.brief
        if not (self.s.visual_ladder and self.s.category_fallback and self.gem and b is not None) or b.generic or b.category_of:
            return None
        top = max((x.score.relevance for x in best if x.score), default=0.0)
        if top >= self.s.category_trigger_relevance:
            return None
        nb = generalise_brief(self.gem, shot)
        if nb is None:
            return None
        log.info("shot %d: nothing close to %r; trying its category %r", shot.id, b.subject, nb.subject)
        alt = shot.model_copy(update={"brief": nb})
        self._dl_base[shot.id] = self._downloads(shot)          # a fresh, small budget for this step
        cap = self.s.category_attempts
        return self._youtube(alt, best, rounds=1, cap=cap) or self._stock(alt, best, cap=cap)

    def _ai_flagged(self, shot: Shot, d: Decision) -> bool:
        """Run the AI-footage check that `verify_clip` deferred for this clip, now that it is about
        to be used as a weak match. True = it is AI-generated and has been rejected (and deleted).
        A judge outage propagates: a clip is never promoted unjudged while a judge is configured."""
        if not self.gem or AI_DEFERRED not in d.notes:
            return False
        cache_id = f"{d.provider}:{d.source_id}:{int(d.src_start)}"
        if not self.gem.confirm_ai_generated(Path(d.clip_path), cache_key=("aicheck-v2", cache_id)):
            return False
        log.info("shot %d: weak candidate %s is AI-generated; dropped", shot.id, Path(d.clip_path).name)
        self.proj.log_decision(d.model_copy(update={
            "status": "rejected", "hard_rejects": d.hard_rejects + ["AI-generated / animated / CGI (AI)"]}))
        Path(d.clip_path).unlink(missing_ok=True)
        return True

    def _cleanup(self, shot: Shot, keep: Decision, best: list[Decision]):
        for x in best:
            if x.clip_path and x.clip_path != keep.clip_path:
                Path(x.clip_path).unlink(missing_ok=True)
        for f in self.proj.path("clips").glob(f"shot{shot.id:03d}_*"):
            if str(f) != keep.clip_path:
                f.unlink(missing_ok=True)
