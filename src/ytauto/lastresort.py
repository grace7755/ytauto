"""The last-resort ways to fill a shot, tried in this order after new footage failed: a clip whose only defect
is a corner logo (re-cut closer), a licensed photograph, and the same footage as an earlier shot. Split out of
`select.py` as a mixin: they share its claim/ledger machinery (`self._claim`, `self._decide`, ...) but none of
its search/screen/download loop."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .assemble import CORNERS, corner_crop, cut_clip
from .benchmark import evaluate
from .inspect.checks import run_checks
from .inspect.probe import probe
from .inspect.verify import verify_clip, verify_still
from .llm.router import JudgeError, QuotaExhausted
from .log import log
from .models import Candidate, Decision, Shot
from .sources.images import make_photo_providers
from .stills import motion_clip, motion_for
from .timing import stage

REUSED = "REUSED footage"             # in a decision's notes: the same footage as an earlier shot (see `_reuse`)


class LastResort:
    """Mixin for `select.Selector`."""

    _CROP_OK = ("watermark / channel logo", "cleanliness", "score")     # rejects a logo alone explains

    # -- shared: judge a clip these paths made, and record the outcome ---------------------------------
    def _judge_made_clip(self, shot: Shot, c: Candidate, out: Path, win: tuple[float, float], verify, note: str, *,
                         height: float | None = None, is_still: bool = False, record_reject: bool = False,
                         min_relevance: float | None = None) -> Decision | None:
        """Check a clip a last-resort path just made the way every clip is checked (local checks, the AI verdict
        from `verify()`, the benchmark) and return the accepted Decision, or None (the file is deleted).
        `height`: the source's real height when the clip was cut closer than the source (a crop). A still
        passes `is_still`: a slow push on one picture is not a frozen clip. `min_relevance` is a stricter floor
        than the benchmark's. `verify` may raise JudgeError (the chain is down): it propagates after the file is
        removed."""
        try:
            with stage("local checks"):
                info, chk = probe(out), run_checks(out)
            if height is not None and info:
                info.height = min(int(height), info.height)
            if is_still:
                chk.frozen = False
            with stage("verify (AI)"):
                verdict = verify()
        except JudgeError as e:
            out.unlink(missing_ok=True)
            if record_reject and not isinstance(e, QuotaExhausted):
                self._decide(shot, c, "unavailable", rejects=[f"judge chain exhausted during verify: {str(e)[:160]}"])
            raise
        sc, rejects = evaluate(self.s, c, info, chk, verdict, 70.0)
        if verdict is None:
            rejects.append("AI verify gave no verdict")
        elif min_relevance is not None and self.s.min_relevance <= sc.relevance < min_relevance:
            rejects.append(f"relevance {sc.relevance:.0f} < {min_relevance:.0f} (stills must show what the narration is about)")
        notes = f"{note} | " + ((verdict or {}).get("notes") or "")[:160]
        if rejects:
            out.unlink(missing_ok=True)
            log.info("shot %d: %s does not fit: %s", shot.id, c.source_id, "; ".join(rejects)[:160])
            if record_reject:
                self._decide(shot, c, "rejected", score=sc, rejects=rejects, notes=notes)
            return None
        return self._decide(shot, c, "accepted", clip=out, win=win, score=sc, notes=notes)

    # -- corner logos: re-cut closer so the logo falls outside the picture ------------------------------
    def _keep_for_crop(self, shot, c, src, a, want, offset, cut_free, avail, info, rejects, verdict, sc):
        """Remember footage that is good in every way except a small logo in one corner, so the shot can
        use it if nothing clean is found. Nothing is cut or judged until then."""
        if not self.s.watermark_crop or not verdict or verdict.get("watermark_corner") not in CORNERS:
            return
        if not verdict.get("watermark_or_logo") or any(not r.startswith(self._CROP_OK) for r in rejects):
            return
        if info.height / self.s.watermark_crop_zoom < self.s.min_clip_height:
            return                                              # the closer cut would be soft: not allowed
        self._wm[shot.id].append(dict(c=c, src=src, a=a, want=want, offset=offset, cut_free=cut_free, avail=avail,
                                      corner=verdict["watermark_corner"], info=info, total=sc.total))

    def _crop_logo(self, shot: Shot) -> Decision | None:
        for e in sorted(self._wm.pop(shot.id, []), key=lambda x: -x["total"]):
            if not self.gem:
                return None
            c, a, offset = e["c"], e["a"], e["offset"]
            key, win = f"youtube:{c.source_id}", (a + offset, a + offset + e["want"])
            if not self._claim(key, win, shot.id):
                continue
            out = self.proj.path("clips") / f"shot{shot.id:03d}_{c.provider}_{c.source_id}_{int(win[0] * 10)}_crop.mp4"
            zoom = self.s.watermark_crop_zoom
            try:
                log.info("shot %d: nothing clean found; trying %s with its %s logo cropped out", shot.id, c.source_id, e["corner"])
                with stage("crop logo"):
                    cut_clip(self.s, e["src"], a, e["want"], out, avail=None if e["cut_free"] else e["avail"],
                             crop=corner_crop(e["corner"], zoom))
                d = self._judge_made_clip(
                    shot, c, out, win,
                    lambda: verify_clip(self.gem, out, shot, f"{c.provider}:{c.source_id}:{int(win[0])}:crop", self.s.min_relevance),
                    f"logo cropped out of the {e['corner']} corner ({zoom}x)", height=e["info"].height / zoom)
                if d:
                    return d
            except JudgeError:
                raise
            except Exception as ex:
                out.unlink(missing_ok=True)
                log.warning("shot %d: logo crop failed: %s", shot.id, str(ex)[:160])
            finally:
                self._release(key, win)
        return None

    # -- reuse: the same footage in a second shot --------------------------------------------------------
    def _reuse(self, shot: Shot) -> Decision | None:
        """Put an earlier shot's footage on this shot as well, only when no new footage, photograph or
        cropped clip fitted. Rules: at most `reuse_max_uses` appearances of one clip in the video, every
        pair of appearances at least `reuse_min_gap` s apart on the timeline (which also rules out
        neighbouring shots), stills never (a photograph plays once), and the clip must still pass the
        full check against THIS shot's brief. Candidates are ranked by local CLIP, best first."""
        if self.s.reuse_max_uses < 2 or self.s.reuse_tries <= 0 or not self.gem or shot.brief is None:
            return None
        idx = self._tile_index()
        if idx is None:
            return None
        from .inspect.embed import query_text
        from .review import frame_of

        me = self.shots[self.sid[shot.id]]
        pool: dict[tuple, list[Decision]] = defaultdict(list)
        for sh in self.shots:
            d = self.proj.accepted(sh.id)                       # the current pick only
            if (d and sh.id != shot.id and d.status in ("accepted", "weak") and d.clip_path
                    and not d.provider.startswith("still_") and Path(d.clip_path).exists()):
                pool[(d.provider, d.source_id, round(d.src_start, 1), round(d.src_end, 1))].append(d)
        q = idx.clip.text(query_text(shot.brief))
        ranked = []
        for ident, uses in pool.items():
            if len(uses) + 1 > self.s.reuse_max_uses:               # this would be one appearance too many
                continue
            gaps = []
            for u in uses:
                other = self.shots[self.sid[u.shot]]
                gaps.append(max(me.start - other.end, other.start - me.end))
            if min(gaps) < self.s.reuse_min_gap:
                continue
            f = frame_of(Path(uses[0].clip_path))
            if f is None:
                continue
            sim = float(idx.clip.images([f])[0] @ q)
            if sim >= self.s.library_min_sim:
                ranked.append((sim, ident, uses[0], min(gaps)))
        for sim, ident, d, gap in sorted(ranked, key=lambda x: -x[0])[: self.s.reuse_tries]:
            with self._lock:
                if ident in self._reusing:
                    continue
                self._reusing.add(ident)
            want = self._clip_len(shot)
            out = self.proj.path("clips") / f"shot{shot.id:03d}_{d.provider}_{d.source_id}_{int(d.src_start * 10)}_reuse.mp4"
            c = Candidate(provider=d.provider, source_id=d.source_id, url=d.url, title=d.title, license=d.license)
            try:
                log.info("shot %d: nothing new fits; reusing shot %d's footage (CLIP %.2f, %.0f s apart)", shot.id, d.shot, sim, gap)
                info = probe(Path(d.clip_path))
                # an earlier shot's clip is exactly that shot's length: a longer shot cannot be cut from it, and
                # stretching it to fit failed on the real run ("clip still short after stretch"), so only a clip
                # at least as long as this shot (within 0.1 s) is reused
                if not info or info.duration < want - 0.1:
                    continue
                with stage("reuse cut"):
                    cut_clip(self.s, Path(d.clip_path), 0.0, want, out)
                got = self._judge_made_clip(
                    shot, c, out, (d.src_start, d.src_end),
                    lambda: verify_clip(self.gem, out, shot, f"{d.provider}:{d.source_id}:{int(d.src_start)}:reuse:{shot.id}", self.s.min_relevance),
                    f"{REUSED} of shot {d.shot} ({gap:.0f} s apart)")
                if got:
                    return got
            except JudgeError:
                raise
            except Exception as ex:
                out.unlink(missing_ok=True)
                log.warning("shot %d: reuse failed: %s", shot.id, str(ex)[:160])
            finally:
                with self._lock:
                    self._reusing.discard(ident)
        return None

    # -- still pictures ---------------------------------------------------------------------------------
    def _still(self, shot: Shot, only: list[str] | None = None) -> Decision | None:
        """A licensed photograph with a slow move when no video (YouTube, stock) fits the shot.
        Judged exactly like a clip: relevance, watermark/text, and the dedicated AI question."""
        if not self.s.stills or not self.gem or shot.brief is None:
            return None
        if self._photos is None:
            self._photos = make_photo_providers(self.s, self.cache)
        b = shot.brief
        # the place goes into the first query: "reef" alone finds any reef, the brief wants this one
        queries = list(dict.fromkeys(q.strip() for q in [f"{b.subject} {b.setting}", b.subject, *(b.stock_queries or [])] if q.strip()))[:3]
        left = self.s.still_tries
        for name, prov in self._photos.items():
            if not prov.available() or (only is not None and name not in only):
                continue
            for q in queries:
                for c in prov.search(q, 4):
                    if left <= 0:
                        return None
                    key = f"{c.provider}:{c.source_id}"
                    if any(t.startswith(key + ":") for t in self._tried(shot)):
                        continue
                    if self._skip_irrelevant(c, shot):
                        continue
                    win = (0.0, 1e6)
                    if not self._claim(key, win, shot.id):          # a picture is used once per project
                        continue
                    try:
                        with stage("still fetch"):
                            path = prov.fetch(c, self.proj.path("sources", "stills"))
                        if not path:
                            continue
                        left -= 1
                        d = self._try_still(shot, c, path)
                    finally:
                        self._release(key, win)
                    if d:
                        return d
        return None

    def _context(self, shot: Shot) -> str:
        """The narration just before and after this shot (other sentences only), so a picture is judged
        in its place in the story and not on a keyword."""
        i = self.sid[shot.id]
        before = next((self.shots[j].text for j in range(i - 1, -1, -1) if self.shots[j].text != shot.text), "")
        after = next((self.shots[j].text for j in range(i + 1, len(self.shots)) if self.shots[j].text != shot.text), "")
        return f'Narration before: "{before}"\nNarration after: "{after}"' if before or after else ""

    def _try_still(self, shot: Shot, c: Candidate, path: Path) -> Decision | None:
        want = self._clip_len(shot)
        out = self.proj.path("clips") / f"shot{shot.id:03d}_{c.provider}_{c.source_id}_0.mp4"
        try:
            with stage("still clip"):
                motion_clip(self.s, path, out, want, motion_for(shot.id))
        except Exception as e:
            self._decide(shot, c, "rejected", rejects=[f"still clip failed: {str(e)[:100]}"])
            return None
        return self._judge_made_clip(
            shot, c, out, (0.0, 0.0),
            lambda: verify_still(self.gem, path, shot, f"{c.provider}:{c.source_id}", self.s.min_relevance,
                                 context=self._context(shot)),
            f"still by {c.channel} ({c.license})", is_still=True, record_reject=True, min_relevance=self.s.still_min_relevance)
