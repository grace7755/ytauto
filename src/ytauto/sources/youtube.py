"""YouTube via yt-dlp: flat search, metadata (no download), and section-only downloads."""
from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path

import yt_dlp
from yt_dlp.utils import download_range_func

from ..cache import Cache
from ..config import Settings
from ..log import log
from ..models import Candidate
from . import storyboard

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> set[str]:
    return set(_WORD.findall(s.lower()))


def text_score(cand: Candidate, queries: list[str], entities: list[str]) -> float:
    """Cheap lexical relevance 0-100 from title/description/tags vs. the brief."""
    hay_title = _tokens(cand.title)
    hay_all = hay_title | _tokens(cand.description[:1500]) | _tokens(" ".join(cand.tags))
    ent_tok = [_tokens(e) for e in entities if e]
    score = 0.0
    if ent_tok:
        hits = sum(1 for t in ent_tok if t and t <= hay_all)
        title_hits = sum(1 for t in ent_tok if t and t <= hay_title)
        score += 55 * hits / len(ent_tok) + 25 * title_hits / len(ent_tok)
    q_tok = set().union(*[_tokens(q) for q in queries]) if queries else set()
    q_tok -= {"4k", "the", "in", "of", "and", "a", "to"}
    if q_tok:
        score += 20 * len(q_tok & hay_title) / len(q_tok)
    bonus = {"4k", "drone", "cinematic", "b-roll", "broll", "walk", "walking", "tour", "timelapse", "aerial"}
    score += 5 * len(bonus & hay_title)
    return min(score, 100.0)


class YouTube:
    def __init__(self, settings: Settings, cache: Cache):
        self.s = settings
        self.cache = cache

    def _opts(self, **extra) -> dict:
        o = {"quiet": True, "no_warnings": True, "noprogress": True, "retries": 5,
             "fragment_retries": 5, "socket_timeout": 30}
        if self.s.cookies_from_browser:
            o["cookiesfrombrowser"] = (self.s.cookies_from_browser,)
        o.update(extra)
        return o

    # ---- search ------------------------------------------------------
    def search(self, query: str, n: int | None = None) -> list[Candidate]:
        n = n or self.s.results_per_query
        cached = self.cache.get("yt_search", query, n)
        if cached is None:
            try:
                with yt_dlp.YoutubeDL(self._opts(extract_flat=True, skip_download=True)) as ydl:
                    info = ydl.extract_info(f"ytsearch{n}:{query}", download=False)
                cached = [e for e in (info.get("entries") or []) if e]
                self.cache.set("yt_search", cached, query, n)
            except Exception as e:  # network / extractor failures must not kill the run
                log.warning("YouTube search failed for %r: %s", query, e)
                return []
        out = []
        for e in cached:
            vid = e.get("id")
            if not vid or len(vid) != 11:
                continue
            out.append(Candidate(
                provider="youtube", source_id=vid, url=f"https://www.youtube.com/watch?v={vid}",
                title=e.get("title") or "", channel=e.get("channel") or e.get("uploader") or "",
                duration=float(e.get("duration") or 0), description=(e.get("description") or "")))
        return out

    def prefilter(self, c: Candidate) -> str | None:
        """Return a rejection reason, or None if the (flat) candidate may proceed."""
        if c.duration and not (self.s.min_source_duration <= c.duration <= self.s.max_source_duration):
            return f"duration {c.duration:.0f}s outside range"
        low = c.title.lower()
        for bad in self.s.title_blacklist:
            if bad in low:
                return f"title blacklist: {bad}"
        return None

    # ---- metadata ----------------------------------------------------
    def details(self, c: Candidate) -> Candidate:
        """Fill height/upload_date/license/tags/description without downloading."""
        meta = self.cache.get("yt_meta", c.source_id)
        if meta is None:
            try:
                with yt_dlp.YoutubeDL(self._opts(skip_download=True)) as ydl:
                    info = ydl.extract_info(c.url, download=False)
                heights = [f.get("height") or 0 for f in info.get("formats", []) if f.get("vcodec") not in (None, "none")]
                meta = {
                    "height": max(heights or [0]), "upload_date": info.get("upload_date") or "",
                    "license": info.get("license") or "Standard YouTube License",
                    "description": (info.get("description") or "")[:3000], "tags": (info.get("tags") or [])[:40],
                    "duration": info.get("duration") or c.duration, "title": info.get("title") or c.title,
                    "channel": info.get("channel") or c.channel, "live": bool(info.get("is_live")),
                    "age_limit": info.get("age_limit") or 0,
                    "chapters": [(ch.get("start_time"), ch.get("title")) for ch in (info.get("chapters") or [])][:30],
                    # captured now so storyboard screening needs no second metadata call
                    "storyboard": storyboard.spec_from_info(info),
                }
                self.cache.set("yt_meta", meta, c.source_id)
            except Exception as e:
                log.warning("metadata failed for %s: %s", c.url, e)
                c.screen_flags["meta_error"] = str(e)[:200]
                return c
        c.height, c.upload_date, c.license = meta["height"], meta["upload_date"], meta["license"]
        c.description, c.tags = meta["description"], meta["tags"]
        c.duration, c.title, c.channel = meta["duration"], meta["title"], meta["channel"]
        c.screen_flags["live"] = meta["live"]
        c.screen_flags["chapters"] = meta["chapters"]
        return c

    def storyboard(self, c: Candidate) -> storyboard.Storyboard | None:
        """Numbered contact sheet of the video's seek-bar previews (cached on disk), or None."""
        meta = self.cache.get("yt_meta", c.source_id) or {}

        def refetch():
            try:
                with yt_dlp.YoutubeDL(self._opts(skip_download=True)) as ydl:
                    info = ydl.extract_info(c.url, download=False)
            except Exception as e:
                log.debug("storyboard metadata failed for %s: %s", c.url, str(e)[:160])
                return None
            spec = storyboard.spec_from_info(info)
            if meta:
                self.cache.set("yt_meta", dict(meta, storyboard=spec), c.source_id)
            return spec

        return storyboard.load(c.source_id, meta.get("storyboard"), c.duration or meta.get("duration") or 0,
                               self.s.cache_dir, refetch_spec=refetch)

    def age_years(self, c: Candidate) -> float | None:
        if len(c.upload_date) != 8:
            return None
        try:
            return (datetime.now() - datetime.strptime(c.upload_date, "%Y%m%d")).days / 365.25
        except ValueError:
            return None

    # ---- download ----------------------------------------------------
    def download_section(self, c: Candidate, start: float, end: float, dest_dir: Path, pad: float = 1.5) -> Path | None:
        """Download only [start-pad, end+pad] of the video (video stream only, <=1080p). Cached on disk.

        Stream copy, no `force_keyframes_at_cuts`: that re-encoded every section at x264 defaults
        before `cut_clip` encodes it again, a lossy extra generation. Measured on two videos: the
        copied section's frames line up with the re-encoded one's to the frame (so a screened
        window still maps to file time by subtracting the section start), and it is ~1 s faster.
        Most of a section's ~12 s is the transfer itself (~2 s extraction, then streaming at
        roughly real time)."""
        a, b = max(0.0, start - pad), end + pad
        dest_dir.mkdir(parents=True, exist_ok=True)
        out = dest_dir / f"{c.source_id}_{int(a * 10):06d}_{int(b * 10):06d}.mp4"
        if out.exists() and out.stat().st_size > 10_000:
            return out
        opts = self._opts(
            format=f"bestvideo[height<={self.s.height}][vcodec^=avc1]/bestvideo[height<={self.s.height}]/best[height<={self.s.height}]",
            download_ranges=download_range_func(None, [(a, b)]),
            outtmpl=str(out.with_suffix(".%(ext)s")),
            merge_output_format="mp4", overwrites=False)
        for attempt in range(2):          # a section transfer can die mid-stream (seen: ffmpeg exit 3436169992)
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    ydl.download([c.url])
                break
            except Exception as e:
                log.warning("section download failed %s [%.1f-%.1f]%s: %s", c.url, a, b,
                            " - retrying" if attempt == 0 else "", str(e)[:200])
                if attempt:
                    return None
                time.sleep(2)
        if out.exists() and out.stat().st_size > 10_000:
            return out
        for f in dest_dir.glob(out.stem + ".*"):  # container other than mp4 (webm/mkv)
            if f.suffix.lower() in (".mkv", ".webm") and f.stat().st_size > 10_000:
                return f
        return None
