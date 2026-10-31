"""YouTube via yt-dlp: flat search, metadata (no download), and section-only downloads."""
from __future__ import annotations

import re
import shutil
import subprocess
import threading
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

# One lock per output file: a section may be fetched ahead in the background while the selector
# asks for the same window, and two yt-dlp runs must never write one file.
_DL_GUARD = threading.Lock()
_DL_LOCKS: dict[str, threading.Lock] = {}

_WORD = re.compile(r"[a-z0-9]+")


_GONE = ("video is not available", "private video", "video unavailable", "has been removed",
         "account associated with this video has been terminated", "no longer available")


def is_gone(err: str) -> bool:
    """A permanent "this video is gone" error, as opposed to a network or bot-check failure."""
    low = err.lower()
    return any(g in low for g in _GONE) and "sign in to confirm" not in low


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
        self._streams: dict[str, tuple[str, dict, float]] = {}   # see _remember_stream

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
            gone = self.cache.get("yt_gone", c.source_id)
            if gone:
                c.screen_flags["meta_error"] = gone
                return c
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
                self._remember_stream(c.source_id, info)
            except Exception as e:
                log.warning("metadata failed for %s: %s", c.url, e)
                c.screen_flags["meta_error"] = str(e)[:200]
                if is_gone(str(e)):
                    # Measured: 104 metadata calls failed this way in one 87-shot run, the same
                    # videos turning up again in later shots' searches. Remembered in the shared
                    # cache, so no project asks again.
                    self.cache.set("yt_gone", str(e)[:200], c.source_id)
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
        with _DL_GUARD:
            lock = _DL_LOCKS.setdefault(str(out), threading.Lock())
        with lock:                        # a concurrent caller waits here, then finds the file
            return self._download_section_locked(c, a, b, dest_dir, out)

    def _remember_stream(self, source_id: str, info: dict):
        """Keep the direct stream URL from a metadata call for this run's section downloads.

        Every yt-dlp section download starts with its own page extraction, ~2.2 s measured,
        about a quarter of a download - for a video whose metadata call fetched the same formats
        seconds earlier. yt-dlp then runs `ffmpeg -ss A -t D <headers> -i URL -c copy`; this does
        the same with the remembered URL. In memory only: the URLs expire after a few hours."""
        fmts = [f for f in info.get("formats") or [] if f.get("vcodec") not in (None, "none")
                and (f.get("acodec") in (None, "none")) and (f.get("height") or 0) <= self.s.height
                and str(f.get("protocol", "")).startswith("http") and f.get("url")]
        if not fmts:
            return
        best = max(fmts, key=lambda f: (str(f.get("vcodec", "")).startswith("avc1"), f.get("height") or 0,
                                        f.get("tbr") or 0))
        m = re.search(r"[?&]expire=(\d+)", best["url"])
        expire = float(m.group(1)) if m else time.time() + 3600
        with _DL_GUARD:
            if not hasattr(self, "_streams"):
                self._streams = {}
            self._streams[source_id] = (best["url"], best.get("http_headers") or info.get("http_headers") or {},
                                        expire)

    def _direct_section(self, source_id: str, a: float, b: float, out: Path) -> bool:
        with _DL_GUARD:
            got = getattr(self, "_streams", {}).get(source_id)
        if not got or got[2] - time.time() < 600 or not shutil.which("ffmpeg"):
            return False
        url, headers, _ = got
        cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}"]
        if headers:
            cmd += ["-headers", "".join(f"{k}: {v}\r\n" for k, v in headers.items())]
        cmd += ["-i", url, "-map", "0:v:0", "-c", "copy", str(out)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except (subprocess.TimeoutExpired, OSError) as e:
            log.debug("direct section download failed for %s: %s", source_id, e)
            out.unlink(missing_ok=True)
            return False
        if r.returncode != 0 or not out.exists() or out.stat().st_size <= 10_000:
            log.debug("direct section download failed for %s: %s", source_id, r.stderr[:160])
            out.unlink(missing_ok=True)
            with _DL_GUARD:
                self._streams.pop(source_id, None)      # stale or refused: let yt-dlp do it
            return False
        return True

    def _download_section_locked(self, c: Candidate, a: float, b: float, dest_dir: Path, out: Path) -> Path | None:
        if out.exists() and out.stat().st_size > 10_000:
            return out
        if self._direct_section(c.source_id, a, b, out):
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
