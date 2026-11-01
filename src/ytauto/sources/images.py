"""Still-image sources for shots no video can fill: Pexels photos, Pixabay photos, Wikimedia Commons.

Each provider exposes search(query, n) -> [Candidate] (provider "still_<name>", `direct_url` the file,
`license`/`channel` the licence and author) and fetch(candidate, dest_dir) -> Path | None. Only
licences that allow reuse are returned; Commons files with a restriction or a non-free flag are
skipped. Coverr and Mixkit have no photographs (checked: their API and site list videos, music and sound
effects only), so they stay video fallbacks in sources/stock.py.
"""
from __future__ import annotations

import html
import re
import threading
import time
from pathlib import Path

import requests

from ..config import Settings
from ..log import log
from ..models import Candidate
from .stock import UA, _download

MIN_RATIO, MAX_RATIO = 1.5, 2.4   # near 16:9: cropping to it must not lose a third of the picture (or a banner)
_COMMONS_OK = re.compile(r"^(public domain|pd\b|cc0|cc[ -]by(?:[ -]sa)?\b|attribution|no restrictions)", re.I)


class PhotoProvider:
    name = ""

    def __init__(self, s: Settings, cache=None):
        self.s = s
        self.cache = cache                 # optional disk cache for search answers (Commons uses it)

    def available(self) -> bool:
        return True

    def search(self, query: str, n: int = 6) -> list[Candidate]:
        raise NotImplementedError

    def fetch(self, c: Candidate, dest_dir: Path) -> Path | None:
        if not c.direct_url:
            return None
        return _download(c.direct_url, dest_dir / f"{c.provider}_{c.source_id}.jpg")

    def _ok(self, w: int, h: int) -> bool:
        return w >= self.s.still_min_width and h > 0 and MIN_RATIO <= w / h <= MAX_RATIO


class PexelsPhotos(PhotoProvider):
    name = "pexels"

    def available(self):
        return bool(self.s.pexels_key)

    def search(self, query, n=6):
        try:
            r = requests.get("https://api.pexels.com/v1/search", timeout=30,
                             headers={"Authorization": self.s.pexels_key, **UA},
                             params={"query": query, "per_page": n * 2, "orientation": "landscape", "size": "large"})
            r.raise_for_status()
        except Exception as e:
            log.warning("pexels photo search failed: %s", str(e)[:160])
            return []
        out = []
        for p in r.json().get("photos", []):
            if not self._ok(p.get("width", 0), p.get("height", 0)):
                continue
            src = (p.get("src") or {})
            out.append(Candidate(provider="still_pexels", source_id=str(p["id"]), url=p.get("url", ""),
                                 title=p.get("alt") or "", channel=p.get("photographer", ""),
                                 height=p["height"], license="Pexels License",
                                 direct_url=src.get("original") or src.get("large2x")))
        return out[:n]


class PixabayPhotos(PhotoProvider):
    name = "pixabay"

    def available(self):
        return bool(self.s.pixabay_key)

    def search(self, query, n=6):
        try:
            r = requests.get("https://pixabay.com/api/", timeout=30, headers=UA,
                             params={"key": self.s.pixabay_key, "q": query, "image_type": "photo",
                                     "orientation": "horizontal", "per_page": max(3, n * 2), "safesearch": "true"})
            r.raise_for_status()
        except Exception as e:
            log.warning("pixabay photo search failed: %s", str(e)[:160])
            return []
        out = []
        for h in r.json().get("hits", []):
            w, hh = h.get("imageWidth", 0), h.get("imageHeight", 0)
            url = h.get("largeImageURL")
            if not url or not self._ok(w, hh):
                continue
            out.append(Candidate(provider="still_pixabay", source_id=str(h["id"]), url=h.get("pageURL", ""),
                                 title=h.get("tags", ""), channel=h.get("user", ""), height=hh,
                                 tags=[t.strip() for t in h.get("tags", "").split(",")],
                                 license="Pixabay Content License", direct_url=url))
        return out[:n]


class Commons(PhotoProvider):
    """Wikimedia Commons: free, and the API reports licence, author and restrictions per file.

    Polite by construction: Commons answers HTTP 429 to a fast, anonymous client (video-4's coverage run was refused from
    shot ~39 on). All instances share one lock and clock: calls are `commons_gap` seconds apart, a 429 pauses Commons for the
    time its `Retry-After` names (60 s if absent) and is logged once, and answers are cached on disk so a redo or re-run does
    not ask again. Errors are never cached."""
    name = "commons"
    _lock = threading.Lock()
    _last = 0.0                    # monotonic time of the last request
    _blocked_until = 0.0           # monotonic time before which Commons is not asked

    def _headers(self) -> dict:
        c = (getattr(self.s, "contact", "") or "").strip()
        return {"User-Agent": f"ytauto/0.1 ({c})" if c else UA["User-Agent"]}

    def search(self, query, n=6):
        hit = self.cache.get("commons-v1", query, n) if self.cache else None
        if hit is not None:
            return [Candidate(**d) for d in hit]
        with Commons._lock:                           # one request at a time, spaced out
            now = time.monotonic()
            if now < Commons._blocked_until:
                return []
            wait = Commons._last + self.s.commons_gap - now
            if wait > 0:
                time.sleep(wait)
            Commons._last = time.monotonic()
            out = self._request(query, n)
        if out is not None and self.cache:
            self.cache.set("commons-v1", [c.model_dump() for c in out], query, n)
        return out or []

    def _request(self, query, n) -> list[Candidate] | None:
        """The candidates, or None when the request failed (not cached, so it is asked again later)."""
        try:
            r = requests.get("https://commons.wikimedia.org/w/api.php", timeout=30, headers=self._headers(), params={
                "action": "query", "generator": "search", "gsrsearch": f"{query} filetype:bitmap",
                "gsrnamespace": 6, "gsrlimit": n * 3, "prop": "imageinfo",
                "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 3840,
                "iiextmetadatafilter": "LicenseShortName|Artist|NonFree|Restrictions", "format": "json",
                "formatversion": 2})
            if getattr(r, "status_code", 200) == 429:
                try:
                    secs = float(r.headers.get("Retry-After", 60))
                except (TypeError, ValueError):
                    secs = 60.0
                secs = min(max(secs, 1.0), 600.0)
                Commons._blocked_until = time.monotonic() + secs
                log.warning("commons rate-limited (429): not asking again for %.0f s", secs)
                return None
            r.raise_for_status()
        except Exception as e:
            log.warning("commons search failed: %s", str(e)[:160])
            return None
        out = []
        for p in sorted(r.json().get("query", {}).get("pages", []), key=lambda p: p.get("index", 99)):
            info = (p.get("imageinfo") or [{}])[0]
            meta = info.get("extmetadata") or {}
            lic = (meta.get("LicenseShortName") or {}).get("value", "")
            nonfree = str((meta.get("NonFree") or {}).get("value", "")).lower() == "true"
            if (info.get("mime") != "image/jpeg" or nonfree or (meta.get("Restrictions") or {}).get("value")
                    or not _COMMONS_OK.match(lic) or not self._ok(info.get("width", 0), info.get("height", 0))):
                continue
            artist = html.unescape(re.sub(r"<[^>]+>", "", (meta.get("Artist") or {}).get("value", ""))).strip()
            url = info["thumburl"] if info["width"] > 3840 and info.get("thumburl") else info["url"]
            out.append(Candidate(provider="still_commons", source_id=str(p["pageid"]),
                                 url=info.get("descriptionurl", ""), title=p.get("title", "").removeprefix("File:"),
                                 channel=artist[:80], height=info["height"], license=lic, direct_url=url))
        return out[:n]


def make_photo_providers(s: Settings, cache=None) -> dict[str, PhotoProvider]:
    all_ = {"pexels": PexelsPhotos(s, cache), "pixabay": PixabayPhotos(s, cache), "commons": Commons(s, cache)}
    return {k: all_[k] for k in s.still_order if k in all_}
