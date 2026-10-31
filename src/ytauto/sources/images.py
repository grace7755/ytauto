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

    def __init__(self, s: Settings):
        self.s = s

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
    """Wikimedia Commons: free, and the API reports licence, author and restrictions per file."""
    name = "commons"

    def search(self, query, n=6):
        try:
            r = requests.get("https://commons.wikimedia.org/w/api.php", timeout=30, headers=UA, params={
                "action": "query", "generator": "search", "gsrsearch": f"{query} filetype:bitmap",
                "gsrnamespace": 6, "gsrlimit": n * 3, "prop": "imageinfo",
                "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 3840,
                "iiextmetadatafilter": "LicenseShortName|Artist|NonFree|Restrictions", "format": "json",
                "formatversion": 2})
            r.raise_for_status()
        except Exception as e:
            log.warning("commons search failed: %s", str(e)[:160])
            return []
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


def make_photo_providers(s: Settings) -> dict[str, PhotoProvider]:
    all_ = {"pexels": PexelsPhotos(s), "pixabay": PixabayPhotos(s), "commons": Commons(s)}
    return {k: all_[k] for k in s.still_order if k in all_}
