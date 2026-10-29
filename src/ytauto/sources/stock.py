"""Fallback stock sources: Pexels, Pixabay, Coverr, Mixkit (via the user's mixkit-dl package).
Each provider exposes search(query, n) -> [Candidate]; fetch(candidate, dest_dir) -> Path|None."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
from pathlib import Path

import requests

from ..config import Settings
from ..log import log
from ..models import Candidate

UA = {"User-Agent": "ytauto/0.1 (+local automation)"}
_MIXKIT_LOCK = threading.Lock()


def _probe_dims(path: Path) -> tuple[int, int] | None:
    """(width, height) of a local video via ffprobe, or None if it cannot be read."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0:nk=1", str(path)],
            capture_output=True, text=True, timeout=30)
        w, h = (int(x) for x in r.stdout.strip().split(",")[:2])
        return w, h
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _download(url: str, dest: Path) -> Path | None:
    if dest.exists() and dest.stat().st_size > 50_000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    try:
        with requests.get(url, stream=True, timeout=60, headers=UA) as r:
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        tmp.replace(dest)
        return dest
    except Exception as e:
        log.warning("stock download failed %s: %s", url, str(e)[:160])
        tmp.unlink(missing_ok=True)
        return None


class Provider:
    name = ""

    def __init__(self, s: Settings):
        self.s = s

    def available(self) -> bool:
        return True

    def search(self, query: str, n: int = 8) -> list[Candidate]:
        raise NotImplementedError

    def fetch(self, c: Candidate, dest_dir: Path) -> Path | None:
        if c.local_path and Path(c.local_path).exists():
            return Path(c.local_path)
        if not c.direct_url:
            return None
        return _download(c.direct_url, dest_dir / f"{c.provider}_{c.source_id}.mp4")


class Pexels(Provider):
    name = "pexels"

    def available(self):
        return bool(self.s.pexels_key)

    def search(self, query, n=8):
        try:
            r = requests.get("https://api.pexels.com/videos/search", timeout=30,
                             headers={"Authorization": self.s.pexels_key, **UA},
                             params={"query": query, "per_page": n, "orientation": "landscape", "size": "large"})
            r.raise_for_status()
        except Exception as e:
            log.warning("pexels search failed: %s", str(e)[:160])
            return []
        out = []
        for v in r.json().get("videos", []):
            files = [f for f in v.get("video_files", []) if f.get("file_type") == "video/mp4" and f.get("width")]
            files = [f for f in files if f["width"] >= f.get("height", 0)]  # landscape
            if not files:
                continue
            best = min(files, key=lambda f: (abs(f["height"] - self.s.height), -f["height"]))
            out.append(Candidate(provider="pexels", source_id=str(v["id"]), url=v["url"],
                                 title=(v.get("url", "").rstrip("/").split("/")[-1].rsplit("-", 1)[0]).replace("-", " "),
                                 channel=(v.get("user") or {}).get("name", ""), duration=float(v.get("duration", 0)),
                                 height=best["height"], license="Pexels License", direct_url=best["link"]))
        return out


class Pixabay(Provider):
    name = "pixabay"

    def available(self):
        return bool(self.s.pixabay_key)

    def search(self, query, n=8):
        try:
            r = requests.get("https://pixabay.com/api/videos/", timeout=30, headers=UA,
                             params={"key": self.s.pixabay_key, "q": query, "per_page": max(3, n),
                                     "video_type": "film", "safesearch": "true"})
            r.raise_for_status()
        except Exception as e:
            log.warning("pixabay search failed: %s", str(e)[:160])
            return []
        out = []
        for h in r.json().get("hits", []):
            vids = [v for v in h.get("videos", {}).values() if v.get("url") and v.get("height")]
            if not vids:
                continue
            best = min(vids, key=lambda v: (abs(v["height"] - self.s.height), -v["height"]))
            out.append(Candidate(provider="pixabay", source_id=str(h["id"]), url=h.get("pageURL", ""),
                                 title=h.get("tags", ""), channel=h.get("user", ""), duration=float(h.get("duration", 0)),
                                 height=best["height"], tags=[t.strip() for t in h.get("tags", "").split(",")],
                                 license="Pixabay Content License", direct_url=best["url"]))
        return out


class Coverr(Provider):
    name = "coverr"

    def available(self):
        return bool(self.s.coverr_key)

    def search(self, query, n=8):
        try:
            r = requests.get("https://api.coverr.co/videos", timeout=30,
                             headers={"Authorization": f"Bearer {self.s.coverr_key}", **UA},
                             params={"query": query, "page_size": n, "urls": "true"})
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log.warning("coverr search failed: %s", str(e)[:160])
            return []
        out = []
        for h in data.get("hits", data.get("videos", [])):
            if h.get("is_vertical") or h.get("is_ai_generated") or h.get("is_premium"):
                continue
            urls = h.get("urls") or {}
            link = urls.get("mp4_download") or urls.get("mp4") or urls.get("mp4_preview")
            if not link:
                continue
            h_px = int(h.get("max_height") or (1080 if urls.get("mp4_download") else 720))
            out.append(Candidate(provider="coverr", source_id=str(h.get("id")), url=f"https://coverr.co/videos/{h.get('id')}",
                                 title=h.get("title", ""), duration=float(h.get("duration") or 0), height=h_px,
                                 tags=h.get("tags") or [], description=h.get("description", ""),
                                 license="Coverr License", direct_url=link))
        return out


# Mixkit has no API; `mixkit-dl` scrapes it. A multi-word phrase IS a real search: the slug
# /free-stock-video/<slug>/ redirects to /free-stock-video/discover/<slug>/ (verified 2026-10-29).
# Longer phrases match fewer clips, so follow the GUIDE's "keyword ladder": specific -> broad.
# Broad terms to append once the shot's own queries are exhausted.
MIXKIT_BROAD = {
    "temple": "travel", "shrine": "travel", "tourist": "travel", "landmark": "travel",
    "volcano": "mountain", "peak": "mountain", "hill": "mountain", "fuji": "mountain",
    "lake": "water", "river": "water", "sea": "sea", "waves": "sea", "waterfall": "water",
    "urban": "city", "downtown": "city", "skyline": "city", "buildings": "city",
    "traffic": "street", "trees": "forest", "woods": "forest", "jungle": "forest",
    "garden": "flowers", "blossom": "flowers", "snowy": "snow", "ice": "winter",
    "sunrise": "sunset", "dusk": "sunset", "clouds": "sky", "crowd": "people",
    "walking": "people", "street": "street", "night": "night", "city": "city",
    "nature": "nature", "travel": "travel", "mountain": "mountain", "forest": "forest",
}
# Exit codes from the documented contract (GUIDE.md Part 2).
MIXKIT_OK = (0, 1)          # success / partial success - read the JSON either way
MIXKIT_EMPTY = 2            # nothing found for those keywords -> try broader
MIXKIT_DISK_FULL = 3        # disk guard tripped -> stop entirely
MIXKIT_RATE_LIMITED = 4     # HTTP 429 -> stop and let the run continue without Mixkit


class Mixkit(Provider):
    """Mixkit via the user's `mixkit-dl` CLI (JSON contract, see its GUIDE.md Part 2)."""

    name = "mixkit"

    def __init__(self, s: Settings):
        super().__init__(s)
        self._blocked = False       # set when Mixkit rate-limits or the disk guard trips

    def available(self):
        if self._blocked:
            return False
        try:
            import mixkit_dl  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def keyword_ladder(query: str, extra: list[str] | None = None) -> list[str]:
        """Search terms from specific to broad, per the GUIDE's recommended logic."""
        q = " ".join(query.split())
        ladder: list[str] = []

        def add(t: str):
            t = t.strip()
            if t and t.lower() not in [x.lower() for x in ladder]:
                ladder.append(t)

        add(q)
        for e in (extra or []):
            add(e)
        words = [w for w in re.findall(r"[A-Za-z]+", q) if len(w) > 2]
        if len(words) > 2:                      # a shorter phrase matches more clips
            add(" ".join(words[:2]))
        for w in words:                         # then each broad term the words imply
            broad = MIXKIT_BROAD.get(w.lower())
            if broad:
                add(broad)
        add("nature")                           # last resort: a category that always has clips
        return ladder

    def _run_cli(self, term: str, n: int) -> tuple[int, dict]:
        """One `mixkit-dl` call. Returns (exit_code, parsed_json). Never raises."""
        cmd = [sys.executable, "-m", "mixkit_dl", term, "--limit", str(n),
               "--quality", "1080", "--out", str(self.s.mixkit_dir),
               "--delay", "1", "--json", "--quiet"]
        try:
            # Mixkit rate-limits aggressively: one call at a time even when shots run in parallel.
            with _MIXKIT_LOCK:
                p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                                   timeout=self.s.mixkit_timeout)
        except subprocess.TimeoutExpired:
            log.warning("mixkit search %r timed out after %ss", term, self.s.mixkit_timeout)
            return 1, {}
        except OSError as e:
            log.warning("could not run mixkit-dl: %s", str(e)[:160])
            return 1, {}
        data = {}
        if p.stdout.strip():                    # --json puts ONLY the report on stdout
            try:
                data = json.loads(p.stdout)
            except json.JSONDecodeError:
                log.warning("mixkit returned unparseable JSON for %r: %s", term, p.stdout[:160])
        return p.returncode, data

    def search(self, query, n=5):
        if self._blocked:
            return []
        out: list[Candidate] = []
        seen: set[str] = set()
        # One run at a time, broadening until we have enough (GUIDE: "well-behaved automation").
        for term in self.keyword_ladder(query)[:self.s.mixkit_max_terms]:
            code, data = self._run_cli(term, n)
            if code == MIXKIT_RATE_LIMITED:
                log.warning("mixkit rate-limited us (429); skipping Mixkit for the rest of this run")
                self._blocked = True
                break
            if code == MIXKIT_DISK_FULL:
                log.warning("mixkit stopped by its disk-space guard; skipping Mixkit for this run")
                self._blocked = True
                break
            if code == MIXKIT_EMPTY:
                log.debug("mixkit: no results for %r, broadening", term)
                continue
            if code not in MIXKIT_OK:
                log.warning("mixkit exited %s for %r", code, term)
                continue
            for it in data.get("items", []):
                # Usable = downloaded or already on disk. Fields may be absent: use .get().
                if it.get("status") not in ("downloaded", "skipped"):
                    continue
                path, vid = it.get("path"), str(it.get("id", ""))
                if not path or not vid or vid in seen or not Path(path).exists():
                    continue
                seen.add(vid)
                # Mixkit mixes vertical clips into results and the JSON has no orientation field,
                # so measure the file: a portrait clip cannot fill a 16:9 frame without ruinous crop.
                dims = _probe_dims(Path(path))
                if dims is None:
                    log.debug("mixkit: could not probe %s, skipping", Path(path).name)
                    continue
                w, h = dims
                if w < h:
                    log.debug("mixkit: skipping vertical clip %s (%dx%d)", Path(path).name, w, h)
                    continue
                out.append(Candidate(
                    provider="mixkit", source_id=vid, url=it.get("page_url") or it.get("url", ""),
                    title=it.get("title", ""), tags=[term], height=h,
                    license="Mixkit Stock Video Free License", local_path=path))
            if len(out) >= n:
                break
        if not out:
            log.warning("mixkit found nothing usable for %r", query)
        return out[:n]


def make_providers(s: Settings) -> dict[str, Provider]:
    all_ = {"pexels": Pexels(s), "pixabay": Pixabay(s), "coverr": Coverr(s), "mixkit": Mixkit(s)}
    return {k: all_[k] for k in s.fallback_order if k in all_}
