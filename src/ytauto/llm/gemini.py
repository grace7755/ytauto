"""Thin google-genai wrapper: JSON-schema responses, retries, caching, usage log."""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..cache import Cache
from ..config import Settings
from ..log import log


class GeminiError(RuntimeError):
    pass


class Gemini:
    def __init__(self, settings: Settings, cache: Cache):
        if not settings.gemini_key:
            raise GeminiError("GEMINI_API_KEY is not set (put it in yt-auto/.env)")
        from google import genai
        self._genai = genai
        self._types = genai.types
        self.client = genai.Client(api_key=settings.gemini_key)
        self.s = settings
        self.cache = cache
        self._down: dict[str, float] = {}
        self.calls = 0
        self.tokens = 0

    FALLBACKS = ["gemini-3-flash-preview", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.7-flash"]

    def _generate(self, model: str, parts: list, schema: dict | None, retries: int = 3) -> str:
        """Try `model`, then fall back to older flash models if it keeps failing (e.g. 503 demand spikes)."""
        last = None
        order = [model] + [f for f in self.FALLBACKS if f != model]
        now = time.time()
        live = [m for m in order if self._down.get(m, 0) < now] or order   # skip models that just failed
        for m in live:
            try:
                return self._generate_one(m, parts, schema, 2 if m != live[-1] else retries)
            except GeminiError as e:
                last = e
                self._down[m] = time.time() + 300      # circuit breaker: skip this model for 5 min
                log.warning("model %s unavailable, skipping it for 5 min", m)
        raise last

    def _generate_one(self, model: str, parts: list, schema: dict | None, retries: int) -> str:
        t = self._types
        cfg = t.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json" if schema else None,
            response_schema=schema if schema else None)
        err = None
        for attempt in range(retries):
            try:
                r = self.client.models.generate_content(
                    model=model, contents=t.Content(role="user", parts=parts), config=cfg)
                self.calls += 1
                u = getattr(r, "usage_metadata", None)
                if u and getattr(u, "total_token_count", None):
                    self.tokens += u.total_token_count
                if not r.text:
                    raise GeminiError("empty response")
                return r.text
            except Exception as e:  # rate limit / 5xx / transient
                err = e
                if "429" in str(e)[:12] or "404" in str(e)[:12]:   # quota exhausted / model gone: don't retry
                    break
                wait = min(60, 3 * 2 ** attempt)
                log.warning("Gemini call failed (%s); retry in %ss", str(e)[:160], wait)
                time.sleep(wait)
        raise GeminiError(f"Gemini failed after {retries} attempts: {err}")

    def json(self, prompt: str, schema: dict, *, model: str | None = None, cache_key: tuple | None = None):
        if cache_key:
            hit = self.cache.get("gemini", *cache_key)
            if hit is not None:
                return hit
        text = self._generate(model or self.s.gemini_plan_model, [self._types.Part(text=prompt)], schema)
        data = json.loads(text)
        if cache_key:
            self.cache.set("gemini", data, *cache_key)
        return data

    def json_youtube(self, url: str, prompt: str, schema: dict, *, start: float | None = None,
                     end: float | None = None, fps: float = 1.0, cache_key: tuple | None = None):
        """Ask Gemini about a public YouTube URL without downloading it."""
        if cache_key:
            hit = self.cache.get("gemini", *cache_key)
            if hit is not None:
                return hit
        t = self._types
        meta = {"fps": fps}
        if start is not None:
            meta["start_offset"] = f"{int(start)}s"
        if end is not None:
            meta["end_offset"] = f"{int(end)}s"
        parts = [t.Part(file_data=t.FileData(file_uri=url), video_metadata=t.VideoMetadata(**meta)),
                 t.Part(text=prompt)]
        data = json.loads(self._generate(self.s.gemini_model, parts, schema))
        if cache_key:
            self.cache.set("gemini", data, *cache_key)
        return data

    def json_images(self, prompt: str, images: list[bytes], schema: dict):
        """Ask Gemini about one or more inline JPEGs (no caching here; the router caches)."""
        t = self._types
        parts = [t.Part(inline_data=t.Blob(data=b, mime_type="image/jpeg")) for b in images]
        parts.append(t.Part(text=prompt))
        return json.loads(self._generate(self.s.gemini_model, parts, schema))

    def json_video_file(self, path: Path, prompt: str, schema: dict, *, fps: float = 2.0,
                        cache_key: tuple | None = None):
        """Ask Gemini about a short local clip (sent inline, < 20 MB)."""
        if cache_key:
            hit = self.cache.get("gemini", *cache_key)
            if hit is not None:
                return hit
        t = self._types
        parts = [t.Part(inline_data=t.Blob(data=Path(path).read_bytes(), mime_type="video/mp4"),
                        video_metadata=t.VideoMetadata(fps=fps)),
                 t.Part(text=prompt)]
        data = json.loads(self._generate(self.s.gemini_model, parts, schema))
        if cache_key:
            self.cache.set("gemini", data, *cache_key)
        return data
