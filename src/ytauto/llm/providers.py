"""Read-only access to a shared OpenAI-compatible provider config (``HEART_CONFIG``).

The config file belongs to another project. We **only ever read it** - never write, never copy -
so a key rotation there is picked up here automatically and this project cannot break that one.

Expected shape (extra keys ignored)::

    {"providers": {"<name>": {"baseUrl": "...", "apiKey": "..."}, ...}}
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..log import log

DIRECT = "_direct"   # sentinel provider meaning "the native google-genai client + GEMINI_API_KEY"


class ProviderConfig:
    """Loaded `providers` map from the shared config file."""

    def __init__(self, providers: dict[str, dict], source: Path | None = None):
        self.providers = providers
        self.source = source

    @classmethod
    def load(cls, path: str | Path | None = None) -> ProviderConfig:
        """Load from `path` or $HEART_CONFIG. Missing/unreadable file -> empty (never raises)."""
        raw = path or os.getenv("HEART_CONFIG")
        if not raw:
            return cls({})
        p = Path(str(raw).strip().strip('"').strip("'")).expanduser()
        if not p.exists():
            log.warning("HEART_CONFIG not found at %s - skipping proxy providers", p)
            return cls({})
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            log.warning("HEART_CONFIG unreadable (%s) - skipping proxy providers", str(e)[:160])
            return cls({})
        provs = data.get("providers")
        if not isinstance(provs, dict):
            log.warning("HEART_CONFIG has no usable 'providers' map - skipping proxy providers")
            return cls({})
        return cls(provs, p)

    def endpoint(self, provider: str) -> tuple[str, str] | None:
        """(base_url, api_key) for `provider`, or None when it is absent/incomplete."""
        p = self.providers.get(provider)
        if not isinstance(p, dict):
            return None
        base, key = p.get("baseUrl"), p.get("apiKey")
        if not base or not key:
            return None
        return str(base).rstrip("/"), str(key)

    def has(self, provider: str) -> bool:
        return provider == DIRECT or self.endpoint(provider) is not None
