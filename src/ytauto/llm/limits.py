"""ChatGPT plan usage limits, as printed by HeartCode and Codex.

Luna, Terra and the Codex CLI all draw from ONE ChatGPT plan limit. Measured on video-4 (4 h 09 min): the ChatGPT link
answered 871 calls and said "Usage limit reached" 13 times; each time only that single link was skipped for 5 minutes, so the
next call tried the same exhausted plan through another model. The reply says when the limit lifts ("try again at 7:06 PM"),
so every ChatGPT link can be paused until then and the chain moves straight on to Gemini.
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timedelta

# "Usage limit reached", "You've hit your usage limit", "limit reached" - not a per-minute rate limit of another provider
_LIMIT = re.compile(r"usage limit|hit your (?:\w+ )?limit|limit reached", re.I)
_AT = re.compile(r"(?:try again|resets?|available again|come back)\s+(?:at|on)?\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", re.I)
_IN = re.compile(r"(?:in|after)\s+(\d+(?:\.\d+)?)\s*(second|sec|minute|min|hour|hr|day)s?\b", re.I)
_UNIT = {"second": 1, "sec": 1, "minute": 60, "min": 60, "hour": 3600, "hr": 3600, "day": 86400}

DEFAULT_PAUSE = 900.0         # no time in the message: look again in 15 min (Gemini carries the run meanwhile)
MIN_PAUSE = 60.0


def plan_limit_reset(text: str, now: float | None = None, default: float = DEFAULT_PAUSE) -> float | None:
    """Epoch time until which the plan is out, or None when `text` is not a usage-limit message.

    Understands "try again at 7:06 PM" (local time; tomorrow if that time has already passed today) and
    "try again in 20 minutes". Without a time, `default` seconds from now."""
    if not text or not _LIMIT.search(text):
        return None
    now = time.time() if now is None else now
    m = _IN.search(text)
    if m:
        return now + max(MIN_PAUSE, float(m.group(1)) * _UNIT[m.group(2).lower()])
    m = _AT.search(text)
    if m:
        hour, minute, ampm = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").lower()
        if ampm == "pm" and hour < 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        if hour < 24 and minute < 60:
            local = datetime.fromtimestamp(now)
            at = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if at.timestamp() <= now:
                at += timedelta(days=1)
            return max(at.timestamp(), now + MIN_PAUSE)
    return now + default
