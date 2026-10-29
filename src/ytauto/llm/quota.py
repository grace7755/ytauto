"""Client-side quota for judge links: a per-minute token bucket and a persisted daily counter.

Free judge links are capped in *requests per day*, per project (Gemini resets at midnight
Pacific). Finding that out by hitting 429s costs a failed round-trip per call, and once every
link is spent a long run turns every remaining shot into an "unavailable" blocker. Counting
requests ourselves lets the chain skip a spent link without calling it and lets the pipeline
stop cleanly - or wait for the reset - instead.

Counters live in the shared sqlite `Cache`, so they survive restarts and are shared by every
project on this machine (the providers count per key, not per project).
"""
from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone

from ..cache import Cache
from ..config import JudgeLink

QUOTA_DOWN_SECONDS = 3600       # quota 429 from a link with no known reset time: retry hourly


def _us_dst(dt_utc: datetime) -> bool:
    """US daylight saving: second Sunday of March 02:00 to first Sunday of November 02:00 local.
    Computed by hand because Windows Python has no tz database unless `tzdata` is installed."""
    y = dt_utc.year
    mar = datetime(y, 3, 8 + (6 - datetime(y, 3, 8).weekday()) % 7, 10, tzinfo=timezone.utc)  # 02:00 PST
    nov = datetime(y, 11, 1 + (6 - datetime(y, 11, 1).weekday()) % 7, 9, tzinfo=timezone.utc)  # 02:00 PDT
    return mar <= dt_utc < nov


def pacific_now(now: float | None = None) -> datetime:
    utc = datetime.fromtimestamp(time.time() if now is None else now, timezone.utc)
    return utc.astimezone(timezone(timedelta(hours=-7 if _us_dst(utc) else -8)))


def pacific_day(now: float | None = None) -> str:
    return pacific_now(now).strftime("%Y-%m-%d")


def next_reset(now: float | None = None) -> float:
    """Epoch seconds of the next midnight Pacific (when Gemini daily quotas reset)."""
    p = pacific_now(now)
    midnight = (p + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    # the offset can change overnight (DST); recompute it for the target moment
    utc_guess = midnight.astimezone(timezone.utc)
    off = -7 if _us_dst(utc_guess) else -8
    return midnight.replace(tzinfo=timezone(timedelta(hours=off))).timestamp()


def resets_daily(link: JudgeLink) -> bool:
    """Links whose quota 429 means "spent until midnight Pacific": an explicit `rpd`, or Gemini."""
    return link.rpd is not None or "gemini" in link.model.lower()


class Quota:
    def __init__(self, cache: Cache, clock=time.time, sleep=time.sleep):
        self.cache, self.clock, self.sleep = cache, clock, sleep
        self._lock = threading.Lock()
        self._minute: dict[str, deque] = {}

    @staticmethod
    def tag(link: JudgeLink) -> str:
        return f"{link.provider}/{link.model}"

    # -- daily ---------------------------------------------------------------
    def used(self, link: JudgeLink) -> int:
        return int(self.cache.get("quota_used", self.tag(link), pacific_day(self.clock())) or 0)

    def out_until(self, link: JudgeLink) -> float:
        """Epoch until which the link is known to be out of quota (0 if it is not)."""
        until = float(self.cache.get("quota_out", self.tag(link)) or 0)
        if until > self.clock():
            return until
        if link.rpd is not None and self.used(link) >= link.rpd:
            return next_reset(self.clock())
        return 0.0

    def exhaust(self, link: JudgeLink) -> float:
        """The provider said the quota is spent: remember it (across restarts) and return until when."""
        now = self.clock()
        until = next_reset(now) if resets_daily(link) else now + QUOTA_DOWN_SECONDS
        self.cache.set("quota_out", until, self.tag(link))
        return until

    # -- per request -----------------------------------------------------------
    def acquire(self, link: JudgeLink) -> bool:
        """Count one request against the link. False (and nothing counted) if the daily cap is
        reached; otherwise block as needed to stay under `rpm`, then return True."""
        tag = self.tag(link)
        while True:
            with self._lock:
                if self.out_until(link):
                    return False
                now = self.clock()
                q = self._minute.setdefault(tag, deque())
                while q and q[0] <= now - 60:
                    q.popleft()
                if link.rpm is None or len(q) < link.rpm:
                    q.append(now)
                    day = pacific_day(now)
                    self.cache.set("quota_used", self.used(link) + 1, tag, day)
                    return True
                wait = q[0] + 60 - now
            self.sleep(max(0.05, wait))

    def status(self, links: list[JudgeLink]) -> list[dict]:
        out = []
        for l in links:
            until = self.out_until(l)
            out.append({"link": self.tag(l), "used_today": self.used(l), "rpd": l.rpd, "rpm": l.rpm,
                        "out_until": until or None})
        return out
