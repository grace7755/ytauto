"""Judge quota: free links are capped per day, so we count requests instead of discovering the
cap one 429 at a time, and a run that spends it pauses cleanly instead of failing shot by shot."""
from datetime import datetime, timezone

import pytest

from ytauto.cache import Cache
from ytauto.config import JudgeLink, Settings
from ytauto.llm import quota as Q
from ytauto.llm import router as R
from ytauto.llm.providers import ProviderConfig
from ytauto.models import Brief, Shot

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}}


def _ts(*a):
    return datetime(*a, tzinfo=timezone.utc).timestamp()


def test_pacific_day_follows_us_daylight_saving():
    assert Q.pacific_day(_ts(2026, 7, 1, 6, 59)) == "2026-06-30"        # PDT, UTC-7
    assert Q.pacific_day(_ts(2026, 7, 1, 7, 0)) == "2026-07-01"
    assert Q.pacific_day(_ts(2026, 12, 1, 7, 59)) == "2026-11-30"       # PST, UTC-8
    assert Q.pacific_day(_ts(2026, 12, 1, 8, 0)) == "2026-12-01"


def test_next_reset_is_the_coming_pacific_midnight():
    assert Q.next_reset(_ts(2026, 7, 1, 12, 0)) == _ts(2026, 7, 2, 7, 0)
    # across the November switch the reset moves from UTC-7 to UTC-8
    assert Q.next_reset(_ts(2026, 11, 1, 6, 0)) == _ts(2026, 11, 1, 7, 0)
    assert Q.next_reset(_ts(2026, 11, 1, 9, 0)) == _ts(2026, 11, 2, 8, 0)


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_rpm_bucket_never_exceeds_the_cap(tmp_path):
    clock = Clock(_ts(2026, 7, 1, 12, 0))
    q = Q.Quota(Cache(tmp_path), clock=clock, sleep=clock.sleep)
    link = JudgeLink(provider="p", model="m", rpm=3)
    sent = []
    for _ in range(7):
        assert q.acquire(link)
        sent.append(clock.t)
    for i in range(len(sent)):
        window = [t for t in sent if sent[i] <= t < sent[i] + 60]
        assert len(window) <= 3


def test_daily_counter_persists_and_resets_at_pacific_midnight(tmp_path):
    clock = Clock(_ts(2026, 7, 1, 12, 0))
    link = JudgeLink(provider="p", model="m", rpd=2)
    q = Q.Quota(Cache(tmp_path), clock=clock, sleep=clock.sleep)
    assert q.acquire(link) and q.acquire(link)
    again = Q.Quota(Cache(tmp_path), clock=clock, sleep=clock.sleep)       # a restarted run
    assert not again.acquire(link), "the cap holds across restarts"
    assert again.out_until(link) == _ts(2026, 7, 2, 7, 0)
    clock.t = _ts(2026, 7, 2, 7, 1)
    assert again.acquire(link), "a new Pacific day starts a new count"


def test_quota_429_on_gemini_lasts_until_the_reset(tmp_path):
    clock = Clock(_ts(2026, 7, 1, 12, 0))
    q = Q.Quota(Cache(tmp_path), clock=clock, sleep=clock.sleep)
    assert q.exhaust(JudgeLink(provider="p", model="models/gemini-2.5-flash-lite")) == _ts(2026, 7, 2, 7, 0)
    assert q.exhaust(JudgeLink(provider="p", model="glm")) == clock.t + Q.QUOTA_DOWN_SECONDS


# --- the judge -----------------------------------------------------------------
def _judge(monkeypatch, tmp_path, chain, **kw):
    monkeypatch.setattr(ProviderConfig, "load",
                        classmethod(lambda cls, p=None: cls({"zai": {"baseUrl": "https://example.invalid/z", "apiKey": "k"},
                                                             "g": {"baseUrl": "https://example.invalid/g", "apiKey": "k"}})))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    s = Settings(judge_chain=chain, cache_dir=tmp_path, judge_patience=0, **kw)
    return R.Judge(s, Cache(tmp_path))


def test_a_spent_link_is_skipped_without_a_request(monkeypatch, tmp_path):
    a = JudgeLink(provider="g", model="gemini-a", vision=True, rpd=1)
    b = JudgeLink(provider="zai", model="b", vision=True)
    j = _judge(monkeypatch, tmp_path, [a, b])
    asked = []
    monkeypatch.setattr(j, "_post", lambda link, *a, **k: asked.append(link.model) or '{"ok": true}')
    for _ in range(3):
        j.json("p", SCHEMA)
    assert asked == ["gemini-a", "b", "b"]


def test_every_link_spent_raises_quota_exhausted_with_the_reset_time(monkeypatch, tmp_path):
    j = _judge(monkeypatch, tmp_path, [JudgeLink(provider="g", model="gemini-a", vision=True)])
    monkeypatch.setattr(j, "_post", lambda *a, **k: (_ for _ in ()).throw(
        R.JudgeError("HTTP 429: You exceeded your current quota, please check your plan")))
    with pytest.raises(R.QuotaExhausted) as e:
        j.json("p", SCHEMA)
    assert e.value.until == Q.next_reset()
    # and the next call does not even try: the provider would only say no again
    monkeypatch.setattr(j, "_post", lambda *a, **k: pytest.fail("called a spent link"))
    with pytest.raises(R.QuotaExhausted):
        j.json("q", SCHEMA)


def test_retry_after_shortens_the_trip(monkeypatch, tmp_path):
    j = _judge(monkeypatch, tmp_path, [JudgeLink(provider="zai", model="m", vision=True)])
    j._trip(j.chain[0], "HTTP 429: slow down [retry-after 7s]")
    assert 0 < j.down_links()["zai/m"] <= 7


def test_quota_pause_stops_new_shots_and_records_nothing(tmp_path, monkeypatch):
    from ytauto.config import load_settings
    from ytauto.pipeline import Pipeline
    from ytauto.state import Project

    s = load_settings(None)
    s.shot_workers = 1
    brief = Brief(visual="v", entities=["e"], youtube_queries=["q"])
    p = Pipeline.__new__(Pipeline)
    p.s, p.vo = s, None
    p.shots = [Shot(id=i, segment=i, start=5 * i, end=5 * i + 5, text="t", brief=brief) for i in range(4)]
    p.proj = Project(tmp_path / "proj")
    p.cache, p.gem, p.yt, p.providers = None, None, None, {}
    started = []

    class Sel:
        def __init__(self, *a, **kw):
            pass

        def select(self, shot):
            started.append(shot.id)
            if shot.id == 1:
                raise R.QuotaExhausted("spent", until=123.0)
            return object()

    monkeypatch.setattr("ytauto.pipeline.Selector", Sel)
    blockers = Pipeline.select_all(p)
    assert started == [0, 1], "no shot starts once the quota is spent"
    assert p.quota_until == 123.0
    assert len(blockers) == 3 and all("quota" in b for b in blockers)
    assert p.proj.ledger() == []


def test_a_per_minute_429_is_not_a_daily_cap(monkeypatch, tmp_path):
    """Gemini's per-minute 429 also says "exceeded your current quota"; it must not take the
    link out until midnight Pacific. Message shapes from Google's real replies."""
    j = _judge(monkeypatch, tmp_path, [JudgeLink(provider="g", model="gemini-a", vision=True)])
    link = j.chain[0]
    j._trip(link, "429 RESOURCE_EXHAUSTED. You exceeded your current quota. Quota exceeded for metric: "
                  "generate_content_free_tier_requests, quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
                  " 'retryDelay': '23s'")
    assert not j.quota.out_until(link)
    assert 0 < j.down_links()["g/gemini-a"] <= 23
    j._trip(link, "429 RESOURCE_EXHAUSTED. You exceeded your current quota. Generate content requests per "
                  "day limit exceeded on Free Tier. Please retry in 15h55m1.6s. 'retryDelay': '57301s'")
    assert j.quota.out_until(link) == Q.next_reset()


def test_direct_links_call_their_own_model_with_no_hidden_fallback(monkeypatch, tmp_path):
    """Free quota is per model; the chain counts per link, so `_direct` must not hop models."""
    monkeypatch.setattr(ProviderConfig, "load", classmethod(lambda cls, p=None: cls({})))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    chain = [JudgeLink(provider="_direct", model="lite-a", vision=True),
             JudgeLink(provider="_direct", model="lite-b", vision=True)]
    j = R.Judge(Settings(judge_chain=chain, cache_dir=tmp_path, judge_patience=0), Cache(tmp_path))
    asked = []

    class G:
        def json_images(self, prompt, images, schema, model=None):
            asked.append(model)
            if model == "lite-a":
                raise R.GeminiError("503 UNAVAILABLE high demand")
            return {"ai_generated": False}

    monkeypatch.setattr(j, "_direct_client", lambda: G())
    assert j.json_images("p", [b"x"], SCHEMA) == {"ai_generated": False}
    assert asked == ["lite-a", "lite-b"]


def test_the_gemini_client_in_chain_mode_never_switches_model(monkeypatch, tmp_path):
    from ytauto.llm import gemini as GM
    g = GM.Gemini.__new__(GM.Gemini)
    g.fallback, g._down = False, {}
    tried = []

    def one(m, parts, schema, retries):
        tried.append(m)
        raise GM.GeminiError("down")

    g._generate_one = one
    with pytest.raises(GM.GeminiError):
        g._generate("lite-a", [], None)
    assert tried == ["lite-a"]


def test_a_demand_spike_trips_a_link_briefly_not_for_five_minutes(monkeypatch, tmp_path):
    """Google's 503 "high demand" clears in seconds (dozens per run were measured); the circuit
    breaker should not take the model away for the usual 5 minutes."""
    j = _judge(monkeypatch, tmp_path, [JudgeLink(provider="g", model="gemini-a", vision=True)])
    j._trip(j.chain[0], "503 UNAVAILABLE. This model is currently experiencing high demand.")
    assert 0 < j.down_links()["g/gemini-a"] <= R.OVERLOAD_SECONDS
    assert not j.quota.out_until(j.chain[0])
    j._trip(j.chain[0], "HTTP 500: something else broke")
    assert j.down_links()["g/gemini-a"] > R.OVERLOAD_SECONDS


def test_chain_mode_gemini_does_not_sleep_on_a_demand_spike(monkeypatch):
    from ytauto.llm import gemini as GM
    g = GM.Gemini.__new__(GM.Gemini)
    g.fallback, g.calls, g.tokens = False, 0, 0
    calls = []

    class Models:
        def generate_content(self, **kw):
            calls.append(1)
            raise RuntimeError("503 UNAVAILABLE. high demand")

    g.client = type("C", (), {"models": Models()})()
    g._types = type("T", (), {"GenerateContentConfig": lambda **k: k, "Content": lambda **k: k})
    monkeypatch.setattr(GM.time, "sleep", lambda s: pytest.fail("slept on a 503 in chain mode"))
    with pytest.raises(GM.GeminiError):
        g._generate_one("m", [], None, 2)
    assert len(calls) == 1, "straight on to the next chain link, no second attempt"
