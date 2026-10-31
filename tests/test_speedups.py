"""Speed-ups that must not change what gets selected: hedged judge calls, a deferred AI check,
and section downloads fetched ahead of the decision loop."""
import threading
import time
from pathlib import Path

import pytest

import ytauto.select as SEL
import ytauto.sources.youtube as Y
from ytauto.cache import Cache
from ytauto.config import JudgeLink, Settings
from ytauto.inspect.verify import verify_clip
from ytauto.llm import router as R
from ytauto.llm.providers import ProviderConfig
from ytauto.models import Brief, Candidate, Decision, ClipScore, Shot
from ytauto.select import AI_DEFERRED, Selector
from ytauto.state import Project

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}}


def _judge(monkeypatch, tmp_path, **kw):
    monkeypatch.setattr(ProviderConfig, "load", classmethod(lambda cls, p=None: cls({
        "zai": {"baseUrl": "https://example.invalid/z", "apiKey": "k"},
        "geminiopenai": {"baseUrl": "https://example.invalid/g", "apiKey": "k"}})))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    chain = [JudgeLink(provider="zai", model="slow", vision=True),
             JudgeLink(provider="geminiopenai", model="fast", vision=True)]
    return R.Judge(Settings(judge_chain=chain, cache_dir=tmp_path, judge_patience=0, **kw), Cache(tmp_path))


def _post_by_model(calls, delays, fail=()):
    def post(link, content, schema, timeout=None):
        calls.append(link.model)
        time.sleep(delays.get(link.model, 0))
        if link.model in fail:
            raise R.JudgeError("HTTP 500: boom")
        return '{"ok": true}' if link.model == "fast" else '{"ok": false}'
    return post


# --- hedged judge calls ------------------------------------------------------
def test_a_slow_link_is_hedged_and_the_first_answer_wins(monkeypatch, tmp_path):
    j = _judge(monkeypatch, tmp_path, judge_hedge_after=0.1)
    calls = []
    monkeypatch.setattr(j, "_post", _post_by_model(calls, {"slow": 1.5}))
    t0 = time.time()
    assert j.json("p", SCHEMA) == {"ok": True}, "the fast link's answer is used"
    assert time.time() - t0 < 1.0, "did not wait for the slow link"
    assert calls == ["slow", "fast"]


def test_without_hedging_the_first_link_is_awaited(monkeypatch, tmp_path):
    j = _judge(monkeypatch, tmp_path, judge_hedge_after=0)
    calls = []
    monkeypatch.setattr(j, "_post", _post_by_model(calls, {"slow": 0.3}))
    assert j.json("p", SCHEMA) == {"ok": False}
    assert calls == ["slow"]


def test_a_link_that_answers_in_time_is_never_hedged(monkeypatch, tmp_path):
    j = _judge(monkeypatch, tmp_path, judge_hedge_after=5)
    calls = []
    monkeypatch.setattr(j, "_post", _post_by_model(calls, {}))
    assert j.json("p", SCHEMA) == {"ok": False}
    assert calls == ["slow"], "no second request, so no extra quota is spent"


def test_hedged_walk_falls_through_a_failing_link(monkeypatch, tmp_path):
    j = _judge(monkeypatch, tmp_path, judge_hedge_after=5)
    calls = []
    monkeypatch.setattr(j, "_post", _post_by_model(calls, {}, fail={"slow"}))
    assert j.json("p", SCHEMA) == {"ok": True}
    assert calls == ["slow", "fast"]
    assert "zai/slow" in j.down_links(), "the failure still trips the circuit breaker"


def test_all_links_failing_still_raises_when_hedging(monkeypatch, tmp_path):
    j = _judge(monkeypatch, tmp_path, judge_hedge_after=0.05)
    monkeypatch.setattr(j, "_post", _post_by_model([], {"slow": 0.2}, fail={"slow", "fast"}))
    with pytest.raises(R.JudgeError):
        j.json("p", SCHEMA)


def test_a_link_that_is_always_slow_is_hedged_later(monkeypatch, tmp_path):
    j = _judge(monkeypatch, tmp_path, judge_hedge_after=0.05)
    j._latency["zai/slow"] = 1.0                 # usually ~1 s: 0.3 s is normal for it, not a spike
    calls = []
    monkeypatch.setattr(j, "_post", _post_by_model(calls, {"slow": 0.3}))
    assert j.json("p", SCHEMA) == {"ok": False}
    assert calls == ["slow"]
    assert j._hedge_delay(j.chain[0]) > 0.05


def test_no_hedge_when_no_in_flight_slot_is_free(monkeypatch, tmp_path):
    j = _judge(monkeypatch, tmp_path, judge_hedge_after=0.05, judge_concurrency=1)
    calls = []
    monkeypatch.setattr(j, "_post", _post_by_model(calls, {"slow": 0.4}))
    assert j.json("p", SCHEMA) == {"ok": False}
    assert calls == ["slow"], "a hedge must never starve other calls of the in-flight cap"


# --- deferred AI check -------------------------------------------------------
class _VJ:
    def __init__(self):
        self.ai_checks = 0

    def json_video_file(self, *a, **kw):
        return {"relevance": 90, "quality": 80}

    def confirm_ai_generated(self, *a, **kw):
        self.ai_checks += 1
        return False


SHOT = Shot(id=0, segment=0, start=0, end=5, text="t", brief=Brief(visual="v", entities=["e"]))


def test_ai_check_is_skipped_for_a_clip_rejected_anyway_and_marked(tmp_path):
    j = _VJ()
    v = verify_clip(j, tmp_path / "x.mp4", SHOT, "id", 60, rejected_anyway=lambda v: True)
    assert j.ai_checks == 0 and v["ai_check_deferred"] is True


def test_ai_check_still_runs_when_the_clip_could_pass(tmp_path):
    j = _VJ()
    v = verify_clip(j, tmp_path / "x.mp4", SHOT, "id", 60, rejected_anyway=lambda v: False)
    assert j.ai_checks == 1 and "ai_check_deferred" not in v


def _selector(tmp_path, gem, **kw):
    return Selector(Settings(**kw), Project(tmp_path / "p"), Cache(tmp_path / "c"), None, gem, {}, [SHOT])


def _weak(tmp_path, name, total, deferred):
    clip = tmp_path / name
    clip.write_bytes(b"x" * 20)
    return Decision(shot=0, status="rejected", provider="youtube", source_id=name, url="u", src_start=7.0,
                    src_end=12.0, clip_path=str(clip), score=ClipScore(total=total),
                    notes="n" + (AI_DEFERRED if deferred else ""))


def test_weak_promotion_runs_the_deferred_ai_check_and_drops_ai_footage(tmp_path):
    class G:
        def confirm_ai_generated(self, clip, cache_key=None):
            return Path(clip).name == "ai.mp4"

    sel = _selector(tmp_path, G())
    ai, real = _weak(tmp_path, "ai.mp4", 74, True), _weak(tmp_path, "real.mp4", 70, True)
    sel._youtube = lambda shot, best: (best.extend([ai, real]), None)[1]
    sel._stock = lambda shot, best: None
    d = sel.select(SHOT)
    assert d.status == "weak" and d.source_id == "real.mp4", "the better-scored AI clip is passed over"
    assert not Path(ai.clip_path).exists()
    assert [r.status for r in sel.proj.ledger() if r.source_id == "ai.mp4"] == ["rejected"]


def test_clips_without_the_marker_are_not_rechecked(tmp_path):
    class G:
        def confirm_ai_generated(self, *a, **k):
            raise AssertionError("already checked once")

    sel = _selector(tmp_path, G())
    assert sel._ai_flagged(SHOT, _weak(tmp_path, "a.mp4", 70, False)) is False


# --- download ahead ----------------------------------------------------------
def test_concurrent_requests_for_one_section_download_it_once(monkeypatch, tmp_path):
    yt = Y.YouTube.__new__(Y.YouTube)
    yt.s = Settings()
    runs = []

    class FakeDL:
        def __init__(self, opts):
            self.out = Path(opts["outtmpl"].replace(".%(ext)s", ".mp4"))

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def download(self, urls):
            runs.append(1)
            time.sleep(0.2)
            self.out.write_bytes(b"x" * 20_000)

    monkeypatch.setattr(Y.yt_dlp, "YoutubeDL", FakeDL)
    c = Candidate(provider="youtube", source_id="vid", url="https://youtu.be/vid", duration=60)
    got = []
    ts = [threading.Thread(target=lambda: got.append(yt.download_section(c, 10, 20, tmp_path))) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(runs) == 1 and len({str(g) for g in got}) == 1 and all(got)


def _ahead_selector(tmp_path, monkeypatch, **kw):
    fetched, ev = [], threading.Event()

    class YT:
        def download_section(self, c, a, b, d):
            fetched.append((c.source_id, a, b))
            ev.set()

    sel = Selector(Settings(**kw), Project(tmp_path / "p"), Cache(tmp_path / "c"), YT(), object(), {}, [SHOT])
    return sel, fetched, ev


def _res(**kw):
    base = {"usable": True, "windows": [{"start": 8.0, "end": 20.0, "match": 90}]}
    base.update(kw)
    return base


def test_only_later_usable_candidates_are_fetched_ahead(tmp_path, monkeypatch):
    sel, fetched, ev = _ahead_selector(tmp_path, monkeypatch)
    wave = [Candidate(provider="youtube", source_id=v, url="u") for v in "abcd"]
    results = {"a": _res(), "b": _res(), "c": _res(usable=False), "d": _res(overlay_suspect=True)}
    sel._download_ahead(SHOT, wave, results)
    assert ev.wait(2)
    sel._dl_pool.shutdown(wait=True)
    assert fetched == [("b", 8.0, 20.0)], "not the first (the loop fetches it), not unusable, not overlay-suspect"


def test_download_ahead_can_be_turned_off(tmp_path, monkeypatch):
    sel, fetched, _ = _ahead_selector(tmp_path, monkeypatch, download_ahead=False)
    wave = [Candidate(provider="youtube", source_id=v, url="u") for v in "ab"]
    sel._download_ahead(SHOT, wave, {"a": _res(), "b": _res()})
    assert sel._dl_pool is None and fetched == []


def test_a_window_already_tried_is_not_fetched_ahead(tmp_path, monkeypatch):
    sel, fetched, _ = _ahead_selector(tmp_path, monkeypatch)
    sel.proj.log_decision(Decision(shot=0, status="rejected", provider="youtube", source_id="b", url="u",
                                   src_start=8.0, src_end=20.0))
    wave = [Candidate(provider="youtube", source_id=v, url="u") for v in "ab"]
    sel._download_ahead(SHOT, wave, {"a": _res(), "b": _res()})
    assert sel._dl_pool is None and fetched == []


# --- verify sends the native Gemini link a frame grid -------------------------
def test_direct_link_gets_the_frame_grid_not_inline_video(monkeypatch, tmp_path):
    monkeypatch.setattr(ProviderConfig, "load", classmethod(lambda cls, p=None: cls({})))
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(R, "extract_grid_data_url", lambda *a, **k: "data:image/jpeg;base64,AAAA")

    seen = {}

    class G:
        def json_images(self, prompt, images, schema, model=None):
            seen["images"], seen["prompt"] = images, prompt
            return {"ok": True}

        def json_video_file(self, *a, **k):
            raise AssertionError("inline video must not be sent")

    chain = [JudgeLink(provider="_direct", model="m", vision=True)]
    j = R.Judge(Settings(judge_chain=chain, cache_dir=tmp_path), Cache(tmp_path))
    monkeypatch.setattr(j, "_direct_client", lambda: G())
    assert j.json_video_file(tmp_path / "c.mp4", "PROMPT", SCHEMA) == {"ok": True}
    assert seen["images"] == [b"\x00\x00\x00"] and "PROMPT" in seen["prompt"] and "frames" in seen["prompt"]


def test_inline_video_can_be_switched_back_on(monkeypatch, tmp_path):
    monkeypatch.setattr(ProviderConfig, "load", classmethod(lambda cls, p=None: cls({})))
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(R, "extract_grid_data_url", lambda *a, **k: "data:image/jpeg;base64,AAAA")

    class G:
        def json_video_file(self, *a, **k):
            return {"video": True}

    chain = [JudgeLink(provider="_direct", model="m", vision=True)]
    j = R.Judge(Settings(judge_chain=chain, cache_dir=tmp_path, direct_verify_video=True), Cache(tmp_path))
    monkeypatch.setattr(j, "_direct_client", lambda: G())
    assert j.json_video_file(tmp_path / "c.mp4", "P", SCHEMA) == {"video": True}


# --- channel reputation --------------------------------------------------------
def _cand(v, ch="chan"):
    return Candidate(provider="youtube", source_id=v, url="u", channel=ch)


def test_a_channel_with_two_watermarked_videos_and_no_accept_is_skipped(tmp_path):
    sel = _selector(tmp_path, None)
    assert not sel.bad_channel(_cand("a"))
    sel._decide(SHOT, _cand("a"), "rejected", rejects=["watermark / channel logo (AI)"])
    assert not sel.bad_channel(_cand("b")), "one bad video is not enough"
    sel._decide(SHOT, _cand("b"), "rejected", rejects=["burnt-in text / branding overlay (local: onride)"])
    assert sel.bad_channel(_cand("c")) and sel.channel_bias(_cand("c")) == -2


def test_the_same_bad_video_counts_once_and_an_accept_protects_the_channel(tmp_path):
    sel = _selector(tmp_path, None)
    for _ in range(3):
        sel._decide(SHOT, _cand("a"), "rejected", rejects=["watermark / channel logo (AI)"])
    assert not sel.bad_channel(_cand("x"))
    sel._decide(SHOT, _cand("ok"), "accepted")
    sel._decide(SHOT, _cand("b"), "rejected", rejects=["watermark / channel logo (AI)"])
    assert not sel.bad_channel(_cand("x")), "a channel with an accepted clip is never skipped"


def test_soft_rejects_do_not_hurt_a_channel(tmp_path):
    sel = _selector(tmp_path, None)
    for v in "abc":
        sel._decide(SHOT, _cand(v), "rejected", rejects=["relevance 30 < 60", "score 50 < 70"])
    assert sel.channel_bias(_cand("z")) == 0


def test_channel_reputation_ignores_stock_and_unnamed_channels(tmp_path):
    sel = _selector(tmp_path, None)
    sel._decide(SHOT, _cand("a", ch=""), "rejected", rejects=["watermark / channel logo (AI)"])
    sel._decide(SHOT, Candidate(provider="pexels", source_id="1", url="u", channel="chan"), "accepted")
    assert sel.channel_bias(_cand("a", ch="")) == 0 and sel.channel_bias(_cand("q")) == 0


def test_a_hedge_goes_to_the_fastest_known_link_not_just_the_next(monkeypatch, tmp_path):
    j = _judge(monkeypatch, tmp_path)
    a, b, c = (JudgeLink(provider="zai", model=m, vision=True) for m in "abc")
    j._latency.update({"zai/b": 10.0, "zai/c": 2.0})
    q = [b, c, a]
    j._fastest_first(q)
    assert q[0] is c, "2 s beats 10 s by far"
    j._latency["zai/c"] = 8.0
    q = [b, c, a]
    j._fastest_first(q)
    assert q[0] is b, "not clearly faster: keep chain order"
    q = [a, b]
    j._fastest_first(q)
    assert q[0] is a or q[0] is b
