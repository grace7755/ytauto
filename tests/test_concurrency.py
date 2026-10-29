"""Parallel selection must be as correct as sequential selection, only faster.

Shots and screening calls now run on threads. What can go wrong: two shots cutting the same
footage from one video, more judge calls in flight than the providers' rate limits allow, and a
parallel round choosing a different clip than the sequential one would have.
"""
import threading
import time

import pytest

from ytauto.cache import Cache
from ytauto.config import JudgeLink, Settings
from ytauto.llm import router as R
from ytauto.llm.providers import ProviderConfig
from ytauto.models import Brief, Candidate, Decision, Shot
from ytauto.select import Selector
from ytauto.state import Project


def _selector(tmp_path, **kw):
    s = Settings(**kw)
    proj = Project(tmp_path / "p")
    return Selector(s, proj, Cache(tmp_path / "c"), None, None, {}, []), proj


def test_one_window_cannot_be_claimed_twice_at_once(tmp_path):
    sel, _ = _selector(tmp_path, max_shots_per_source=3)
    wins = []
    barrier = threading.Barrier(8)

    def grab():
        barrier.wait()
        wins.append(sel._claim("youtube:x", (10.0, 20.0)))

    ts = [threading.Thread(target=grab) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert wins.count(True) == 1, "exactly one shot may hold an overlapping window"


def test_claims_respect_the_per_source_cap_including_in_flight(tmp_path):
    sel, proj = _selector(tmp_path, max_shots_per_source=2)
    proj.log_decision(Decision(shot=0, status="accepted", provider="youtube", source_id="x", url="u",
                               src_start=0, src_end=5))
    assert sel._claim("youtube:x", (100.0, 110.0))           # 1 used + 1 in flight = cap
    assert not sel._claim("youtube:x", (200.0, 210.0)), "the cap counts windows still being worked on"
    sel._release("youtube:x", (100.0, 110.0))
    assert sel._claim("youtube:x", (200.0, 210.0))


def test_windows_already_used_in_the_ledger_are_refused(tmp_path):
    sel, proj = _selector(tmp_path)
    proj.log_decision(Decision(shot=0, status="accepted", provider="youtube", source_id="x", url="u",
                               src_start=50, src_end=55))
    assert not sel._claim("youtube:x", (48.0, 60.0))
    assert sel._claim("youtube:x", (55.0, 65.0)), "touching but not overlapping is fine"


def test_ledger_survives_concurrent_writers(tmp_path):
    proj = Project(tmp_path / "p")

    def write(i):
        for k in range(50):
            proj.log_decision(Decision(shot=i, status="rejected", provider="youtube",
                                       source_id=f"{i}-{k}", url="u", notes="x" * 200))

    ts = [threading.Thread(target=write, args=(i,)) for i in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    rows = proj.ledger()                                      # raises if any line is interleaved
    assert len(rows) == 300


# --- judge throttle and patience ---------------------------------------------
def _judge(monkeypatch, tmp_path, chain, **kw):
    monkeypatch.setattr(ProviderConfig, "load",
                        classmethod(lambda cls, p=None: cls({"zai": {"baseUrl": "https://example.invalid/z", "apiKey": "k"},
                                                             "geminiopenai": {"baseUrl": "https://example.invalid/g", "apiKey": "k"}})))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    s = Settings(judge_chain=chain, cache_dir=tmp_path, **kw)
    return R.Judge(s, Cache(tmp_path))


def test_judge_calls_in_flight_never_exceed_the_cap(monkeypatch, tmp_path):
    j = _judge(monkeypatch, tmp_path, [JudgeLink(provider="zai", model="m", vision=True)],
               judge_concurrency=2, judge_patience=0)
    live, peak, lock = [0], [0], threading.Lock()

    def slow_post(link, content, schema, timeout=None):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.05)
        with lock:
            live[0] -= 1
        return '{"ok": true}'

    monkeypatch.setattr(j, "_post", slow_post)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    ts = [threading.Thread(target=j.json, args=(f"p{i}", schema)) for i in range(10)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert peak[0] == 2


def test_patience_waits_once_for_a_recovering_link(monkeypatch, tmp_path):
    j = _judge(monkeypatch, tmp_path, [JudgeLink(provider="zai", model="m", vision=True)],
               judge_patience=30)
    slept, attempts = [], []
    monkeypatch.setattr(R.time, "sleep", lambda s: slept.append(s))

    def flaky(link, content, schema, timeout=None):
        attempts.append(1)
        if len(attempts) == 1:
            raise R.JudgeError("HTTP 500: internal")
        return '{"ok": true}'

    monkeypatch.setattr(j, "_post", flaky)
    assert j.json("p", {"type": "object", "properties": {"ok": {"type": "boolean"}}}) == {"ok": True}
    assert slept == [30] and len(attempts) == 2


def test_no_waiting_on_a_link_that_is_out_of_daily_quota(monkeypatch, tmp_path):
    """Measured 2026-10-29: "You exceeded your current quota" is a daily cap. Waiting a minute
    for it, on every call, for the rest of the day, would only slow the run down."""
    j = _judge(monkeypatch, tmp_path, [JudgeLink(provider="zai", model="m", vision=True)],
               judge_patience=30)
    slept = []
    monkeypatch.setattr(R.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(j, "_post", lambda *a, **k: (_ for _ in ()).throw(
        R.JudgeError("HTTP 429: You exceeded your current quota, please check your plan")))
    with pytest.raises(R.JudgeError):
        j.json("p", {"type": "object", "properties": {"ok": {"type": "boolean"}}})
    assert slept == []
    assert j.down_links()["zai/m"] > R.DOWN_SECONDS, "a quota trip lasts longer than a blip"


# --- parallel screening keeps the sequential decision ------------------------
def test_parallel_round_picks_the_same_clip_as_sequential(tmp_path, monkeypatch):
    """Screens finish in any order; the decision must still follow the ranked candidate order."""
    import ytauto.select as SEL

    def run(workers):
        s = Settings(screen_workers=workers, rounds_per_shot=1, screen_top_k=3, prerank="off", screen_batch=1)
        proj = Project(tmp_path / f"p{workers}")

        class YT:
            def search(self, q, n):
                return [Candidate(provider="youtube", source_id=v, url=f"https://youtu.be/{v}",
                                  height=1080, duration=600, title=v) for v in ("a", "b", "c")]

            def details(self, c):
                return c

            def prefilter(self, c):
                return None

            def download_section(self, c, a, b, d):
                return tmp_path / f"{c.source_id}.mp4"

        delays = {"a": 0.06, "b": 0.0, "c": 0.03}            # 'a' answers last, yet ranks first

        def fake_screen(gem, c, shot, fps=1.0, *, yt=None, mode="storyboard"):
            time.sleep(delays[c.source_id])
            return {"usable": True, "windows": [{"start": 10.0, "end": 20.0, "match": 90}]}

        monkeypatch.setattr(SEL, "screen", fake_screen)
        monkeypatch.setattr(SEL, "text_score", lambda c, q, e: {"a": 90, "b": 80, "c": 70}[c.source_id])
        shot = Shot(id=0, segment=0, start=0, end=5, text="t",
                    brief=Brief(visual="v", entities=["e"], youtube_queries=["q"]))
        sel = Selector(s, proj, Cache(tmp_path / "c"), YT(), object(), {}, [shot])
        picked = []
        monkeypatch.setattr(sel, "_try_file", lambda shot, c, path, hint, prior, off: (
            picked.append(c.source_id),
            (Decision(shot=0, status="accepted", provider="youtube", source_id=c.source_id, url="u"), None))[1])
        sel._youtube(shot, [])
        return picked

    assert run(1) == run(3) == ["a"]


def test_batched_waves_keep_ranked_order_and_stop_after_an_accept(tmp_path, monkeypatch):
    """Batches of 2: the first wave answers in one call; once a clip is accepted from it the
    second wave is never screened - that is where the saving on free quota comes from."""
    import ytauto.select as SEL
    from ytauto.sources.storyboard import Storyboard

    s = Settings(rounds_per_shot=1, screen_top_k=4, prerank="off", screen_batch=2)
    proj = Project(tmp_path / "p")

    class YT:
        def search(self, q, n):
            return [Candidate(provider="youtube", source_id=v, url=f"https://youtu.be/{v}",
                              height=1080, duration=600, title=v) for v in "abcd"]

        def details(self, c):
            return c

        def prefilter(self, c):
            return None

        def storyboard(self, c):
            return Storyboard(c.source_id, b"j", [0.0, 10.0, 20.0], 10.0, 30.0)

        def download_section(self, c, a, b, d):
            return tmp_path / f"{c.source_id}.mp4"

    batches = []

    def fake_batch(gem, cands, shot, sbs):
        batches.append([c.source_id for c in cands])
        return [{"usable": c.source_id == "b", "reason": "", "presenter_on_camera": False,
                 "black_and_white": False, "windows": [{"start": 8.0, "end": 20.0, "match": 90}]}
                for c in cands]

    monkeypatch.setattr(SEL, "screen_storyboards", fake_batch)
    monkeypatch.setattr(SEL, "text_score", lambda c, q, e: {"a": 90, "b": 80, "c": 70, "d": 60}[c.source_id])
    shot = Shot(id=0, segment=0, start=0, end=5, text="t",
                brief=Brief(visual="v", entities=["e"], youtube_queries=["q"]))
    sel = Selector(s, proj, Cache(tmp_path / "c"), YT(), object(), {}, [shot])
    picked = []
    monkeypatch.setattr(sel, "_try_file", lambda shot, c, path, hint, prior, off: (
        picked.append(c.source_id),
        (Decision(shot=0, status="accepted", provider="youtube", source_id=c.source_id, url="u"), None))[1])
    d = sel._youtube(shot, [])
    assert d.source_id == "b" and picked == ["b"]
    assert batches == [["a", "b"]], "one call for the first two; the second wave is never screened"
    assert [r.source_id for r in proj.ledger() if r.status == "rejected"] == ["a"]


def test_batched_screen_maps_letters_back_to_the_right_videos(tmp_path):
    from ytauto.screen import screen_storyboards
    from ytauto.sources.storyboard import Storyboard

    class J:
        cache = Cache(tmp_path)

        def json_images(self, prompt, images, schema, **kw):
            assert len(images) == 2 and "A to B" in prompt
            return {"videos": [{"video": "B", "usable": False, "tiles": [], "match": 0,
                                "presenter_on_camera": False, "black_and_white": False},
                               {"video": "A", "usable": True, "tiles": [2], "match": 88,
                                "presenter_on_camera": False, "black_and_white": False}]}

    shot = Shot(id=0, segment=0, start=0, end=5, text="t", brief=Brief(subject="s", visual="v", entities=["e"]))
    cands = [Candidate(provider="youtube", source_id=v, url="u") for v in "xy"]
    sbs = [Storyboard(v, b"j", [0.0, 10.0, 20.0], 10.0, 30.0) for v in "xy"]
    r = screen_storyboards(J(), cands, shot, sbs)
    assert r[0]["usable"] and r[0]["windows"][0]["start"] == 18.0
    assert not r[1]["usable"]

    class Never(J):
        def json_images(self, *a, **kw):
            raise AssertionError("both verdicts are cached per video")

    again = screen_storyboards(Never(), cands, shot, sbs)
    assert again[0]["usable"] and not again[1]["usable"]


def test_overlay_suspects_are_tried_after_the_clean_candidates(tmp_path, monkeypatch):
    """A usable video whose every proposed thumbnail shows text or a logo is downloaded only once
    the clean ones failed - deferred, never rejected (it still wins when it is all there is)."""
    import ytauto.select as SEL
    from ytauto.sources.storyboard import Storyboard

    s = Settings(rounds_per_shot=1, screen_top_k=3, prerank="off", screen_batch=1)
    proj = Project(tmp_path / "p")

    class YT:
        def search(self, q, n):
            return [Candidate(provider="youtube", source_id=v, url=f"https://youtu.be/{v}",
                              height=1080, duration=600, title=v) for v in "abc"]

        def details(self, c):
            return c

        def prefilter(self, c):
            return None

        def storyboard(self, c):
            return Storyboard(c.source_id, b"j", [0.0, 10.0, 20.0], 10.0, 30.0)

        def download_section(self, c, a, b, d):
            return tmp_path / f"{c.source_id}.mp4"

    def fake_screen(gem, c, shot, **kw):
        return {"usable": True, "reason": "", "presenter_on_camera": False, "black_and_white": False,
                "overlay_suspect": c.source_id == "a",
                "windows": [{"start": 8.0, "end": 20.0, "match": 90, "overlay": c.source_id == "a"}]}

    monkeypatch.setattr(SEL, "screen", fake_screen)
    monkeypatch.setattr(SEL, "text_score", lambda c, q, e: {"a": 90, "b": 80, "c": 70}[c.source_id])
    shot = Shot(id=0, segment=0, start=0, end=5, text="t",
                brief=Brief(visual="v", entities=["e"], youtube_queries=["q"]))
    sel = Selector(s, proj, Cache(tmp_path / "c"), YT(), object(), {}, [shot])
    picked = []

    def try_file(shot, c, path, hint, prior, off):
        picked.append(c.source_id)
        ok = c.source_id == "a"                   # only the suspect turns out fine
        return Decision(shot=0, status="accepted" if ok else "rejected", provider="youtube",
                        source_id=c.source_id, url="u"), None

    monkeypatch.setattr(sel, "_try_file", try_file)
    d = sel._youtube(shot, [])
    assert picked == ["b", "c", "a"], "the top-ranked suspect waits for the clean ones"
    assert d.source_id == "a"


def _pipeline_with(tmp_path, monkeypatch, sel_cls, n_shots=6, **kw):
    import ytauto.pipeline as PL
    s = Settings(shot_workers=2, **kw)
    pipe = PL.Pipeline.__new__(PL.Pipeline)
    pipe.s, pipe.proj, pipe.cache = s, Project(tmp_path / "p"), Cache(tmp_path / "c")
    pipe.yt, pipe.gem, pipe.providers = None, object(), {}
    pipe.shots = [Shot(id=i, segment=i, start=i * 5.0, end=i * 5.0 + 5, text="t",
                       brief=Brief(visual="v", entities=["e"], youtube_queries=["q"])) for i in range(n_shots)]
    monkeypatch.setattr(PL, "Selector", sel_cls)
    return pipe


def test_prefetch_prepares_upcoming_shots_while_workers_are_busy(tmp_path, monkeypatch):
    """Judge-free work for shot i+1.. runs while shot i waits on the judge - never more than
    `prefetch_ahead` shots in front of the workers, and a failing warm-up changes nothing."""
    warmed, selected, gate = [], [], threading.Event()

    class Sel:
        def __init__(self, *a, **k):
            pass

        def warm(self, shot):
            warmed.append(shot.id)
            if shot.id == 3:
                raise RuntimeError("network blip")

        def select(self, shot):
            selected.append(shot.id)
            if shot.id in (0, 1):
                gate.wait(5)                     # the first two shots sit on the (slow) judge
            return Decision(shot=shot.id, status="accepted", provider="youtube", source_id="x", url="u")

    pipe = _pipeline_with(tmp_path, monkeypatch, Sel, prefetch_ahead=2)
    t = threading.Thread(target=pipe.select_all)
    t.start()
    deadline = time.time() + 5
    while time.time() < deadline and len(warmed) < 2:
        time.sleep(0.02)
    assert warmed == [2, 3][: len(warmed)] and len(warmed) == 2, "warms shots 2-3 while 0-1 are busy"
    assert 4 not in warmed, "stays prefetch_ahead shots in front of the workers"
    gate.set()
    t.join(10)
    assert not t.is_alive()
    assert sorted(selected) == list(range(6)), "a failed warm-up never fails or skips a shot"


def test_prefetch_off_and_offline_do_not_start_a_thread(tmp_path, monkeypatch):
    class Sel:
        def __init__(self, *a, **k):
            pass

        def warm(self, shot):
            pytest.fail("prefetch must be off")

        def select(self, shot):
            return Decision(shot=shot.id, status="accepted", provider="youtube", source_id="x", url="u")

    _pipeline_with(tmp_path, monkeypatch, Sel, prefetch_ahead=0).select_all()
    p = _pipeline_with(tmp_path, monkeypatch, Sel, prefetch_ahead=3)
    p.gem = None                                   # offline mode: no judge, nothing to wait on
    p.select_all()
