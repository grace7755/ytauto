"""Second speed round: low-match deferral, sibling shots from one download, remembered dead videos,
direct section downloads, judge lanes, background Remotion export, and the footage library."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import ytauto.select as SEL
import ytauto.sources.youtube as Y
from ytauto.cache import Cache
from ytauto.config import JudgeLink, Settings
from ytauto.llm import router as R
from ytauto.llm.providers import ProviderConfig
from ytauto.models import Brief, Candidate, ClipScore, Decision, Shot
from ytauto.select import Selector
from ytauto.state import Project

HAS_FFMPEG = bool(shutil.which("ffmpeg"))


def _sel(tmp_path, shots, **kw):
    s = Settings(cache_dir=tmp_path / "cache", projects_dir=tmp_path / "p", library=False, **kw)
    proj = Project(tmp_path / "p" / "demo")
    return Selector(s, proj, Cache(tmp_path / "cache"), None, None, {}, shots)


def _shot(i, seg, variant=0, start=None):
    st = i * 4.0 if start is None else start
    return Shot(id=i, segment=seg, start=st, end=st + 4.0, text=f"shot {i}", variant=variant,
                brief=Brief(subject="reef", visual="reef", setting="Philippines"))


# --- low-match deferral -------------------------------------------------------------
def test_candidates_the_screen_rated_low_are_tried_last(tmp_path):
    sel = _sel(tmp_path, [_shot(0, 0)], screen_min_match=70)
    good = {"usable": True, "windows": [{"start": 1, "end": 6, "match": 85}]}
    low = {"usable": True, "windows": [{"start": 1, "end": 6, "match": 55}]}
    cands = [Candidate(provider="youtube", source_id=x, url=x) for x in ("low", "good")]
    sel._screened = lambda shot, ready: iter([(cands[0], low), (cands[1], good)])
    order = [c.source_id for c, _ in sel._screen_waves(_shot(0, 0), cands)]
    assert order == ["good", "low"], "deferred, not dropped"


def test_low_match_floor_zero_keeps_the_ranked_order(tmp_path):
    sel = _sel(tmp_path, [_shot(0, 0)], screen_min_match=0)
    low = {"usable": True, "windows": [{"start": 1, "end": 6, "match": 10}]}
    assert not sel._low_match(low)


# --- siblings ------------------------------------------------------------------------
def test_the_first_shot_of_a_sentence_downloads_extra_for_its_sibling(tmp_path):
    shots = [_shot(0, 0), _shot(1, 0, 1), _shot(2, 1)]
    sel = _sel(tmp_path, shots, sibling_extend=9.0)
    assert sel._extend(shots[0]) == 9.0
    assert sel._extend(shots[1]) == 0.0, "last shot of its sentence: nothing to share"
    assert _sel(tmp_path, shots, sibling_extend=0)._extend(shots[0]) == 0.0


def test_section_file_lookup_covers_the_used_window(tmp_path):
    shots = [_shot(0, 0), _shot(1, 0, 1)]
    sel = _sel(tmp_path, shots)
    d = sel.proj.path("sources", "youtube")
    for name in ("vid_000985_001100.mp4", "vid_000985_001190.mp4", "vid_000100_000200.mp4", "other_000985_001190.mp4"):
        (d / name).write_bytes(b"x")
    prev = Decision(shot=0, status="accepted", provider="youtube", source_id="vid", url="u", src_start=100.0, src_end=104.4)
    path, start = sel._section_with(prev)
    assert path.name == "vid_000985_001190.mp4" and start == pytest.approx(98.5), "the longest covering section"


def test_a_sibling_cuts_a_different_scene_of_the_same_download(tmp_path, monkeypatch):
    shots = [_shot(0, 0), _shot(1, 0, 1)]
    sel = _sel(tmp_path, shots, sibling_extend=9.0)
    src = sel.proj.path("sources", "youtube") / "vid_000985_001190.mp4"
    src.write_bytes(b"x")
    prev = Decision(shot=0, status="accepted", provider="youtube", source_id="vid", url="u",
                    src_start=100.0, src_end=104.4, clip_path=str(src))
    sel.proj.log_decision(prev)
    import ytauto.inspect.scenes as SC
    monkeypatch.setattr(SC, "scene_spans", lambda p: [(0.0, 6.5), (6.5, 20.4)])   # file time
    monkeypatch.setattr(SEL, "probe", lambda p: type("I", (), {"duration": 20.4, "width": 1920, "height": 1080})())
    seen = {}

    def fake_try(shot, c, path, hint, prior, offset):
        seen.update(hint=hint, offset=offset, src=c.source_id)
        return Decision(shot=shot.id, status="accepted", provider="youtube", source_id=c.source_id, url=c.url,
                        src_start=hint[0] + offset, src_end=hint[0] + offset + 4.0), None

    monkeypatch.setattr(sel, "_try_file", fake_try)
    sel.yt = type("Y", (), {"details": lambda self, c: c})()
    d = sel._sibling(shots[1])
    assert d is not None and seen["src"] == "vid"
    assert seen["hint"] == (6.5, 20.4), "the scene after the one shot 0 used (1.5-5.9 in the file)"
    assert seen["offset"] == pytest.approx(98.5)


def test_no_sibling_attempt_without_an_accepted_youtube_predecessor(tmp_path):
    shots = [_shot(0, 0), _shot(1, 0, 1)]
    sel = _sel(tmp_path, shots)
    assert sel._sibling(shots[1]) is None
    assert sel._sibling(shots[0]) is None, "a first shot has no sibling"


# --- dead videos -----------------------------------------------------------------------
def test_gone_errors_are_told_apart_from_bot_checks():
    assert Y.is_gone("ERROR: [youtube] abc: This video is not available")
    assert Y.is_gone("Private video. Sign in if you've been granted access")
    assert not Y.is_gone("Sign in to confirm you're not a bot. This video is not available")
    assert not Y.is_gone("HTTP Error 503")


def test_a_gone_video_is_remembered_and_never_asked_again(tmp_path, monkeypatch):
    yt = Y.YouTube(Settings(cache_dir=tmp_path), Cache(tmp_path))
    calls = []

    class DL:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def extract_info(self, url, download=False):
            calls.append(url)
            raise Exception("ERROR: [youtube] gone1: This video is not available")

    monkeypatch.setattr(Y.yt_dlp, "YoutubeDL", DL)
    for _ in range(3):
        c = yt.details(Candidate(provider="youtube", source_id="gone1", url="u1"))
        assert c.screen_flags.get("meta_error")
    assert len(calls) == 1


# --- direct section download -------------------------------------------------------------
def test_stream_url_is_remembered_from_metadata_and_prefers_avc1(tmp_path):
    yt = Y.YouTube(Settings(cache_dir=tmp_path), Cache(tmp_path))
    info = {"formats": [
        {"vcodec": "vp9", "acodec": "none", "height": 1080, "protocol": "https", "url": "https://x/vp9?expire=9999999999"},
        {"vcodec": "avc1.640028", "acodec": "none", "height": 1080, "protocol": "https", "url": "https://x/avc?expire=9999999999"},
        {"vcodec": "avc1.640033", "acodec": "none", "height": 2160, "protocol": "https", "url": "https://x/4k"},
        {"vcodec": "avc1", "acodec": "mp4a", "height": 360, "protocol": "https", "url": "https://x/muxed"}]}
    yt._remember_stream("v", info)
    url, _, expire = yt._streams["v"]
    assert url.endswith("/avc?expire=9999999999") and expire == 9999999999


def test_an_expired_stream_falls_back_to_yt_dlp(tmp_path):
    yt = Y.YouTube(Settings(cache_dir=tmp_path), Cache(tmp_path))
    yt._streams["v"] = ("https://x/a", {}, 0.0)
    assert not yt._direct_section("v", 1, 5, tmp_path / "o.mp4")


# --- judge lanes ---------------------------------------------------------------------------
def _judge(monkeypatch, tmp_path, **kw):
    monkeypatch.setattr(ProviderConfig, "load", classmethod(lambda cls, p=None: cls({
        "zai": {"baseUrl": "https://example.invalid/z", "apiKey": "k"},
        "geminiopenai": {"baseUrl": "https://example.invalid/g", "apiKey": "k"}})))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    chain = [JudgeLink(provider="zai", model="a", vision=True), JudgeLink(provider="geminiopenai", model="b", vision=True)]
    return R.Judge(Settings(judge_chain=chain, cache_dir=tmp_path, judge_patience=0, **kw), Cache(tmp_path))


def test_lanes_send_the_call_to_the_link_with_the_shorter_expected_wait(monkeypatch, tmp_path):
    j = _judge(monkeypatch, tmp_path, judge_lanes=2)
    links = list(j.chain)
    assert [l.model for l in j._lanes(links)] == ["a", "b"], "no data yet: chain order"
    j._latency = {"zai/a": 6.0, "geminiopenai/b": 6.0}
    j._inflight = {"zai/a": 2}
    assert [l.model for l in j._lanes(links)] == ["b", "a"], "a has two calls running"
    j._inflight = {}
    j._latency = {"zai/a": 20.0, "geminiopenai/b": 5.0}
    assert [l.model for l in j._lanes(links)] == ["b", "a"], "b answers four times faster"


def test_one_lane_keeps_strict_chain_order(monkeypatch, tmp_path):
    j = _judge(monkeypatch, tmp_path, judge_lanes=1)
    j._latency = {"zai/a": 50.0, "geminiopenai/b": 1.0}
    assert [l.model for l in j._lanes(list(j.chain))] == ["a", "b"]


# --- remotion export --------------------------------------------------------------------------
def _clip(path: Path, dur=2, src="testsrc2"):
    sep = ":" if "=" in src else "="
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"{src}{sep}size=1280x720:rate=25:duration={dur}",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def _probe(p):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height,r_frame_rate,codec_type:format=duration",
                          "-of", "json", str(p)], capture_output=True, text=True).stdout
    return json.loads(out)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_export_makes_1080p30_with_a_cloned_tail_and_baked_overlays(tmp_path):
    from ytauto.export import Look, export_clip

    src = _clip(tmp_path / "s.mp4")
    grain = _clip(tmp_path / "g.mp4", 1)            # 1 s: must loop to cover the 2.6 s clip
    out = export_clip(src, tmp_path / "out" / "s00.mp4", pad=0.6, look=Look(grain=grain, dust=grain))
    info = _probe(out)
    v = [s for s in info["streams"] if s["codec_type"] == "video"]
    assert len(info["streams"]) == 1, "no audio"
    assert (v[0]["width"], v[0]["height"], v[0]["r_frame_rate"]) == (1920, 1080, "30/1")
    assert float(info["format"]["duration"]) == pytest.approx(2.6, abs=0.1), "looping overlays never stretch it"


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_exporter_writes_shots_json_and_skips_unchanged_clips(tmp_path, monkeypatch):
    import ytauto.export as E

    src = _clip(tmp_path / "s.mp4")
    shots = [_shot(0, 0), _shot(1, 0, 1)]
    ex = E.Exporter(tmp_path / "out", None)
    ex.submit(shots[0], str(src))
    ex.submit(shots[1], None)                       # no clip: nothing to export
    rows = json.loads(ex.finish(shots).read_text())
    assert [r["id"] for r in rows] == ["s00"] and rows[0]["t0"] == 0.0 and rows[0]["clip"] == "s00.mp4"
    calls = []
    monkeypatch.setattr(E, "export_clip", lambda *a, **k: calls.append(a))
    again = E.Exporter(tmp_path / "out", None)
    again.submit(shots[0], str(src))
    again.finish(shots)
    assert calls == [], "same source clip: not re-encoded"


# --- footage library ---------------------------------------------------------------------------
class FakeClip:
    """CLIP stand-in: an image's embedding is its mean colour, a text's is a fixed vector."""
    vecs = {"red": np.array([1, 0, 0] + [0] * 509, np.float32), "blue": np.array([0, 0, 1] + [0] * 509, np.float32)}

    def images(self, imgs):
        out = []
        for im in imgs:
            a = np.asarray(im, np.float32).reshape(-1, 3).mean(0)
            v = np.zeros(512, np.float32)
            v[:3] = a / max(np.linalg.norm(a), 1e-8)
            out.append(v)
        return np.stack(out)

    def text(self, s):
        return self.vecs["red" if "red" in s else "blue"]


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_library_finds_the_matching_clip_and_respects_setting_length_and_reuse(tmp_path):
    from ytauto.library import Library

    red = _clip(tmp_path / "red.mp4", 5, "color=c=red")
    blue = _clip(tmp_path / "blue.mp4", 5, "color=c=blue")
    lib = Library(tmp_path / "lib", FakeClip())
    kw = dict(title="t", license="l", src_start=10.0, src_end=15.0, text="x")
    assert lib.add(clip=red, provider="youtube", source_id="r", url="u", subject="red", setting="Philippines", project="p1", **kw)
    assert lib.add(clip=blue, provider="pexels", source_id="b", url="u", subject="blue", setting="Philippines", project="p1", **kw)
    assert not lib.add(clip=red, provider="youtube", source_id="r", url="u", subject="red", setting="Philippines", project="p1", **kw)
    q = FakeClip().text("red")
    hits = lib.search(q, want=4.0, setting="Philippines", project="p2", min_sim=0.5)
    assert [e["source_id"] for e, _ in hits] == ["r"]
    assert not lib.search(q, want=4.0, setting="Alaska", project="p2", min_sim=0.5), "other setting"
    assert not lib.search(q, want=6.0, setting="Philippines", project="p2", min_sim=0.5), "too short"
    assert not lib.search(q, want=4.0, setting="Philippines", project="p1", min_sim=0.5), "its own project"
    for p in ("p2", "p3", "p4"):
        lib.note_use(hits[0][0]["key"], p)
    assert not lib.search(q, want=4.0, setting="Philippines", project="p5", min_sim=0.5), "used by 3 videos"
    again = Library(tmp_path / "lib", FakeClip())
    assert len(again.entries) == 2 and again.emb.shape == (2, 512), "persisted"


# --- AI flag: two strikes before a whole video is banned ------------------------------------------
def test_one_ai_flag_does_not_ban_the_whole_video_two_do(tmp_path):
    sel = _sel(tmp_path, [_shot(0, 0)])
    c = Candidate(provider="youtube", source_id="drone1", url="u")
    sel._note_defects(c, ["AI-generated / animated / CGI (AI)"])
    assert sel.bad_video(c) is None, "real drone footage is often flagged once; other scenes stay usable"
    sel._note_defects(c, ["AI-generated / animated / CGI (AI)"])
    assert sel.bad_video(c) == "AI-generated / animated / CGI"


def test_a_channel_logo_still_bans_the_video_at_once(tmp_path):
    sel = _sel(tmp_path, [_shot(0, 0)])
    c = Candidate(provider="youtube", source_id="logo1", url="u")
    sel._note_defects(c, ["watermark / channel logo (AI)"])
    assert sel.bad_video(c) == "watermark / channel logo"


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_exporter_marks_clips_made_from_photographs_for_the_composition(tmp_path):
    import ytauto.export as E

    src = _clip(tmp_path / "shot001_still_pexels_7_0.mp4")
    shots = [_shot(0, 0), _shot(1, 0, 1)]
    ex = E.Exporter(tmp_path / "out", None)
    ex.submit(shots[0], str(_clip(tmp_path / "shot000_youtube_a_0.mp4")))
    ex.submit(shots[1], str(src))
    rows = {r["id"]: r for r in json.loads(ex.finish(shots).read_text())}
    assert "kind" not in rows["s00"]
    assert rows["s01"]["kind"] == "still" and rows["s01"]["label"] == "reef"
