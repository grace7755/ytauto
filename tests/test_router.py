"""Offline tests for the multi-provider judge: parsing, chain building, circuit breaker, grid."""
from __future__ import annotations

import json
import subprocess
import time

import pytest

from ytauto.cache import Cache
from ytauto.config import JudgeLink, Settings
from ytauto.inspect.grid import extract_grid, extract_grid_data_url
from ytauto.llm import router as R
from ytauto.llm.providers import DIRECT, ProviderConfig

PROVIDERS = {"geminiopenai": {"baseUrl": "https://example.invalid/g", "apiKey": "k1"},
             "zai": {"baseUrl": "https://example.invalid/z", "apiKey": "k2"}}


def _settings(chain, **kw):
    # judge_patience=0: offline tests must not sleep waiting for a provider to recover
    kw.setdefault("judge_patience", 0)
    return Settings(judge_chain=chain, **kw)


def _judge(monkeypatch, chain, providers=PROVIDERS, tmp_path=None, direct_key=None):
    monkeypatch.setattr(ProviderConfig, "load", classmethod(lambda cls, p=None: cls(dict(providers))))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    if direct_key:
        monkeypatch.setenv("GEMINI_API_KEY", direct_key)
    s = _settings(chain)
    if tmp_path is not None:
        s = s.model_copy(update={"cache_dir": tmp_path})
    return R.Judge(s, Cache(s.cache_dir))


# ---- JSON extraction -------------------------------------------------------
@pytest.mark.parametrize("raw,expect", [
    ('{"a":1}', {"a": 1}),
    ('```json\n{"a":1}\n```', {"a": 1}),
    ('```\n{"a":1}\n```', {"a": 1}),
    ('Sure! {"a":1} hope that helps', {"a": 1}),
    ('{"a":{"b":[1,2]}}', {"a": {"b": [1, 2]}}),
])
def test_extract_json_tolerates_fences_and_prose(raw, expect):
    assert R._extract_json(raw) == expect


@pytest.mark.parametrize("raw,expect", [
    # Malformed shapes real providers emit; none may circuit-break a healthy link.
    ('{"a":1,"b":2,}', {"a": 1, "b": 2}),
    ('{"w":[{"s":"1"},{"s":"2"},]}', {"w": [{"s": "1"}, {"s": "2"}]}),
    ('{"usable": True, "flag": False, "x": None}', {"usable": True, "flag": False, "x": None}),
    ("{'usable': true, 'windows': []}", {"usable": True, "windows": []}),
    # raw newline inside a string literal (illegal control character)
    ('{"reason":"line one' + chr(10) + 'line two"}', {"reason": "line one" + chr(10) + "line two"}),
])
def test_extract_json_repairs_malformed_replies(raw, expect):
    assert R._extract_json(raw) == expect


def test_extract_json_recovers_a_truncated_reply():
    """finish_reason=length cuts mid-structure; keep the windows that did arrive."""
    raw = ('{"usable": true, "reason": "drone shots of Fuji", "windows": ['
           '{"start": "08:02", "end": "08:12", "match": 95},{"start": "08:36", "end"')
    got = R._extract_json(raw)
    assert got["usable"] is True
    assert got["windows"] == [{"start": "08:02", "end": "08:12", "match": 95}]


@pytest.mark.parametrize("raw", ["no json here at all", "I cannot help with that.", "",
                                 "[1,2,3]", "null"])
def test_extract_json_still_rejects_non_objects(raw):
    with pytest.raises(ValueError):
        R._extract_json(raw)


def test_schema_hint_lists_required_keys():
    schema = {"type": "object",
              "properties": {"relevance": {"type": "integer", "description": "0-100"},
                             "ok": {"type": "boolean"}},
              "required": ["relevance", "ok"]}
    hint = R._schema_hint(schema)
    assert '"relevance": <integer>' in hint and "// 0-100" in hint
    assert '"ok": <boolean>' in hint
    assert "relevance, ok" in hint
    assert R._schema_hint(None) == ""


def test_schema_hint_spells_out_array_item_keys():
    """Without the item shape, providers invent key names (start_time vs start)."""
    schema = {"type": "object", "properties": {"windows": {"type": "array", "items": {
        "type": "object", "properties": {
            "start": {"type": "string", "description": "MM:SS"},
            "match": {"type": "integer", "description": "0-100"}}}}}}
    hint = R._schema_hint(schema)
    assert '"start": <string MM:SS>' in hint
    assert '"match": <integer 0-100>' in hint


# ---- chain construction ----------------------------------------------------
def test_chain_drops_providers_missing_from_config(monkeypatch, tmp_path):
    chain = [JudgeLink(provider="geminiopenai", model="m1", vision=True),
             JudgeLink(provider="nope", model="m2", vision=True)]
    j = _judge(monkeypatch, chain, tmp_path=tmp_path)
    assert [l.provider for l in j.chain] == ["geminiopenai"]


def test_direct_link_requires_a_key(monkeypatch, tmp_path):
    chain = [JudgeLink(provider="geminiopenai", model="m1"),
             JudgeLink(provider=DIRECT, model="g")]
    assert len(_judge(monkeypatch, chain, tmp_path=tmp_path).chain) == 1
    assert len(_judge(monkeypatch, chain, tmp_path=tmp_path, direct_key="x").chain) == 2


def test_no_usable_provider_raises(monkeypatch, tmp_path):
    with pytest.raises(R.JudgeError):
        _judge(monkeypatch, [JudgeLink(provider="nope", model="m")], providers={}, tmp_path=tmp_path)


def test_make_judge_returns_none_for_offline_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(ProviderConfig, "load", classmethod(lambda cls, p=None: cls({})))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    s = Settings(judge_chain=[JudgeLink(provider="nope", model="m")]).model_copy(
        update={"cache_dir": tmp_path})
    assert R.make_judge(s, Cache(tmp_path)) is None


# ---- capability filtering --------------------------------------------------
def test_live_filters_by_capability(monkeypatch, tmp_path):
    chain = [JudgeLink(provider="zai", model="v", vision=True, youtube=False),
             JudgeLink(provider="geminiopenai", model="vy", vision=True, youtube=True)]
    j = _judge(monkeypatch, chain, tmp_path=tmp_path)
    assert [l.model for l in j._live(need_vision=True)] == ["v", "vy"]
    assert [l.model for l in j._live(need_youtube=True)] == ["vy"]


# ---- circuit breaker -------------------------------------------------------
def test_breaker_skips_failed_link_then_recovers(monkeypatch, tmp_path):
    chain = [JudgeLink(provider="zai", model="bad", vision=True),
             JudgeLink(provider="geminiopenai", model="good", vision=True)]
    j = _judge(monkeypatch, chain, tmp_path=tmp_path)
    tried: list[str] = []

    def fake_post(link, content, schema, timeout=None):
        tried.append(link.model)
        if link.model == "bad":
            raise R.JudgeError("HTTP 403: nope")
        return '{"ok":true}'

    monkeypatch.setattr(j, "_post", fake_post)
    assert j.json("p", {"type": "object", "properties": {"ok": {"type": "boolean"}}}) == {"ok": True}
    assert tried == ["bad", "good"]
    assert "zai/bad" in j.down_links()

    tried.clear()
    j.json("p2", {"type": "object", "properties": {"ok": {"type": "boolean"}}})
    assert tried == ["good"], "a tripped link must be skipped on the next call"

    # Once the window expires the link is retried again.
    j._down["zai/bad"] = time.time() - 1
    assert j.down_links() == {}
    tried.clear()
    j.json("p3", {"type": "object", "properties": {"ok": {"type": "boolean"}}})
    assert tried == ["bad", "good"]


def test_all_links_down_raises_judge_error(monkeypatch, tmp_path):
    j = _judge(monkeypatch, [JudgeLink(provider="zai", model="bad", vision=True)], tmp_path=tmp_path)
    monkeypatch.setattr(j, "_post", lambda *a: (_ for _ in ()).throw(R.JudgeError("HTTP 500")))
    with pytest.raises(R.JudgeError, match="all judge providers failed"):
        j.json("p", {"type": "object", "properties": {"ok": {"type": "boolean"}}})


def test_empty_content_and_truncation_are_failures(monkeypatch, tmp_path):
    """A 200 with content=null must fail over, not poison the scores."""
    j = _judge(monkeypatch, [JudgeLink(provider="zai", model="m", vision=True)], tmp_path=tmp_path)

    for payload, expect in [
        ({"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}, "empty content"),
        ({"choices": [{"message": {"content": None}, "finish_reason": "length"}]}, "truncated"),
        ({"choices": []}, "no choices"),
    ]:
        class FakeResp:
            def __enter__(self_in): return self_in
            def __exit__(self_in, *a): return False
            def read(self_in): return json.dumps(payload).encode()

        monkeypatch.setattr(R.urllib.request, "urlopen", lambda *a, **k: FakeResp())
        with pytest.raises(R.JudgeError, match=expect):
            j._post(j.chain[0], [{"type": "text", "text": "x"}], None)


# ---- caching ---------------------------------------------------------------
def test_cache_hit_avoids_a_second_call(monkeypatch, tmp_path):
    j = _judge(monkeypatch, [JudgeLink(provider="zai", model="m", vision=True)], tmp_path=tmp_path)
    calls = []
    monkeypatch.setattr(j, "_post", lambda *a: (calls.append(1), '{"ok":true}')[1])
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    key = ("verify", "clip-1")
    assert j.json("p", schema, cache_key=key) == {"ok": True}
    assert j.json("p", schema, cache_key=key) == {"ok": True}
    assert len(calls) == 1


# ---- frame grid (needs ffmpeg) --------------------------------------------
def _have_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=20)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not on PATH")
def test_extract_grid_builds_one_tiled_jpeg(tmp_path):
    clip = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=1280x720:rate=30", "-t", "5", "-pix_fmt", "yuv420p", str(clip)],
                   check=True, capture_output=True)
    raw = extract_grid(clip, cols=3, rows=2, fps=1.2)
    assert raw and raw[:2] == b"\xff\xd8", "expected a JPEG"
    assert extract_grid_data_url(clip).startswith("data:image/jpeg;base64,")


@pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not on PATH")
def test_extract_grid_returns_none_for_unreadable_input(tmp_path):
    bad = tmp_path / "not-a-video.mp4"
    bad.write_bytes(b"garbage")
    assert extract_grid(bad) is None


# ---- Mixkit: CLI keyword ladder + exit-code contract (GUIDE.md Part 2) -----
def test_mixkit_keyword_ladder_goes_specific_to_broad():
    from ytauto.sources.stock import Mixkit
    l = Mixkit.keyword_ladder("Mount Fuji snow-capped volcano above a calm lake")
    assert l[0] == "Mount Fuji snow-capped volcano above a calm lake", "the real query comes first"
    assert "mountain" in l and "water" in l, "broad terms the words imply follow"
    assert len(l) == len(set(x.lower() for x in l)), "no duplicates"
    assert Mixkit.keyword_ladder("tokyo") == ["tokyo"]


def test_mixkit_never_falls_back_to_an_unrelated_category():
    """Russian-firearms film: the old unconditional "nature" step returned the same waterfall for 55 gun shots."""
    from ytauto.sources.stock import Mixkit
    for q in ("Russian sniper rifle bolt action", "pistol accessory rail", "PPK-20 blowback action"):
        assert "nature" not in [t.lower() for t in Mixkit.keyword_ladder(q)]
    assert "sea" in Mixkit.keyword_ladder("open sea waves at sunset"), "a shot that names nature still gets it"


@pytest.mark.parametrize("code,expect_blocked,expect_terms", [
    (4, True, 1),     # rate limited -> stop at once, stay out for the run
    (3, True, 1),     # disk guard -> same
    (2, False, 3),    # nothing found -> broaden through the ladder
    (1, False, 3),    # partial failure -> keep trying
])
def test_mixkit_honours_exit_codes(code, expect_blocked, expect_terms, monkeypatch):
    import sys, types
    # mixkit-dl is a separate local package CI does not install; available() only needs it importable
    monkeypatch.setitem(sys.modules, "mixkit_dl", types.ModuleType("mixkit_dl"))
    from ytauto.config import Settings
    from ytauto.sources.stock import Mixkit

    m = Mixkit(Settings())
    tried: list[str] = []
    m._run_cli = lambda term, n: (tried.append(term), (code, {}))[1]
    assert m.search("tokyo street at night", 3) == []
    assert m._blocked is expect_blocked
    assert m.available() is (not expect_blocked)
    assert len(tried) == expect_terms


def test_mixkit_keeps_only_usable_landscape_items(monkeypatch, tmp_path):
    """Usable = downloaded/skipped; portrait clips cannot fill a 16:9 frame."""
    from ytauto.config import Settings
    from ytauto.sources.stock import Mixkit

    good, tall = tmp_path / "a.mp4", tmp_path / "b.mp4"
    good.write_bytes(b"x")
    tall.write_bytes(b"x")
    dims = {str(good): (1920, 1080), str(tall): (1080, 1920)}
    monkeypatch.setattr("ytauto.sources.stock._probe_dims", lambda p: dims.get(str(p)))

    m = Mixkit(Settings())
    m._run_cli = lambda term, n: (0, {"items": [
        {"status": "downloaded", "id": 1, "title": "wide", "path": str(good), "quality": "1080"},
        {"status": "downloaded", "id": 2, "title": "tall", "path": str(tall), "quality": "1080"},
        {"status": "premium_only", "id": 3, "path": str(good)},
        {"status": "failed", "id": 4, "path": str(good), "reason": "x"},
        {"status": "skipped", "id": 1, "path": str(good)},          # duplicate id
        {"status": "downloaded", "id": 5, "path": str(tmp_path / "missing.mp4")},
    ]})
    got = m.search("tokyo", 5)
    assert [c.source_id for c in got] == ["1"], "only the wide, present, non-duplicate clip"
    assert got[0].height == 1080 and got[0].provider == "mixkit"
    assert "Mixkit" in got[0].license


def test_zai_link_has_a_raised_token_budget():
    """GLM-4.6V-Flash is a reasoning model: measured 2026-10-29, 2048 tokens were consumed by
    reasoning before it emitted any answer, so the link failed with "truncated before any answer"
    on every real screening prompt. Keep its budget well above the 2048 default."""
    from ytauto.config import DEFAULT_JUDGE_CHAIN

    zai = [l for l in DEFAULT_JUDGE_CHAIN if l.provider == "zai"]
    assert zai, "the chain should still contain the zai link"
    for link in zai:
        assert link.max_tokens >= 8192, f"{link.model} needs room to reason before answering"


def test_link_stats_count_who_answered(monkeypatch, tmp_path):
    """video-4's log could not say which link answered a call; logs/judge-links*.json now does."""
    from ytauto.cache import Cache
    from ytauto.llm import router as R
    from ytauto.llm.providers import ProviderConfig
    monkeypatch.setattr(ProviderConfig, "load", classmethod(lambda cls, p=None: cls(
        {"g": {"baseUrl": "https://example.invalid/g", "apiKey": "k"}})))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    j = R.Judge(Settings(judge_chain=[JudgeLink(provider="g", model="m1", vision=True)], cache_dir=tmp_path,
                         judge_patience=0), Cache(tmp_path))
    monkeypatch.setattr(j, "_post", lambda *a, **k: '{"ok": true}')
    for _ in range(2):
        j.json("p", {"type": "object", "properties": {"ok": {"type": "boolean"}}})
    assert j.link_stats()["answers"] == {"g/m1": 2}
