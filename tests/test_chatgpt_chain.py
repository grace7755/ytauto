"""GPT Luna first, Terra second, Gemini when the ChatGPT plan is at its limit; Sol never; effort per task.

Measured on video-4 (4 h 09 min): the ChatGPT link answered 871 calls and said "Usage limit reached" 13 times. Each time only that
one link was skipped for 5 minutes; Luna, Terra and Codex all draw from ONE plan, so the next call asked the same spent plan again."""
from __future__ import annotations

import json
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from ytauto.cache import Cache
from ytauto.config import JudgeLink, Settings, load_settings
from ytauto.llm import limits as L
from ytauto.llm import router as R
from ytauto.llm.heart import HeartError
from ytauto.llm.providers import ProviderConfig

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
LUNA, TERRA = "chatgpt gpt-6-luna", "chatgpt gpt-6.1-terra"


def _at(h, m=0):
    return datetime(2026, 11, 1, h, m).timestamp()


# --- the reset time in a usage-limit message -------------------------------------------------------
def test_try_again_at_a_clock_time_today():
    now = _at(18, 43)
    assert L.plan_limit_reset("You've hit your usage limit. Upgrade to Pro. Try again at 7:06 PM.", now) == _at(19, 6)


def test_a_clock_time_already_past_means_tomorrow():
    now = _at(20, 0)
    until = L.plan_limit_reset("Usage limit reached, try again at 7:06 PM", now)
    assert until == _at(19, 6) + 86400


def test_try_again_in_minutes_or_hours():
    now = _at(12, 0)
    assert L.plan_limit_reset("usage limit reached. Try again in 20 minutes", now) == now + 1200
    assert L.plan_limit_reset("You've hit your usage limit, try again in 2 hours.", now) == now + 7200


def test_a_limit_without_a_time_pauses_for_a_default_and_other_errors_are_not_limits():
    now = _at(12, 0)
    assert L.plan_limit_reset("Usage limit reached", now) == now + L.DEFAULT_PAUSE
    for text in ("HTTP 503 high demand", "heart exit 1: not logged in", "You exceeded your current quota per minute", ""):
        assert L.plan_limit_reset(text, now) is None


def test_am_pm_edge_cases():
    now = _at(10, 0)
    assert L.plan_limit_reset("usage limit reached, try again at 12:30 AM", now) == _at(0, 30) + 86400
    assert L.plan_limit_reset("usage limit reached, try again at 12:30 PM", now) == _at(12, 30)


# --- Sol is excluded, Luna and Terra are the ChatGPT models ----------------------------------------------
@pytest.mark.parametrize("model", ["chatgpt gpt-6.1-sol", "gpt-6-sol", "Sol"])
def test_gpt_sol_is_refused(model):
    with pytest.raises(ValidationError, match="Sol is excluded"):
        JudgeLink(provider="_heart", model=model)


@pytest.mark.parametrize("model", [LUNA, TERRA, "models/gemini-flash-lite-latest", "gemini-3.1-flash-lite", "GLM-4.6V-Flash",
                                   "nvidia/nemotron-3.5-super-vl:free", "solar-pro"])
def test_other_models_are_allowed(model):
    assert JudgeLink(provider="x", model=model).model == model


def test_effort_must_be_a_known_level():
    assert JudgeLink(provider="_heart", model=LUNA, effort="Low").effort == "low"
    with pytest.raises(ValidationError):
        JudgeLink(provider="_heart", model=LUNA, effort="extreme")


def test_the_default_chain_is_luna_terra_then_gemini_and_every_call_starts_on_luna():
    s = load_settings(None)
    chain = [(l.provider, l.model) for l in s.judge_chain]
    assert chain[0] == ("_heart", LUNA) and chain[1] == ("_heart", TERRA)
    first_gemini = next(i for i, (p, m) in enumerate(chain) if "gemini" in m.lower())
    assert first_gemini == 2, "Gemini only after both ChatGPT models"
    assert not any("sol" in m.lower().split() for _, m in chain)
    assert s.judge_lanes == 1, "no spreading over the first two links: Terra is a fallback, not a lane"
    assert s.heart_concurrency == s.judge_concurrency


# --- the router -------------------------------------------------------------------------------------------
class FakeHeart:
    installed = True

    def __init__(self, error=None):
        self.asked, self.error = [], error

    def ask(self, model, content, timeout=None, effort=None):
        self.asked.append((model, effort))
        if self.error:
            raise HeartError(self.error)
        return '{"ok": true}'


def _judge(monkeypatch, tmp_path, chain, heart, **kw):
    monkeypatch.setenv("YTAUTO_HEART_CMD", f"{sys.executable}|x")
    monkeypatch.setattr(ProviderConfig, "load", classmethod(lambda cls, p=None: cls(
        {"g": {"baseUrl": "https://example.invalid/g", "apiKey": "k"}})))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    j = R.Judge(Settings(judge_chain=chain, cache_dir=tmp_path, judge_patience=0, judge_lanes=1, **kw), Cache(tmp_path))
    j._heart = heart
    return j


def _chain():
    return [JudgeLink(provider="_heart", model=LUNA, vision=True), JudgeLink(provider="_heart", model=TERRA, vision=True),
            JudgeLink(provider="g", model="gemini-x", vision=True)]


def test_a_plan_limit_pauses_luna_and_terra_together_and_gemini_answers(monkeypatch, tmp_path):
    heart = FakeHeart("Usage limit reached. Try again at 11:59 PM.")
    j = _judge(monkeypatch, tmp_path, _chain(), heart)
    asked = []
    monkeypatch.setattr(j, "_post", lambda link, *a, **k: asked.append(link.model) or '{"ok": true}')
    real_post = R.Judge._post
    monkeypatch.setattr(j, "_post", lambda link, *a, **k: (real_post(j, link, *a, **k) if link.provider == "_heart"
                                                           else (asked.append(link.model) or '{"ok": true}')))
    assert j.json("p", SCHEMA) == {"ok": True}
    assert [m for m, _ in heart.asked] == [LUNA], "Terra is never asked: it shares the plan that just said no"
    assert asked == ["gemini-x"]
    for l in _chain()[:2]:
        assert j.quota.out_until(l) > 0, "both ChatGPT links are paused until the reset time"
    assert not j.quota.out_until(_chain()[2])
    j.json("q", SCHEMA)
    assert [m for m, _ in heart.asked] == [LUNA], "and the next call does not ask the spent plan at all"


def test_an_ordinary_luna_failure_falls_to_terra(monkeypatch, tmp_path):
    heart = FakeHeart("empty content")
    j = _judge(monkeypatch, tmp_path, _chain(), heart)
    real_post = R.Judge._post
    monkeypatch.setattr(j, "_post", lambda link, *a, **k: (real_post(j, link, *a, **k) if link.provider == "_heart" else '{"ok": true}'))
    j.json("p", SCHEMA)
    assert [m for m, _ in heart.asked] == [LUNA, TERRA], "no usage-limit text: Terra gets its turn"


def test_terra_answers_when_luna_fails_and_the_plan_is_fine(monkeypatch, tmp_path):
    class Flaky(FakeHeart):
        def ask(self, model, content, timeout=None, effort=None):
            self.asked.append((model, effort))
            if model == LUNA:
                raise HeartError("reply cut off")
            return '{"ok": true}'
    heart = Flaky()
    j = _judge(monkeypatch, tmp_path, _chain(), heart)
    assert j.json("p", SCHEMA) == {"ok": True}
    assert [m for m, _ in heart.asked] == [LUNA, TERRA]


def test_effort_for_a_task_reaches_the_heart_command_and_is_absent_when_unset(monkeypatch, tmp_path):
    heart = FakeHeart()
    j = _judge(monkeypatch, tmp_path, _chain(), heart, judge_effort={"aicheck": "low"})
    j.json("p", SCHEMA, task="plan")
    j.json("p2", SCHEMA, task="aicheck")
    assert heart.asked == [(LUNA, None), (LUNA, "low")]


def test_a_links_own_effort_beats_the_task_setting(monkeypatch, tmp_path):
    heart = FakeHeart()
    chain = [JudgeLink(provider="_heart", model=LUNA, vision=True, effort="high")] + _chain()[1:]
    j = _judge(monkeypatch, tmp_path, chain, heart, judge_effort={"plan": "low"})
    j.json("p", SCHEMA, task="plan")
    assert heart.asked == [(LUNA, "high")]


def test_chatgpt_links_share_a_group_and_a_hedge_never_goes_to_the_same_plan():
    luna, terra, g = _chain()
    assert R.Judge._group(luna) == R.Judge._group(terra) == "chatgpt"
    assert R.Judge._group(JudgeLink(provider="_codex", model="gpt-6.1-terra")) == "chatgpt"
    assert R.Judge._group(g) != R.Judge._group(luna)


# --- the command lines ------------------------------------------------------------------------------------
def _fake_cli(tmp_path):
    script = tmp_path / "fake_cli.py"
    script.write_text(textwrap.dedent(f"""
        import json, sys, pathlib
        a = sys.argv[1:]
        sys.stdin.read()
        open(r"{tmp_path / 'seen.json'}", "w").write(json.dumps(a))
        if "-o" in a:
            open(a[a.index("-o") + 1], "w").write('{{"ok": true}}')
        else:
            print(json.dumps({{"type": "result", "subtype": "success", "is_error": False, "result": '{{"ok": true}}'}}))
    """), encoding="utf-8")
    return f"{sys.executable}|{script}"


def test_heart_gets_the_effort_flag_only_when_set(tmp_path, monkeypatch):
    from ytauto.llm import heart as H
    monkeypatch.setenv("YTAUTO_HEART_CMD", _fake_cli(tmp_path))
    H.HeartClient().ask(LUNA, [{"type": "text", "text": "hi"}], effort="low")
    a = json.loads((tmp_path / "seen.json").read_text())
    assert a[a.index("--effort") + 1] == "low"
    H.HeartClient().ask(LUNA, [{"type": "text", "text": "hi"}])
    assert "--effort" not in json.loads((tmp_path / "seen.json").read_text())


def test_codex_gets_the_reasoning_effort_override_only_when_set(tmp_path, monkeypatch):
    from ytauto.llm import codex as C
    monkeypatch.setenv("YTAUTO_CODEX_CMD", _fake_cli(tmp_path))
    C.CodexClient().ask("gpt-6.1-terra", [{"type": "text", "text": "hi"}], effort="medium")
    a = json.loads((tmp_path / "seen.json").read_text())
    assert a[a.index("-c") + 1] == 'model_reasoning_effort="medium"'
    C.CodexClient().ask("gpt-6.1-terra", [{"type": "text", "text": "hi"}])
    assert "-c" not in json.loads((tmp_path / "seen.json").read_text())


def test_a_codex_usage_limit_pauses_the_whole_chatgpt_plan_too(monkeypatch, tmp_path):
    from ytauto.llm.codex import CodexError

    class Cx:
        installed = True

        def ask(self, *a, **k):
            raise CodexError("You've hit your usage limit. Try again at 11:59 PM.")
    monkeypatch.setenv("YTAUTO_HEART_CMD", f"{sys.executable}|x")
    monkeypatch.setenv("YTAUTO_CODEX_CMD", f"{sys.executable}|x")
    monkeypatch.setattr(ProviderConfig, "load", classmethod(lambda cls, p=None: cls(
        {"g": {"baseUrl": "https://example.invalid/g", "apiKey": "k"}})))
    chain = [JudgeLink(provider="_codex", model="gpt-6.1-terra", vision=True), _chain()[0], _chain()[2]]
    j = R.Judge(Settings(judge_chain=chain, cache_dir=tmp_path, judge_patience=0, judge_lanes=1), Cache(tmp_path))
    j._codex, j._heart = Cx(), FakeHeart()
    real_post = R.Judge._post
    monkeypatch.setattr(j, "_post", lambda link, *a, **k: (real_post(j, link, *a, **k) if link.provider != "g" else '{"ok": true}'))
    assert j.json("p", SCHEMA) == {"ok": True}
    assert j._heart.asked == [], "Heart shares the Codex plan, so it is paused as well"
