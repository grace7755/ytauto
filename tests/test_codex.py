"""OpenAI Codex CLI as a judge link: prompt/image handling, answer file, limits, and the router treating
it like any other link. A fake `codex` (a tiny Python script) stands in for the CLI."""
import base64
import json
import sys
import textwrap

import pytest

from ytauto.cache import Cache
from ytauto.config import JudgeLink, Settings
from ytauto.llm import codex as C
from ytauto.llm import router as R
from ytauto.llm.providers import ProviderConfig

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
IMG = "data:image/jpeg;base64," + base64.b64encode(b"JPEGDATA").decode()


def _fake_codex(tmp_path, reply='{"ok": true}', exit_code=0, sleep=0.0, stderr=""):
    script = tmp_path / "fake_codex.py"
    script.write_text(textwrap.dedent(f"""
        import json, sys, time, pathlib
        a = sys.argv[1:]
        imgs = [pathlib.Path(a[i + 1]).read_bytes().decode("latin1") for i, x in enumerate(a) if x == "-i"]
        open(r"{tmp_path / 'seen.json'}", "w").write(json.dumps({{"argv": a, "stdin": sys.stdin.read(), "imgs": imgs}}))
        time.sleep({sleep})
        sys.stderr.write({stderr!r})
        if {exit_code} == 0:
            open(a[a.index("-o") + 1], "w").write({reply!r})
        sys.exit({exit_code})
    """), encoding="utf-8")
    return f"{sys.executable}|{script}"


def test_parts_split_into_prompt_and_image_files():
    text, imgs = C.split_parts([{"type": "text", "text": "judge"}, {"type": "image_url", "image_url": {"url": IMG}},
                                {"type": "text", "text": "again"}])
    assert text == "judge\n\nagain" and imgs == [(b"JPEGDATA", ".jpg")]


def test_a_url_image_is_refused():
    with pytest.raises(C.CodexError):
        C.split_parts([{"type": "image_url", "image_url": {"url": "https://youtube.com/watch?v=x"}}])


def test_client_runs_codex_exec_headless_with_prompt_on_stdin_and_images(tmp_path, monkeypatch):
    monkeypatch.setenv("YTAUTO_CODEX_CMD", _fake_codex(tmp_path))
    text = C.CodexClient().ask("gpt-5", [{"type": "text", "text": "is it a gun?"},
                                         {"type": "image_url", "image_url": {"url": IMG}}])
    assert json.loads(text) == {"ok": True}
    seen = json.loads((tmp_path / "seen.json").read_text())
    a = seen["argv"]
    assert a[0] == "exec" and a[a.index("-m") + 1] == "gpt-5" and a[a.index("-s") + 1] == "read-only"
    for flag in ("--skip-git-repo-check", "--ephemeral", "--ignore-user-config", "-o", "-i"):
        assert flag in a
    assert "is it a gun?" in seen["stdin"] and seen["stdin"].startswith(C.SYSTEM)
    assert seen["imgs"] == ["JPEGDATA"], "the picture reaches codex as a file"
    assert not any("is it a gun" in x for x in a), "the prompt is on stdin, never a positional argument"


def test_a_hung_cli_is_killed_at_the_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("YTAUTO_CODEX_CMD", _fake_codex(tmp_path, sleep=5))
    with pytest.raises(C.CodexError, match="timed out"):
        C.CodexClient(timeout=1).ask("m", [{"type": "text", "text": "hi"}])


def test_failure_and_empty_answer_raise(tmp_path, monkeypatch):
    monkeypatch.setenv("YTAUTO_CODEX_CMD", _fake_codex(tmp_path, exit_code=1, stderr="not logged in"))
    with pytest.raises(C.CodexError, match="not logged in"):
        C.CodexClient().ask("m", [{"type": "text", "text": "hi"}])
    monkeypatch.setenv("YTAUTO_CODEX_CMD", _fake_codex(tmp_path, reply="  "))
    with pytest.raises(C.CodexError, match="empty"):
        C.CodexClient().ask("m", [{"type": "text", "text": "hi"}])


def test_missing_cli_is_reported_not_crashed(monkeypatch):
    monkeypatch.delenv("YTAUTO_CODEX_CMD", raising=False)
    monkeypatch.setattr(C.shutil, "which", lambda name: None)
    c = C.CodexClient()
    assert not c.installed
    with pytest.raises(C.CodexError, match="not installed"):
        c.ask("m", [{"type": "text", "text": "hi"}])


def test_plan_limit_text_is_recognised():
    assert C.is_limit("You've hit your usage limit. Try again at 3:41 PM.")
    assert not C.is_limit("codex exit 1: not logged in")


# --- the router ---------------------------------------------------------------
def _judge(monkeypatch, tmp_path, chain):
    monkeypatch.setattr(ProviderConfig, "load", classmethod(lambda cls, p=None: cls({})))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    return R.Judge(Settings(judge_chain=chain, cache_dir=tmp_path, judge_patience=0), Cache(tmp_path))


def test_router_uses_a_codex_link_like_any_other(monkeypatch, tmp_path):
    monkeypatch.setenv("YTAUTO_CODEX_CMD", _fake_codex(tmp_path))
    j = _judge(monkeypatch, tmp_path, [JudgeLink(provider="_codex", model="gpt-5", vision=True)])
    assert j.json("hi", SCHEMA) == {"ok": True}
    assert j.json_images("is it a gun?", [b"JPEGDATA"], SCHEMA) == {"ok": True}


def test_codex_link_is_skipped_when_not_installed(monkeypatch, tmp_path):
    monkeypatch.delenv("YTAUTO_CODEX_CMD", raising=False)
    monkeypatch.setattr(C.shutil, "which", lambda name: None)
    with pytest.raises(R.JudgeError, match="no usable judge providers"):
        _judge(monkeypatch, tmp_path, [JudgeLink(provider="_codex", model="gpt-5", vision=True)])


def test_a_plan_limit_pauses_the_run_instead_of_failing_the_shot(monkeypatch, tmp_path):
    monkeypatch.setenv("YTAUTO_CODEX_CMD", _fake_codex(
        tmp_path, exit_code=1, stderr="You've hit your usage limit. Try again at 3:41 PM."))
    j = _judge(monkeypatch, tmp_path, [JudgeLink(provider="_codex", model="gpt-5", vision=True)])
    with pytest.raises(R.QuotaExhausted):
        j.json("hi", SCHEMA)
