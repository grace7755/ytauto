"""ChatGPT through HeartCode as a judge link: message building, result parsing, and the router
treating it like any other link. A fake `heart` (a tiny Python script) stands in for the CLI."""
import json
import sys
import textwrap

import pytest

from ytauto.cache import Cache
from ytauto.config import JudgeLink, Settings
from ytauto.llm import heart as H
from ytauto.llm import router as R
from ytauto.llm.providers import ProviderConfig

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}}


def test_parts_become_one_message_with_images_first():
    msg = H.to_message([{"type": "text", "text": "judge"},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}}])
    blocks = msg["message"]["content"]
    assert msg["type"] == "user" and [b["type"] for b in blocks] == ["image", "text"]
    assert blocks[0]["source"] == {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"}


def test_a_url_image_is_refused():
    with pytest.raises(H.HeartError):
        H.to_message([{"type": "image_url", "image_url": {"url": "https://youtube.com/watch?v=x"}}])


def test_result_line_is_found_among_log_lines():
    out = 'requests: C:\\tmp\n{"type":"result","subtype":"success","is_error":false,"result":"{\\"ok\\": true}"}\n'
    assert H.parse_result(out) == '{"ok": true}'


@pytest.mark.parametrize("out", [
    '{"type":"result","subtype":"error","is_error":true,"result":"403: an HTML error page"}',
    '{"type":"result","subtype":"success","is_error":false,"result":"  "}',
    "nothing useful here",
])
def test_errors_and_empty_answers_raise(out):
    with pytest.raises(H.HeartError):
        H.parse_result(out)


def _fake_heart(tmp_path, reply='{"ok": true}', exit_code=0, sleep=0.0):
    script = tmp_path / "fake_heart.py"
    script.write_text(textwrap.dedent(f"""
        import json, sys, time
        msg = json.loads(sys.stdin.readline())
        open(r"{tmp_path / 'seen.json'}", "w").write(json.dumps({{"argv": sys.argv[1:], "msg": msg}}))
        time.sleep({sleep})
        print("requests: somewhere")
        print(json.dumps({{"type": "result", "subtype": "success", "is_error": False, "result": {reply!r}}}))
        sys.exit({exit_code})
    """), encoding="utf-8")
    return f"{sys.executable}|{script}"


def test_client_runs_the_cli_headless_and_returns_the_answer(tmp_path, monkeypatch):
    monkeypatch.setenv("YTAUTO_HEART_CMD", _fake_heart(tmp_path))
    c = H.HeartClient()
    text = c.ask("chatgpt gpt-6-luna", [{"type": "text", "text": "hi"}])
    assert json.loads(text) == {"ok": True}
    seen = json.loads((tmp_path / "seen.json").read_text())
    for flag in ("-p", "--temp", "--input-format", "stream-json", "--output-format", "json"):
        assert flag in seen["argv"]
    assert seen["argv"][seen["argv"].index("-m") + 1] == "chatgpt gpt-6-luna"
    assert seen["msg"]["message"]["content"][0]["text"] == "hi"


def test_a_hung_cli_is_killed_at_the_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("YTAUTO_HEART_CMD", _fake_heart(tmp_path, sleep=5))
    with pytest.raises(H.HeartError, match="timed out"):
        H.HeartClient(timeout=1).ask("m", [{"type": "text", "text": "hi"}])


def test_missing_cli_is_reported_not_crashed(monkeypatch):
    monkeypatch.delenv("YTAUTO_HEART_CMD", raising=False)
    monkeypatch.setattr(H.shutil, "which", lambda name: None)
    c = H.HeartClient()
    assert not c.installed
    with pytest.raises(H.HeartError):
        c.ask("m", [])


def _judge(monkeypatch, tmp_path, chain, **kw):
    monkeypatch.setattr(ProviderConfig, "load", classmethod(lambda cls, p=None: cls({
        "zai": {"baseUrl": "https://example.invalid/z", "apiKey": "k"}})))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    return R.Judge(Settings(judge_chain=chain, cache_dir=tmp_path, judge_patience=0, **kw), Cache(tmp_path))


def test_router_uses_a_heart_link_like_any_other(tmp_path, monkeypatch):
    monkeypatch.setenv("YTAUTO_HEART_CMD", _fake_heart(tmp_path, reply='```json\n{"ok": true}\n```'))
    j = _judge(monkeypatch, tmp_path, [JudgeLink(provider="_heart", model="chatgpt gpt-6-luna", vision=True)])
    assert j.json_images("is it ok?", [b"\xff\xd8jpeg"], SCHEMA) == {"ok": True}, "fenced JSON repaired too"
    seen = json.loads((tmp_path / "seen.json").read_text())
    assert [b["type"] for b in seen["msg"]["message"]["content"]] == ["image", "text"]


def test_a_failing_heart_link_falls_through_to_the_next(tmp_path, monkeypatch):
    monkeypatch.setenv("YTAUTO_HEART_CMD", _fake_heart(tmp_path, reply="", exit_code=1))
    j = _judge(monkeypatch, tmp_path, [JudgeLink(provider="_heart", model="m", vision=True),
                                       JudgeLink(provider="zai", model="z", vision=True)])
    real = R.Judge._post
    monkeypatch.setattr(R.Judge, "_post", lambda self, link, content, schema, timeout=None:
                        real(self, link, content, schema, timeout) if link.provider == "_heart"
                        else '{"ok": false}')
    assert j.json("p", SCHEMA) == {"ok": False}
    assert "_heart/m" in j.down_links(), "an expired ChatGPT login trips the link, the run goes on"


def test_heart_link_is_dropped_when_the_cli_is_not_installed(tmp_path, monkeypatch):
    monkeypatch.delenv("YTAUTO_HEART_CMD", raising=False)
    monkeypatch.setattr(H.shutil, "which", lambda name: None)
    j = _judge(monkeypatch, tmp_path, [JudgeLink(provider="_heart", model="m", vision=True),
                                       JudgeLink(provider="zai", model="z", vision=True)])
    assert [l.provider for l in j.chain] == ["zai"]
