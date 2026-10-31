"""`ytauto doctor` and `ytauto batch`."""
from pathlib import Path

from typer.testing import CliRunner

import ytauto.cli as CLI
from ytauto import doctor as D
from ytauto.cache import Cache
from ytauto.config import JudgeLink, Settings

runner = CliRunner()


def test_doctor_checks_never_raise_and_report_tools():
    names = [c[0] for c in D.check_tools()]
    assert names[:2] == ["ffmpeg", "ffprobe"] and "opencv" in names


def test_doctor_reports_missing_tools(monkeypatch):
    monkeypatch.setattr(D.shutil, "which", lambda n: None)
    assert D._tool("ffmpeg") == ("ffmpeg", False, "not on PATH")


def test_doctor_flags_a_link_out_of_quota_without_calling_it(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    link = JudgeLink(provider="_direct", model="gemini-x", vision=True)
    s = Settings(judge_chain=[link], cache_dir=tmp_path)
    cache = Cache(tmp_path)
    from ytauto.llm.quota import Quota

    Quota(cache).exhaust(link)
    (name, ok, detail), = D.check_judge(s, cache, live=True)
    assert not ok and "out of daily quota" in detail


def test_doctor_without_any_judge_says_so(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    s = Settings(judge_chain=[JudgeLink(provider="_direct", model="m")], cache_dir=tmp_path)
    (name, ok, _), = D.check_judge(s, Cache(tmp_path))
    assert name == "judge chain" and not ok


class _FakePipe:
    def __init__(self, f, fail):
        self.f, self.fail = f, fail
        self.proj = type("P", (), {"path": lambda self, *a: Path("out") / "/".join(a)})()

    def run(self, wait=False):
        if self.fail:
            raise RuntimeError("boom")
        return {"shots": [{"shot": 0, "status": "accepted"}], "blockers": [], "quota_paused_until": None}


def test_batch_runs_every_script_and_survives_a_failure(tmp_path, monkeypatch):
    for n in ("a.txt", "b.txt", "c.srt", "notes.md"):
        (tmp_path / n).write_text("00:00-00:05 hi", encoding="utf-8")
    seen = []

    def fake(f, title, vo, config, verbose):
        seen.append(f.name)
        return _FakePipe(f, fail=f.name == "b.txt")

    monkeypatch.setattr(CLI, "_pipe", fake)
    r = runner.invoke(CLI.app, ["batch", str(tmp_path)])
    assert seen == ["a.txt", "b.txt", "c.srt"], "folders expand to transcripts only, in order"
    assert "FAILED   b.txt" in r.output and r.output.count("done") == 2
    assert r.exit_code == 1, "a failed video makes the batch exit non-zero"


def test_batch_all_good_exits_zero(tmp_path, monkeypatch):
    f = tmp_path / "a.txt"
    f.write_text("00:00-00:05 hi", encoding="utf-8")
    monkeypatch.setattr(CLI, "_pipe", lambda f, *a: _FakePipe(f, fail=False))
    r = runner.invoke(CLI.app, ["batch", str(f)])
    assert r.exit_code == 0 and "batch summary" in r.output
