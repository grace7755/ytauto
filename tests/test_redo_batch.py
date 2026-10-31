"""Batch `redo` and the Remotion workflow without the unused ffmpeg render (all offline, no ffmpeg).

Measured on the 5-guns film: 12 one-shot redos took 39 min, ~4 min each, and every one rebuilt the
ffmpeg final.mp4 + QA that the Remotion workflow never uses."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from ytauto.cli import _shot_ids
from ytauto.config import Settings
from ytauto.models import Decision, Shot
from ytauto.pipeline import Pipeline
from ytauto.qa import run_qa_clips
from ytauto.state import Project


def _clip(path, colour, pattern=0):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 90))
    f = np.full((90, 160, 3), colour, np.uint8)
    cv2.rectangle(f, (10 + 40 * pattern, 10), (50 + 40 * pattern, 70), (255, 255, 255), -1)
    for _ in range(10):
        w.write(f)
    w.release()
    return path


def _shots(n):
    return [Shot(id=i, segment=i, start=i * 4.0, end=i * 4.0 + 4.0, text=f"line {i}") for i in range(n)]


def _pipe(tmp_path, **cfg):
    p = Pipeline.__new__(Pipeline)                 # no transcript, judge or network
    p.s = Settings(**cfg)
    p.proj = Project(tmp_path / "proj")
    p.shots = _shots(3)
    p.title = "t"
    p.quota_until = None
    return p


@pytest.mark.parametrize("cfg,expect", [
    ({}, True),                                                         # plain run: render as before
    ({"remotion_export": "X:/clips"}, False),                           # Remotion renders the film
    ({"remotion_export": "X:/clips", "ffmpeg_render": "always"}, True),
    ({"ffmpeg_render": "never"}, False),
])
def test_ffmpeg_render_only_when_someone_uses_it(tmp_path, cfg, expect):
    assert _pipe(tmp_path, **cfg).wants_render() is expect


def test_shot_ids_accept_one_or_many():
    assert _shot_ids("7") == [7]
    assert _shot_ids("3,7,12") == [3, 7, 12]
    assert _shot_ids("3, 7 12") == [3, 7, 12]


def test_redo_selects_all_shots_in_one_pass_and_skips_the_unused_render(tmp_path, monkeypatch):
    p = _pipe(tmp_path, remotion_export=str(tmp_path / "clips"))
    calls = {"prepare": 0, "select": [], "render": 0, "invalidated": []}
    p.prepare = lambda: calls.__setitem__("prepare", calls["prepare"] + 1)
    p.select_all = lambda only=None: (calls["select"].append(only), [])[1]
    p.render = lambda: calls.__setitem__("render", calls["render"] + 1)
    p.make_exporter = lambda out_dir=None: None
    p.proj.invalidate = lambda sid, reason="": calls["invalidated"].append(sid)
    monkeypatch.setattr("ytauto.pipeline.build", lambda proj, title, shots, qa, blockers: {"qa": qa, "blockers": blockers})

    data = p.redo([2, 0, 2])
    assert calls["prepare"] == 1, "one prepare for all shots"
    assert calls["select"] == [{0, 2}], "one select_all over every shot (they run side by side)"
    assert calls["invalidated"] == [2, 0]
    assert calls["render"] == 0, "no ffmpeg final.mp4 in the Remotion workflow"
    assert data["qa"].info["mode"].startswith("clips")
    assert (p.proj.root / "logs" / "timings-redo.json").exists()
    assert not (p.proj.root / "logs" / "timings.json").exists(), "the run's own timings are kept"


def test_redo_refuses_unknown_shot_ids_before_touching_the_ledger(tmp_path):
    p = _pipe(tmp_path)
    p.prepare = lambda: None
    touched = []
    p.proj.invalidate = lambda sid, reason="": touched.append(sid)
    with pytest.raises(ValueError, match="no such shot"):
        p.redo([1, 9])
    assert touched == []


def test_clip_qa_finds_duplicates_and_missing_shots_without_a_render(tmp_path):
    proj = Project(tmp_path / "proj")
    a = _clip(tmp_path / "a.mp4", (20, 120, 30), 0)
    b = _clip(tmp_path / "b.mp4", (20, 120, 30), 0)        # same picture as a: a duplicate
    for i, c in ((0, a), (1, b)):
        proj.log_decision(Decision(shot=i, status="accepted", provider="youtube", source_id=f"v{i}",
                                   url="u", clip_path=str(c)))
    r = run_qa_clips(Settings(), proj, _shots(3))
    assert not r.passed
    assert 1 in r.bad_shots and any("duplicates" in i for i in r.issues)
    assert 2 in r.bad_shots and any("shot 2 has no accepted clip" in i for i in r.issues)
    assert (proj.root / "output" / "contact_sheet.jpg").exists(), "the reviewer still gets its sheet"


def test_clip_qa_passes_distinct_clips(tmp_path):
    proj = Project(tmp_path / "proj")
    colours = [(200, 30, 30), (30, 30, 200), (30, 200, 30)]
    for i in range(3):
        c = _clip(tmp_path / f"c{i}.mp4", colours[i], i)
        proj.log_decision(Decision(shot=i, status="accepted", provider="youtube", source_id=f"v{i}",
                                   url="u", clip_path=str(c)))
    r = run_qa_clips(Settings(), proj, _shots(3))
    assert r.passed, r.issues
