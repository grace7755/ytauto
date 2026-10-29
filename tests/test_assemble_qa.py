import subprocess

import pytest

from ytauto.assemble import assemble, cut_clip
from ytauto.config import Settings
from ytauto.models import Decision, Shot
from ytauto.qa import run_qa
from ytauto.state import Project

S = Settings(width=640, height=360, fps=30)


def _src(path, src: str, dur=8):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"{src}{':' if '=' in src else '='}size=1280x720:rate=30",
                    "-t", str(dur), "-pix_fmt", "yuv420p", str(path)], check=True)


@pytest.fixture()
def built(tmp_path):
    proj = Project(tmp_path / "p")
    shots = [Shot(id=0, segment=0, start=0, end=4.5, text="a"), Shot(id=1, segment=0, start=4.5, end=9, text="b"),
             Shot(id=2, segment=1, start=9, end=13.5, text="c")]
    srcs = {0: "testsrc2", 1: "mandelbrot", 2: "testsrc"}
    clips = {}
    for sh in shots:
        raw = tmp_path / f"raw{sh.id}.mp4"
        _src(raw, srcs[sh.id])
        clips[sh.id] = cut_clip(S, raw, 1.0, sh.duration, proj.path("clips", f"{sh.id}.mp4"))
        proj.log_decision(Decision(shot=sh.id, status="accepted", provider="youtube", source_id=str(sh.id), url="u",
                                   clip_path=str(clips[sh.id])))
    return proj, shots, clips


def test_assemble_spec_and_duration(built, tmp_path):
    proj, shots, clips = built
    out = assemble(S, proj, shots, clips, None, proj.path("output", "final.mp4"))
    qa = run_qa(S, proj, shots, out, 13.5)
    assert not [i for i in qa.issues if "resolution" in i or "fps" in i or "duration" in i], qa.issues
    assert qa.info["width"] == 640 and qa.info["height"] == 360


def test_qa_flags_black_shot(built, tmp_path):
    proj, shots, clips = built
    black = tmp_path / "black.mp4"
    _src(black, "color=c=black")
    clips[1] = cut_clip(S, black, 0, 4.5, proj.path("clips", "1.mp4"))
    out = assemble(S, proj, shots, clips, None, proj.path("output", "final.mp4"))
    qa = run_qa(S, proj, shots, out, 13.5)
    assert not qa.passed and 1 in qa.bad_shots
    assert any("black" in i for i in qa.issues)


def test_short_source_is_stretched_to_fill_its_slot(tmp_path):
    """Measured 2026-10-29: a section download ended 6 frames early, the clip came out 144 frames
    for a 150-frame slot, and the 60.0 s video rendered as 59.8 s and failed QA. A small shortfall
    must be stretched to exactly fill the slot."""
    from ytauto.assemble import count_frames

    raw = tmp_path / "short.mp4"
    _src(raw, "testsrc2", dur=4.8)                       # 144 frames of source for a 5 s slot
    out = cut_clip(S, raw, 0.0, 5.0, tmp_path / "c.mp4")  # avail=None: caller believed it was enough
    assert count_frames(out) == 150


def test_source_far_too_short_is_refused_not_shipped_short(tmp_path):
    from ytauto.assemble import FFmpegError

    raw = tmp_path / "tiny.mp4"
    _src(raw, "testsrc2", dur=3)                         # would need a 1.67x slow-down
    with pytest.raises(FFmpegError, match="ran out"):
        cut_clip(S, raw, 0.0, 5.0, tmp_path / "c.mp4")
    assert not (tmp_path / "c.mp4").exists()
