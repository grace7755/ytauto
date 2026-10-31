"""Local burnt-in text detector: the matching rules, and (when OCR is installed) a real clip."""
import numpy as np
import pytest

from ytauto.inspect import overlay as O


def box(t, x=0.5, y=0.9, w=0.1, h=0.03):
    return (t, x, y, w, h)


def test_text_in_every_frame_at_the_same_place_persists():
    frames = [[box("onride")] for _ in range(4)]
    assert O.persistent_text(frames) == ["onride"]


def test_text_missing_from_one_frame_does_not_persist():
    frames = [[box("onride")], [box("onride")], [], [box("onride")]]
    assert O.persistent_text(frames) == []


def test_text_that_moves_is_scenery_not_a_watermark():
    frames = [[box("tsutaya", x=0.5 + 0.03 * i)] for i in range(4)]
    assert O.persistent_text(frames) == []


def test_text_that_changes_size_is_scenery():
    frames = [[box("sapporo", w=0.1 + 0.05 * i, h=0.03 + 0.02 * i)] for i in range(4)]
    assert O.persistent_text(frames) == []


def test_different_strings_never_match_each_other():
    frames = [[box("alpha")], [box("bravo")], [box("alpha")], [box("alpha")]]
    assert O.persistent_text(frames) == []


def test_one_frame_is_not_enough_evidence():
    assert O.persistent_text([[box("onride")]]) == []


def test_burnt_in_needs_a_moving_scene_behind_the_text():
    assert O.OverlayResult(persistent=["x"], motion=O.MOTION_MIN + 1).burnt_in
    assert not O.OverlayResult(persistent=["x"], motion=0.5).burnt_in, "tripod shot of a sign: leave it to the AI"
    assert not O.OverlayResult(persistent=[], motion=50).burnt_in


def _clip(path, moving, text="WATERMARK"):
    import cv2

    w, h, fps, n = 960, 540, 10, 20
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rng = np.random.default_rng(1)
    base = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    for i in range(n):
        fr = np.roll(base, i * 40, axis=1) if moving else base.copy()
        cv2.rectangle(fr, (300, 400), (700, 480), (0, 0, 0), -1)
        cv2.putText(fr, text, (320, 460), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 5)
        vw.write(fr)
    vw.release()
    return path


@pytest.mark.skipif(not O.available(), reason="rapidocr-onnxruntime not installed")
def test_a_pinned_watermark_over_moving_footage_is_found(tmp_path):
    r = O.detect(_clip(tmp_path / "m.mp4", moving=True))
    assert r is not None and r.burnt_in and any("watermark" in t for t in r.persistent)


@pytest.mark.skipif(not O.available(), reason="rapidocr-onnxruntime not installed")
def test_the_same_text_on_a_static_scene_is_not_rejected(tmp_path):
    r = O.detect(_clip(tmp_path / "s.mp4", moving=False))
    assert r is not None and not r.burnt_in


def test_detect_returns_none_for_an_unreadable_file(tmp_path):
    (tmp_path / "x.mp4").write_bytes(b"not a video")
    assert O.detect(tmp_path / "x.mp4") is None
