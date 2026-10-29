from ytauto.benchmark import evaluate
from ytauto.config import Settings
from ytauto.inspect.checks import CheckResult
from ytauto.inspect.probe import MediaInfo
from ytauto.models import Candidate

S = Settings()
C = Candidate(provider="youtube", source_id="x", url="u", upload_date="20260101")
INFO = MediaInfo(1920, 1080, 30, 5, False, "h264")
GOOD = CheckResult(blur=300, saturation=0.3, black=0, frozen=False, letterbox=0, overlay=0.1, face=0, n_frames=9)
VERDICT = dict(relevance=90, quality=85, authenticity=90, composition=80, recency=90, cleanliness=95, narrative=85,
               presenter_on_camera=False, watermark_or_logo=False, black_and_white=False, ai_or_animated=False)


def test_good_clip_passes():
    sc, rej = evaluate(S, C, INFO, GOOD, VERDICT, 80)
    assert rej == [] and sc.total >= 70


def test_high_score_cannot_override_hard_reject():
    v = dict(VERDICT, presenter_on_camera=True)
    sc, rej = evaluate(S, C, INFO, GOOD, v, 95)
    assert sc.total > 80 and any("presenter" in r for r in rej)


def test_bw_blur_lowres_rejected():
    bad = CheckResult(blur=10, saturation=0.0, black=0, frozen=False, letterbox=0, overlay=0, face=0, n_frames=9)
    _, rej = evaluate(S, C, MediaInfo(1280, 720, 30, 5, False, "h264"), bad, VERDICT, 90)
    assert any("black & white" in r for r in rej)
    assert any("blurry" in r for r in rej)
    assert any("resolution" in r for r in rej)


def test_large_face_is_rejected():
    c = CheckResult(blur=300, saturation=0.3, face=0.3, overlay=0.0, n_frames=9)
    _, rej = evaluate(S, C, INFO, c, None, 80)
    assert any("face" in r for r in rej)


def test_local_overlay_hint_never_hard_rejects_on_its_own():
    """chk.overlay misses translucent watermarks and fires on static scenery, so it only
    costs cleanliness score - the AI judge's flags are what actually reject an overlay."""
    c = CheckResult(blur=300, saturation=0.3, face=0.0, overlay=0.95, n_frames=9)
    sc, rej = evaluate(S, C, INFO, c, None, 80)
    assert not any("overlay" in r or "watermark" in r for r in rej)
    assert sc.cleanliness < 50, "it must still hurt the score"

    # ...while the AI judge's verdict does hard-reject.
    v = dict(VERDICT, watermark_or_logo=True)
    _, rej2 = evaluate(S, C, INFO, GOOD, v, 80)
    assert any("watermark" in r for r in rej2)


def test_old_footage_is_soft_penalty_not_reject():
    old = Candidate(provider="youtube", source_id="o", url="u", upload_date="20120101")
    sc, rej = evaluate(S, old, INFO, GOOD, VERDICT, 80)
    assert sc.recency < 60 and not any("recen" in r for r in rej)


def test_overlay_detector_ignores_static_scenery():
    """A static camera on a detailed scene must not look like a corner watermark."""
    import numpy as np
    from ytauto.inspect.checks import static_overlay_corners

    rng = np.random.default_rng(0)
    # Tripod shot: a fixed, detailed scene with real movement in the middle of frame.
    base = rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)
    frames = []
    for i in range(9):
        f = base.copy()
        f[120:240, 200:440] = rng.integers(0, 255, (120, 240, 3), dtype=np.uint8)  # moving subject
        frames.append(f)
    assert static_overlay_corners(frames) == 0.0, "static scenery in many corners is not a logo"

    # A near-static shot gives this heuristic nothing to work with -> no signal, no reject.
    still = [base.copy() for _ in range(9)]
    assert static_overlay_corners(still) == 0.0
