"""Benchmark: hard rejects (never overridden by a high score) + weighted score + reasons."""
from __future__ import annotations

from datetime import datetime

from .config import Settings
from .inspect.checks import CheckResult
from .inspect.probe import MediaInfo
from .models import Candidate, ClipScore


def _clamp(x: float) -> float:
    return max(0.0, min(100.0, x))


def recency_score(c: Candidate, soft_years: int) -> float:
    if len(c.upload_date) != 8:
        return 70.0
    try:
        age = (datetime.now() - datetime.strptime(c.upload_date, "%Y%m%d")).days / 365.25
    except ValueError:
        return 70.0
    if age <= soft_years / 2:
        return 100.0
    return _clamp(100 - 12 * (age - soft_years / 2))   # soft penalty only; never a hard reject


def local_rejects(s: Settings, info: MediaInfo | None, chk: CheckResult) -> list[str]:
    """The hard rejects the free local checks can decide alone (no AI verdict, no score): an unreadable,
    low-resolution, black-and-white, black, frozen, blurry, letterboxed or presenter-sized-face clip. A clip
    with any of these is rejected whatever the judge would say, so it is not sent to the judge."""
    if info is None or chk.n_frames == 0:
        return ["unreadable clip"]
    rejects: list[str] = []

    # ---- deterministic hard rejects -----------------------------------
    if info.height < s.min_clip_height:
        rejects.append(f"resolution {info.width}x{info.height} below {s.min_clip_height}p")
    if chk.saturation < s.min_saturation:
        rejects.append(f"black & white / desaturated (sat {chk.saturation:.3f})")
    if chk.black > s.max_black_ratio:
        rejects.append("black frames")
    if chk.frozen:
        rejects.append("frozen / static footage")
    if chk.blur < s.blur_threshold:
        rejects.append(f"too blurry (sharpness {chk.blur:.0f} < {s.blur_threshold:.0f})")
    if chk.letterbox > 0.12:
        rejects.append(f"letterboxed ({chk.letterbox:.0%} black bars)")
    if chk.face > s.max_face_ratio:
        rejects.append(f"large face on screen ({chk.face:.0%} of frame) - likely presenter")
    return rejects


def evaluate(s: Settings, c: Candidate, info: MediaInfo | None, chk: CheckResult, verdict: dict | None,
             prior_relevance: float = 0.0) -> tuple[ClipScore, list[str]]:
    """verdict = AI verify dict (or None in offline mode). prior_relevance = text/screen match 0-100."""
    if info is None or chk.n_frames == 0:
        return ClipScore(), ["unreadable clip"]
    rejects = local_rejects(s, info, chk)

    # NOTE: chk.overlay is a weak hint only (it misses translucent watermarks and fires on static
    # scenery), so it never hard-rejects. Real overlays are caught by the AI judge's text_overlay /
    # watermark_or_logo flags below; here it only costs the clip some cleanliness score.

    # ---- component scores ----------------------------------------------
    quality_local = _clamp(min(chk.blur / 250.0, 1.0) * 70 + (30 if info.height >= 1080 else 10)
                           + (5 if info.height >= 2160 else 0))
    clean_local = _clamp(100 - chk.overlay * 100 - max(0.0, chk.letterbox) * 200)
    sc = ClipScore(
        relevance=prior_relevance, quality=quality_local, authenticity=75.0, composition=70.0,
        recency=recency_score(c, s.max_age_years_soft), cleanliness=clean_local, narrative=prior_relevance)

    if verdict:
        # the AI judge overrides the heuristic components it can actually see
        sc.relevance = float(verdict["relevance"])
        sc.authenticity = float(verdict["authenticity"])
        sc.composition = float(verdict["composition"])
        sc.narrative = float(verdict["narrative"])
        sc.quality = min(quality_local, float(verdict["quality"])) if quality_local < 40 else (quality_local + float(verdict["quality"])) / 2
        sc.cleanliness = min(clean_local, float(verdict["cleanliness"]))
        sc.recency = (sc.recency + float(verdict["recency"])) / 2
        if verdict.get("presenter_on_camera"):
            rejects.append("creator/presenter on camera (AI)")
        if verdict.get("watermark_or_logo"):
            rejects.append("watermark / channel logo (AI)")
        if verdict.get("black_and_white"):
            rejects.append("black & white (AI)")
        if verdict.get("ai_or_animated"):
            rejects.append("AI-generated / animated / CGI (AI)")
        if verdict.get("text_overlay"):
            rejects.append("burnt-in text / branding overlay (AI)")
            sc.cleanliness = min(sc.cleanliness, 40.0)
        elif sc.cleanliness < 50:
            rejects.append(f"cleanliness {sc.cleanliness:.0f} < 50 (overlays/logos)")
    if sc.relevance < s.min_relevance:
        rejects.append(f"relevance {sc.relevance:.0f} < {s.min_relevance:.0f}")

    w = s.weights
    tot_w = w.relevance + w.quality + w.authenticity + w.composition + w.recency + w.cleanliness + w.narrative
    sc.total = round((sc.relevance * w.relevance + sc.quality * w.quality + sc.authenticity * w.authenticity +
                      sc.composition * w.composition + sc.recency * w.recency + sc.cleanliness * w.cleanliness +
                      sc.narrative * w.narrative) / tot_w, 1)
    if sc.total < s.min_score:
        rejects.append(f"score {sc.total:.0f} < {s.min_score:.0f}")
    return sc, rejects
