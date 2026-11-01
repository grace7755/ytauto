"""Final semantic judgement of an extracted clip by the AI judge (relevance, authenticity, overlays...).

The clip is sent as a tiled frame grid, so any vision model can serve this check.
"""
from __future__ import annotations

from pathlib import Path

from ..llm.router import Judge, JudgeError
from ..log import log
from ..models import Shot

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "relevance": {"type": "integer"}, "quality": {"type": "integer"}, "authenticity": {"type": "integer"},
        "composition": {"type": "integer"}, "recency": {"type": "integer"}, "cleanliness": {"type": "integer"},
        "narrative": {"type": "integer"},
        "presenter_on_camera": {"type": "boolean"}, "watermark_or_logo": {"type": "boolean"},
        "text_overlay": {"type": "boolean"}, "black_and_white": {"type": "boolean"},
        "ai_or_animated": {"type": "boolean"}, "notes": {"type": "string"},
        "watermark_corner": {"type": "string", "description": "none | top-left | top-right | bottom-left | bottom-right | other"},
    },
    "required": ["relevance", "quality", "authenticity", "composition", "recency", "cleanliness", "narrative",
                 "presenter_on_camera", "watermark_or_logo", "black_and_white", "ai_or_animated"],
}

PROMPT = """Judge this ~5 second clip as B-roll for a professional faceless YouTube video.
Narration it accompanies: "{text}"
Must show (required): {subject}
Ideal shot (a preference): {visual}
Names involved: {entities}

Score each 0-100 (be strict; 100 is perfect):
relevance (does it clearly show the required subject; a different framing, time of day or camera
move than the ideal shot costs composition/narrative points, not relevance), quality (sharp, stable, no compression artefacts),
authenticity (looks like genuine camera footage, not AI/CGI/animation/slideshow),
composition (framing, lighting, colour, visual appeal), recency (looks modern, not dated),
cleanliness (no watermark/logo/subtitles/burnt-in text/channel branding), narrative (fits the narration's moment).
Set booleans: presenter_on_camera (a person facing/addressing camera), watermark_or_logo, text_overlay,
black_and_white, ai_or_animated. Put a short reason in notes.
When watermark_or_logo is true, set watermark_corner to the frame corner it sits in (top-left, top-right,
bottom-left or bottom-right) ONLY if the whole mark lies inside the outer 12% of the frame in that corner;
otherwise "other". Without a watermark use "none".

BEFORE scoring, decide `ai_or_animated` deliberately. AI-generated stock footage is common on
YouTube and often looks beautiful, which is exactly why it slips through - but real cinematic
footage is also beautiful, so do NOT treat "looks gorgeous", "very symmetrical", "colour graded"
or "unusually still water" as evidence on their own. Those are ordinary properties of good camera work.

For `ai_or_animated`, look for geometry or physics that cannot occur (a reflection that does not
match its subject, objects changing shape, merging or vanishing between frames), textures that
dissolve into smeared painterly mush with no sensor grain, or a dreamlike look where every element
is equally sharp and equally idealised. Do NOT infer it from beauty, symmetry, colour grading,
shallow depth of field or very still water - those are ordinary properties of good cinematography.
When it is genuinely synthetic set authenticity <= 30 even if the clip matches the brief perfectly;
when you are unsure, set ai_or_animated = false and explain the doubt in notes."""


# Verdict flags that hard-reject a clip whatever the AI-generated check would say.
_OTHER_REJECTS = ("presenter_on_camera", "watermark_or_logo", "text_overlay", "black_and_white")


# A category-fallback brief (planner.generalise_brief): the narration names an exact thing no clean footage exists of.
CATEGORY_HINT = """

No usable footage of the exact "{exact}" exists, so this shot illustrates its CATEGORY instead. Judge relevance on the
category: a clip that clearly shows {subject} is relevant (70+) even though it is not that exact model; a different kind
of thing is still not relevant."""


def needs_ai_check(v: dict, min_relevance: float) -> bool:
    """Is the dedicated AI check able to change this clip's outcome? Not if the verdict already
    rejects it for something else: measured on the speed test, 14 of ~35 downloads failed verify
    on a watermark or burnt-in text, and each still spent a second judge call on the AI question."""
    if any(v.get(k) for k in _OTHER_REJECTS):
        return False
    try:
        return float(v.get("relevance", 100)) >= min_relevance
    except (TypeError, ValueError):
        return True


def verify_clip(gem: Judge, clip: Path, shot: Shot, cache_id: str,
                min_relevance: float = 0.0, rejected_anyway=None) -> dict | None:
    """The judge's verdict on a cut clip, or None if no verdict could be had for this clip.

    Raises `JudgeError` when the judge chain itself is down: the caller must not mistake our
    outage for a clip that merely could not be judged (or, worse, accept it unjudged).

    `rejected_anyway(verdict) -> bool`: when it says the clip is already rejected on this verdict,
    the dedicated AI check is skipped (it can only lower the score further) and the verdict is
    marked `ai_check_deferred`, so a caller that later promotes the clip anyway can still run it.
    """
    b = shot.brief
    prompt = PROMPT.format(text=shot.text, subject=b.subject or b.visual, visual=b.visual,
                           entities=b.names())
    if getattr(b, "category_of", ""):
        prompt += CATEGORY_HINT.format(exact=b.category_of, subject=b.subject)
    try:
        v = gem.json_video_file(clip, prompt, VERIFY_SCHEMA, fps=2.0,
                                cache_key=("verify-v2", cache_id, b.subject, b.visual, b.key()), task="verify")
    except JudgeError as e:
        if "frame grid" in str(e):          # this clip's problem, not an outage
            log.warning("verify failed for %s: %s", clip.name, str(e)[:200])
            return None
        raise
    except Exception as e:
        log.warning("verify failed for %s: %s", clip.name, str(e)[:200])
        return None
    if not needs_ai_check(v, min_relevance):
        return v
    if rejected_anyway is not None and rejected_anyway(v):
        return dict(v, ai_check_deferred=True)

    # `ai_or_animated` inside the 12-field verdict is unreliable in both directions: measured on
    # five known clips it called a real Tokyo skyline synthetic AND passed a generated Fuji clip.
    # A dedicated single-question check scored 4/4 on the same clips, so it decides this flag.
    dedicated = gem.confirm_ai_generated(clip, cache_key=("aicheck-v2", cache_id))
    if dedicated is not None and dedicated != bool(v.get("ai_or_animated")):
        log.info("%s: AI flag %s -> %s by the dedicated check",
                 clip.name, bool(v.get("ai_or_animated")), dedicated)
        v["ai_or_animated"] = dedicated
        v["notes"] = (v.get("notes") or "") + f" [dedicated AI check: {dedicated}]"
        if dedicated:
            v["authenticity"] = min(float(v.get("authenticity", 100)), 30.0)
    return v


STILL_PROMPT = PROMPT.replace("Judge this ~5 second clip as B-roll", "Judge this still photograph as a B-roll shot").replace(
    "authenticity (looks like genuine camera footage, not AI/CGI/animation/slideshow)",
    "authenticity (looks like a genuine photograph, not AI-generated, CGI, an illustration or a stock-art composite)")
STILL_RELEVANCE_RULE = """

Relevance for a photograph is about MEANING, not keywords. Ask: does this exact picture show the person, place,
object or event the narration is talking about, in the context of the narration around it? A picture that only
shares a word with the narration (the same word with another meaning, a similar-looking place elsewhere, a
generic image of the topic's category) is NOT relevant: give it relevance 40 or less and say why in notes."""
STILL_AI_PROMPT = (
    "Is this picture a real photograph, or AI-generated / CGI / an illustration? Real (false) includes "
    "polished, colour-graded, perfectly composed professional photos - beauty is NOT evidence of AI. "
    "AI (true) shows impossible geometry, merged or malformed objects, smeared painterly texture with no "
    'sensor grain, or text that is gibberish. Reply ONLY JSON: {"ai_generated": <bool>, "evidence": "<15 words>"}')
STILL_AI_SCHEMA = {"type": "object", "properties": {"ai_generated": {"type": "boolean"},
                                                    "evidence": {"type": "string"}}, "required": ["ai_generated"]}


def verify_still(gem: Judge, image: Path, shot: Shot, cache_id: str, min_relevance: float = 0.0,
                 context: str = "") -> dict | None:
    """The judge's verdict on a still picture (same fields as `verify_clip`), or None when it could
    not be judged. The AI question is asked on its own, as for clips, when the verdict is otherwise
    usable. `JudgeError` (the chain is down) propagates."""
    b = shot.brief
    jpeg = _jpeg(image)
    if jpeg is None:
        return None
    prompt = STILL_PROMPT.format(text=shot.text, subject=b.subject or b.visual, visual=b.visual, entities=b.names())
    prompt += ("\n\n" + context if context else "") + STILL_RELEVANCE_RULE
    try:
        v = gem.json_images(prompt, [jpeg], VERIFY_SCHEMA, cache_key=("verify-still-v2", cache_id, b.subject, b.visual, b.key(), context), task="verify")
    except JudgeError:
        raise
    except Exception as e:
        log.warning("verify failed for %s: %s", image.name, str(e)[:200])
        return None
    if not needs_ai_check(v, min_relevance):
        return v
    ai = gem.json_images(STILL_AI_PROMPT, [jpeg], STILL_AI_SCHEMA, cache_key=("aicheck-still-v2", cache_id), task="aicheck")
    if isinstance(ai, dict) and ai.get("ai_generated") and getattr(getattr(gem, "s", None), "ai_check_votes", 1) >= 2:
        # same rule as for clips (Judge.confirm_ai_generated): an "AI" answer must be repeated before it rejects the picture
        again = gem.json_images(STILL_AI_PROMPT, [jpeg], STILL_AI_SCHEMA, task="aicheck")      # no cache_key: a fresh answer
        if isinstance(again, dict) and "ai_generated" in again and not again["ai_generated"]:
            ai = dict(ai, ai_generated=False, evidence=(ai.get("evidence") or "") + " [second answer: real]")
        gem.cache.set("judge", ai, "aicheck-still-v2", cache_id)
    if isinstance(ai, dict) and "ai_generated" in ai:
        v["ai_or_animated"] = bool(ai["ai_generated"])
        v["notes"] = (v.get("notes") or "") + f" [dedicated AI check: {bool(ai['ai_generated'])}]"
        if v["ai_or_animated"]:
            v["authenticity"] = min(float(v.get("authenticity", 100)), 30.0)
    return v


def _jpeg(image: Path, side: int = 1280) -> bytes | None:
    """The picture as a JPEG no wider than `side` px (what the judge is sent)."""
    import io

    from PIL import Image

    try:
        with Image.open(image) as im:
            im = im.convert("RGB")
            im.thumbnail((side, side))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=88)
            return buf.getvalue()
    except Exception as e:
        log.warning("cannot read picture %s: %s", image.name, str(e)[:120])
        return None
