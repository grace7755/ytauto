"""Final semantic judgement of an extracted clip by the AI judge (relevance, authenticity, overlays...).

The clip is sent as a tiled frame grid, so any vision model can serve this check.
"""
from __future__ import annotations

from pathlib import Path

from ..llm.router import Judge
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
    },
    "required": ["relevance", "quality", "authenticity", "composition", "recency", "cleanliness", "narrative",
                 "presenter_on_camera", "watermark_or_logo", "black_and_white", "ai_or_animated"],
}

PROMPT = """Judge this ~5 second clip as B-roll for a professional faceless YouTube video.
Narration it accompanies: "{text}"
Wanted shot: {visual}
Must show: {entities}

Score each 0-100 (be strict; 100 is perfect):
relevance (does it clearly show what is wanted), quality (sharp, stable, no compression artefacts),
authenticity (looks like genuine camera footage, not AI/CGI/animation/slideshow),
composition (framing, lighting, colour, visual appeal), recency (looks modern, not dated),
cleanliness (no watermark/logo/subtitles/burnt-in text/channel branding), narrative (fits the narration's moment).
Set booleans: presenter_on_camera (a person facing/addressing camera), watermark_or_logo, text_overlay,
black_and_white, ai_or_animated. Put a short reason in notes.

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


def verify_clip(gem: Judge, clip: Path, shot: Shot, cache_id: str) -> dict | None:
    b = shot.brief
    prompt = PROMPT.format(text=shot.text, visual=b.visual, entities=", ".join(b.entities) or "-")
    try:
        v = gem.json_video_file(clip, prompt, VERIFY_SCHEMA, fps=2.0,
                                cache_key=("verify", cache_id, b.visual, ",".join(b.entities)))
    except Exception as e:
        log.warning("verify failed for %s: %s", clip.name, str(e)[:200])
        return None

    # `ai_or_animated` inside the 12-field verdict is unreliable in both directions: measured on
    # five known clips it called a real Tokyo skyline synthetic AND passed a generated Fuji clip.
    # A dedicated single-question check scored 4/4 on the same clips, so it decides this flag.
    dedicated = gem.confirm_ai_generated(clip, cache_key=("aicheck", cache_id))
    if dedicated is not None and dedicated != bool(v.get("ai_or_animated")):
        log.info("%s: AI flag %s -> %s by the dedicated check",
                 clip.name, bool(v.get("ai_or_animated")), dedicated)
        v["ai_or_animated"] = dedicated
        v["notes"] = (v.get("notes") or "") + f" [dedicated AI check: {dedicated}]"
        if dedicated:
            v["authenticity"] = min(float(v.get("authenticity", 100)), 30.0)
    return v
