"""Turn transcript shots into visual briefs + search queries (AI judge, with offline fallback).

Planning is **per shot**, not per segment. A 10 s segment becomes two ~5 s shots, and if both
carry the same brief the selector fetches two clips of the same subject and the cut reads as a
stutter rather than an edit. Each shot therefore gets its own `visual`, varying the framing and
camera move while keeping the segment's subject, which is what the `variant` index is for.

The prompt also forbids briefs the pipeline cannot satisfy by construction: `inspect/scenes.py`
extracts a single cut-free window, so a brief asking for a montage, a sequence of quick cuts, or a
move between two different places can never be matched and burns a shot's whole retry budget
before falling back to stock.
"""
from __future__ import annotations

import re

from .cache import Cache
from .config import Settings
from .llm.router import Judge
from .log import log
from .models import Brief, Segment, Shot

_STOP = set("a an the and or but of in on at to for with from by is are was were be been it its this that these those "
            "number one two three four five top best which would you your we our they their there here as where "
            "never exist side visit first famous home thousands".split())

# Framing ladder used to differentiate shots split from the same segment. The subject stays the
# same; only the lens and the move change, so consecutive clips cut together instead of repeating.
_ANGLES = [
    "a wide establishing view of",
    "a closer detail view of",
    "a slow aerial or drone move over",
    "a low-angle, ground-level view of",
    "a static locked-off view of",
]

PLAN_SCHEMA = {
    "type": "object",
    "properties": {"shots": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "visual": {"type": "string"},
            "entities": {"type": "array", "items": {"type": "string"}},
            "youtube_queries": {"type": "array", "items": {"type": "string"}},
            "stock_queries": {"type": "array", "items": {"type": "string"}},
            "avoid": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["id", "visual", "entities", "youtube_queries", "stock_queries"]}},
        "topic": {"type": "string", "description": "the one filmable place/subject the video is about"}},
    "required": ["shots", "topic"],
}

PLAN_PROMPT = """You are a documentary B-roll researcher for a faceless YouTube video.
Overall video topic/title: {title}

Below is the shot list. Each line is ONE shot of about {dur} seconds that will be filled with a
single real-world clip. Shots sharing a segment number narrate the same sentence and must show the
same subject from DIFFERENT framings, never the same shot twice.

Return `topic`: the one filmable place or subject the whole video is about (e.g. "Japan"), used for
shots whose narration names nothing (an intro, a closing question).

Return one entry per shot in `shots` (keep id):
- visual: one sentence describing the ideal shot (subject, setting, camera style).
- entities: the 1-4 specific named places/objects/things that must appear, the MAIN SUBJECT FIRST
  (for a takoyaki close-up in Dotonbori: ["Takoyaki", "Dotonbori"], not ["Dotonbori", ...]).
- youtube_queries: {nq} distinct YouTube search queries likely to surface high-quality real footage with no presenter
  (prefer wording such as "4K", "drone", "walking tour", "cinematic b-roll", "timelapse"; mix wide and detail angles).
  At least half of them must name the main subject itself, not only the area around it.
- stock_queries: 2 short generic queries (2-4 words) for stock sites.
- avoid: things that would ruin the shot (e.g. "vlogger talking to camera", "map graphic", "cartoon").

HARD RULES for `visual` - a brief breaking any of these cannot be filled and wastes the shot:
1. ONE continuous shot from ONE camera. Never describe a montage, a series, a sequence, quick cuts,
   or "various" shots.
2. ONE place. Never describe moving, transitioning, cutting or travelling from one location or
   landmark to another - such footage does not exist as a single clip.
3. Something a real camera has actually filmed. No graphics, maps, text, split screens or collages.
4. Shots sharing a segment number must differ from each other in framing, lens or camera move
   (for example wide establishing, then a closer detail, then an aerial), not in subject.
5. Keep it findable: ONE main subject plus AT MOST ONE extra condition (a time of day, OR the
   weather, OR one camera move). Do not stack details such as "close-up, slow-motion, mist rising,
   reflection, at dawn" - a real 5-second clip almost never has all of them at once, and every
   extra condition is another reason to reject good footage.

Shot list:
{shots}
"""


# Title words that describe the *video*, not anything a camera could film.
_GENERIC = set("destinations destination places place things reasons ways tips facts guide ultimate "
               "amazing beautiful full validation test video part episode list ranking ranked "
               "must see visit travel trip tour vlog do does why how what when who should need know "
               "go unreal incredible insane hidden secret perfect".split())


def topic_from_title(title: str) -> str:
    """Offline guess at the filmable subject of a video from its title: "5 Best Destinations in
    Japan" -> "Japan". The video title itself is never a usable search subject - measured on the
    12-shot run, the closing beat searched for "5 Best Destinations in Japan - Full Validation 4k"."""
    # The first run of consecutive capitalised, non-generic words is the name: "Top 10 Things to
    # Do in Kyoto" -> "Kyoto", "Why Iceland Is Unreal" -> "Iceland", "Mount Fuji Guide" -> "Mount Fuji".
    runs, cur = [], []
    for w in re.findall(r"[A-Za-z][A-Za-z'-]*|\S", title):
        ok = w[0].isupper() and w.lower() not in _STOP and w.lower() not in _GENERIC
        if ok:
            cur.append(w)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return " ".join(runs[0]) if runs else ""


def _fallback_brief(shot: Shot, seg: Segment, title: str, nq: int, topic: str = "") -> Brief:
    """No-LLM heuristic: capitalised phrases are entities; build simple queries.

    `shot.variant` picks a framing from `_ANGLES` and rotates the query suffixes, so shots split
    from one segment do not come out identical here either. A beat that names nothing ("Which
    would you visit first?") falls back to the video's `topic`, never to its title.
    """
    caps = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", seg.text)
    ents = [c for c in caps if c.lower() not in _STOP][:3]
    words = [w for w in re.findall(r"[A-Za-z]+", seg.text) if w.lower() not in _STOP and len(w) > 3]
    topic = topic or topic_from_title(title)
    if not ents and topic:
        ents = [topic]
    core = " ".join(ents) if ents else " ".join(words[:3]) or topic or title
    suffix = ["4k", "drone footage", "cinematic b-roll", "walking tour 4k", "timelapse"]
    k = shot.variant % len(_ANGLES)
    rot = suffix[k:] + suffix[:k]
    return Brief(visual=f"{_ANGLES[k].capitalize()} {core}.",
                 entities=ents or words[:2],
                 youtube_queries=[f"{core} {s}" for s in rot[:nq]],
                 stock_queries=[core, " ".join(words[:2]) or core],
                 avoid=["person talking to camera", "text overlay"])


# Wording that describes footage no single clip can contain. `inspect/scenes.py` extracts ONE
# cut-free window, so a brief asking for several shots, or for a move between two places, can never
# be matched: measured 2026-10-29 such a brief burned 24 candidates and dropped the shot to stock.
# Checked against `visual` only, and kept narrow - these must not fire on legitimate camera moves
# ("a drone shot gliding over", "a pan across") which are single continuous takes.
_IMPOSSIBLE = [
    (re.compile(r"\bmontage\b", re.I), "a montage"),
    (re.compile(r"\bquick cuts?\b", re.I), "quick cuts"),
    (re.compile(r"\b(?:series|sequence|succession|collage|compilation)\s+of\b", re.I), "a series of shots"),
    (re.compile(r"\bvarious\s+(?:shots|clips|scenes|views)\b", re.I), "various shots"),
    (re.compile(r"\bsplit[- ]screen\b", re.I), "a split screen"),
    # "transitions from X to Y", "cuts to", "travelling from ... to ..." between two places
    (re.compile(r"\btransition(?:s|ing)?\s+(?:from|to|into)\b", re.I), "a transition between places"),
    (re.compile(r"\bcut(?:s|ting)?\s+(?:to|between)\b", re.I), "a cut to another shot"),
    (re.compile(r"\b(?:graphic|text overlay|title card|map)\b", re.I), "a graphic rather than footage"),
]


def _impossible_reason(visual: str) -> str | None:
    """Name the first reason `visual` cannot be filled by one continuous clip, else None."""
    for rx, why in _IMPOSSIBLE:
        if rx.search(visual):
            return why
    return None


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2}


def primary_subject(b: Brief) -> str:
    """The entity the shot is actually *of*: the one named earliest in `visual`.

    The planner is asked to list the main subject first but does not reliably do so - on the
    12-shot run a takoyaki close-up came back as ["Dotonbori", "Osaka", "Takoyaki"]."""
    v = b.visual.lower()
    best, pos = None, len(v) + 1
    for e in b.entities:
        toks = [t for t in re.findall(r"[a-z0-9]+", e.lower()) if len(t) > 2]
        hits = [v.find(t) for t in toks if t in v]
        if toks and len(hits) == len(toks) and min(hits) < pos:
            best, pos = e, min(hits)
    return best or (b.entities[0] if b.entities else "")


def _align_queries(shots: list[Shot], briefs: dict[int, Brief], nq: int) -> list[str]:
    """Make at least half of each shot's YouTube queries name its main subject.

    Measured on the 12-shot run: the takoyaki close-up's queries were mostly about the Dotonbori
    canal, so YouTube returned long canal walking tours with no takoyaki in them, all 8 were
    rejected and the shot fell to stock. Queries about the area find the area, not the subject.
    """
    fixed = []
    for sh in shots:
        b = briefs.get(sh.id)
        subject = primary_subject(b) if b else ""
        # any significant word of the name counts: "Kyoto bamboo forest" is on "Arashiyama Bamboo
        # Grove"; requiring every word re-aimed shots that had succeeded
        st = {t for t in _tokens(subject) if len(t) >= 4}
        if not b or not st or not b.youtube_queries:
            continue
        on = [q for q in b.youtube_queries if st & _tokens(q)]
        need = (len(b.youtube_queries) + 1) // 2
        if len(on) >= need:
            continue
        extra = [f"{subject} 4k", f"{subject} cinematic b-roll", f"{subject} close up 4k"]
        extra = [q for q in extra if q.lower() not in {x.lower() for x in b.youtube_queries}]
        add = extra[: need - len(on)]
        # subject-first: the selector screens the first queries first
        off = [q for q in b.youtube_queries if q not in on]
        qs = (on + add + off)[: max(nq, len(on) + len(add))]
        briefs[sh.id] = b.model_copy(update={"youtube_queries": qs})
        fixed.append(f"shot {sh.id} ({subject})")
    return fixed


def _reject_impossible(shots: list[Shot], briefs: dict[int, Brief], segments: dict[int, Segment],
                       title: str, nq: int, topic: str = "") -> list[str]:
    """Replace briefs that no single clip can satisfy with the heuristic brief for that shot.

    The prompt forbids these, but the planner is a text model and still produced "a montage of
    quick, exciting clips ... ending with a question mark graphic" on a live call. A brief like
    that cannot be screened successfully by construction, so it is cheaper to discard it here than
    to spend a shot's whole retry budget discovering the same thing from the judge.
    """
    replaced = []
    for sh in shots:
        b = briefs.get(sh.id)
        if b is None:
            continue
        why = _impossible_reason(b.visual)
        if not why:
            continue
        # Keep the planner's research (entities and queries are still useful); replace the shot
        # description only, since that is what screening and verification are judged against.
        # Prefer its own entities as the subject: a narration-only beat ("Which would you visit
        # first?") yields no entity from the text, and the heuristic would otherwise fall back to
        # the video title, which is not a filmable subject.
        angle = _ANGLES[sh.variant % len(_ANGLES)]
        # ONE subject: joining two entities rebuilt the very two-place brief being replaced
        # (measured: "a wide establishing view of Japanese garden, Tokyo street").
        subject = primary_subject(b) or topic
        if subject:
            visual = f"{angle.capitalize()} {subject}."
        else:
            visual = _fallback_brief(sh, segments[sh.segment], title, nq, topic).visual
        briefs[sh.id] = b.model_copy(update={"visual": visual})
        replaced.append(f"shot {sh.id} ({why})")
    return replaced


def _differentiate(shots: list[Shot], briefs: dict[int, Brief]) -> list[str]:
    """Repair shots of one segment that still share a `visual`, naming the groups repaired.

    The model is told not to do this, but it is a text model and sometimes does it anyway. Rather
    than let two identical briefs reach the selector, prefix each repeat with its framing from
    `_ANGLES` so the screening prompt, the search queries and the cache key all differ.
    """
    groups: dict[tuple[int, str], list[Shot]] = {}
    for sh in shots:
        b = briefs.get(sh.id)
        if b is not None:
            groups.setdefault((sh.segment, b.visual.strip().lower()), []).append(sh)
    repaired = []
    for (seg_idx, _), group in sorted(groups.items(), key=lambda kv: kv[1][0].id):
        if len(group) < 2:
            continue
        repaired.append(f"{'/'.join(str(x.id) for x in group)} (segment {seg_idx})")
        for sh in group[1:]:
            b = briefs[sh.id]
            angle = _ANGLES[sh.variant % len(_ANGLES)]
            briefs[sh.id] = b.model_copy(update={"visual": f"{angle.capitalize()} {b.visual}"})
    return repaired


def plan(segments: list[Segment], shots: list[Shot], title: str, s: Settings, cache: Cache,
         gem: Judge | None) -> dict[int, Brief]:
    """Return a brief per **shot id** (not per segment id)."""
    briefs: dict[int, Brief] = {}
    by_seg = {seg.idx: seg for seg in segments}
    topic = ""
    if gem:
        payload = "\n".join(
            f"[{sh.id}] segment {sh.segment}, shot {sh.variant + 1} of this segment, "
            f"{sh.duration:.1f}s: {sh.text}" for sh in shots)
        prompt = PLAN_PROMPT.format(title=title, nq=s.queries_per_segment, shots=payload,
                                    dur=f"{s.shot_target:.0f}")
        try:
            data = gem.json(prompt, PLAN_SCHEMA,
                            cache_key=("plan-shots-v2", title, payload, s.queries_per_segment))
            topic = str(data.get("topic") or "").strip()
            for d in data.get("shots") or []:
                if isinstance(d.get("id"), int):
                    briefs[d["id"]] = Brief(visual=d.get("visual", ""), entities=d.get("entities", []),
                                            youtube_queries=d.get("youtube_queries", []),
                                            stock_queries=d.get("stock_queries", []),
                                            avoid=d.get("avoid", []))
        except Exception as e:
            log.warning("AI planning failed (%s); using heuristic briefs", e)
    topic = topic or topic_from_title(title)
    for sh in shots:
        b = briefs.get(sh.id)
        if b is None or not b.youtube_queries or not b.visual.strip():
            briefs[sh.id] = _fallback_brief(sh, by_seg[sh.segment], title, s.queries_per_segment, topic)
    bad = _reject_impossible(shots, briefs, by_seg, title, s.queries_per_segment, topic)
    if bad:
        log.warning("planner asked for footage that cannot exist as one clip: %s; using a plain "
                    "single-shot brief instead", "; ".join(bad))
    aligned = _align_queries(shots, briefs, s.queries_per_segment)
    if aligned:
        log.info("search queries re-aimed at the shot's main subject: %s", "; ".join(aligned))
    dupes = _differentiate(shots, briefs)
    if dupes:
        log.warning("planner gave shots %s the same brief; differentiated them by framing",
                    ", ".join(dupes))
    return briefs
