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
from .llm.router import Judge, QuotaExhausted
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
            "subject": {"type": "string", "description": "the ONE filmable thing the clip must show"},
            "visual": {"type": "string"},
            "entities": {"type": "array", "items": {"type": "string"}},
            "youtube_queries": {"type": "array", "items": {"type": "string"}},
            "stock_queries": {"type": "array", "items": {"type": "string"}},
            "avoid": {"type": "array", "items": {"type": "string"}},
            "generic": {"type": "boolean", "description": "ordinary b-roll any stock library has; names nothing specific"},
        },
        "required": ["id", "subject", "visual", "entities", "youtube_queries", "stock_queries"]}},
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
- subject: the ONE filmable thing the clip must show, 2-6 words (e.g. "Fushimi Inari torii gates",
  "Mount Fuji", "takoyaki"). It must be named in, or clearly meant by, the narration (or be the video
  topic when the narration names nothing). A clip showing it is usable even if the framing differs.
- visual: one sentence describing the ideal shot of that subject (setting, camera style). This is a
  preference, not a requirement.
- entities: the 1-4 specific named places/objects/things that must appear, the MAIN SUBJECT FIRST
  (for a takoyaki close-up in Dotonbori: ["Takoyaki", "Dotonbori"], not ["Dotonbori", ...]).
- youtube_queries: {nq} distinct YouTube search queries likely to surface high-quality real footage with no presenter
  (prefer wording such as "4K", "drone", "walking tour", "cinematic b-roll", "timelapse"; mix wide and detail angles).
  At least half of them must name the main subject itself, not only the area around it.
- stock_queries: 2 short generic queries (2-4 words) for stock sites.
- avoid: things that would ruin the shot (e.g. "vlogger talking to camera", "map graphic", "cartoon").
- generic: true ONLY when the subject is ordinary b-roll that any stock-footage library has (open sea,
  waves, aerial coastline, sunset, forest, rain, a crowd walking, street food being cooked) AND the
  narration names no specific place, person, event, landmark or object. False whenever the shot
  must show something named (a town, a tribe, a volcano, a person, a dish, a festival).

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


CRITIC_SCHEMA = {
    "type": "object",
    "properties": {"shots": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "subject": {"type": "string"},
                       "visual": {"type": "string"},
                       "entities": {"type": "array", "items": {"type": "string"}}},
        "required": ["id", "subject", "visual", "entities"]}}},
    "required": ["shots"],
}

CRITIC_PROMPT = """You are checking B-roll briefs for a faceless YouTube video before anyone searches
for footage. Video topic: {topic}

For each shot below you get the narration and the draft brief. Fix every brief so it can be filled by
ONE real, ordinary 5-second clip of stock-style footage:
1. Delete any person, object, vehicle, activity or detail the narration does not mention or clearly
   imply. Researched names of what the narration describes are fine (narration "thousands of red
   gates" -> "Fushimi Inari"); invented extras are not (a "snow groomer" for a ski-resort line).
2. subject: the ONE filmable thing, 2-6 words.
3. visual: that subject plus AT MOST ONE extra condition (a time of day, OR the weather, OR one camera
   move). Remove stacked details.
4. entities: 1-4 names, the subject first; only things the clip must show.
5. Shots of the same segment must keep different framings (wide, closer, aerial ...).
If a brief is already fine, return it unchanged. Return every shot, keep the ids.

{shots}
"""


def _critic(gem: Judge, batch: list[Shot], briefs: dict[int, Brief], topic: str, cache_key: tuple) -> int:
    """Second pass over one planner batch that strips details the narration does not support.

    The plan prompt already forbids stacked and invented details, and the model still produced
    "a snow groomer at dawn" for a line that never mentions one, and a pagoda *and* a skyline in
    one shot (speed test, 2026-10-29). A reviewer pass costs one call per ~40 shots. Only the
    subject/visual/entities are taken from it; the search queries stay the planner's.
    Returns the number of briefs it changed.
    """
    lines = []
    for sh in batch:
        b = briefs.get(sh.id)
        if b is None:
            continue
        lines.append(f'[{sh.id}] segment {sh.segment} | narration: "{sh.text}" | subject: {b.subject} | '
                     f'visual: {b.visual} | entities: {", ".join(b.entities)}')
    if not lines:
        return 0
    try:
        data = gem.json(CRITIC_PROMPT.format(topic=topic or "-", shots="\n".join(lines)), CRITIC_SCHEMA,
                        cache_key=("plan-critic-v1",) + cache_key)
    except QuotaExhausted:
        raise
    except Exception as e:
        log.warning("brief review failed (%s); keeping the planner's briefs", e)
        return 0
    changed, ids = 0, {sh.id for sh in batch}
    for d in data.get("shots") or []:
        if not isinstance(d, dict) or d.get("id") not in ids or d["id"] not in briefs:
            continue
        b = briefs[d["id"]]
        upd = {}
        for k in ("subject", "visual"):
            v = str(d.get(k) or "").strip()
            if v and v != getattr(b, k):
                upd[k] = v
        ents = [str(e).strip() for e in d.get("entities") or [] if str(e).strip()]
        if ents and ents != b.entities:
            upd["entities"] = ents[:4]
        if upd:
            briefs[d["id"]] = b.model_copy(update=upd)
            changed += 1
    return changed


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


PLAN_CHUNK = 40     # shots per planner call


def chunk_shots(shots: list[Shot], size: int = PLAN_CHUNK) -> list[list[Shot]]:
    """Split the shot list into planner batches of about `size`, never splitting a segment.

    One call for the whole list does not scale: a 25-minute video is ~300 shots, ~45k tokens of
    briefs, past the output limit of the free models - the reply is truncated and every shot
    after the cut silently gets a heuristic brief. Shots of one segment must stay together so
    the model can vary their framing (`_differentiate` only repairs what it misses)."""
    out: list[list[Shot]] = []
    for sh in shots:
        if out and len(out[-1]) >= size and out[-1][-1].segment != sh.segment:
            out.append([])
        if not out:
            out.append([])
        out[-1].append(sh)
    return out


def plan(segments: list[Segment], shots: list[Shot], title: str, s: Settings, cache: Cache,
         gem: Judge | None) -> dict[int, Brief]:
    """Return a brief per **shot id** (not per segment id)."""
    briefs: dict[int, Brief] = {}
    by_seg = {seg.idx: seg for seg in segments}
    topic = ""
    for batch in (chunk_shots(shots) if gem else []):
        payload = "\n".join(
            f"[{sh.id}] segment {sh.segment}, shot {sh.variant + 1} of this segment, "
            f"{sh.duration:.1f}s: {sh.text}" for sh in batch)
        prompt = PLAN_PROMPT.format(title=title, nq=s.queries_per_segment, shots=payload,
                                    dur=f"{s.shot_target:g}")
        ids = {sh.id for sh in batch}
        try:
            key = (title, payload, s.queries_per_segment)
            data = gem.json(prompt, PLAN_SCHEMA, cache_key=("plan-shots-v4",) + key)
            topic = topic or str(data.get("topic") or "").strip()
            for d in data.get("shots") or []:
                if isinstance(d.get("id"), int) and d["id"] in ids:
                    briefs[d["id"]] = Brief(subject=str(d.get("subject") or "").strip(),
                                            visual=d.get("visual", ""), entities=d.get("entities", []),
                                            youtube_queries=d.get("youtube_queries", []),
                                            stock_queries=d.get("stock_queries", []),
                                            avoid=d.get("avoid", []), generic=bool(d.get("generic", False)))
            if s.plan_critic:
                n = _critic(gem, batch, briefs, topic, key)
                if n:
                    log.info("brief review simplified %d of %d briefs (shots %d-%d)",
                             n, len(batch), batch[0].id, batch[-1].id)
        except QuotaExhausted:
            raise                 # heuristic briefs would quietly lower the whole video's quality
        except Exception as e:
            log.warning("AI planning failed for shots %d-%d (%s); using heuristic briefs for them",
                        batch[0].id, batch[-1].id, e)
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
    for sh in shots:                     # heuristic and older cached briefs have no subject
        b = briefs[sh.id]
        if not b.subject.strip():
            briefs[sh.id] = b.model_copy(update={"subject": primary_subject(b) or topic})
    for sh in shots:
        briefs[sh.id] = with_setting(briefs[sh.id], topic, sh.text)
    return briefs


def _proper(entity: str, narration: str) -> bool:
    """A named place or thing ("Hakuba", "Mount Fuji"), not a generic noun ("House")."""
    caps = [w for w in entity.split() if w[:1].isupper()]
    return bool(caps) and (entity in narration or len(caps) >= 2)


def with_setting(b: Brief, topic: str, narration: str = "") -> Brief:
    """Pin the video's setting on a brief, and add it to search queries that name no place:
    "House 4k" found houses anywhere on earth for an Alaska video. A query that names the
    setting or one of the brief's named places ("hakuba ski 4k") is left as the planner wrote it."""
    topic = (topic or "").strip()
    if not topic:
        return b
    words = {w.lower()[:5] for w in re.findall(r"[A-Za-z]{4,}", topic)}   # "Alaska" ~ "Alaskan"
    places = [e.lower() for e in b.entities if _proper(e, narration)]

    def placed(q: str) -> bool:
        low = q.lower()
        return any(w in low for w in words) or any(p in low for p in places)

    queries = [q if placed(q) else f"{topic} {q}" for q in b.youtube_queries]
    return b.model_copy(update={"setting": topic, "youtube_queries": queries})
