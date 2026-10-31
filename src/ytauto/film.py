"""`film.json`: everything a Remotion film needs, built from a finished ytauto project.

Why: each film used to be a hand-written composition (a ~290-line component plus a data file with shot
cameras, chapters, stat cards, place chips and sound cues). The reusable `DocFilm` composition in
`D:\\remotion\\my-video\\src\\doc-kit` reads this one file instead (Remotion's documented way for a data-driven
film: `calculateMetadata` fetches JSON and sets the duration), so a new film needs no new code.

What is decided here, and how:
- shots (from `export-remotion`'s shots.json): a camera move and a transition per shot, from fixed rotations;
  a shot that opens a chapter gets the bigger "whip" transition. Deterministic, as before.
- chapters, stat cards, place chips, the tagline: one AI call over the narration (`extract_facts`). The AI
  only picks WHAT; every claim is checked against the narration here (a stat's spoken number must occur in its
  sentence, a place name must occur in the narration or title), and WHEN comes from the word timings, so a card
  appears as the number is spoken.
- sound: `sfx.plan` (hook, chapter whooshes, soft zoom whooshes, stat ticks, impacts), levelled against the voice.
"""
from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

from .log import log
from .models import Segment

# Fixed rotations (the same ones the Philippines film used), so a film's look stays consistent run to run.
ZOOMS = [(1.0, 1.12), (1.12, 1.0), (1.02, 1.14), (1.1, 1.02)]
PANS_X = [(0, -2), (-2, 2), (2, -2), (0, 0), (-3, 1)]
PANS_Y = [(0, 0), (1.5, -1.5), (-1.5, 1.5), (0, 0)]
TRANSITIONS = ["fade", "cut", "fade", "zoom", "fade", "cut", "dip", "fade"]
CHAPTER_SNAP = 0.9           # a shot starting this close to a chapter time opens that chapter
MIN_CHAPTER_GAP = 20.0
STAT_SECONDS, PLACE_SECONDS, TITLE_SECONDS = 5.0, 7.0, 5.1
STAT_MIN_GAP = 3.0           # a second card closer than this to the previous one is dropped
STAT_SIDE_GAP = 5.0          # closer than this (but >= STAT_MIN_GAP): the second card goes on the other side

FACTS_SCHEMA = {"type": "object", "properties": {
    "tagline": {"type": "string"},
    "chapters": {"type": "array", "items": {"type": "object", "properties": {"segment": {"type": "integer"}}, "required": ["segment"]}},
    "stats": {"type": "array", "items": {"type": "object", "properties": {
        "segment": {"type": "integer"}, "say": {"type": "string"}, "big": {"type": "string"},
        "unit": {"type": "string"}, "label": {"type": "string"}, "major": {"type": "boolean"}},
        "required": ["segment", "say", "big", "label"]}},
    "places": {"type": "array", "items": {"type": "object", "properties": {
        "segment": {"type": "integer"}, "name": {"type": "string"}, "sub": {"type": "string"}}, "required": ["segment", "name"]}},
}, "required": ["chapters", "stats", "places"]}

FACTS_PROMPT = """You are the graphics editor of a documentary YouTube film titled "{title}".
Below are its narration sentences, numbered. Decide where on-screen graphics go. Use ONLY what the narration says:
never add a fact, number or place that is not in the sentence you attach it to.

Return:
- chapters: the sentence numbers where the story moves to a new place or a new big idea (at most {max_chapters},
  at least 20 seconds apart, never sentence 0). Every place card below should sit on a chapter.
- stats: sentences that state a striking number worth a big on-screen card (at most one per 25 seconds).
    say   = the number exactly as it is written in the sentence (e.g. "7,641", "37%", "2,954").
    big   = the number to show, digits only with commas or a decimal point, at most 8 characters (e.g. "7,641").
    unit  = a short suffix shown next to it ("%", " km", " m", " days") or "" for none.
    label = what the number counts, UPPERCASE, at most 6 words (e.g. "ISLANDS", "OF THE WORLD'S CORAL SPECIES").
    major = true for the 3 most striking numbers of the film, else false.
- places: the first time each distinct named place or people is introduced (at most {max_places}).
    name = the place in UPPERCASE as the narration names it (e.g. "PALAWAN"); sub = at most 6 words of what it is,
    taken from the narration (e.g. "Puerto Princesa Underground River").
- tagline: five words or fewer that sum the film up, lowercase, no quotation marks.

Sentences:
{lines}
"""


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def extract_facts(gem, segments: list[Segment], title: str, seconds: float) -> dict:
    """The AI's picks, checked against the narration. {} when there is no judge or the call fails."""
    if gem is None:
        return {}
    lines = "\n".join(f"[{s.idx}] ({s.start:.0f}-{s.end:.0f}s) {s.text}" for s in segments)
    prompt = FACTS_PROMPT.format(title=title, lines=lines, max_chapters=max(3, int(seconds // 35)),
                                 max_places=max(3, int(seconds // 40)))
    try:
        raw = gem.json(prompt, FACTS_SCHEMA, cache_key=("film-facts-v1", title, lines))
    except Exception as e:
        log.warning("film facts unavailable (%s); the film gets no stat cards, place chips or chapters", str(e)[:120])
        return {}
    return check_facts(raw, segments, title)


def _spell(name: str, vocab: list[str]) -> str:
    """The place name written as the narration spells it ("TUBATAHA" -> "TUBBATAHA"), or "" when a word of it
    is nowhere in the narration (an invented place). The AI mis-spells rare names; the narration is the truth."""
    out = []
    for tok in re.findall(r"\w+", name):
        if tok.lower() in vocab:
            out.append(tok)
        else:
            near = difflib.get_close_matches(tok.lower(), vocab, 1, 0.8)
            if not near:
                return ""
            out.append(near[0])
    return " ".join(out).upper()


def check_facts(raw: dict, segments: list[Segment], title: str) -> dict:
    """Drop every pick the narration does not support."""
    by_idx = {s.idx: s for s in segments}
    stats = []
    for st in raw.get("stats") or []:
        seg = by_idx.get(st.get("segment"))
        say = str(st.get("say") or "").strip()
        if seg and say and _norm(say) and _norm(say) in _norm(seg.text) and len(str(st.get("big") or "")) <= 10:
            stats.append({**st, "say": say})
    vocab = sorted({w.lower() for w in re.findall(r"\w+", " ".join(s.text for s in segments) + " " + title)})
    places = []
    for p in raw.get("places") or []:
        name = _spell(str(p.get("name") or ""), vocab)
        if p.get("segment") in by_idx and name:
            places.append({**p, "name": name})
    picked = [c if isinstance(c, int) else (c or {}).get("segment") for c in raw.get("chapters") or []]
    chapters = sorted({c for c in picked if c in by_idx and c > 0})
    return {"tagline": re.sub(r"[\"“”]", "", str(raw.get("tagline") or "")).strip(), "chapters": chapters,
            "stats": stats, "places": places}


def _word_time(words: list[dict], seg: Segment, say: str) -> float:
    """When the number is spoken: the first word inside the sentence's span that starts like it
    (the sentence's own start when no word matches)."""
    toks = [_norm(x) for x in re.split(r"[\s\-–—/]+", say) if _norm(x)]
    first = toks[0] if toks else ""
    for w in words:
        if seg.start - 0.3 <= w["start"] <= seg.end + 0.3:
            n = _norm(w["text"])
            if n and first and (n == first or n.startswith(first) or (first.startswith(n) and len(n) >= 2)):
                return float(w["start"])
    return float(seg.start)


def place_cards(facts: dict, segments: list[Segment], words: list[dict], seconds: float) -> tuple[list[dict], list[dict], list[float]]:
    """(stats, places, chapter times) with their moments from the sentences and word timings."""
    by_idx = {s.idx: s for s in segments}
    chapters: list[float] = []
    for idx in facts.get("chapters", []):
        t = by_idx[idx].start
        if not chapters or t - chapters[-1] >= MIN_CHAPTER_GAP:
            chapters.append(round(t, 2))
    stats: list[dict] = []
    for st in sorted(facts.get("stats", []), key=lambda s: _word_time(words, by_idx[s["segment"]], s["say"])):
        at = _word_time(words, by_idx[st["segment"]], st["say"])
        if at + STAT_SECONDS > seconds:
            continue
        side = "left"
        if stats:
            gap = at - stats[-1]["at"]
            if gap < STAT_MIN_GAP:
                continue
            if gap < STAT_SIDE_GAP:
                if stats[-1]["side"] == "right":
                    continue                              # two cards are already on screen
                side = "right"
        stats.append({"at": round(at, 2), "to": round(at + STAT_SECONDS, 2), "big": str(st["big"]), "unit": st.get("unit") or "",
                      "label": str(st["label"]).upper(), "side": side, "major": bool(st.get("major"))})
    places: list[dict] = []
    for p in sorted(facts.get("places", []), key=lambda p: by_idx[p["segment"]].start):
        at = by_idx[p["segment"]].start + 0.25
        if places and at - places[-1]["at"] < PLACE_SECONDS:
            continue
        places.append({"at": round(at, 2), "to": round(min(at + PLACE_SECONDS, seconds), 2),
                       "name": str(p["name"]).upper(), "sub": str(p.get("sub") or "")})
    return stats, places, chapters


def style_shots(raw_shots: list[dict], chapters: list[float]) -> list[dict]:
    out = []
    for i, s in enumerate(raw_shots):
        big = any(abs(c - s["t0"]) < CHAPTER_SNAP for c in chapters)
        tr = "cut" if i == 0 else "whip" if big else TRANSITIONS[i % len(TRANSITIONS)]
        row = {**s, "tr": tr, "big": big, "z": list(ZOOMS[i % len(ZOOMS)]), "px": list(PANS_X[i % len(PANS_X)]),
               "py": list(PANS_Y[i % len(PANS_Y)])}
        if s.get("kind") == "still":
            row["z"] = [1.0, 1.0]            # a still already moves (stills.py): no second camera move on top
            row["px"] = row["py"] = [0, 0]
        out.append(row)
    return out


def sfx_events(shots: list[dict], stats: list[dict], title: dict | None, chapters: list[float]) -> list[dict]:
    ev = [{"kind": "hook", "at": 0.1}]
    ev += [{"kind": "chapter", "at": s["t0"]} for s in shots if s["big"]]
    ev += [{"kind": "zoom", "at": s["t0"]} for s in shots if s["tr"] == "zoom" and not s["big"]]
    ev += [{"kind": "statbig" if st["major"] else "stat", "at": st["at"]} for st in stats]
    if title:
        ev.append({"kind": "impact", "at": title["at"] - 0.1})
    return ev


def build(*, slug: str, title: str, segments: list[Segment], shots: list[dict], words: list[dict], seconds: float,
          facts: dict, fps: int = 30, tail: float = 1.6, next_video: str | None = None, show_title: bool = True,
          music: str | None = None) -> dict:
    stats, places, chapters = place_cards(facts, segments, words, seconds)
    title_card = None
    if show_title and title:
        at = round(chapters[0] if chapters else 3.0, 2)
        title_card = {"at": at, "to": round(at + TITLE_SECONDS, 2), "kicker": "A DOCUMENTARY", "name": title.upper(),
                      "tagline": facts.get("tagline", "")}
    if title_card:      # a chip for the film's own subject would sit on the title card and say the same thing
        places = [p for p in places if not (_norm(p["name"]) in _norm(title) or _norm(title) in _norm(p["name"]))
                  and not (title_card["at"] - 0.5 < p["at"] < title_card["to"])]
    end = {"at": round(max(seconds - 15.0, 0.0), 2), "kicker": "NEXT VIDEO", "title": next_video} if next_video else None
    styled = style_shots(shots, chapters)
    return {"version": 1, "slug": slug, "fps": fps, "seconds": round(seconds, 2), "tail": tail, "clipsDir": "clips",
            "vo": "vo.mp3", "music": music, "title": title_card, "end": end, "chapters": chapters, "stats": stats,
            "places": places, "shots": styled, "words": words, "sfx": None,
            "events": sfx_events(styled, stats, title_card, chapters)}


def write(film: dict, out: Path) -> Path:
    out.write_text(json.dumps(film, indent=1, ensure_ascii=False), encoding="utf-8")
    return out
