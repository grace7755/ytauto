"""Pre-download screening: the judge watches a public YouTube URL and proposes matching windows."""
from __future__ import annotations

from .llm.router import Judge, JudgeError
from .log import log
from .models import Candidate, Shot
from .sources.storyboard import Storyboard
from .timing import stage

SCREEN_SCHEMA = {
    "type": "object",
    "properties": {
        "usable": {"type": "boolean"},
        "reason": {"type": "string"},
        "windows": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "start": {"type": "string", "description": "MM:SS"},
                "end": {"type": "string", "description": "MM:SS"},
                "match": {"type": "integer", "description": "0-100 how well it shows the requested shot"},
                "description": {"type": "string"}},
            "required": ["start", "end", "match"]}},
        "presenter_on_camera": {"type": "boolean"},
        "watermark_or_logo": {"type": "boolean"},
        "black_and_white": {"type": "boolean"},
        "ai_or_animated": {"type": "boolean"},
        "looks_dated": {"type": "boolean"},
    },
    "required": ["usable", "windows", "presenter_on_camera", "watermark_or_logo", "black_and_white",
                 "ai_or_animated", "looks_dated"],
}

PROMPT = """You are screening a YouTube video for REAL, high-quality B-roll for a faceless documentary video.
Must show (required): {subject}
Ideal shot (a preference - do not reject footage of the subject for a different framing): {visual}
Names involved: {entities}
Narration it will accompany: "{text}"
Avoid: {avoid}

Find up to 3 windows (each 5-12 seconds, continuous, no hard cuts inside) that show the wanted shot clearly and
look sharp, stable, well-lit and colourful.
Each window MUST use exactly the keys "start", "end" and "match": "start"/"end" as MM:SS timestamps measured from
the start of the video (not start_time/end_time, not seconds), and "match" as an integer 0-100.
If usable is true you MUST return at least one window; if you cannot give timestamps, set usable to false.
Set the flags for the video as a whole:
presenter_on_camera (a person addresses the camera / talking head dominates), watermark_or_logo (persistent channel
logo or burned-in watermark), black_and_white, ai_or_animated (AI-generated, CGI, cartoon, game or slideshow),
looks_dated (visibly old footage/low-fi look for a topic that should look modern).
If the video has nothing usable, set usable=false and windows=[].
"""


def mmss(t: str) -> float:
    parts = [float(p) for p in str(t).strip().split(":")]
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return sec


def screen_url(gem: Judge, c: Candidate, shot: Shot, fps: float = 1.0) -> dict | None:
    """The judge watches the YouTube URL. Return the screening dict (windows converted to
    seconds) or None on failure.

    Raises `JudgeError` when the whole judge chain is exhausted. That is an outage on our side,
    not a verdict about this candidate, and the caller must not record it as a footage reject or
    it will keep burning candidates and fall through to stock for the wrong reason.
    """
    b = shot.brief
    prompt = PROMPT.format(subject=b.subject or b.visual, visual=b.visual,
                           entities=b.names(), text=shot.text,
                           avoid=", ".join(b.avoid) or "-")
    # long videos: look at the first 20 min only (cost control); chapters could narrow this later
    end = min(c.duration, 1200) if c.duration > 1200 else None
    try:
        data = gem.json_youtube(c.url, prompt, SCREEN_SCHEMA, end=end, fps=fps,
                                cache_key=("screen-v2", c.source_id, b.subject, b.visual, b.key()), task="screen")
    except JudgeError:
        raise                     # chain exhausted: our outage, not this candidate's fault
    except Exception as e:
        log.warning("screening failed for %s: %s", c.url, str(e)[:200])
        return None
    wins = []
    for w in data.get("windows", []):
        if not isinstance(w, dict):
            continue
        # Key names vary by provider (no enforced schema on OpenAI-compatible endpoints):
        # start/end, start_time/end_time, from/to have all been observed.
        raw_a = next((w[k] for k in ("start", "start_time", "from", "begin") if w.get(k) is not None), None)
        raw_z = next((w[k] for k in ("end", "end_time", "to", "finish") if w.get(k) is not None), None)
        if raw_a is None or raw_z is None:
            log.debug("screen: unparseable window %s", w)
            continue
        try:
            a, z = mmss(raw_a), mmss(raw_z)       # mmss handles MM:SS and HH:MM:SS
        except (ValueError, TypeError):
            log.debug("screen: unparseable timestamps %s", w)
            continue
        if z - a >= 3 and (c.duration == 0 or z <= c.duration + 1):
            try:
                match = int(float(w.get("match", 0) or 0))
            except (ValueError, TypeError):
                match = 0
            wins.append({"start": a, "end": z, "match": match, "description": w.get("description", "")})
    if data.get("usable") and not wins and data.get("windows"):
        log.warning("screening for %s returned %d window(s), none usable (unparseable, shorter "
                    "than 3 s, or past the end): %s", c.url, len(data["windows"]), str(data["windows"])[:200])
    data["windows"] = sorted(wins, key=lambda w: -w["match"])
    return data


SB_SCHEMA = {
    "type": "object",
    "properties": {
        "usable": {"type": "boolean"},
        "reason": {"type": "string"},
        "tiles": {"type": "array", "items": {"type": "integer"}},
        "match": {"type": "integer", "description": "0-100 for the best tile"},
        "overlay_tiles": {"type": "array", "items": {"type": "integer"},
                          "description": "thumbnails among tiles with text, captions, a title or a logo drawn over the footage"},
        "presenter_on_camera": {"type": "boolean"},
        "black_and_white": {"type": "boolean"},
    },
    "required": ["usable", "tiles", "match", "overlay_tiles", "presenter_on_camera", "black_and_white"],
}

SB_PROMPT = """This image is a contact sheet of ONE YouTube video: {n} numbered thumbnails taken in time
order (left-to-right, then top-to-bottom), about {step:.0f} seconds apart.

We need ONE continuous ~5 second shot of REAL camera footage for a faceless documentary:
- must show (required): {subject}
- ideal shot (a preference, use it to rank tiles; never a reason to reject the subject): {visual}
- names involved: {entities}
- narration it will accompany: "{text}"
- avoid: {avoid}

Return:
- usable: true if some thumbnails clearly show the required subject as real camera footage
- tiles: numbers of the thumbnails that show the subject, closest to the ideal shot first, at most 5
  ([] if none)
- match: 0-100, how well the best of those thumbnails matches the ideal shot of the subject
- overlay_tiles: those of your tiles with text, subtitles, a title or a channel logo drawn over the
  footage ([] if none; signs and shop names that are part of the scene do not count)
- presenter_on_camera: true if, in those thumbnails, a person is talking to the camera
- black_and_white: true if that footage is black and white
- reason: one sentence
Judge the subject and setting; small details may be unreadable at this size and will be checked later
on the full-resolution clip. Title cards, maps and text slides are not usable footage."""


def _sb_cache_key(c: Candidate, shot: Shot) -> tuple:
    b = shot.brief
    return ("screen-sb2", c.source_id, b.subject, b.visual, b.key())


def _tiles(raw) -> list[int]:
    out = []
    for x in raw or []:
        try:
            out.append(int(x))
        except (ValueError, TypeError):
            continue
    return out


TILE_FACE_MIN = 24     # px: a presenter-sized face (12% of the frame) is ~60 px wide in a 240x135 tile


def _face_check(gem) -> float | None:
    """`max_face_ratio` when `storyboard_face_check` is on, else None."""
    s = getattr(gem, "s", None)
    return s.max_face_ratio if s is not None and getattr(s, "storyboard_face_check", False) else None


def _tile_face(sb: Storyboard, tile: int, limit: float) -> bool:
    """Does this thumbnail show a presenter-sized face? Free (local Haar on the cached sheet)."""
    from .inspect.checks import largest_face_ratio

    im = sb.tile(tile)
    return im is not None and largest_face_ratio([im], TILE_FACE_MIN) > limit


def _sb_result(data: dict, sb: Storyboard, face_limit: float | None = None) -> dict:
    """Map a storyboard verdict (tile numbers) to the screening result shape (windows in seconds).

    `overlay_tiles` is a soft signal only: thumbnails are ~160 px wide, too small to rule on a
    watermark (verify and the benchmark do that on the full-resolution clip). It only reorders -
    clean tiles are downloaded first, and a video whose every tile is flagged is `overlay_suspect`,
    tried after the round's clean candidates. Measured: 12 of 30 downloads in a 12-shot run were
    rejected for a watermark or burnt-in text that screening never looked for.

    `face_limit` (`storyboard_face_check`): a tile with a presenter-sized face is tried last and a
    video whose every tile has one is `face_suspect`, held back like `overlay_suspect`. Also only a
    reorder: the face check on the downloaded clip and the judge's `presenter_on_camera` decide."""
    try:
        best = int(float(data.get("match", 0) or 0))
    except (ValueError, TypeError):
        best = 0
    flagged = set(_tiles(data.get("overlay_tiles")))
    wins, seen = [], set()
    for k, tile in enumerate(_tiles(data.get("tiles"))):
        win = sb.window(tile)
        if win is None or tile in seen or win[1] - win[0] < 3:
            continue
        seen.add(tile)
        # keep the judge's best-first order; later tiles rank a little lower
        wins.append({"start": win[0], "end": win[1], "match": max(0, best - 3 * k),
                     "description": f"storyboard tile {tile}", "overlay": tile in flagged,
                     "face": face_limit is not None and _tile_face(sb, tile, face_limit)})
    wins.sort(key=lambda w: (w["overlay"], w["face"]))   # stable: clean tiles first, judge order kept
    return {"usable": bool(data.get("usable")), "reason": data.get("reason", ""), "windows": wins,
            "overlay_suspect": bool(wins) and all(w["overlay"] for w in wins),
            "face_suspect": bool(wins) and all(w["face"] for w in wins),
            "presenter_on_camera": bool(data.get("presenter_on_camera")),
            "black_and_white": bool(data.get("black_and_white")),
            "ai_or_animated": False, "watermark_or_logo": False, "looks_dated": False,
            "via": "storyboard"}


BATCH_SCHEMA = {
    "type": "object",
    "properties": {"videos": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "video": {"type": "string", "description": "the image letter, A, B, ..."},
            "usable": {"type": "boolean"},
            "tiles": {"type": "array", "items": {"type": "integer"}},
            "match": {"type": "integer", "description": "0-100 for the best tile"},
            "overlay_tiles": {"type": "array", "items": {"type": "integer"},
                              "description": "thumbnails among tiles with text, captions, a title or a logo drawn over the footage"},
            "presenter_on_camera": {"type": "boolean"},
            "black_and_white": {"type": "boolean"},
            "reason": {"type": "string"}},
        "required": ["video", "usable", "tiles", "match", "overlay_tiles", "presenter_on_camera",
                     "black_and_white"]}}},
    "required": ["videos"],
}

BATCH_PROMPT = """You get {k} images, A to {last}. Each image is a contact sheet of a DIFFERENT YouTube
video: numbered thumbnails in time order (left-to-right, then top-to-bottom). Judge each video on its
own; the videos have nothing to do with each other.

We need ONE continuous ~5 second shot of REAL camera footage for a faceless documentary:
- must show (required): {subject}
- ideal shot (a preference, use it to rank tiles; never a reason to reject the subject): {visual}
- names involved: {entities}
- narration it will accompany: "{text}"
- avoid: {avoid}

Return `videos`: one entry per image, in order A, B, ..., each with:
- video: the image letter
- usable: true if some thumbnails of THAT image clearly show the required subject as real camera footage
- tiles: that image's thumbnail numbers that show the subject, closest to the ideal shot first, at
  most 5 ([] if none)
- match: 0-100, how well that image's best thumbnail matches the ideal shot of the subject
- overlay_tiles: those of that image's tiles with text, subtitles, a title or a channel logo drawn
  over the footage ([] if none; signs and shop names that are part of the scene do not count)
- presenter_on_camera: true if, in those thumbnails, a person is talking to the camera
- black_and_white: true if that footage is black and white
- reason: one sentence
Judge the subject and setting; small details may be unreadable at this size and will be checked later
on the full-resolution clip. Title cards, maps and text slides are not usable footage."""


def screen_storyboards(gem: Judge, cands: list[Candidate], shot: Shot, sbs: list[Storyboard]) -> list[dict | None]:
    """Screen several candidates' storyboards in ONE judge call; results in input order, each in
    the `screen_storyboard` shape (None = no usable verdict for that one).

    Free links cap requests per day, not tokens, so one call for k videos costs a k-th of the
    quota. Verdicts are cached per (video, brief), the same key a single screen uses, so a batch
    and a single screen of the same pair never pay twice. Raises `JudgeError` on chain exhaustion.
    """
    if len(cands) == 1:
        return [screen_storyboard(gem, cands[0], shot, sbs[0])]
    b = shot.brief
    letters = [chr(ord("A") + i) for i in range(len(cands))]
    out: list[dict | None] = [None] * len(cands)
    todo = []
    for i, (c, sb) in enumerate(zip(cands, sbs)):
        hit = gem.cache.get("judge", *_sb_cache_key(c, shot)) if getattr(gem, "cache", None) else None
        if hit is not None:
            out[i] = _sb_result(hit, sb, _face_check(gem))
        else:
            todo.append(i)
    if len(todo) == 1:
        i = todo[0]
        out[i] = screen_storyboard(gem, cands[i], shot, sbs[i])
        return out
    if not todo:
        return out
    prompt = BATCH_PROMPT.format(k=len(todo), last=letters[len(todo) - 1], subject=b.subject or b.visual,
                                 visual=b.visual, entities=b.names(), text=shot.text,
                                 avoid=", ".join(b.avoid) or "-")
    try:
        data = gem.json_images(prompt, [sbs[i].jpeg for i in todo], BATCH_SCHEMA, task="screen")
    except JudgeError:
        raise
    except Exception as e:
        log.warning("batched storyboard screening failed (%s); screening one by one", str(e)[:160])
        for i in todo:
            out[i] = screen_storyboard(gem, cands[i], shot, sbs[i])
        return out
    by_letter = {str(v.get("video", "")).strip().upper()[:1]: v
                 for v in data.get("videos") or [] if isinstance(v, dict)}
    for n, i in enumerate(todo):
        v = by_letter.get(letters[n])
        if v is None and len(data.get("videos") or []) == len(todo):
            v = data["videos"][n]                    # unlabelled but complete: trust the order
        if not isinstance(v, dict):
            out[i] = screen_storyboard(gem, cands[i], shot, sbs[i])   # the reply skipped it
            continue
        if getattr(gem, "cache", None):
            gem.cache.set("judge", v, *_sb_cache_key(cands[i], shot))
        out[i] = _sb_result(v, sbs[i], _face_check(gem))
    return out


def screen_storyboard(gem: Judge, c: Candidate, shot: Shot, sb: Storyboard) -> dict | None:
    """Screen from the storyboard contact sheet; same return shape as `screen_url`.

    The AI-generated flag is deliberately NOT read from thumbnails: even at full resolution the
    combined verdict erred both ways on it, which is why `Judge.confirm_ai_generated` exists, and
    it runs on every downloaded clip. Here it would only add false rejects.
    """
    b = shot.brief
    prompt = SB_PROMPT.format(n=len(sb.times), step=sb.step, subject=b.subject or b.visual, visual=b.visual,
                              entities=b.names(), text=shot.text,
                              avoid=", ".join(b.avoid) or "-")
    try:
        data = gem.json_images(prompt, [sb.jpeg], SB_SCHEMA, cache_key=_sb_cache_key(c, shot), task="screen")
    except JudgeError:
        raise                     # chain exhausted: our outage, not this candidate's fault
    except Exception as e:
        log.warning("storyboard screening failed for %s: %s", c.url, str(e)[:200])
        return None
    return _sb_result(data, sb, _face_check(gem))


def screen(gem: Judge, c: Candidate, shot: Shot, fps: float = 1.0, *, yt=None,
           mode: str = "storyboard") -> dict | None:
    """Screen a YouTube candidate. Storyboard first (fast, cheap, any vision link); the judge
    watching the URL is the fallback when a video has no storyboard, or `mode="url"`.

    Raises `JudgeError` when the whole judge chain is exhausted (see `screen_url`).
    """
    if mode in ("storyboard", "storyboard_only") and yt is not None:
        with stage("storyboard fetch"):
            sb = yt.storyboard(c)
        if sb is not None:
            with stage("screen (storyboard)"):
                return screen_storyboard(gem, c, shot, sb)
        if mode == "storyboard_only":
            return {"usable": False, "reason": "no storyboard (URL screening is off)", "windows": []}
        log.debug("no storyboard for %s; screening the URL instead", c.url)
    with stage("screen (url)"):
        return screen_url(gem, c, shot, fps)
