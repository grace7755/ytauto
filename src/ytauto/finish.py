"""Finishing pass: turn the silent B-roll cut into a publishable video.

Adds the voice-over, word-timed captions (the spoken word highlighted), title / lower-third
text, a music bed ducked under the voice, SFX hits, film grain, a vignette, a light grade, an
optional particle overlay and fades. One ffmpeg render, no new dependencies: captions are an ASS
file burnt in with libass, the ducking is `sidechaincompress` keyed by the voice.

Caption text comes from the transcript (correct spelling); its timing from faster-whisper word
timestamps on the voice-over, matched word by word. Words whisper missed are interpolated
inside their segment, so a bad transcription never moves a caption out of its sentence.
"""
from __future__ import annotations

import difflib
import json
import random
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .log import log
from .models import Segment

_TOK = re.compile(r"[a-z0-9']+")


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class TextCard:
    text: str
    start: float
    end: float
    kind: str = "lower"          # "title" (centre, large) or "lower" (lower-left label)


@dataclass
class Sfx:
    file: Path
    at: float
    gain_db: float = -14.0


@dataclass
class FinishSpec:
    vo: Path
    music: Path | None = None
    music_db: float = -14.0      # music bed loudness relative to the -16 LUFS voice, before ducking
    sfx: list[Sfx] = field(default_factory=list)
    cards: list[TextCard] = field(default_factory=list)
    grain: Path | None = None
    grain_opacity: float = 0.10
    particles: Path | None = None
    particles_from: float | None = None   # seconds; screen-blended from here to the end
    tail: float = 1.5            # seconds the last frame holds after the voice ends
    captions: bool = True
    font: str = "Arial Black"


# ---------------------------------------------------------------- word timing
def _norm(w: str) -> str:
    t = _TOK.findall(w.lower())
    return t[0] if t else ""


def align_words(segments: list[Segment], heard: list[tuple[str, float, float]]) -> list[Word]:
    """Transcript words, each with the time whisper heard it (matched in order), interpolated
    inside its own segment where whisper has no match."""
    words: list[tuple[str, int]] = []                # (display text, segment index)
    for i, s in enumerate(segments):
        words += [(w, i) for w in s.text.split()]
    a = [_norm(w) for w, _ in words]
    b = [t for t, _, _ in heard]
    times: list[tuple[float, float] | None] = [None] * len(words)
    for blk in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            _, st, en = heard[blk.b + k]
            times[blk.a + k] = (st, en)
    out: list[Word] = []
    for i, s in enumerate(segments):
        idx = [k for k, (_, si) in enumerate(words) if si == i]
        if not idx:
            continue
        # keep matched times only if they fall inside the segment (+-0.5 s)
        for k in idx:
            if times[k] and not (s.start - 0.5 <= times[k][0] <= s.end + 0.5):
                times[k] = None
        known = [k for k in idx if times[k]]
        lens = [max(1, len(words[k][0])) for k in idx]
        if not known:                                 # spread by word length over the segment
            total, t = sum(lens), s.start
            for k, n in zip(idx, lens):
                d = (s.end - s.start) * n / total
                times[k] = (t, t + d)
                t += d
        else:
            pos = 0
            while pos < len(idx):
                if times[idx[pos]]:
                    pos += 1
                    continue
                end = pos                                 # run of unmatched words idx[pos:end]
                while end < len(idx) and not times[idx[end]]:
                    end += 1
                lo = times[idx[pos - 1]][1] if pos > 0 else s.start
                hi = times[idx[end]][0] if end < len(idx) else s.end
                step = max(0.05, (hi - lo) / (end - pos))
                for r, k in enumerate(idx[pos:end]):
                    times[k] = (lo + r * step, lo + (r + 1) * step)
                pos = end
        for k in idx:
            st, en = times[k]
            out.append(Word(words[k][0], max(s.start, st), min(max(en, st + 0.08), s.end)))
    for w, nxt in zip(out, out[1:]):                  # monotonic
        if nxt.start < w.start:
            nxt.start = w.start
        w.end = min(w.end, max(nxt.start, w.start + 0.05))
    return out


def heard_words(vo: Path) -> list[tuple[str, float, float]]:
    try:
        from .align import transcribe_words

        return transcribe_words(vo)[0]
    except Exception as e:                            # whisper missing: proportional timing
        log.warning("word timing unavailable (%s); captions spread by word length", str(e)[:120])
        return []


# ---------------------------------------------------------------- captions (ASS)
def chunks(words: list[Word], max_words: int = 4, max_chars: int = 26) -> list[list[Word]]:
    out, cur = [], []
    for w in words:
        if cur and (len(cur) >= max_words or len(" ".join(x.text for x in cur + [w])) > max_chars):
            out.append(cur)
            cur = []
        cur.append(w)
        if re.search(r"[.!?,;:]$", w.text):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def _ts(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("{", "(").replace("}", ")")


def build_ass(words: list[Word], cards: list[TextCard], font: str, w: int = 1920, h: int = 1080,
              captions: bool = True) -> str:
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{font},64,&H00FFFFFF,&H00FFFFFF,&H00101010,&H80000000,0,0,0,0,100,100,1,0,1,4,2,2,120,120,96,1
Style: Title,Georgia,104,&H00FFFFFF,&H00FFFFFF,&H00101010,&H64000000,1,0,0,0,100,100,6,0,1,3,3,5,120,120,0,1
Style: Lower,Segoe UI Semibold,46,&H00FFFFFF,&H00FFFFFF,&H00101010,&H96000000,0,0,0,0,100,100,3,0,1,2,2,1,110,110,170,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev = []
    for c in cards:
        style = "Title" if c.kind == "title" else "Lower"
        text = _esc(c.text).replace("\n", "\\N")
        fx = "{\\fad(600,700)}" if style == "Title" else "{\\fad(400,500)\\move(70,%d,110,%d)}" % (h - 170, h - 170)
        ev.append(f"Dialogue: 2,{_ts(c.start)},{_ts(c.end)},{style},,0,0,0,,{fx}{text}")
    if captions:
        gold = "{\\c&H0035C8FF&}"                     # warm gold, BGR
        white = "{\\c&H00FFFFFF&}"
        groups = chunks(words)
        for gi, g in enumerate(groups):
            end_g = g[-1].end
            if gi + 1 < len(groups) and groups[gi + 1][0].start - end_g < 0.35:
                end_g = groups[gi + 1][0].start
            for k, wd in enumerate(g):
                st = wd.start if k else g[0].start
                en = g[k + 1].start if k + 1 < len(g) else end_g
                if en <= st:
                    continue
                txt = " ".join((gold if j == k else white) + _esc(x.text) for j, x in enumerate(g))
                ev.append(f"Dialogue: 1,{_ts(st)},{_ts(en)},Caption,,0,0,0,,{txt}")
    return head + "\n".join(ev) + "\n"


# ---------------------------------------------------------------- render
def _dur(p: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                       capture_output=True, text=True)
    return float(r.stdout.strip() or 0)


def render(video: Path, spec: FinishSpec, segments: list[Segment], out: Path, workdir: Path,
           fps: int = 30, crf: int = 18) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    vo_len = _dur(spec.vo)
    total = round(max(_dur(video), vo_len) + spec.tail, 2)
    words = align_words(segments, heard_words(spec.vo)) if spec.captions else []
    (workdir / "words.json").write_text(json.dumps([w.__dict__ for w in words], indent=1), encoding="utf-8")
    (workdir / "captions.ass").write_text(build_ass(words, spec.cards, spec.font, captions=spec.captions),
                                          encoding="utf-8")

    inputs = ["-i", str(video), "-i", str(spec.vo)]
    n = 2
    vf = [f"[0:v]tpad=stop_mode=clone:stop_duration={spec.tail + 2},trim=0:{total},setpts=PTS-STARTPTS,"
          f"fps={fps},format=yuv420p,eq=contrast=1.05:saturation=0.95,colorbalance=bs=0.03:bm=0.015,"
          f"vignette=PI/5[base]"]
    last = "base"
    if spec.grain:
        inputs += ["-stream_loop", "-1", "-i", str(spec.grain)]
        # blends run in RGB: in YUV a screen/overlay blend also shifts the chroma planes, which
        # turned a sunset purple in the first Alaska render
        vf.append(f"[{n}:v]scale=1920:1080,fps={fps},format=gbrp,trim=0:{total},setpts=PTS-STARTPTS[g]")
        vf.append(f"[{last}]format=gbrp[bg];[bg][g]blend=all_mode=overlay:all_opacity={spec.grain_opacity},"
                  f"format=yuv420p[vg]")
        last, n = "vg", n + 1
    if spec.particles and spec.particles_from is not None:
        inputs += ["-stream_loop", "-1", "-i", str(spec.particles)]
        vf.append(f"[{n}:v]scale=1920:1080,fps={fps},format=gbrp,trim=0:{total},setpts=PTS-STARTPTS[p]")
        vf.append(f"[{last}]format=gbrp[bp];[bp][p]blend=all_mode=screen:all_opacity=0.35:"
                  f"enable='gte(t,{spec.particles_from})',format=yuv420p[vp]")
        last, n = "vp", n + 1
    vf.append(f"[{last}]subtitles=captions.ass,fade=t=in:st=0:d=0.8,fade=t=out:st={total - 1.3}:d=1.3[vout]")

    af = [f"[1:a]aresample=48000,loudnorm=I=-16:TP=-1.5:LRA=11,apad,atrim=0:{total}[vo]",
          "[vo]asplit=2[vomix][vokey]"]
    mix = ["[vomix]"]
    if spec.music:
        inputs += ["-stream_loop", "-1", "-i", str(spec.music)]
        af.append(f"[{n}:a]aresample=48000,atrim=0:{total},asetpts=PTS-STARTPTS,loudnorm=I=-16:TP=-2,"
                  f"volume={spec.music_db}dB,afade=t=in:st=0:d=2.5,afade=t=out:st={total - 3}:d=3[mus]")
        af.append("[mus][vokey]sidechaincompress=threshold=0.02:ratio=5:attack=30:release=600:makeup=1[duck]")
        mix.append("[duck]")
        n += 1
    else:
        af.append("[vokey]anullsink")
    for i, fx in enumerate(spec.sfx):
        inputs += ["-i", str(fx.file)]
        ms = int(fx.at * 1000)
        af.append(f"[{n}:a]aresample=48000,volume={fx.gain_db}dB,adelay={ms}|{ms},apad,atrim=0:{total}[s{i}]")
        mix.append(f"[s{i}]")
        n += 1
    af.append(f"{''.join(mix)}amix=inputs={len(mix)}:normalize=0:duration=first,"
              f"alimiter=limit=0.89,aresample=48000,aformat=sample_rates=48000:channel_layouts=stereo[aout]")

    cmd = ["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", ";".join(vf + af),
           "-map", "[vout]", "-map", "[aout]", "-t", str(total),
           "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)]
    log.info("finishing render -> %s (%.1fs)", out, total)
    r = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg finish failed: {r.stderr[-800:]}")
    return out


# ---------------------------------------------------------------- asset picking
def pick(assets: Path, sub: str, *keys: str, seed: int = 0) -> Path | None:
    """First file under assets/sub whose name contains any key (keys in priority order)."""
    files = sorted((assets / sub).glob("*.*")) if (assets / sub).exists() else []
    for k in keys:
        hit = [f for f in files if k in f.name.lower()]
        if hit:
            return random.Random(seed).choice(hit) if len(hit) > 1 and k == "" else hit[0]
    return None
