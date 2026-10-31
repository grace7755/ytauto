"""Sound-effect plan and audit for a Remotion film.

`plan` turns story events (hook, chapter changes, zooms, stats, impacts) into a short cue list with
levels set against the voice-over; `audit` checks the finished plan, and optionally the rendered
video, number by number. Rules: SFX only where they help - one hook sound, one whoosh per chapter
change, a soft whoosh only on big zooms, ticks on stats, impacts on key numbers - never two cues
close together, and every cue clearly under the narration.

Levels are set from measured files, not guessed: a cue's peak sits `under` dB below the voice-over's
peak, an ambience bed's average sits `BED_UNDER` dB below the voice's average.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

# kind -> (priority, default sound, dB under the voice peak). Higher priority survives a clash.
KINDS: dict[str, tuple[int, str, float]] = {
    "hook": (5, "riser", 8.0),
    "impact": (4, "impact_low", 8.0),
    "chapter": (3, "whoosh_deep", 10.0),
    "statbig": (2, "shimmer", 12.0),
    "stat": (1, "tick", 12.0),
    "zoom": (0, "whoosh_soft", 14.0),
}
LENGTH = {"hook": 3.0, "impact": 2.0, "chapter": 2.2, "statbig": 2.0, "stat": 2.0, "zoom": 1.4}
LEAD = {"chapter": 0.12, "zoom": 0.05}      # a whoosh starts just before the cut it announces
MIN_GAP = 3.0                               # seconds between two kept cues
MAX_PER_MINUTE = 8
BED_UNDER = 20.0                            # ambience average, dB under the voice average
MIN_MARGIN = 6.0                            # audit: a cue must stay at least this far under speech
SPEECH_DB = -35.0                           # a window whose voice peak is above this has speech in it


def _run(args: list[str], check: bool = False) -> str:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", *args], capture_output=True, text=True, timeout=900)
    if check and r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr[-400:]}")
    return r.stderr


def measure(path: Path, start: float = 0.0, dur: float | None = None) -> tuple[float, float]:
    """(mean dB, peak dB) of an audio file or a window of it, by ffmpeg `volumedetect`."""
    win = ["-ss", f"{start:.3f}"] + (["-t", f"{dur:.3f}"] if dur else [])
    err = _run([*win, "-i", str(path), "-af", "volumedetect", "-vn", "-f", "null", "-"])
    mean = re.search(r"mean_volume: (-?[\d.]+|-inf) dB", err)
    peak = re.search(r"max_volume: (-?[\d.]+|-inf) dB", err)
    if not mean or not peak:
        raise RuntimeError(f"cannot measure {path}: {err[-200:]}")

    def num(m):
        return -120.0 if m.group(1) == "-inf" else float(m.group(1))
    return num(mean), num(peak)


@dataclass
class Cue:
    name: str
    at: float
    len: float
    vol: float
    kind: str
    anchor: float            # the moment it announces (a cut, a number being spoken)


def _vol(target_db: float, file_db: float) -> float:
    return round(min(1.0, 10 ** ((target_db - file_db) / 20)), 3)


def _fits(anchors: list[float], new: float, per_minute: int) -> bool:
    """Would adding `new` keep every 60 s window at or under `per_minute` cues?"""
    pts = sorted(anchors + [new])
    return all(sum(1 for q in pts if p <= q <= p + 60) <= per_minute for p in pts)


def prune(events: list[dict], min_gap: float = MIN_GAP, per_minute: int = MAX_PER_MINUTE) -> tuple[list[dict], list[dict]]:
    """Keep the important events: highest priority first, none within `min_gap` seconds of a kept
    one, and at most `per_minute` in any 60 s. Returns (kept sorted by time, dropped sorted by time)."""
    kept: list[dict] = []
    dropped: list[dict] = []
    for ev in sorted(events, key=lambda e: (-KINDS[e["kind"]][0], e["anchor"])):
        near = any(abs(ev["anchor"] - k["anchor"]) < min_gap for k in kept)
        if near or not _fits([k["anchor"] for k in kept], ev["anchor"], per_minute):
            dropped.append(ev)
        else:
            kept.append(ev)
    return sorted(kept, key=lambda e: e["anchor"]), sorted(dropped, key=lambda e: e["anchor"])


SNAP = 1.5      # a chapter or zoom event this close to a cut moves onto the cut


def _snap(at: float, cuts: list[float]) -> float:
    near = min(cuts, key=lambda c: abs(c - at), default=None)
    return near if near is not None and abs(near - at) <= SNAP else at


def plan(events: list[dict], vo: Path, audio_dir: Path, total: float, *, beds: list[dict] | None = None,
         sounds: dict[str, str] | None = None, cuts: list[float] | None = None) -> dict:
    """Build `sfx.json`. `events`: [{"kind": ..., "at": seconds}] (+ optional "name" to pick the
    sound). `beds`: [{"name", "from", "to"}] ambience under the voice. `cuts`: shot start times;
    chapter and zoom events snap onto the nearest one so the whoosh lands on the cut. Sounds are
    looked up as `<audio_dir>/<name>.mp3`. A cue over speech is set `MIN_MARGIN + 2` dB under the
    voice's own peak in that window, so a quiet sentence is not covered by a loud effect."""
    sounds = {k: v[1] for k, v in KINDS.items()} | (sounds or {})
    vo_mean, vo_peak = measure(vo)
    evs = [{**e, "anchor": _snap(float(e["at"]), cuts or []) if e["kind"] in LEAD else float(e["at"])}
           for e in events if e["kind"] in KINDS and 0 <= float(e["at"]) < total]
    kept, dropped = prune(evs)
    cues = []
    for e in kept:
        kind = e["kind"]
        name = e.get("name") or sounds[kind]
        _, file_peak = measure(audio_dir / f"{name}.mp3")
        at = max(0.0, e["anchor"] - LEAD.get(kind, 0.0))
        target = vo_peak - KINDS[kind][2]
        _, local = measure(vo, at, LENGTH[kind])
        if local > SPEECH_DB:
            target = min(target, local - MIN_MARGIN - 2)
        cues.append(Cue(name, round(at, 3), LENGTH[kind], _vol(target, file_peak), kind, e["anchor"]))
    out_beds = []
    for b in beds or []:
        mean, _ = measure(audio_dir / f"{b['name']}.mp3")
        out_beds.append({"name": b["name"], "from": b["from"], "to": b["to"], "vol": _vol(vo_mean - BED_UNDER, mean)})
    return {"total": total, "vo_mean_db": vo_mean, "vo_peak_db": vo_peak, "cues": [asdict(c) for c in cues],
            "beds": out_beds, "dropped": [{"kind": d["kind"], "at": d["anchor"]} for d in dropped]}


# ---------------------------------------------------------------------------------------------
def _mix(p: dict, vo: Path, audio_dir: Path, out: Path):
    """Voice + cues + beds as one wav, to measure the loudness and peak of what the plan adds."""
    ins, parts = ["-i", str(vo)], ["[0:a]aresample=48000,aformat=channel_layouts=stereo[a0]"]
    labels = ["[a0]"]
    items = [(c["name"], c["at"], c["vol"], c["len"], False) for c in p["cues"]]
    items += [(b["name"], b["from"], b["vol"], b["to"] - b["from"], True) for b in p["beds"]]
    for i, (name, at, vol, ln, loop) in enumerate(items, 1):
        ins += (["-stream_loop", "-1"] if loop else []) + ["-i", str(audio_dir / f"{name}.mp3")]
        ms = int(at * 1000)
        parts.append(f"[{i}:a]atrim=0:{ln:.3f},aresample=48000,aformat=channel_layouts=stereo,volume={vol},"
                     f"adelay={ms}|{ms}[a{i}]")
        labels.append(f"[a{i}]")
    parts.append("".join(labels) + f"amix=inputs={len(labels)}:normalize=0:duration=first[m]")
    _run(["-y", *ins, "-filter_complex", ";".join(parts), "-map", "[m]", "-c:a", "pcm_s16le", str(out)], check=True)


def loudness(path: Path) -> tuple[float, float]:
    """(integrated LUFS, true peak dBTP) of an audio or video file."""
    err = _run(["-i", str(path), "-vn", "-af", "ebur128=peak=true", "-f", "null", "-"])
    tail = err[err.rfind("Summary:"):]
    i = re.search(r"I:\s+(-?[\d.]+) LUFS", tail)
    pk = re.search(r"Peak:\s+(-?[\d.]+) dBFS", tail)
    if not i or not pk:
        raise RuntimeError(f"cannot read loudness of {path}")
    return float(i.group(1)), float(pk.group(1))


def audit(p: dict, vo: Path, audio_dir: Path, shots: list[dict], video: Path | None = None,
          work: Path | None = None) -> dict:
    """Check every cue and bed of plan `p`, then the whole mix. Returns {"issues": [...], "rows": [...],
    "mix_lufs", "mix_peak"}; an empty `issues` means the audio passes. `shots`: [{"t0": seconds}, ...].
    With `video` (the rendered file) its loudness and peak are checked too."""
    issues: list[str] = []
    rows = []
    total = p["total"]
    cuts = sorted(float(s["t0"]) for s in shots)
    cues = sorted(p["cues"], key=lambda c: c["at"])
    for c in cues:
        row = {"at": c["at"], "kind": c["kind"], "name": c["name"]}
        rows.append(row)
        f = audio_dir / f"{c['name']}.mp3"
        if not f.exists():
            issues.append(f"{c['at']:.1f}s {c['name']}: sound file missing")
            continue
        if c["at"] + c["len"] > total + 0.5:
            issues.append(f"{c['at']:.1f}s {c['name']}: runs past the end of the video")
        _, fpeak = measure(f)
        cue_peak = fpeak + 20 * math.log10(max(c["vol"], 1e-6))
        _, vpeak = measure(vo, c["at"], c["len"])
        row.update(cue_peak=round(cue_peak, 1), voice_peak=round(vpeak, 1))
        if vpeak > SPEECH_DB:
            margin = vpeak - cue_peak
            row["margin"] = round(margin, 1)
            if margin < MIN_MARGIN:
                issues.append(f"{c['at']:.1f}s {c['name']}: only {margin:.1f} dB under the voice (needs {MIN_MARGIN:.0f})")
        if c["kind"] in ("chapter", "zoom") and cuts:
            d = min(abs(x - c["anchor"]) for x in cuts)
            row["cut_offset"] = round(d, 2)
            if d > 0.3:
                issues.append(f"{c['at']:.1f}s {c['name']}: {d:.2f} s from the nearest cut (not in sync)")
    for a, b in zip(cues, cues[1:]):
        if b["anchor"] - a["anchor"] < MIN_GAP - 0.2:
            issues.append(f"{a['at']:.1f}s {a['name']} and {b['at']:.1f}s {b['name']}: cues closer than {MIN_GAP:.0f} s")
    for c in cues:
        n = sum(1 for k in cues if c["at"] <= k["at"] <= c["at"] + 60)
        if n > MAX_PER_MINUTE:
            issues.append(f"{c['at']:.1f}s: {n} cues in one minute (max {MAX_PER_MINUTE})")
            break
    for b in p["beds"]:
        f = audio_dir / f"{b['name']}.mp3"
        if not f.exists():
            issues.append(f"bed {b['name']}: sound file missing")
            continue
        mean, _ = measure(f)
        margin = p["vo_mean_db"] - (mean + 20 * math.log10(max(b["vol"], 1e-6)))
        if margin < BED_UNDER - 3:
            issues.append(f"bed {b['name']} {b['from']:.0f}-{b['to']:.0f}s: only {margin:.1f} dB under the voice average")
        if b["to"] > total + 0.5:
            issues.append(f"bed {b['name']}: runs past the end of the video")
    if any("missing" in i for i in issues):          # the mix cannot be rendered without every file
        return {"rows": rows, "mix_lufs": float("nan"), "mix_peak": float("nan"), "issues": issues}
    mixed = (work or vo.parent) / "_sfx_mix.wav"
    _mix(p, vo, audio_dir, mixed)
    lufs, peak = loudness(mixed)
    mixed.unlink(missing_ok=True)
    res = {"rows": rows, "mix_lufs": lufs, "mix_peak": peak, "issues": issues}
    if peak > -1.0:
        issues.append(f"voice + effects peak at {peak:.1f} dBTP (limit -1)")
    if video:
        v_lufs, v_peak = loudness(video)
        res.update(video_lufs=v_lufs, video_peak=v_peak)
        if v_peak > -1.0:
            issues.append(f"rendered video peaks at {v_peak:.1f} dBTP (limit -1)")
        if not -18.0 <= v_lufs <= -13.0:
            issues.append(f"rendered video is {v_lufs:.1f} LUFS (target about -16)")
    return res


def load(path: Path) -> dict:
    """An sfx plan: an `sfx.json` from `plan`, or a `film.json` (its "sfx" entry)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "cues" not in data and isinstance(data.get("sfx"), dict):
        return data["sfx"]
    if "cues" not in data:
        raise ValueError(f"{path} has no sound plan (run `ytauto film --audio-dir ...` or `ytauto sfx-plan`)")
    return data
