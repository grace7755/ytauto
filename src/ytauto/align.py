"""Re-time transcript segments to a supplied voice-over using faster-whisper word timestamps."""
from __future__ import annotations

import difflib
import re
from pathlib import Path

from .log import log
from .models import Segment

_W = re.compile(r"[a-z0-9']+")


def _toks(s: str) -> list[str]:
    return _W.findall(s.lower())


def load_audio(audio: Path, sr: int = 16000):
    """Mono float32 samples at `sr` via ffmpeg. faster-whisper's own decoder passes
    `metadata_errors` to PyAV's open(), which current PyAV no longer accepts (TypeError), so
    every `--vo` alignment silently fell back to the transcript timing."""
    import subprocess

    import numpy as np

    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(audio), "-ac", "1", "-ar", str(sr),
                        "-f", "f32le", "-"], capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


def transcribe_words(audio: Path, model_size: str = "base") -> tuple[list[tuple[str, float, float]], float]:
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segs, info = model.transcribe(load_audio(audio), word_timestamps=True, vad_filter=True)
    words = []
    for sg in segs:
        for w in sg.words or []:
            t = _toks(w.word)
            if t:
                words.append((t[0], float(w.start), float(w.end)))
    return words, float(info.duration)


def align_segments(segments: list[Segment], audio: Path, model_size: str = "base") -> list[Segment]:
    """Return segments whose start/end follow the voice-over. Falls back to the original timing on failure."""
    try:
        words, total = transcribe_words(audio, model_size)
    except Exception as e:
        log.warning("VO alignment failed (%s); keeping transcript timestamps", e)
        return segments
    if not words:
        return segments
    seg_tokens = [_toks(s.text) for s in segments]
    flat = [t for toks in seg_tokens for t in toks]
    owner = [i for i, toks in enumerate(seg_tokens) for _ in toks]
    sm = difflib.SequenceMatcher(None, flat, [w[0] for w in words], autojunk=False)
    first: dict[int, float] = {}
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            seg = owner[blk.a + k]
            first.setdefault(seg, words[blk.b + k][1])
    starts, last = [], 0.0
    for i in range(len(segments)):
        t = first.get(i)
        if t is None or t < last:
            t = last if i else 0.0
        starts.append(t)
        last = t
    starts[0] = 0.0
    out = []
    for i, s in enumerate(segments):
        end = starts[i + 1] if i + 1 < len(segments) else total
        if end - starts[i] < 1.0:      # degenerate match: keep original length
            end = starts[i] + max(1.0, s.duration)
        out.append(Segment(idx=s.idx, start=round(starts[i], 3), end=round(end, 3), text=s.text))
    log.info("aligned %d segments to voice-over (%.1fs)", len(out), total)
    return out
