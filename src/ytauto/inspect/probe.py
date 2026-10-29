"""ffprobe wrapper."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MediaInfo:
    width: int
    height: int
    fps: float
    duration: float
    has_audio: bool
    codec: str


def _fps(rate: str) -> float:
    try:
        n, d = rate.split("/")
        return float(n) / float(d) if float(d) else 0.0
    except Exception:
        return 0.0


def probe(path: Path) -> MediaInfo | None:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
            capture_output=True, text=True, timeout=60)
        j = json.loads(r.stdout)
        v = next(s for s in j["streams"] if s["codec_type"] == "video")
        dur = float(v.get("duration") or j["format"].get("duration") or 0)
        return MediaInfo(int(v["width"]), int(v["height"]), _fps(v.get("avg_frame_rate", "0/1")), dur,
                         any(s["codec_type"] == "audio" for s in j["streams"]), v.get("codec_name", ""))
    except Exception:
        return None
