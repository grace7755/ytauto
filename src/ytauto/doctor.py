"""`ytauto doctor`: check the machine and the judge chain before a long run.

Every check returns (name, ok, detail) and never raises, so one broken piece does not hide the
others. Things it catches that otherwise surface minutes into a run: a missing ffmpeg, an
`opencv-python` install shadowing the headless build (no `cv2.CascadeClassifier`, a crash at the
first local check), an expired ChatGPT login, a Gemini model out of its daily quota.
"""
from __future__ import annotations

import shutil
import subprocess
import time

from .config import Settings

Check = tuple[str, bool, str]


def _tool(name: str) -> Check:
    path = shutil.which(name)
    if not path:
        return name, False, "not on PATH"
    try:
        out = subprocess.run([name, "-version"], capture_output=True, text=True, timeout=20).stdout
        return name, True, (out.splitlines() or [path])[0][:70]
    except Exception as e:                       # noqa: BLE001 - a doctor must not crash
        return name, False, str(e)[:80]


def check_tools() -> list[Check]:
    out = [_tool("ffmpeg"), _tool("ffprobe")]
    try:
        import yt_dlp

        out.append(("yt-dlp", True, yt_dlp.version.__version__))
    except Exception as e:                       # noqa: BLE001
        out.append(("yt-dlp", False, str(e)[:80]))
    try:
        import cv2

        ok = hasattr(cv2, "CascadeClassifier")
        out.append(("opencv", ok, cv2.__version__ + ("" if ok else
                    " - no CascadeClassifier: uninstall opencv-python, reinstall opencv-python-headless<5")))
    except Exception as e:                       # noqa: BLE001
        out.append(("opencv", False, str(e)[:80]))
    return out


def check_optional(s: Settings) -> list[Check]:
    from .inspect import overlay
    from .llm.heart import command

    out = []
    if s.local_overlay_check:
        out.append(("local OCR (watermarks)", overlay.available(),
                    "ready" if overlay.available() else
                    "off - pip install --no-deps rapidocr-onnxruntime pyclipper shapely six"))
    clip_dir = s.cache_dir / "models"
    out.append(("CLIP model", clip_dir.exists() and any(clip_dir.rglob("*.onnx")),
                str(clip_dir) if clip_dir.exists() else "downloads on first run (~150 MB)"))
    if any(l.provider == "_heart" for l in s.judge_chain):
        cmd = command()
        out.append(("HeartCode (ChatGPT)", cmd is not None, " ".join(cmd)[:70] if cmd else "heart not installed"))
    if any(l.provider == "_codex" for l in s.judge_chain):
        from .llm.codex import command as codex_command

        cmd = codex_command()
        out.append(("Codex (ChatGPT plan)", cmd is not None,
                    " ".join(cmd)[:70] if cmd else "codex not installed; then run `codex login`"))
    for name, key in (("Pexels key", s.pexels_key), ("Pixabay key", s.pixabay_key), ("Coverr key", s.coverr_key)):
        out.append((name, bool(key), "set" if key else "missing - that stock fallback is skipped"))
    return out


PING_SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


def check_judge(s: Settings, cache, live: bool = False) -> list[Check]:
    """The usable links in chain order; with `live`, one tiny request each (costs one request
    of each link's quota) to see which answer right now and how fast."""
    from .llm.router import Judge, JudgeError

    try:
        j = Judge(s, cache)
    except JudgeError as e:
        return [("judge chain", False, str(e)[:120])]
    out: list[Check] = []
    for link in j.chain:
        tag = f"judge {link.provider}/{link.model}"
        until = j.quota.out_until(link)
        if until:
            out.append((tag, False, "out of daily quota until " + time.strftime("%H:%M", time.localtime(until))))
            continue
        if not live:
            out.append((tag, True, f"configured, {j.quota.used(link)} requests today"))
            continue
        single = s.model_copy(update={"judge_chain": [link], "judge_patience": 0, "judge_hedge_after": 0})
        t0 = time.time()
        try:
            one = Judge(single, cache)
            got = one.json('Reply with the JSON object {"ok": true}.', PING_SCHEMA)
            out.append((tag, bool(got.get("ok")), f"answered in {time.time() - t0:.1f}s"))
        except Exception as e:                   # noqa: BLE001
            out.append((tag, False, f"{str(e)[:90]} ({time.time() - t0:.0f}s)"))
    return out


def run_all(s: Settings, cache, live: bool = False) -> list[Check]:
    return check_tools() + check_optional(s) + check_judge(s, cache, live)
