"""HeartCode (`heart`) as a judge link: the ChatGPT models connected through Nibedita.

There is no HTTP endpoint for those models, only the HeartCode CLI with its "Sign in with
ChatGPT" login, so a link with `provider: _heart` runs `heart -p` headless per request: the
OpenAI-style message parts (text + data-URL images) become one stream-json user message, and
the JSON result line holds the answer. Measured live 2026-10-30 with gpt-6-luna: 4-5 s per
frame-grid verdict (Gemini lite: 2-5 s, with 20-60 s spikes), right on a watermarked, a wrong
and a right clip. It is a separate quota pool from every Gemini key, which is the point.

Kept deliberately thin: the router treats it like any other link (circuit breaker, quota,
hedging, JSON repair). Failures raise `JudgeError`, so an expired login just trips the link.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from ..log import log

HEART = "_heart"   # sentinel provider for judge_chain entries

SYSTEM = ("You are a strict visual judge called by a program. Answer directly from the images "
          "and text you are given. Do not use tools, do not read or write files, do not ask "
          "questions. Reply with the JSON object only.")


class HeartError(RuntimeError):
    pass


def command() -> list[str] | None:
    """How to start HeartCode, or None when it is not installed. Node is called directly where
    possible: the npm `.cmd` shim would re-parse arguments through cmd.exe."""
    override = os.getenv("YTAUTO_HEART_CMD")
    if override:
        return override.split("|")
    exe = shutil.which("heart")
    if not exe:
        return None
    script = Path(exe).parent / "node_modules" / "heartcode" / "bin" / "heart.js"
    node = shutil.which("node")
    if script.exists() and node:
        return [node, str(script)]
    return [exe]


def to_message(content: list[dict]) -> dict:
    """OpenAI-style parts -> one stream-json user message (Claude-style content blocks)."""
    blocks = []
    for part in content:
        if part.get("type") == "text":
            blocks.append({"type": "text", "text": part["text"]})
        elif part.get("type") == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            if not url.startswith("data:"):
                raise HeartError("HeartCode link takes inline images only, not URLs")
            head, b64 = url.split(",", 1)
            mime = head[5:].split(";", 1)[0] or "image/jpeg"
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}})
    # images first, then the question: the order the live probe was measured with
    blocks.sort(key=lambda b: b["type"] != "image")
    return {"type": "user", "message": {"role": "user", "content": blocks}}


def parse_result(stdout: str) -> str:
    """The answer text from `--output-format json` output (one result line among log lines)."""
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") == "result":
            if obj.get("is_error") or obj.get("subtype") != "success":
                raise HeartError(f"heart error: {str(obj.get('result') or obj)[:200]}")
            text = obj.get("result") or ""
            if not text.strip():
                raise HeartError("empty content")
            return text
    raise HeartError(f"no result line in heart output: {stdout[-200:]!r}")


class HeartClient:
    def __init__(self, concurrency: int = 2, timeout: float = 90.0):
        self.cmd = command()
        self._slots = threading.BoundedSemaphore(max(1, concurrency))
        self.timeout = timeout
        self._cwd = Path(tempfile.gettempdir()) / "ytauto-heart"    # empty folder: nothing to read
        self._cwd.mkdir(exist_ok=True)

    @property
    def installed(self) -> bool:
        return self.cmd is not None

    def ask(self, model: str, content: list[dict], timeout: float | None = None, effort: str | None = None) -> str:
        if not self.cmd:
            raise HeartError("HeartCode (heart) is not installed")
        msg = json.dumps(to_message(content)) + "\n"
        args = self.cmd + ["-p", "--temp", "-m", model, "-C", str(self._cwd),
                           "--input-format", "stream-json", "--output-format", "json",
                           "--append-system-prompt", SYSTEM]
        if effort:
            args += ["--effort", effort]
        with self._slots:
            try:
                r = subprocess.run(args, input=msg, capture_output=True, text=True, encoding="utf-8",
                                   errors="replace", timeout=timeout or self.timeout, cwd=self._cwd)
            except subprocess.TimeoutExpired as e:
                raise HeartError(f"heart timed out after {e.timeout:.0f}s") from e
            except OSError as e:
                raise HeartError(f"could not start heart: {e}") from e
        if r.returncode != 0 and '"type":"result"' not in (r.stdout or ""):
            raise HeartError(f"heart exit {r.returncode}: {(r.stderr or r.stdout)[-200:]}")
        text = parse_result(r.stdout or "")
        log.debug("heart %s answered %d chars", model, len(text))
        return text
