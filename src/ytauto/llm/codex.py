"""OpenAI's Codex CLI as a judge link: ChatGPT models through your ChatGPT plan, no API key.

`provider: _codex` runs `codex exec` headless once per request, signed in by `codex login` ("Logged
in using ChatGPT"). The OpenAI-style message parts (text + data-URL images) become the prompt on stdin
plus `--image` files in a temp folder; the model's final message is read from `--output-last-message`.

Flags used, all from `codex exec --help` (codex-cli 0.166.0): `-m`, `-s read-only`, `-C`,
`--skip-git-repo-check`, `--ephemeral`, `--ignore-user-config`, `-i`, `-o`. The prompt goes on stdin
with no positional argument, because `-i <FILE>...` would swallow one.

Same contract as `heart.py`: failures raise `JudgeError`-compatible `CodexError`, so the router's circuit
breaker, quota and JSON repair apply, and an expired login or a plan limit just trips the link.
A ChatGPT plan has its own message limit per few hours; when Codex says so, `is_limit()` tells the
router to treat it as a spent quota instead of a failure of the footage.
"""
from __future__ import annotations

import base64
import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from ..log import log

CODEX = "_codex"   # sentinel provider for judge_chain entries

SYSTEM = ("You are a strict visual judge called by a program. Answer directly from the images and text "
          "you are given. Do not run commands, do not read or write files, do not ask questions. "
          "Reply with the JSON object only.\n\n")

_LIMIT = re.compile(r"usage limit|rate limit|too many requests|try again (?:in|at|later)|quota", re.I)
_EXT = {"image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png", "image/webp": ".webp"}


class CodexError(RuntimeError):
    pass


def command() -> list[str] | None:
    """How to start Codex, or None when it is not installed. `YTAUTO_CODEX_CMD` (a `|`-separated
    command) overrides it, which is how the tests stand in a fake."""
    override = os.getenv("YTAUTO_CODEX_CMD")
    if override:
        return override.split("|")
    exe = shutil.which("codex")
    return [exe] if exe else None


def is_limit(text: str) -> bool:
    return bool(_LIMIT.search(text or ""))


def split_parts(content: list[dict]) -> tuple[str, list[tuple[bytes, str]]]:
    """OpenAI-style parts -> (prompt text, [(image bytes, extension)]) with images in order."""
    texts, images = [], []
    for part in content:
        if part.get("type") == "text":
            texts.append(part["text"])
        elif part.get("type") == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            if not url.startswith("data:"):
                raise CodexError("Codex link takes inline images only, not URLs")
            head, b64 = url.split(",", 1)
            mime = head[5:].split(";", 1)[0] or "image/jpeg"
            images.append((base64.b64decode(b64), _EXT.get(mime, ".jpg")))
    return "\n\n".join(texts), images


class CodexClient:
    def __init__(self, concurrency: int = 2, timeout: float = 120.0):
        self.cmd = command()
        self._slots = threading.BoundedSemaphore(max(1, concurrency))
        self.timeout = timeout
        self._cwd = Path(tempfile.gettempdir()) / "ytauto-codex"      # empty folder: nothing to read
        self._cwd.mkdir(exist_ok=True)

    @property
    def installed(self) -> bool:
        return self.cmd is not None

    def ask(self, model: str, content: list[dict], timeout: float | None = None) -> str:
        if not self.cmd:
            raise CodexError("Codex (codex) is not installed")
        prompt, images = split_parts(content)
        with tempfile.TemporaryDirectory(prefix="job-", dir=self._cwd) as tmp:
            tmpdir = Path(tmp)
            out = tmpdir / "answer.txt"
            args = self.cmd + ["exec", "-m", model, "-s", "read-only", "-C", str(tmpdir),
                               "--skip-git-repo-check", "--ephemeral", "--ignore-user-config",
                               "-o", str(out)]
            for i, (data, ext) in enumerate(images):
                p = tmpdir / f"img{i}{ext}"
                p.write_bytes(data)
                args += ["-i", str(p)]
            with self._slots:
                try:
                    r = subprocess.run(args, input=SYSTEM + prompt, capture_output=True, text=True,
                                       encoding="utf-8", errors="replace",
                                       timeout=timeout or self.timeout, cwd=tmpdir)
                except subprocess.TimeoutExpired as e:
                    raise CodexError(f"codex timed out after {e.timeout:.0f}s") from e
                except OSError as e:
                    raise CodexError(f"could not start codex: {e}") from e
            text = out.read_text(encoding="utf-8", errors="replace").strip() if out.exists() else ""
        if r.returncode != 0 or not text:
            tail = ((r.stderr or "") + (r.stdout or ""))[-300:].strip()
            if r.returncode == 0:
                raise CodexError("empty content")
            raise CodexError(f"codex exit {r.returncode}: {tail}")
        log.debug("codex %s answered %d chars", model, len(text))
        return text
