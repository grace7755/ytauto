"""Multi-provider judge: same interface as `Gemini`, but backed by an ordered fallback chain.

Why: a single Gemini key is a single point of failure (503 demand spikes, 429 quota, models
retired for new users). The chain walks provider+model pairs until one answers, so a run only
degrades to offline mode when *every* link is down.

Verified failure modes this guards against (probed live 2026-10-29):
  * HTTP 503/429/403/500  - provider down, rate limited, or key rejected
  * empty `content`       - model advertises vision but returns nothing (nemotron-3.5-super-vl)
  * markdown-fenced JSON  - OpenAI-compat has no response_schema, so models wrap JSON in ```
"""
from __future__ import annotations

import base64
import json
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from ..cache import Cache
from ..config import JudgeLink, Settings
from ..inspect.grid import extract_grid_data_url, grid_hint
from ..log import log
from .gemini import Gemini, GeminiError
from .providers import DIRECT, ProviderConfig
from .quota import Quota

TIMEOUT = 150               # a judge watching a whole YouTube video can legitimately take minutes
IMAGE_TIMEOUT = 90          # one contact sheet / frame grid: measured median ~12 s
DOWN_SECONDS = 300          # circuit breaker: skip a failing link for 5 minutes
OVERLOAD_SECONDS = 60       # Google's 503 "high demand" / 504 clear fast; measured: dozens per run on 2026-10-29
_OVERLOADED = re.compile(r"\b(?:503|504)\b|UNAVAILABLE|DEADLINE_EXCEEDED")
QUOTA_DOWN_SECONDS = 3600   # "exceeded your current quota" is a daily cap, not a blip
_QUOTA = re.compile(r"exceeded your current quota|quota exceeded|RESOURCE_EXHAUSTED.*per day", re.I)
_RETRY_AFTER = re.compile(r"\[retry-after (\d+(?:\.\d+)?)s\]")
# Gemini's own 429 body names the quota that ran out and when to retry. A *per-minute* quota says
# "exceeded your current quota" too, and must not take the link out for the rest of the day.
_PER_MINUTE = re.compile(r"per ?minute", re.I)
_PER_DAY = re.compile(r"per ?day", re.I)
_RETRY_DELAY = re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s")
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I)


class JudgeError(RuntimeError):
    pass


class QuotaExhausted(JudgeError):
    """Every link that could answer is out of its daily quota. Unlike a flapping provider this
    will not clear in minutes, so the pipeline stops starting shots (or waits for `until`, an
    epoch time) instead of turning each remaining shot into an "unavailable" blocker."""

    def __init__(self, msg: str, until: float):
        super().__init__(msg)
        self.until = until


def _strip_fence(text: str) -> str:
    return _FENCE.sub("", text.strip())


def _repair(s: str) -> str:
    """Best-effort repairs for the malformed JSON models actually emit."""
    s = re.sub(r",\s*([}\]])", r"\1", s)                      # trailing commas
    s = re.sub(r"\bTrue\b", "true", s)                        # python literals
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    # raw newlines/tabs inside string literals are illegal control characters
    out, in_str, esc = [], False, False
    for ch in s:
        if esc:
            out.append(ch); esc = False; continue
        if ch == "\\":
            out.append(ch); esc = True; continue
        if ch == '"':
            in_str = not in_str
        if in_str and ch in "\n\r\t":
            out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[ch]); continue
        out.append(ch)
    return "".join(out)


def _close_truncated(s: str) -> str:
    """Close a reply that was cut off mid-structure (finish_reason=length)."""
    depth_obj = depth_arr = 0
    in_str = esc = False
    last_safe = None          # end of the last complete top-level-ish element
    for i, ch in enumerate(s):
        if esc:
            esc = False; continue
        if ch == "\\" and in_str:
            esc = True; continue
        if ch == '"':
            in_str = not in_str; continue
        if in_str:
            continue
        if ch == "{":
            depth_obj += 1
        elif ch == "}":
            depth_obj -= 1
        elif ch == "[":
            depth_arr += 1
        elif ch == "]":
            depth_arr -= 1
        if not in_str and ch in "}]" and depth_obj >= 1:
            last_safe = i
    if in_str:                                   # cut inside a string: drop the partial value
        s = s[:s.rfind('"')]
        s = s[:s.rfind(",")] if "," in s else s
    elif last_safe is not None and (depth_obj > 1 or depth_arr > 0):
        # drop the incomplete tail after the last complete element
        s = s[:last_safe + 1]
    # re-count and close what is still open
    depth_obj = depth_arr = 0
    in_str = esc = False
    for ch in s:
        if esc:
            esc = False; continue
        if ch == "\\" and in_str:
            esc = True; continue
        if ch == '"':
            in_str = not in_str; continue
        if in_str:
            continue
        depth_obj += (ch == "{") - (ch == "}")
        depth_arr += (ch == "[") - (ch == "]")
    s = re.sub(r",\s*$", "", s.rstrip())
    return s + "]" * max(0, depth_arr) + "}" * max(0, depth_obj)


def _extract_json(text: str) -> dict:
    """Parse a model's JSON reply, tolerating fences, prose, and malformed/truncated output.

    OpenAI-compatible endpoints have no enforced response schema, so replies arrive with code
    fences, trailing commas, Python literals, raw newlines in strings, or simply cut off. A
    healthy provider should not be circuit-broken over any of those.
    """
    cleaned = _strip_fence(text)
    # Narrow to the outermost object span, dropping any surrounding prose.
    i, j = cleaned.find("{"), cleaned.rfind("}")
    span = cleaned[i:j + 1] if i != -1 and j > i else cleaned[i:] if i != -1 else cleaned

    repaired = _repair(span)
    # Single-quoted (python-style) object: only safe to convert when there are no double quotes.
    single = re.sub(r"'", '"', repaired) if "'" in repaired and '"' not in repaired else None

    for attempt in (cleaned, span, repaired, _close_truncated(repaired), single):
        if not attempt or not attempt.strip():
            continue
        try:
            data = json.loads(attempt)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict):
            return data
    raise ValueError(f"no JSON object in reply: {cleaned[:200]!r}")


def _schema_hint(schema: dict | None) -> str:
    """OpenAI-compat endpoints lack response_schema, so state the contract in the prompt."""
    if not schema:
        return ""
    props = schema.get("properties") or {}
    if not props:
        return "\n\nReply with a single JSON object and nothing else."
    fields = []
    for name, spec in props.items():
        t = spec.get("type", "string")
        desc = spec.get("description")
        if t == "array":
            # Spell out the item shape, or providers invent their own key names
            # (start_time/end_time instead of start/end was observed in production).
            item = (spec.get("items") or {}).get("properties") or {}
            if item:
                inner = ", ".join(
                    f'"{k}": <{v.get("type", "string")}'
                    + (f' {v["description"]}' if v.get("description") else "") + ">"
                    for k, v in item.items())
                t = f"array of objects, each exactly {{{inner}}}"
        fields.append(f'  "{name}": <{t}>' + (f"  // {desc}" if desc else ""))
    required = schema.get("required") or list(props)
    return ("\n\nReply with ONE JSON object, no markdown fences, no prose. Shape:\n{\n"
            + ",\n".join(fields) + "\n}\n"
            + f"All of these keys are required: {', '.join(required)}.")


class Judge:
    """Ordered provider chain with per-link circuit breaking and the Gemini method signatures."""

    def __init__(self, settings: Settings, cache: Cache):
        self.s = settings
        self.cache = cache
        self.cfg = ProviderConfig.load(settings.heart_config)
        self._down: dict[str, float] = {}
        self._quota_down: dict[str, float] = {}     # links out of their daily quota
        self._direct: Gemini | None = None
        self.quota = Quota(cache)
        self._direct_lock = threading.Lock()
        # One cap on in-flight judge calls across all shot and screening threads: provider rate
        # limits, not CPU, are what parallelism runs into (the free links return 429 even serially).
        self._slots = threading.BoundedSemaphore(max(1, settings.judge_concurrency))
        self._direct_failed = False
        self.calls = 0
        self.tokens = 0

        self.chain: list[JudgeLink] = [l for l in settings.judge_chain if self._usable(l)]
        if not self.chain:
            raise JudgeError(
                "no usable judge providers: set HEART_CONFIG to a provider config and/or "
                "GEMINI_API_KEY in yt-auto/.env")
        log.info("judge chain: %s", " -> ".join(f"{l.provider}/{l.model}" for l in self.chain))

    # -- chain bookkeeping ---------------------------------------------------
    def _usable(self, link: JudgeLink) -> bool:
        if link.provider == DIRECT:
            return bool(self.s.gemini_key)
        if not self.cfg.has(link.provider):
            return False
        return True

    @staticmethod
    def _tag(link: JudgeLink) -> str:
        return f"{link.provider}/{link.model}"

    def _fit(self, need_vision: bool, need_youtube: bool) -> list[JudgeLink]:
        return [l for l in self.chain if (l.vision or not need_vision) and (l.youtube or not need_youtube)
                and not (l.provider == DIRECT and self._direct_failed)]

    def _live(self, *, need_vision: bool = False, need_youtube: bool = False) -> list[JudgeLink]:
        """Links worth trying now. Raises `QuotaExhausted` when every fitting link is out of its
        daily quota: those are never retried "anyway", since the provider will only say no."""
        now = time.time()
        fit = [l for l in self._fit(need_vision, need_youtube) if not self.quota.out_until(l)]
        if not fit:
            self._raise_quota(need_vision, need_youtube)
        live = [l for l in fit if self._down.get(self._tag(l), 0) < now]
        return live or fit        # all tripped -> try them anyway rather than fail outright

    def _raise_quota(self, need_vision: bool, need_youtube: bool):
        outs = [self.quota.out_until(l) for l in self._fit(need_vision, need_youtube)]
        until = min([u for u in outs if u] or [time.time() + QUOTA_DOWN_SECONDS])
        raise QuotaExhausted("every judge link is out of its daily quota until "
                             + time.strftime("%Y-%m-%d %H:%M", time.localtime(until)), until)

    def out_of_quota(self, *, need_vision: bool = False, need_youtube: bool = False) -> bool:
        fit = self._fit(need_vision, need_youtube)
        return bool(fit) and all(self.quota.out_until(l) for l in fit)

    def down_links(self) -> dict[str, float]:
        """Links currently circuit-broken -> seconds remaining (for logs and the report)."""
        now = time.time()
        return {tag: round(until - now, 1) for tag, until in self._down.items() if until > now}

    def _trip(self, link: JudgeLink, why: str):
        # Measured 2026-10-29: once a free key's daily quota is spent, retrying it every 5 min for
        # the rest of the day only adds a failed round-trip to every call.
        quota = bool(_QUOTA.search(why)) and bool(_PER_DAY.search(why) or not _PER_MINUTE.search(why))
        secs = DOWN_SECONDS
        ra = _RETRY_AFTER.search(why) or _RETRY_DELAY.search(why)
        if ra:                                  # a per-minute 429 that says when to come back
            secs = min(DOWN_SECONDS, max(1.0, float(ra.group(1))))
        elif _OVERLOADED.search(why[:120]):
            secs = OVERLOAD_SECONDS               # a demand spike: back off briefly, not for 5 minutes
        if quota:
            # persisted, so a restarted run does not spend a request rediscovering it
            secs = self.quota.exhaust(link) - time.time()
            self._quota_down[self._tag(link)] = time.time() + secs
        self._down[self._tag(link)] = time.time() + secs
        log.warning("judge %s unavailable (%s); skipping it for %ds",
                    self._tag(link), why[:160], secs)

    # -- transport -----------------------------------------------------------
    def _post(self, link: JudgeLink, content: list[dict], schema: dict | None,
              timeout: float = TIMEOUT) -> str:
        ep = self.cfg.endpoint(link.provider)
        if ep is None:
            raise JudgeError(f"{link.provider} missing from provider config")
        base, key = ep
        body = {"model": link.model, "temperature": 0.0, "max_tokens": link.max_tokens,
                "messages": [{"role": "user", "content": content}]}
        req = urllib.request.Request(
            base + "/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read()[:200].decode("utf8", "replace").replace("\n", " ")
            ra = (e.headers.get("Retry-After") or "").strip() if e.headers else ""
            if ra.replace(".", "", 1).isdigit():
                detail += f" [retry-after {ra}s]"
            raise JudgeError(f"HTTP {e.code}: {detail}") from e
        except Exception as e:
            raise JudgeError(str(e)[:200]) from e

        choices = payload.get("choices") or []
        if not choices:
            raise JudgeError(f"no choices in reply: {str(payload)[:160]}")
        choice = choices[0]
        text = (choice.get("message") or {}).get("content") or ""
        if not text.strip():
            # Observed: reasoning models spend the whole budget on `reasoning` and return
            # content=null with finish_reason="length". Name it so the log is actionable.
            why = "empty content"
            if choice.get("finish_reason") == "length":
                why = ("truncated before any answer (reasoning used the whole max_tokens budget; "
                       "raise max_tokens for this link)")
            raise JudgeError(why)
        self.calls += 1
        usage = payload.get("usage") or {}
        if isinstance(usage.get("total_tokens"), int):
            self.tokens += usage["total_tokens"]
        return text

    def _send(self, link: JudgeLink, content: list[dict], schema: dict | None,
              timeout: float = TIMEOUT) -> str | None:
        """`_post` behind the daily counter, the per-minute bucket and the in-flight cap.
        None means the link's daily cap is reached and nothing was sent."""
        if not self.quota.acquire(link):
            return None
        with self._slots:
            return self._post(link, content, schema, timeout)

    def _direct_client(self) -> Gemini:
        with self._direct_lock:              # shots run on several threads
            if self._direct is None:
                self._direct = Gemini(self.s, self.cache, fallback=False)
            return self._direct

    # -- core dispatch -------------------------------------------------------
    def _run(self, content_for: callable, schema: dict | None, *, need_vision: bool = False,
             need_youtube: bool = False, direct: callable | None = None,
             timeout: float = TIMEOUT) -> dict:
        """Walk the chain until a link returns parseable JSON. `content_for(link)` builds the parts.

        If every link fails, wait `judge_patience` seconds once and walk the chain again before
        giving up: 5xx and per-minute 429s usually clear within a minute, and giving up costs a
        whole shot. Links out of their *daily* quota are not worth waiting for.
        """
        last: Exception | None = None
        patience = self.s.judge_patience
        for attempt in range(2):
            for link in self._live(need_vision=need_vision, need_youtube=need_youtube):
                try:
                    if link.provider == DIRECT:
                        if direct is None or self._direct_failed:
                            continue
                        if not self.quota.acquire(link):
                            continue
                        with self._slots:
                            data = direct(self._direct_client(), link.model)
                        self.calls += 1
                        return data
                    reply = self._send(link, content_for(link), schema, timeout)
                    if reply is None:
                        continue                       # daily cap reached: not a failure
                    return _extract_json(reply)
                except (JudgeError, GeminiError, ValueError, json.JSONDecodeError) as e:
                    last = e
                    if link.provider == DIRECT and isinstance(e, GeminiError) and "API_KEY" in str(e):
                        self._direct_failed = True     # no key: never retry this link
                    self._trip(link, str(e))
            if self.out_of_quota(need_vision=need_vision, need_youtube=need_youtube):
                self._raise_quota(need_vision, need_youtube)
            if attempt == 0 and patience > 0 and self._recoverable(need_vision, need_youtube):
                log.warning("every judge link failed; waiting %.0fs for one to recover", patience)
                time.sleep(patience)
                continue
            break
        raise JudgeError(f"all judge providers failed; last error: {last}")

    def _recoverable(self, need_vision: bool, need_youtube: bool) -> bool:
        """Is any link down only briefly (not out of daily quota, not missing its key)?"""
        now = time.time()
        for l in self.chain:
            if (l.vision or not need_vision) and (l.youtube or not need_youtube):
                if l.provider == DIRECT and self._direct_failed:
                    continue
                if self._quota_down.get(self._tag(l), 0) < now and not self.quota.out_until(l):
                    return True
        return False

    # -- public API (mirrors Gemini) ----------------------------------------
    def json(self, prompt: str, schema: dict, *, model: str | None = None,
             cache_key: tuple | None = None):
        if cache_key is not None:
            hit = self.cache.get("judge", *cache_key)
            if hit is not None:
                return hit
        full = prompt + _schema_hint(schema)
        data = self._run(lambda l: [{"type": "text", "text": full}], schema,
                         direct=lambda g, m: g.json(prompt, schema, model=m))
        if cache_key is not None:
            self.cache.set("judge", data, *cache_key)
        return data

    def json_youtube(self, url: str, prompt: str, schema: dict, *, start: float | None = None,
                     end: float | None = None, fps: float = 1.0, cache_key: tuple | None = None):
        """Analyse a public YouTube URL without downloading it.

        Verified: the OpenAI-compatible Gemini proxy accepts a watch URL in `image_url`
        (a `type:"file"` part is rejected with HTTP 400).
        """
        if cache_key is not None:
            hit = self.cache.get("judge", *cache_key)
            if hit is not None:
                return hit
        full = prompt + _schema_hint(schema)

        def content(_link):
            return [{"type": "text", "text": full},
                    {"type": "image_url", "image_url": {"url": url}}]

        data = self._run(content, schema, need_vision=True, need_youtube=True,
                         direct=lambda g, m: g.json_youtube(url, prompt, schema, start=start,
                                                            end=end, fps=fps, model=m))
        if cache_key is not None:
            self.cache.set("judge", data, *cache_key)
        return data

    def json_images(self, prompt: str, images: list[bytes], schema: dict, *,
                    cache_key: tuple | None = None):
        """Judge one or more JPEGs (e.g. a storyboard contact sheet). Every vision link qualifies."""
        if cache_key is not None:
            hit = self.cache.get("judge", *cache_key)
            if hit is not None:
                return hit
        full = prompt + _schema_hint(schema)
        urls = ["data:image/jpeg;base64," + base64.b64encode(b).decode() for b in images]

        def content(_link):
            return [{"type": "text", "text": full}] + [
                {"type": "image_url", "image_url": {"url": u}} for u in urls]

        data = self._run(content, schema, need_vision=True, timeout=IMAGE_TIMEOUT,
                         direct=lambda g, m: g.json_images(prompt, images, schema, model=m))
        if cache_key is not None:
            self.cache.set("judge", data, *cache_key)
        return data

    def json_video_file(self, path: Path, prompt: str, schema: dict, *, fps: float = 2.0,
                        cache_key: tuple | None = None):
        """Judge a local clip. Sends a tiled frame grid, which every vision model accepts."""
        if cache_key is not None:
            hit = self.cache.get("judge", *cache_key)
            if hit is not None:
                return hit
        cols, rows = self.s.grid_cols, self.s.grid_rows
        data_url = extract_grid_data_url(Path(path), cols=cols, rows=rows, fps=self.s.grid_fps)
        if data_url is None:
            raise JudgeError(f"could not build a frame grid for {Path(path).name}")
        full = grid_hint(cols, rows) + "\n\n" + prompt + _schema_hint(schema)

        def content(_link):
            return [{"type": "text", "text": full},
                    {"type": "image_url", "image_url": {"url": data_url}}]

        data = self._run(content, schema, need_vision=True,
                         direct=lambda g, m: g.json_video_file(Path(path), prompt, schema, fps=fps,
                                                               model=m))
        if cache_key is not None:
            self.cache.set("judge", data, *cache_key)
        return data


    # -- second opinion ------------------------------------------------------
    AI_CHECK_PROMPT = (
        "Look at these {n} frames sampled across one ~5 second video clip. Decide ONE thing: is "
        "this real camera footage, or AI-generated/CGI?\n"
        "Real (false) includes gorgeous cinematic drone and tripod shots, perfectly still lakes, "
        "heavy colour grading and slow motion - beauty and stillness are NOT evidence of AI.\n"
        "AI (true) shows reflections that do not match their subject, objects changing shape or "
        "merging between frames, smeared painterly texture with no sensor grain, or impossible "
        "geometry.\n"
        'Reply ONLY JSON: {{"ai_generated": <bool>, "evidence": "<15 words>"}}')
    AI_CHECK_SCHEMA = {"type": "object", "properties": {
        "ai_generated": {"type": "boolean"}, "evidence": {"type": "string"}},
        "required": ["ai_generated"]}

    def confirm_ai_generated(self, clip: Path, cache_key: tuple | None = None) -> bool | None:
        """Dedicated single-question AI check for one clip.

        Asking *only* "is this AI-generated?" is far more reliable than reading the
        `ai_or_animated` field out of the 12-field verdict: on five known clips the dedicated
        check scored 4/4, while the combined verdict produced confident errors in both directions
        (a real Tokyo skyline called synthetic at 95% confidence, and a generated clip passed).
        Returns None when no provider could answer.
        """
        if cache_key is not None:
            hit = self.cache.get("judge", *cache_key)
            if hit is not None:
                return hit.get("ai_generated") if isinstance(hit, dict) else None
        cols, rows = self.s.grid_cols, self.s.grid_rows
        data_url = extract_grid_data_url(Path(clip), cols=cols, rows=rows, fps=self.s.grid_fps)
        if data_url is None:
            return None
        question = self.AI_CHECK_PROMPT.format(n=cols * rows)
        prompt = question + _schema_hint(self.AI_CHECK_SCHEMA)

        for link in self._live(need_vision=True):
            try:
                if link.provider == DIRECT:
                    if self._direct_failed or not self.quota.acquire(link):
                        continue
                    jpeg = base64.b64decode(data_url.split(",", 1)[1])
                    with self._slots:
                        data = self._direct_client().json_images(question, [jpeg], self.AI_CHECK_SCHEMA,
                                                                 model=link.model)
                    self.calls += 1
                else:
                    reply = self._send(link, [{"type": "text", "text": prompt},
                                              {"type": "image_url", "image_url": {"url": data_url}}],
                                       self.AI_CHECK_SCHEMA, IMAGE_TIMEOUT)
                    if reply is None:
                        continue
                    data = _extract_json(reply)
            except (JudgeError, GeminiError, ValueError, json.JSONDecodeError) as e:
                self._trip(link, str(e))
                continue
            if cache_key is not None:
                self.cache.set("judge", data, *cache_key)
            return bool(data.get("ai_generated"))
        return None


def make_judge(settings: Settings, cache: Cache) -> Judge | None:
    """Build the judge, or return None to run in offline heuristic mode."""
    try:
        return Judge(settings, cache)
    except JudgeError as e:
        log.warning("%s -> running in OFFLINE heuristic mode (no AI screening/verification)", e)
        return None
