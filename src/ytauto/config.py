"""Settings: config/default.yaml (+ optional project override) and .env secrets."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[2]


class Weights(BaseModel):
    relevance: float = 35
    quality: float = 20
    authenticity: float = 10
    composition: float = 10
    recency: float = 10
    cleanliness: float = 10
    narrative: float = 5


class JudgeLink(BaseModel):
    """One provider+model link in the judge fallback chain.

    `provider` is a key in the shared provider config (see HEART_CONFIG), or the sentinel
    `_direct` meaning the native google-genai client with GEMINI_API_KEY.
    """
    provider: str
    model: str
    vision: bool = False      # can analyse images / frame grids
    youtube: bool = False     # can analyse a public YouTube URL without downloading
    max_tokens: int = 2048
    # Client-side limits (None = unknown). Free links are capped per day; counting requests
    # ourselves skips a spent link without calling it. See llm/quota.py.
    rpm: int | None = None
    rpd: int | None = None
    # reasoning effort for this link: low | medium | high (None = the tool's own default). Passed to HeartCode
    # (--effort) and Codex (model_reasoning_effort); other links ignore it. A per-task value in `judge_effort` is the
    # fallback when the link sets none.
    effort: str | None = None

    @field_validator("model")
    @classmethod
    def _no_sol(cls, v: str) -> str:
        """GPT Sol is excluded from this system (owner's decision, 1 Nov 2026): Luna by default, Terra when Luna
        cannot answer, Gemini when the ChatGPT plan is at its limit."""
        if re.search(r"(?<![a-z0-9])sol(?![a-z0-9])", v, re.I):
            raise ValueError(f"model {v!r}: GPT Sol is excluded from this project (use Luna or Terra)")
        return v

    @field_validator("effort")
    @classmethod
    def _effort_level(cls, v: str | None) -> str | None:
        if v is not None and v.lower() not in ("low", "medium", "high"):
            raise ValueError("effort must be low, medium or high")
        return v.lower() if v else v


# Verified working 2026-10-29; order = fastest and most accurate first.
DEFAULT_JUDGE_CHAIN = [
    JudgeLink(provider="geminiopenai", model="models/gemini-2.5-flash-lite", vision=True, youtube=True),
    JudgeLink(provider="geminiopenai", model="models/gemini-flash-lite-latest", vision=True, youtube=True),
    # Reasoning model: it spends the budget thinking before it answers, and 2048 was still being
    # exhausted before any answer on real screening prompts (observed 2026-10-29).
    JudgeLink(provider="zai", model="GLM-4.6V-Flash", vision=True, max_tokens=8192),
    JudgeLink(provider="_direct", model="gemini-3-flash-preview", vision=True, youtube=True),
]


class Settings(BaseModel):
    # models
    gemini_model: str = "gemini-3.8-flash"
    gemini_plan_model: str = "gemini-3.8-flash"
    # judge: ordered provider fallback chain (see config/default.yaml)
    judge_chain: list[JudgeLink] = Field(default_factory=lambda: list(DEFAULT_JUDGE_CHAIN))
    # frame grid sent to vision models in place of video
    grid_cols: int = 3
    grid_rows: int = 2
    grid_fps: float = 1.2
    # output
    width: int = 1920
    height: int = 1080
    fps: int = 30
    crf: int = 18
    crossfade: float = 0.0  # seconds between segments; 0 = hard cuts
    # shots
    shot_target: float = 4.5      # no clip in the final video is longer than shot_max
    shot_min: float = 3.0
    shot_max: float = 5.0
    # youtube search
    queries_per_segment: int = 4
    # one extra planner call per ~40 shots that strips details the narration does not support
    plan_critic: bool = True
    results_per_query: int = 15
    min_source_height: int = 1080
    min_source_duration: int = 15
    max_source_duration: int = 3600
    max_age_years_soft: int = 6
    screen_top_k: int = 4
    max_shots_per_source: int = 0     # per YouTube video; 0 = no limit (sections never overlap)
    # hand fixes for single shots, applied over the planner's brief: {shot_id: {subject: ..., visual: ...,
    # youtube_queries: [...], stock_queries: [...]}} - e.g. when no footage of the planned subject exists
    brief_overrides: dict[int, dict] = Field(default_factory=dict)
    # neighbouring shots may not take footage of one video closer than this (seconds); 0 = off
    adjacent_source_gap: float = 30.0
    rounds_per_shot: int = 2
    # clips downloaded and judged per shot in one select() before YouTube is given up (0 = no budget); see Selector._over_budget
    max_downloads_per_shot: int = 8
    # film-wide stock skip: a stock/still item judged irrelevant (relevance <= stock_skip_relevance) on this many OTHER
    # shots is not tried again unless its title shares a word with the shot's subject. 0 = off
    stock_skip_after: int = 2
    stock_skip_relevance: float = 25.0
    # "storyboard": judge the video's seek-bar preview sheet (~12 s, ~3k tokens, any vision link);
    # "url": the judge watches the YouTube URL (~50 s, ~240k tokens, YouTube-capable links only).
    # "storyboard_only": as storyboard, but a video without one is skipped instead of URL-screened
    # (the one call that costs ~240k tokens; only ~5% of candidates have no storyboard).
    screen_mode: Literal["storyboard", "storyboard_only", "url"] = "storyboard_only"
    # storyboards screened per judge call (1 = one call per candidate, all of a round at once)
    screen_batch: int = 2
    # local CLIP pre-ranking of candidates before any judge call: off | rank | filter
    prerank: Literal["off", "rank", "filter"] = "rank"
    prerank_margin: float = 0.04       # filter: drop candidates this far below the best similarity
    # speed: shots selected concurrently, and candidates screened concurrently within a shot.
    # Judge rate limits are the ceiling - raising these past ~4 mostly buys 429s.
    shot_workers: int = 2
    prefetch_ahead: int = 3          # shots whose judge-free work is prepared in the background (0 = off)
    screen_workers: int = 3
    judge_concurrency: int = 4        # max judge calls in flight at once, across all threads
    # seconds before a slow judge call is hedged with the next link (first answer wins); 0 = off
    judge_hedge_after: float = 0.0
    # reject a clip whose text stays pinned in place over moving footage, found by local OCR
    # (inspect/overlay.py); needs the optional rapidocr-onnxruntime package, off without it
    local_overlay_check: bool = True
    # `provider: _heart` judge links (ChatGPT via the HeartCode CLI, see llm/heart.py):
    # processes running at once, and seconds before one is killed
    heart_concurrency: int = 2
    heart_timeout: float = 90.0
    # `provider: _codex` judge links (ChatGPT plan via OpenAI's Codex CLI, see llm/codex.py)
    codex_concurrency: int = 2
    codex_timeout: float = 120.0
    # start the next screened candidate's section download while the current clip is still being
    # cut and judged (a download is ~10 s of pure transfer, the judge call can be 20 s or more)
    download_ahead: bool = True
    # skip the dedicated AI-footage check on a clip that is already rejected on score (it can only
    # lower the score further); run it later only if that clip is promoted to a weak match
    defer_ai_check: bool = True
    # seconds to wait for a judge link to recover before giving a shot up as "unavailable"
    judge_patience: float = 60.0
    # screened candidates whose best window scores below this (0-100) are tried only after the
    # better ones of the round. Measured on the Philippines run (217 downloads): below 70 only 6 of
    # 39 downloads became a usable clip, 80+ gave 55 of 154. A deferral, never a reject; 0 = off
    screen_min_match: int = 70
    # a shot that is the 2nd/3rd shot of one sentence first tries a different scene of the video
    # its previous sibling used (no search, screening or download); 0 = off. The first shot's
    # section is downloaded this many seconds longer so the sibling has footage to cut from
    sibling_extend: float = 9.0
    # judge links at the head of the chain used side by side (least expected wait first) instead
    # of strictly in order; 1 = plain chain order
    judge_lanes: int = 2
    # reasoning effort per task (plan, critic, screen, verify, aicheck, review, facts): low | medium | high.
    # Empty = every tool's own default. Set from a measurement (scripts/measure_effort.py), not by guess.
    judge_effort: dict[str, str] = Field(default_factory=dict)
    # answers needed before a clip or photo is rejected as AI-generated: 2 = an "AI" answer is asked again and must be repeated
    # (GPT Luna flipped on 4 of 24 real clips when asked twice); 1 = the old single answer
    ai_check_votes: int = 2
    # local face check on the storyboard tiles the judge picked: a presenter-sized face only moves a tile /
    # video to the back of the queue (never rejects). Off until measured on your ledgers:
    # scripts/measure_face_tiles.py
    storyboard_face_check: bool = False
    # a judge link whose requests-per-minute bucket would make a call wait longer than this many seconds
    # is skipped for the next live link; the call only waits when every link is busy. 0 = always wait
    judge_spill_after: float = 3.0
    # copy every accepted clip into a shared library (`<cache_dir>/library`) and search it with
    # CLIP before YouTube; a library clip skips search, screening and download, never the checks
    library: bool = True
    library_min_sim: float = 0.28     # CLIP text-image similarity to try a library clip (median for a
    # clip and its own subject, measured on 112 clips; the AI verify still decides)
    library_tries: int = 2
    # a shot the planner marked `generic` (ordinary b-roll) tries stock before YouTube: no search,
    # screening or AI-ban risk, and it keeps YouTube near 65-70% of the footage. False = YouTube first
    stock_first: bool = True
    # reusing earlier footage is the last resort before a weak match: a clip appears at most `reuse_max_uses`
    # times in the video, `reuse_min_gap` seconds apart (timeline), never in neighbouring shots, and it is
    # judged again against the new shot. 0 uses = never reuse
    reuse_max_uses: int = 2
    reuse_min_gap: float = 60.0
    reuse_tries: int = 2
    # export each accepted clip for Remotion as soon as it is accepted (1080p30, padded tail,
    # optional baked look); None = off. See `ytauto export-remotion`
    remotion_export: str | None = None
    remotion_bake: bool = True
    remotion_grain: str | None = None   # grain overlay video baked in (blend overlay), optional
    remotion_dust: str | None = None    # particles-on-black video baked in (blend screen), optional
    # build the ffmpeg final.mp4 (+ QA on it) in `run`/`redo`: auto = only when not exporting to
    # Remotion (that workflow renders in Remotion and never used it) | always | never
    ffmpeg_render: str = "auto"
    title_blacklist: list[str] = Field(default_factory=lambda: [
        "ai generated", "ai-generated", "sora", "veo 3", "kling", "midjourney",
        "reaction", "podcast", "slideshow", "gameplay", "minecraft", "anime"])
    # benchmark
    min_score: float = 70
    min_relevance: float = 60
    min_clip_height: int = 1080
    blur_threshold: float = 40.0       # Laplacian variance (on 1080p-normalised frame); 60 rejected good soft/underwater clips
    min_saturation: float = 0.06       # mean HSV S (0-1); below = B&W
    max_black_ratio: float = 0.15
    max_face_ratio: float = 0.12       # face area / frame area -> talking head
    weights: Weights = Field(default_factory=Weights)
    use_gemini_verify: bool = True
    # mixkit (CLI-based; no API key - see mixkit-dl GUIDE.md)
    mixkit_timeout: int = 300          # seconds for one mixkit-dl call
    mixkit_max_terms: int = 3          # keyword-ladder steps per shot (keeps Mixkit happy)
    # Clips tried per stock provider before moving on. Measured on the 12-shot run: every stock
    # acceptance came at try 1-3, while all-fail shots spent 4 tries on each of 4 providers.
    stock_tries_per_provider: int = 3
    # fallback order after YouTube
    fallback_order: list[str] = Field(default_factory=lambda: ["pexels", "pixabay", "coverr", "mixkit"])
    # last resort for a shot nothing else fills: footage whose only defect is a small logo in one corner is
    # re-cut `watermark_crop_zoom`x closer so the corner falls outside the picture, then judged again.
    # Whether the footage may be reused at all (licence) stays the user's call; the source is in the ledger
    watermark_crop: bool = True
    watermark_crop_zoom: float = 1.15
    # still images: the last resort for a shot no video fills (see sources/images.py, Selector._still).
    # A still is judged like a clip (relevance, no watermark/text, not AI) and becomes a slow push-in,
    # pull-out or pan clip. `still_share_warn` is a soft guide in % of running time, not a limit.
    stills: bool = True
    still_order: list[str] = Field(default_factory=lambda: ["pexels", "pixabay", "commons"])
    still_min_relevance: float = 75.0   # stricter than clips: keyword-level matches are not enough
    still_min_width: int = 2000        # px; a 12% push-in on a smaller picture goes soft
    still_tries: int = 3               # pictures judged per shot, across all providers
    still_share_warn: float = 10.0
    # qa
    qa_max_loops: int = 2
    # paths
    projects_dir: Path = ROOT / "projects"
    cache_dir: Path = ROOT / "cache"
    mixkit_dir: Path = ROOT.parent / "mixkit-downloads"
    cookies_from_browser: str | None = None

    # secrets (from env)
    @property
    def heart_config(self) -> str | None:
        """Path to the shared OpenAI-compatible provider config. Read-only; never written."""
        return os.getenv("HEART_CONFIG")

    @property
    def gemini_key(self) -> str | None:
        return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    @property
    def pexels_key(self) -> str | None:
        return os.getenv("PEXELS_API_KEY")

    @property
    def pixabay_key(self) -> str | None:
        return os.getenv("PIXABAY_API_KEY")

    @property
    def coverr_key(self) -> str | None:
        return os.getenv("COVERR_API_KEY")


def load_settings(extra: Path | None = None) -> Settings:
    load_dotenv(ROOT / ".env")
    data: dict = {}
    for p in (ROOT / "config" / "default.yaml", extra):
        if p and Path(p).exists():
            data.update(yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {})
    return Settings(**data)
