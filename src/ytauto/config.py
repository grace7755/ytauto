"""Settings: config/default.yaml (+ optional project override) and .env secrets."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

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
    shot_target: float = 4.5
    shot_min: float = 3.0
    shot_max: float = 6.0
    # youtube search
    queries_per_segment: int = 4
    results_per_query: int = 15
    min_source_height: int = 1080
    min_source_duration: int = 15
    max_source_duration: int = 3600
    max_age_years_soft: int = 6
    screen_top_k: int = 4
    max_shots_per_source: int = 3
    rounds_per_shot: int = 2
    # "storyboard": judge the video's seek-bar preview sheet (~12 s, ~3k tokens, any vision link);
    # "url": the judge watches the YouTube URL (~50 s, ~240k tokens, YouTube-capable links only).
    screen_mode: Literal["storyboard", "url"] = "storyboard"
    # speed: shots selected concurrently, and candidates screened concurrently within a shot.
    # Judge rate limits are the ceiling - raising these past ~4 mostly buys 429s.
    shot_workers: int = 3
    screen_workers: int = 3
    judge_concurrency: int = 4        # max judge calls in flight at once, across all threads
    # seconds to wait for a judge link to recover before giving a shot up as "unavailable"
    judge_patience: float = 60.0
    title_blacklist: list[str] = Field(default_factory=lambda: [
        "ai generated", "ai-generated", "sora", "veo 3", "kling", "midjourney",
        "reaction", "podcast", "slideshow", "gameplay", "minecraft", "anime"])
    # benchmark
    min_score: float = 70
    min_relevance: float = 60
    min_clip_height: int = 1080
    blur_threshold: float = 60.0       # Laplacian variance (on 1080p-normalised frame)
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
