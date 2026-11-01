from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Segment(BaseModel):
    idx: int
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


class Brief(BaseModel):
    # What the clip MUST show: one filmable thing. Screening and verification judge usability and
    # relevance on this alone; `visual` is the ideal framing and only affects ranking. Measured on
    # the speed test, briefs stacking details made the judge reject footage of the right subject.
    subject: str = ""
    visual: str = ""                       # the ideal shot (framing, light, move): nice to have
    entities: list[str] = Field(default_factory=list)
    youtube_queries: list[str] = Field(default_factory=list)
    stock_queries: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    # Where the whole video is set ("Alaska"). The brief review strips the subject down to one
    # filmable thing ("Remote house"), so without this the judge happily accepted a South Asian
    # river town and a tropical stilt house for an Alaska documentary.
    setting: str = ""
    # Ordinary b-roll that any stock library has (open sea, aerial coast, crowd, sunset) and the
    # narration names no specific place, person, event or landmark: stock is tried before YouTube
    # (faster, licence-clean, no AI-footage ban risk). See `Settings.stock_first`.
    generic: bool = False
    # Decided up front (Round 6 visual ladder): what kind of picture this line needs. footage = video (default, the old
    # behaviour); photo = a still of the named thing first (an obscure product rarely exists as clean footage); graphic = a
    # card drawn over a background (spec numbers, a quote, a list). Graphics are drawn by Remotion; until those cards exist
    # (`graphics_enabled`), a graphic shot is filled like a footage shot.
    visual_kind: str = "footage"
    graphic: str = ""                      # spec | quote | list | compare | diagram | map (visual_kind == "graphic")
    # Set only on a category-fallback brief: the exact subject this broader category stands in for.
    category_of: str = ""

    def names(self) -> str:
        """The prompts' "names involved" line, with the setting as a hard requirement."""
        base = ", ".join(self.entities) or "-"
        if not self.setting:
            return base
        return (f"{base}. Setting of the whole video: {self.setting} - the footage must plausibly be "
                f"filmed there; a clearly different country, climate or culture is not usable "
                f"(relevance at most 20)")

    def key(self) -> str:
        """Cache-key part for the names; unchanged for briefs without a setting."""
        return ",".join(self.entities) + (f"|{self.setting}" if self.setting else "")


class Shot(BaseModel):
    id: int
    segment: int
    start: float
    end: float
    text: str
    variant: int = 0                       # nth shot inside its segment (for variety)
    brief: Brief | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


class Candidate(BaseModel):
    provider: Literal["youtube", "pexels", "pixabay", "coverr", "mixkit", "still_pexels", "still_pixabay", "still_commons"]
    source_id: str
    url: str
    title: str = ""
    channel: str = ""
    duration: float = 0
    height: int = 0
    upload_date: str = ""                  # YYYYMMDD
    license: str = "unknown"
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    direct_url: str | None = None          # stock download URL
    local_path: str | None = None          # stock file already on disk
    text_score: float = 0.0
    window: tuple[float, float] | None = None   # suggested [start,end] in source (seconds)
    screen_flags: dict = Field(default_factory=dict)


class ClipScore(BaseModel):
    relevance: float = 0
    quality: float = 0
    authenticity: float = 0
    composition: float = 0
    recency: float = 0
    cleanliness: float = 0
    narrative: float = 0
    total: float = 0


class Decision(BaseModel):
    shot: int
    # "unavailable" = our judge chain was exhausted, so this candidate was never actually judged.
    # It is deliberately NOT "rejected": nothing is known about the footage.
    status: Literal["accepted", "rejected", "weak", "unavailable"]
    provider: str
    source_id: str
    url: str
    title: str = ""
    license: str = "unknown"
    src_start: float = 0
    src_end: float = 0
    clip_path: str = ""
    score: ClipScore | None = None
    hard_rejects: list[str] = Field(default_factory=list)
    notes: str = ""
    credit: str = ""                       # author of a still picture (for the credits list)
    ts: str = ""
    run: str = ""                          # "run" or "redo-<n>": which pass logged it (old ledgers: "")
