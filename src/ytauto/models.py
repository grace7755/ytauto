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
    provider: Literal["youtube", "pexels", "pixabay", "coverr", "mixkit"]
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
    ts: str = ""
