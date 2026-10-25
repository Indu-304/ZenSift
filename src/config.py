"""Load and validate config.yaml. Secrets (API keys, recipient email) live in .env, not here."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    # Typos like `similarity_treshold` fail loudly instead of being silently ignored.
    model_config = ConfigDict(extra="forbid")


class SifterSettings(_Strict):
    # 0.62 is the middle of the zero-junk range (0.60-0.64 at margin 0.05) across two real runs.
    similarity_threshold: float = Field(0.62, ge=0.0, le=1.0)
    min_margin: float = Field(0.05, ge=-1.0, le=1.0)
    max_articles: int = Field(8, ge=1, le=10)
    exclude_sources: list[str] = Field(default_factory=list)


class DigestSettings(_Strict):
    hard_drop_tone: float = Field(0.15, ge=0.0, le=1.0)
    avoid_drop_tone: float = Field(0.40, ge=0.0, le=1.0)
    avoid_hit_similarity: float = Field(0.60, ge=0.0, le=1.0)


class ZenSiftConfig(_Strict):
    delivery_time: str = Field("02:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    interests: list[str] = Field(min_length=1)
    avoid: list[str] = Field(default_factory=list)
    sifter: SifterSettings = Field(default_factory=SifterSettings)
    digest: DigestSettings = Field(default_factory=DigestSettings)


def load_config(path: str | Path = "config.yaml") -> ZenSiftConfig:
    with open(path, encoding="utf-8") as f:
        return ZenSiftConfig.model_validate(yaml.safe_load(f) or {})
