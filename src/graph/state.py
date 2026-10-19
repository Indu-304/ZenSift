"""Shared graph state and domain models (kept separate to avoid circular imports)."""

from __future__ import annotations

import operator
from datetime import date
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel

from src.config import DigestSettings, SifterSettings


class Article(BaseModel):
    title: str
    url: str
    # Publisher URL resolved by the reader; `url` stays the feed URL so dedup keeps matching.
    link: str = ""
    snippet: str = ""
    body: str = ""
    source: str = ""
    similarity: float | None = None
    avoid_similarity: float | None = None
    summary: str | None = None
    tone_score: float | None = None


class ZenSiftState(TypedDict, total=False):
    # Input
    interests: list[str]
    avoid: list[str]
    sifter: SifterSettings
    digest: DigestSettings
    digest_date: date
    idempotency_key: str

    # Planner output
    rss_urls: list[str]
    hn_queries: list[str]

    # Pipeline data
    raw_articles: list[Article]
    fresh_articles: list[Article]
    sifted_articles: list[Article]
    sanitized_articles: list[str]
    digest_articles: list[Article]
    unreadable_count: int

    # Output
    email_subject: str
    email_html: str
    email_text: str
    email_id: str
    delivery_status: Literal["sent", "already_sent", "failed"]

    # Control flow / observability
    llm_failed: bool
    degraded: bool
    errors: Annotated[list[str], operator.add]
