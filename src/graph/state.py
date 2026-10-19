"""Shared graph state and domain models (kept separate to avoid circular imports)."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel


class Article(BaseModel):
    title: str
    url: str
    snippet: str = ""
    source: str = ""
    similarity: float | None = None
    avoid_similarity: float | None = None
    summary: str | None = None
    tone_score: float | None = None


class ZenSiftState(TypedDict, total=False):
    # Input
    email: str
    interests: str
    avoid: list[str]

    # Planner output
    rss_urls: list[str]
    hn_queries: list[str]

    # Pipeline data
    raw_articles: list[Article]
    fresh_articles: list[Article]
    sifted_articles: list[Article]
    sanitized_articles: list[str]
    digest_articles: list[Article]

    # Output
    email_html: str
    delivery_status: Literal["sent", "failed"]

    # Control flow / observability
    llm_failed: bool
    degraded: bool
    errors: Annotated[list[str], operator.add]
