"""Planner: turn a natural-language goal into Google News RSS URLs and HN keywords."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated
from urllib.parse import urlencode

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, StringConstraints

from src.agents.llm import build_chat_model, llm_retry

if TYPE_CHECKING:
    from src.graph.state import ZenSiftState

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = (
    "You are an expert research librarian. Based on the user's goal, generate "
    "3 Google News RSS queries and 3 Hacker News keywords. Return strictly JSON."
)

GOOGLE_NEWS_RSS_SEARCH = "https://news.google.com/rss/search"
MAX_FALLBACK_QUERY_CHARS = 200

Query = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]


class SearchPlan(BaseModel):
    google_news_queries: list[Query] = Field(
        min_length=3,
        max_length=3,
        description=(
            "Exactly 3 Google News search queries. May use quotes for phrases and "
            "-term to exclude unwanted topics."
        ),
    )
    hackernews_keywords: list[Query] = Field(
        min_length=3,
        max_length=3,
        description="Exactly 3 short keyword phrases for Hacker News search.",
    )


def google_news_rss_url(query: str, hl: str = "en-US", gl: str = "US", ceid: str = "US:en") -> str:
    params = urlencode({"q": query, "hl": hl, "gl": gl, "ceid": ceid})
    return f"{GOOGLE_NEWS_RSS_SEARCH}?{params}"


@llm_retry
def generate_plan(interests: list[str], llm: BaseChatModel) -> SearchPlan:
    structured_llm = llm.with_structured_output(SearchPlan)
    goals = "\n".join(f"- {topic}" for topic in interests)
    return structured_llm.invoke(
        [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f"User goals:\n{goals}"),
        ]
    )


def fallback_query(interests: list[str]) -> str:
    return " OR ".join(f"({topic})" for topic in interests)[:MAX_FALLBACK_QUERY_CHARS]


def planner_node(state: ZenSiftState) -> dict:
    interests = state["interests"]
    try:
        plan = generate_plan(interests, build_chat_model(temperature=0.2))
    except Exception as exc:
        # No fallback edge after the planner; search the raw goals so the email still goes out.
        logger.exception("Planner failed; falling back to raw interests as the query.")
        return {
            "rss_urls": [google_news_rss_url(fallback_query(interests))],
            "hn_queries": [],
            "errors": [f"planner: {exc!r}"],
        }

    return {
        "rss_urls": [google_news_rss_url(q) for q in plan.google_news_queries],
        "hn_queries": plan.hackernews_keywords,
    }
