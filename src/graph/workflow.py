"""LangGraph DAG for the ZenSift daily digest.

Flow:
    planner -> fetcher -> deduplicate -> sifter -[ok]-> sanitizer -> editor -[ok]-> assembler
                                           |                          |              ^
                                           +--[llm_down]--> fallback <+--[llm_down]  |
                                                               +---------------------+
    assembler -> delivery -[delivered]-> record_sent -> END
                          -[failed]----> END
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.planner import planner_node
from src.agents.sifter import sifter_node
from src.graph.state import Article, ZenSiftState
from src.utils.deduplicate import db_path_from_env, get_sent_urls, mark_sent
from src.utils.fetcher import fetcher_node

__all__ = ["Article", "ZenSiftState", "build_graph", "compile_workflow", "run"]

FALLBACK_ARTICLE_COUNT = 3

Node = Callable[[ZenSiftState], dict]

logger = logging.getLogger(__name__)


# --- Nodes -------------------------------------------------------------------


def deduplicate_node(state: ZenSiftState) -> dict:
    """Drop articles already sent (SQLite) or repeated within this run. Read-only."""
    raw = state.get("raw_articles", [])
    try:
        already_sent = get_sent_urls((a.url for a in raw), db_path_from_env())
    except sqlite3.Error as exc:
        # A broken ledger must not block today's email; worst case is a repeat link.
        logger.exception("Dedup lookup failed; continuing without it.")
        return {"fresh_articles": raw, "errors": [f"deduplicate: {exc!r}"]}

    seen: set[str] = set()
    fresh = []
    for article in raw:
        if article.url in already_sent or article.url in seen:
            continue
        seen.add(article.url)
        fresh.append(article)
    return {"fresh_articles": fresh}


def sanitizer_node(state: ZenSiftState) -> dict:
    """Step 5: wrap article text in <article> delimiters."""
    return {"sanitized_articles": []}


def editor_node(state: ZenSiftState) -> dict:
    """Step 5: summarize with the editor persona."""
    return {"digest_articles": state.get("sifted_articles", [])}


def fallback_node(state: ZenSiftState) -> dict:
    """Top 3 fetched articles, untouched by AI. Uses the deduped list to avoid repeats."""
    articles = state.get("fresh_articles", [])[:FALLBACK_ARTICLE_COUNT]
    return {"digest_articles": articles, "degraded": True}


def assembler_node(state: ZenSiftState) -> dict:
    """Step 6: render the HTML email. No DB access."""
    return {"email_html": ""}


def delivery_node(state: ZenSiftState) -> dict:
    """Step 6: send via Resend; "sent" only when Resend confirms with an email id."""
    return {"delivery_status": "failed"}


def record_sent_node(state: ZenSiftState) -> dict:
    """Persist sent URLs. Only reachable after a successful send."""
    urls = [a.url for a in state.get("digest_articles", [])]
    try:
        mark_sent(urls, db_path_from_env())
    except sqlite3.Error as exc:
        logger.exception("Failed to record sent URLs; they may repeat tomorrow.")
        return {"errors": [f"record_sent: {exc!r}"]}
    return {}


def llm_guarded(name: str, node: Node) -> Node:
    """Convert an exception that survived tenacity retries into llm_failed=True."""

    def _run(state: ZenSiftState) -> dict:
        try:
            return node(state)
        except Exception as exc:
            logger.exception("%s failed after retries; degrading to fallback.", name)
            return {"llm_failed": True, "errors": [f"{name}: {exc!r}"]}

    return _run


# --- Routing -----------------------------------------------------------------


def route_on_llm(state: ZenSiftState) -> Literal["ok", "llm_down"]:
    return "llm_down" if state.get("llm_failed") else "ok"


def route_on_delivery(state: ZenSiftState) -> Literal["delivered", "failed"]:
    if state.get("delivery_status") == "sent":
        return "delivered"
    logger.error("Delivery failed; URLs not recorded so they can be retried next run.")
    return "failed"


# --- Graph -------------------------------------------------------------------


def build_graph() -> StateGraph:
    graph = StateGraph(ZenSiftState)

    graph.add_node("planner", planner_node)
    graph.add_node("fetcher", fetcher_node)
    graph.add_node("deduplicate", deduplicate_node)
    graph.add_node("sifter", llm_guarded("sifter", sifter_node))
    graph.add_node("sanitizer", sanitizer_node)
    graph.add_node("editor", llm_guarded("editor", editor_node))
    graph.add_node("fallback", fallback_node)
    graph.add_node("assembler", assembler_node)
    graph.add_node("delivery", delivery_node)
    graph.add_node("record_sent", record_sent_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "fetcher")
    graph.add_edge("fetcher", "deduplicate")
    graph.add_edge("deduplicate", "sifter")
    graph.add_conditional_edges(
        "sifter", route_on_llm, {"ok": "sanitizer", "llm_down": "fallback"}
    )
    graph.add_edge("sanitizer", "editor")
    graph.add_conditional_edges(
        "editor", route_on_llm, {"ok": "assembler", "llm_down": "fallback"}
    )
    graph.add_edge("fallback", "assembler")
    graph.add_edge("assembler", "delivery")
    graph.add_conditional_edges(
        "delivery", route_on_delivery, {"delivered": "record_sent", "failed": END}
    )
    graph.add_edge("record_sent", END)

    return graph


def compile_workflow() -> CompiledStateGraph:
    return build_graph().compile()


def run(email: str, interests: list[str], avoid: list[str] | None = None) -> ZenSiftState:
    app = compile_workflow()
    return app.invoke(
        {"email": email, "interests": interests, "avoid": avoid or [], "errors": []}
    )
