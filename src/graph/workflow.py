"""LangGraph DAG for the ZenSift daily digest.

Flow:
    planner -> fetcher -> deduplicate -> sifter -[ok]-> reader -> sanitizer -> editor -[ok]-> assembler
                                           |                                          |              ^
                                           +--[llm_down]--> fallback <----------------+--[llm_down]  |
                                                               +-------------------------------------+
    assembler -> delivery -[delivered]-> record_sent -> END
                          -[failed]----> END   (includes "already_sent": duplicate blocked)
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Literal
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.editor import editor_node
from src.agents.planner import planner_node
from src.agents.sifter import sifter_node
from src.config import ZenSiftConfig
from src.graph.state import Article, ZenSiftState
from src.utils.assembler import assembler_node
from src.utils.deduplicate import db_path_from_env, get_sent_urls, mark_sent
from src.utils.delivery import delivery_node, idempotency_key
from src.utils.fetcher import fetcher_node
from src.utils.reader import read_articles, reader_node
from src.utils.sanitizer import sanitizer_node

__all__ = ["Article", "ZenSiftState", "build_graph", "compile_workflow", "run"]

FALLBACK_ARTICLE_COUNT = 3
# Extra candidates so a few unreadable pages still leave enough for the fallback.
FALLBACK_CANDIDATES = 6

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


def fallback_node(state: ZenSiftState) -> dict:
    """Top 3 readable articles, untouched by AI, shown as their opening lines."""
    already_read = [a for a in state.get("sifted_articles", []) if a.body]
    if already_read:
        return {"digest_articles": already_read[:FALLBACK_ARTICLE_COUNT], "degraded": True}

    # Uses the deduped list to avoid repeats.
    candidates = state.get("fresh_articles", [])[:FALLBACK_CANDIDATES]
    readable, errors = read_articles(candidates)
    return {
        "digest_articles": readable[:FALLBACK_ARTICLE_COUNT],
        "degraded": True,
        "unreadable_count": len(candidates) - len(readable),
        "errors": errors,
    }


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
    status = state.get("delivery_status")
    if status == "sent":
        return "delivered"
    if status != "already_sent":
        logger.error("Delivery failed; URLs not recorded so they can be retried next run.")
    return "failed"


# --- Graph -------------------------------------------------------------------


def build_graph() -> StateGraph:
    graph = StateGraph(ZenSiftState)

    graph.add_node("planner", planner_node)
    graph.add_node("fetcher", fetcher_node)
    graph.add_node("deduplicate", deduplicate_node)
    graph.add_node("sifter", llm_guarded("sifter", sifter_node))
    graph.add_node("reader", reader_node)
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
        "sifter", route_on_llm, {"ok": "reader", "llm_down": "fallback"}
    )
    graph.add_edge("reader", "sanitizer")
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


def initial_state(
    config: ZenSiftConfig, today: date | None = None, force_send: bool = False
) -> ZenSiftState:
    digest_date = today or datetime.now(UTC).date()
    key = idempotency_key(digest_date)
    if force_send:
        # Manual testing only: a unique key bypasses today's duplicate protection.
        key = f"{key}/manual-{uuid4().hex[:8]}"
    return {
        "interests": config.interests,
        "avoid": config.avoid,
        "sifter": config.sifter,
        "digest": config.digest,
        "digest_date": digest_date,
        "idempotency_key": key,
        "errors": [],
    }


def run(config: ZenSiftConfig, today: date | None = None, force_send: bool = False) -> ZenSiftState:
    return compile_workflow().invoke(initial_state(config, today, force_send))
