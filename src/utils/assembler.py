"""Assembler: apply tone rules, order by relevance, and render the digest as HTML and text."""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from html import escape
from urllib.parse import urlparse

from src.config import DigestSettings
from src.graph.state import Article, ZenSiftState

logger = logging.getLogger(__name__)

DEGRADED_NOTICE = (
    "AI summaries weren't available today, so here are a few top links without "
    "filtering or summaries. Normal service should resume tomorrow."
)
QUIET_DAY = "Nothing cleared the bar today. Enjoy the quiet."


def apply_tone_rules(articles: list[Article], settings: DigestSettings) -> list[Article]:
    """Drop alarmist articles; unscored (fallback) articles pass through untouched."""
    kept = []
    for article in articles:
        tone = article.tone_score
        if tone is not None and tone < settings.hard_drop_tone:
            logger.info("drop (hard tone %.2f)  %s", tone, article.title[:70])
            continue
        near_avoid = (
            article.avoid_similarity is not None
            and article.avoid_similarity >= settings.avoid_hit_similarity
        )
        if near_avoid and tone is not None and tone < settings.avoid_drop_tone:
            logger.info(
                "drop (avoid %.2f + tone %.2f)  %s", article.avoid_similarity, tone, article.title[:70]
            )
            continue
        kept.append(article)
    return kept


def order_by_relevance(articles: list[Article]) -> list[Article]:
    # Most similar first; unscored fallback articles keep their order at the end (sort is stable).
    return sorted(articles, key=lambda a: (a.similarity is None, -(a.similarity or 0.0)))


def _safe_url(url: str) -> str | None:
    return url if urlparse(url).scheme in ("http", "https") else None


def _pretty_date(d: date) -> str:
    return f"{d:%A}, {d.day} {d:%B %Y}"


def build_subject(articles: list[Article], digest_date: date, degraded: bool) -> str:
    day = f"{digest_date:%a} {digest_date.day} {digest_date:%b}"
    if degraded:
        return f"ZenSift · {len(articles)} links (summaries unavailable) · {day}"
    if not articles:
        return f"ZenSift · a quiet day · {day}"
    reads = "calm read" if len(articles) == 1 else "calm reads"
    return f"ZenSift · {len(articles)} {reads} · {day}"


def _html_item(article: Article) -> str:
    title = escape(article.title)
    url = _safe_url(article.url)
    heading = (
        f'<a href="{escape(url, quote=True)}" style="color:#2b4c7e;text-decoration:none;">{title}</a>'
        if url
        else title
    )
    source = f'<div style="color:#888;font-size:13px;margin:2px 0 6px;">{escape(article.source)}</div>' if article.source else ""
    summary = f'<p style="margin:0;">{escape(article.summary)}</p>' if article.summary else ""
    return f'<div style="margin:0 0 24px;"><div style="font-size:17px;">{heading}</div>{source}{summary}</div>'


def render_html(articles: list[Article], digest_date: date, degraded: bool) -> str:
    notice = (
        f'<p style="background:#fff4e0;padding:12px;border-radius:6px;font-size:14px;">{escape(DEGRADED_NOTICE)}</p>'
        if degraded
        else ""
    )
    body = "".join(_html_item(a) for a in articles) or f"<p>{escape(QUIET_DAY)}</p>"
    return (
        '<!doctype html><html><body style="margin:0;background:#f6f5f2;">'
        '<div style="max-width:600px;margin:0 auto;padding:24px;font-family:Georgia,serif;'
        'color:#2b2b2b;line-height:1.55;">'
        '<h1 style="font-size:22px;font-weight:normal;margin:0;">ZenSift</h1>'
        f'<p style="color:#777;margin:0 0 24px;font-size:14px;">{escape(_pretty_date(digest_date))}</p>'
        f"{notice}{body}"
        '<p style="color:#aaa;font-size:12px;margin-top:32px;">Curated by ZenSift. '
        "Summaries are AI-generated; follow the link for the full story.</p>"
        "</div></body></html>"
    )


def render_text(articles: list[Article], digest_date: date, degraded: bool) -> str:
    lines = [f"ZenSift · {_pretty_date(digest_date)}", ""]
    if degraded:
        lines += [DEGRADED_NOTICE, ""]
    if not articles:
        lines.append(QUIET_DAY)
    for article in articles:
        lines.append(article.title)
        if article.source:
            lines.append(f"  {article.source}")
        if article.summary:
            lines.append(f"  {article.summary}")
        lines += [f"  {article.url}", ""]
    return "\n".join(lines).rstrip() + "\n"


def assembler_node(state: ZenSiftState) -> dict:
    settings = state.get("digest") or DigestSettings()
    digest_date = state.get("digest_date") or datetime.now(UTC).date()
    degraded = state.get("degraded", False)

    articles = order_by_relevance(apply_tone_rules(state.get("digest_articles", []), settings))
    logger.info("Assembled digest with %d articles (degraded=%s).", len(articles), degraded)
    return {
        # Overwritten so record_sent only records what was actually emailed.
        "digest_articles": articles,
        "email_subject": build_subject(articles, digest_date, degraded),
        "email_html": render_html(articles, digest_date, degraded),
        "email_text": render_text(articles, digest_date, degraded),
    }
