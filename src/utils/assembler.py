"""Assembler: apply tone rules, order by relevance, and render the digest as HTML and text."""

from __future__ import annotations

import logging
import re
from datetime import UTC, date, datetime
from html import escape
from urllib.parse import urlparse

from src.config import DigestSettings
from src.graph.state import Article, ZenSiftState

logger = logging.getLogger(__name__)

DEGRADED_NOTICE = (
    "AI summaries weren't available today, so here are the opening lines of a few top "
    "stories, without filtering. Normal service should resume tomorrow."
)
QUIET_DAY = "Nothing cleared the bar today. Enjoy the quiet."
NOTHING_READABLE = (
    "Today's picks couldn't be opened (paywalls or sites that block readers), "
    "so there was nothing to summarise. Back tomorrow."
)
FOOTER = "Summaries are AI-generated from the original articles. Follow a link for the full story."
DEGRADED_FOOTER = "Opening lines are quoted from the original articles. Follow a link for the full story."
EXCERPT_CHARS = 320

_DOMAIN = re.compile(r"^[a-z0-9-]+(\.[a-z0-9-]+)+$", re.IGNORECASE)
_SECOND_LEVEL = {"co", "com", "org", "net", "ac", "gov", "edu"}

# Beige-coffee palette.
PAGE_BG = "#f3ede4"
CARD_BG = "#fbf8f3"
ESPRESSO = "#3e2f25"
COFFEE = "#6f4e37"
LATTE = "#9c8672"
DIVIDER = "#e6dccd"
NOTICE_BG = "#efe3d0"
FONT = "Georgia,'Times New Roman',serif"


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


def build_subject(
    articles: list[Article], digest_date: date, degraded: bool, unreadable: bool = False
) -> str:
    day = f"{digest_date:%a} {digest_date.day} {digest_date:%b}"
    if not articles and unreadable:
        return f"ZenSift · nothing readable today · {day}"
    if degraded and articles:
        return f"ZenSift · {len(articles)} stories (summaries unavailable) · {day}"
    if not articles:
        return f"ZenSift · a quiet day · {day}"
    reads = "calm read" if len(articles) == 1 else "calm reads"
    return f"ZenSift · {len(articles)} {reads} · {day}"


def _display_source(source: str) -> str:
    """Turn a bare domain like 'tech-insider.org' into 'Tech Insider' so mail apps don't auto-link it."""
    source = source.strip()
    if not _DOMAIN.match(source):
        return source
    parts = [p for p in source.lower().split(".") if p != "www"][:-1]
    while len(parts) > 1 and parts[-1] in _SECOND_LEVEL:
        parts.pop()
    return parts[-1].replace("-", " ").title() if parts else source


def _link_label(article: Article) -> str:
    source = _display_source(article.source)
    return f"Read at {source}" if source else "Read the full story"


def _excerpt(body: str) -> str:
    text = " ".join(body.split())
    if len(text) <= EXCERPT_CHARS:
        return text
    cut = text[:EXCERPT_CHARS]
    sentence_end = cut.rfind(". ")
    if sentence_end >= EXCERPT_CHARS // 2:
        return cut[: sentence_end + 1]
    return cut.rsplit(" ", 1)[0] + "…"


def _lead_text(article: Article) -> str:
    # Headlines are written to provoke, so they are never shown; the fallback quotes the article instead.
    return article.summary or _excerpt(article.body)


def _href(article: Article) -> str:
    return article.link or article.url


def _empty_message(unreadable: bool) -> str:
    return NOTHING_READABLE if unreadable else QUIET_DAY


def _html_item(article: Article, first: bool) -> str:
    url = _safe_url(_href(article))
    border = "" if first else f"border-top:1px solid {DIVIDER};"
    source_name = _display_source(article.source)
    source = (
        f'<p style="margin:0 0 10px;font-size:11px;letter-spacing:1.5px;text-transform:uppercase;'
        f'color:{LATTE};">{escape(source_name)}</p>'
        if source_name
        else ""
    )
    lead_text = _lead_text(article)
    lead = (
        f'<p style="margin:0;font-size:16px;line-height:1.7;color:{ESPRESSO};">{escape(lead_text)}</p>'
        if lead_text
        else ""
    )
    link = (
        f'<p style="margin:14px 0 0;font-size:14px;">'
        f'<a href="{escape(url, quote=True)}" style="color:{COFFEE};text-decoration:none;'
        f'border-bottom:1px solid {DIVIDER};">{escape(_link_label(article))} &rarr;</a></p>'
        if url
        else ""
    )
    return f'<tr><td style="padding:28px 0;{border}">{source}{lead}{link}</td></tr>'


def render_html(
    articles: list[Article], digest_date: date, degraded: bool, unreadable: bool = False
) -> str:
    notice = (
        f'<tr><td style="padding:0 0 8px;"><p style="margin:0;padding:14px 16px;background:{NOTICE_BG};'
        f'border-radius:6px;font-size:14px;line-height:1.6;color:{ESPRESSO};">{escape(DEGRADED_NOTICE)}</p></td></tr>'
        if degraded
        else ""
    )
    items = "".join(_html_item(a, first=i == 0) for i, a in enumerate(articles)) or (
        f'<tr><td style="padding:28px 0;font-size:16px;color:{ESPRESSO};">{escape(_empty_message(unreadable))}</td></tr>'
    )
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta name="color-scheme" content="light"><meta name="supported-color-schemes" content="light">'
        "</head>"
        f'<body style="margin:0;padding:0;background:{PAGE_BG};">'
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:{PAGE_BG};">'
        '<tr><td align="center" style="padding:32px 12px;">'
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width:720px;'
        f'background:{CARD_BG};border-radius:10px;font-family:{FONT};">'
        '<tr><td style="padding:40px 44px 8px;">'
        f'<p style="margin:0;font-size:13px;letter-spacing:4px;text-transform:uppercase;color:{COFFEE};">ZenSift</p>'
        f'<p style="margin:6px 0 0;font-size:14px;font-style:italic;color:{LATTE};">{escape(_pretty_date(digest_date))}</p>'
        f'<div style="width:32px;height:1px;background:{COFFEE};margin:20px 0 0;line-height:1px;font-size:1px;">&nbsp;</div>'
        "</td></tr>"
        '<tr><td style="padding:8px 44px 16px;">'
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0">{notice}{items}</table>'
        "</td></tr>"
        "</table>"
        f'<p style="max-width:680px;margin:20px auto 0;font-family:{FONT};font-size:12px;line-height:1.6;'
        f'color:{LATTE};text-align:center;">{escape(DEGRADED_FOOTER if degraded else FOOTER)}</p>'
        "</td></tr></table></body></html>"
    )


def render_text(
    articles: list[Article], digest_date: date, degraded: bool, unreadable: bool = False
) -> str:
    lines = [f"ZenSift · {_pretty_date(digest_date)}", ""]
    if degraded:
        lines += [DEGRADED_NOTICE, ""]
    if not articles:
        lines += [_empty_message(unreadable), ""]
    for article in articles:
        if source_name := _display_source(article.source):
            lines.append(source_name.upper())
        if lead_text := _lead_text(article):
            lines.append(lead_text)
        lines += [f"{_link_label(article)}: {_href(article)}", "", "---", ""]
    lines.append(DEGRADED_FOOTER if degraded else FOOTER)
    return "\n".join(lines).rstrip() + "\n"


def assembler_node(state: ZenSiftState) -> dict:
    settings = state.get("digest") or DigestSettings()
    digest_date = state.get("digest_date") or datetime.now(UTC).date()
    degraded = state.get("degraded", False)
    # Only when reading failed; tone rules emptying the digest is a genuinely quiet day.
    unreadable = not state.get("digest_articles") and state.get("unreadable_count", 0) > 0

    articles = order_by_relevance(apply_tone_rules(state.get("digest_articles", []), settings))
    logger.info("Assembled digest with %d articles (degraded=%s).", len(articles), degraded)
    return {
        # Overwritten so record_sent only records what was actually emailed.
        "digest_articles": articles,
        "email_subject": build_subject(articles, digest_date, degraded, unreadable),
        "email_html": render_html(articles, digest_date, degraded, unreadable),
        "email_text": render_text(articles, digest_date, degraded, unreadable),
    }
