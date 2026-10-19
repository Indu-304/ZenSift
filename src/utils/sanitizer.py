"""Sanitizer: wrap untrusted article text in escaped <article> blocks before any LLM sees it."""

from __future__ import annotations

import html
import unicodedata

from src.graph.state import Article, ZenSiftState

MAX_FIELD_CHARS = 2000


def _clean(text: str) -> str:
    # Drop control and invisible format chars (e.g. zero-width spaces) that can hide injected text.
    kept = (
        ch
        for ch in text
        if ch in "\n\t" or unicodedata.category(ch) not in {"Cc", "Cf"}
    )
    # Escaping < and > means article text can never forge a closing </article> tag.
    return html.escape("".join(kept)[:MAX_FIELD_CHARS].strip(), quote=False)


def wrap_article(article: Article, article_id: int) -> str:
    # URLs are deliberately excluded: the LLM never sees or writes links.
    return (
        f'<article id="{article_id}">\n'
        f"<title>{_clean(article.title)}</title>\n"
        f"<source>{_clean(article.source)}</source>\n"
        f"<text>{_clean(article.body)}</text>\n"
        f"</article>"
    )


def sanitizer_node(state: ZenSiftState) -> dict:
    articles = state.get("sifted_articles", [])
    return {"sanitized_articles": [wrap_article(a, i) for i, a in enumerate(articles, start=1)]}
