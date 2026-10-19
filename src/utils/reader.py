"""Reader: fetch each sifted article's full text so the editor summarizes the story, not the headline."""

from __future__ import annotations

import logging
import re
import time
from html.parser import HTMLParser
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from googlenewsdecoder import gnewsdecoder

from src.graph.state import Article, ZenSiftState
from src.utils.fetcher import USER_AGENT

logger = logging.getLogger(__name__)

REQUEST_DELAY_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 15
MAX_PAGE_BYTES = 2 * 1024 * 1024
# Skips menu labels, captions and cookie notices, which are usually shorter than this.
MIN_PARAGRAPH_CHARS = 60
MAX_BODY_CHARS = 2000
GOOGLE_NEWS_HOST = "news.google.com"

_WHITESPACE = re.compile(r"\s+")
_CHARSET = re.compile(r"charset=[\"']?([\w.:-]+)", re.IGNORECASE)


class _ParagraphExtractor(HTMLParser):
    """Collects the text of each <p> element, ignoring scripts and styles."""

    _SKIP_TAGS = {"script", "style", "noscript"}
    _MAIN_TAGS = {"article", "main"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.paragraphs: list[tuple[str, bool]] = []
        self._current: list[str] | None = None
        self._current_in_main = False
        self._skip_depth = 0
        self._main_depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        elif tag in self._MAIN_TAGS:
            self._main_depth += 1
        elif tag == "p":
            self._flush()
            self._current = []
            self._current_in_main = self._main_depth > 0

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self._MAIN_TAGS and self._main_depth:
            self._main_depth -= 1
        elif tag == "p":
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._current is not None and not self._skip_depth:
            self._current.append(data)

    def close(self) -> None:
        super().close()
        self._flush()

    def _flush(self) -> None:
        if self._current is not None:
            text = _WHITESPACE.sub(" ", "".join(self._current)).strip()
            self.paragraphs.append((text, self._current_in_main))
            self._current = None


def extract_paragraphs(page: str) -> str:
    parser = _ParagraphExtractor()
    parser.feed(page)
    parser.close()
    long_enough = [(p, in_main) for p, in_main in parser.paragraphs if len(p) >= MIN_PARAGRAPH_CHARS]
    # Site chrome (newsletter, notification and cookie prompts) usually sits outside <article>/<main>.
    in_main = [p for p, is_main in long_enough if is_main]
    kept = in_main or [p for p, _ in long_enough]
    return "\n\n".join(kept)[:MAX_BODY_CHARS]


def _decode(raw: bytes, content_type: str) -> str:
    match = _CHARSET.search(content_type)
    try:
        return raw.decode(match.group(1) if match else "utf-8", errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def resolve_url(url: str) -> str | None:
    """Return the publisher's URL, or None if a Google News link cannot be decoded."""
    if urlparse(url).netloc != GOOGLE_NEWS_HOST:
        return url
    try:
        result = gnewsdecoder(url, interval=1)
    except Exception:
        logger.warning("Google News decode raised for %s", url, exc_info=True)
        return None
    if not result.get("success"):
        return None
    return result["decoded_url"]


def fetch_article_text(url: str, timeout: float = REQUEST_TIMEOUT_SECONDS) -> str:
    """Return the paragraph text of an HTML article, or "" for non-HTML content."""
    if urlparse(url).scheme not in {"http", "https"}:
        raise ValueError(f"Refusing non-web article URL: {url}")
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            return ""
        page = _decode(response.read(MAX_PAGE_BYTES), content_type)
    return extract_paragraphs(page)


def read_articles(
    articles: list[Article], delay: float = REQUEST_DELAY_SECONDS
) -> tuple[list[Article], list[str]]:
    """Attach full text to each article. Articles with no readable text are dropped."""
    kept: list[Article] = []
    errors: list[str] = []
    for i, article in enumerate(articles):
        if i > 0:
            time.sleep(delay)
        try:
            target = resolve_url(article.url)
            body = fetch_article_text(target) if target else ""
        except Exception as exc:
            logger.warning("Dropping %s: could not read it (%r)", article.url, exc)
            errors.append(f"reader: {article.url}: {exc!r}")
            continue
        if not body:
            logger.warning("Dropping %s: no readable article text.", article.url)
            errors.append(f"reader: {article.url}: no readable article text")
            continue
        kept.append(article.model_copy(update={"body": body, "link": target}))
    return kept, errors


def reader_node(state: ZenSiftState) -> dict:
    articles = state.get("sifted_articles", [])
    kept, errors = read_articles(articles)
    if articles and not kept:
        logger.error("None of the %d sifted articles had readable text.", len(articles))
    logger.info("Read full text for %d of %d articles.", len(kept), len(articles))
    return {
        "sifted_articles": kept,
        "unreadable_count": len(articles) - len(kept),
        "errors": errors,
    }
