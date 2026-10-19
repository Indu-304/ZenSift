"""Reader: fetch each sifted article's full text so the editor summarizes the story, not the headline."""

from __future__ import annotations

import logging
import re
import time
from html.parser import HTMLParser
from urllib.error import HTTPError
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
RETRY_STATUSES = {429, 503}
DEFAULT_RETRY_WAIT_SECONDS = 5.0
MAX_RETRY_WAIT_SECONDS = 15.0

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


def _retry_wait(exc: HTTPError) -> float:
    try:
        wait = float(exc.headers.get("Retry-After", DEFAULT_RETRY_WAIT_SECONDS))
    except (TypeError, ValueError):
        # Retry-After may be an HTTP date; not worth parsing for one retry.
        wait = DEFAULT_RETRY_WAIT_SECONDS
    return min(max(wait, 1.0), MAX_RETRY_WAIT_SECONDS)


def fetch_with_retry(url: str) -> str:
    """Fetch once more after a rate-limit or overload response, honouring Retry-After."""
    try:
        return fetch_article_text(url)
    except HTTPError as exc:
        if exc.code not in RETRY_STATUSES:
            raise
        wait = _retry_wait(exc)
        logger.info("%s answered %d; retrying once in %.0fs.", url, exc.code, wait)
        time.sleep(wait)
        return fetch_article_text(url)


def read_articles(
    articles: list[Article], delay: float = REQUEST_DELAY_SECONDS
) -> tuple[list[Article], list[str]]:
    """Attach full text to each article. Articles with no readable text are dropped."""
    kept: list[Article] = []
    errors: list[str] = []
    for i, article in enumerate(articles):
        if i > 0:
            time.sleep(delay)
        target = None
        try:
            target = resolve_url(article.url)
            body = fetch_with_retry(target) if target else ""
        except Exception as exc:
            # The publisher URL is shorter and says which site refused, unlike the Google link.
            shown = target or article.url
            logger.warning("Dropping %s: could not read it (%r)", shown, exc)
            errors.append(f"reader: {shown}: {exc!r}")
            continue
        if not body:
            shown = target or article.url
            logger.warning("Dropping %s: no readable article text.", shown)
            errors.append(f"reader: {shown}: no readable article text")
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
