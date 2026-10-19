"""Fetcher: download Google News RSS and HN search results politely, normalize into Articles."""

from __future__ import annotations

import html
import json
import logging
import re
import time
from html.parser import HTMLParser
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

import feedparser

from src.graph.state import Article, ZenSiftState

logger = logging.getLogger(__name__)

REQUEST_DELAY_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 15
MAX_FEED_BYTES = 2 * 1024 * 1024
MAX_ITEMS_PER_FEED = 15
MAX_SNIPPET_CHARS = 500

# HN has no reliable RSS search (hnrss.org 502s on multi-word queries); Algolia is HN's official search.
HN_SEARCH_HOST = "hn.algolia.com"
HN_SEARCH_API = f"https://{HN_SEARCH_HOST}/api/v1/search"
HN_ITEM_URL = "https://news.ycombinator.com/item?id={}"
HN_MIN_POINTS = 10
HN_LOOKBACK_SECONDS = 7 * 24 * 3600

USER_AGENT = "ZenSift/0.1 (personal RSS digest)"

_WHITESPACE = re.compile(r"\s+")


class _TextExtractor(HTMLParser):
    _SKIP_TAGS = {"script", "style"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


def strip_html(text: str) -> str:
    parser = _TextExtractor()
    parser.feed(text or "")
    parser.close()
    return _WHITESPACE.sub(" ", html.unescape(" ".join(parser.parts))).strip()


def hn_search_url(
    keyword: str, count: int = MAX_ITEMS_PER_FEED, now: float | None = None
) -> str:
    since = int((time.time() if now is None else now) - HN_LOOKBACK_SECONDS)
    params = urlencode(
        {
            "query": keyword,
            "tags": "story",
            "numericFilters": f"points>{HN_MIN_POINTS},created_at_i>{since}",
            "hitsPerPage": count,
        }
    )
    return f"{HN_SEARCH_API}?{params}"


def download_feed(url: str, timeout: float = REQUEST_TIMEOUT_SECONDS) -> bytes:
    # urlopen also handles file:// and ftp://; only allow remote https feeds.
    if urlparse(url).scheme != "https":
        raise ValueError(f"Refusing non-https feed URL: {url}")
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read(MAX_FEED_BYTES)


def parse_feed(content: bytes, default_source: str, max_items: int = MAX_ITEMS_PER_FEED) -> list[Article]:
    feed = feedparser.parse(content)
    if feed.bozo and not feed.entries:
        raise ValueError(f"Unparseable feed: {feed.get('bozo_exception')!r}")

    articles = []
    for entry in feed.entries[:max_items]:
        title = strip_html(entry.get("title", ""))
        url = (entry.get("link") or "").strip()
        if not title or not url:
            continue
        source = (entry.get("source") or {}).get("title") or default_source
        articles.append(
            Article(
                title=title,
                url=url,
                snippet=strip_html(entry.get("summary", ""))[:MAX_SNIPPET_CHARS],
                source=source,
            )
        )
    return articles


def parse_hn_search(content: bytes, max_items: int = MAX_ITEMS_PER_FEED) -> list[Article]:
    payload = json.loads(content)
    hits = payload.get("hits") if isinstance(payload, dict) else None
    if not isinstance(hits, list):
        raise ValueError("Unexpected HN search response shape")

    articles = []
    for hit in hits[:max_items]:
        title = strip_html(hit.get("title") or "")
        object_id = hit.get("objectID")
        # Ask/Launch HN posts have no external URL; link to the discussion instead.
        url = hit.get("url") or (HN_ITEM_URL.format(object_id) if object_id else "")
        if not title or not url:
            continue
        articles.append(
            Article(
                title=title,
                url=url,
                snippet=strip_html(hit.get("story_text") or "")[:MAX_SNIPPET_CHARS],
                source="Hacker News",
            )
        )
    return articles


def _parse(url: str, content: bytes) -> list[Article]:
    host = urlparse(url).netloc
    if host == HN_SEARCH_HOST:
        return parse_hn_search(content)
    return parse_feed(content, "Google News" if host == "news.google.com" else host)


def fetch_all(urls: list[str], delay: float = REQUEST_DELAY_SECONDS) -> tuple[list[Article], list[str]]:
    """Fetch sources sequentially with backpressure. A failing source is logged and skipped."""
    articles: list[Article] = []
    errors: list[str] = []
    for i, url in enumerate(urls):
        if i > 0:
            time.sleep(delay)
        try:
            feed_articles = _parse(url, download_feed(url))
        except Exception as exc:
            logger.warning("Skipping source %s: %r", url, exc)
            errors.append(f"fetcher: {url}: {exc!r}")
            continue
        logger.info("Fetched %d articles from %s", len(feed_articles), url)
        articles.extend(feed_articles)
    return articles, errors


def fetcher_node(state: ZenSiftState) -> dict:
    urls = list(state.get("rss_urls", [])) + [hn_search_url(k) for k in state.get("hn_queries", [])]
    articles, errors = fetch_all(urls)
    if urls and not articles:
        logger.error("All %d sources failed or were empty.", len(urls))
    return {"raw_articles": articles, "errors": errors}
