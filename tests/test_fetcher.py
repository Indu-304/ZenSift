import json
from unittest.mock import MagicMock
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse

import pytest

from src.utils import fetcher
from src.utils.fetcher import (
    MAX_SNIPPET_CHARS,
    download_feed,
    fetch_all,
    fetcher_node,
    hn_search_url,
    parse_feed,
    parse_hn_search,
    strip_html,
)

GOOGLE_NEWS_RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Google News</title>
<item>
  <title>TU Munich expands AI master's programme - Tagesspiegel</title>
  <link>https://news.google.com/rss/articles/abc</link>
  <description>&lt;a href="https://example.com/a"&gt;TU Munich expands&lt;/a&gt;&amp;nbsp;&amp;nbsp;&lt;font color="#6f6f6f"&gt;Tagesspiegel&lt;/font&gt;</description>
  <source url="https://www.tagesspiegel.de">Tagesspiegel</source>
</item>
<item>
  <title>Open-weight LLM tops benchmark</title>
  <link>https://news.google.com/rss/articles/def</link>
  <description>Plain &amp;amp; simple</description>
</item>
<item>
  <title></title>
  <link>https://news.google.com/rss/articles/no-title</link>
</item>
</channel></rss>"""

HN_JSON = json.dumps(
    {
        "hits": [
            {"objectID": "1", "title": "Show HN: A tiny open-source LLM", "url": "https://github.com/example/tiny-llm", "points": 120},
            {"objectID": "2", "title": "Ask HN: Best open LLM?", "url": None, "story_text": "<p>Curious &amp; asking</p>"},
            {"objectID": "3", "title": "", "url": "https://x"},
        ]
    }
).encode()

HN_URL = "https://hn.algolia.com/api/v1/search?query=x"


def fake_response(body: bytes) -> MagicMock:
    response = MagicMock()
    response.__enter__.return_value.read.return_value = body
    return response


@pytest.fixture
def no_sleep(mocker):
    return mocker.patch.object(fetcher.time, "sleep")


# --- strip_html --------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('<a href="x">Hello</a>&nbsp;<b>world</b>', "Hello world"),
        ("Tom &amp; Jerry", "Tom & Jerry"),
        ("  lots \n\n of   space ", "lots of space"),
        ("<script>alert(1)</script>Safe", "Safe"),
        ("<style>p{color:red}</style><p>Body</p>", "Body"),
        ("", ""),
    ],
)
def test_strip_html(raw, expected):
    assert strip_html(raw) == expected


# --- URL building ------------------------------------------------------------


def test_hn_search_url_encodes_keyword_and_filters():
    url = hn_search_url("C++ & LLMs", count=5, now=1_000_000)

    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == fetcher.HN_SEARCH_API
    assert params["query"] == ["C++ & LLMs"]
    assert params["tags"] == ["story"]
    since = 1_000_000 - fetcher.HN_LOOKBACK_SECONDS
    assert params["numericFilters"] == [f"points>{fetcher.HN_MIN_POINTS},created_at_i>{since}"]
    assert params["hitsPerPage"] == ["5"]


# --- download_feed -----------------------------------------------------------


@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://example.com/feed", "ftp://x/y"])
def test_download_feed_rejects_non_https(url, block_feed_network):
    with pytest.raises(ValueError):
        download_feed(url)
    block_feed_network.assert_not_called()


def test_download_feed_sends_user_agent_timeout_and_caps_size(block_feed_network):
    block_feed_network.side_effect = None
    block_feed_network.return_value = fake_response(b"<rss/>")

    assert download_feed("https://news.google.com/rss/search?q=x", timeout=7) == b"<rss/>"

    request = block_feed_network.call_args.args[0]
    assert request.get_header("User-agent") == fetcher.USER_AGENT
    assert block_feed_network.call_args.kwargs["timeout"] == 7
    read = block_feed_network.return_value.__enter__.return_value.read
    read.assert_called_once_with(fetcher.MAX_FEED_BYTES)


# --- parse_feed --------------------------------------------------------------


def test_parse_feed_extracts_title_url_snippet_and_source():
    articles = parse_feed(GOOGLE_NEWS_RSS, "Google News")

    assert [a.url for a in articles] == [
        "https://news.google.com/rss/articles/abc",
        "https://news.google.com/rss/articles/def",
    ]
    first, second = articles
    assert first.title == "TU Munich expands AI master's programme - Tagesspiegel"
    assert first.snippet == "TU Munich expands Tagesspiegel"
    assert first.source == "Tagesspiegel"
    assert second.snippet == "Plain & simple"
    assert second.source == "Google News"


def test_parse_feed_skips_entries_without_title_or_link():
    assert all(a.title for a in parse_feed(GOOGLE_NEWS_RSS, "x"))


def test_parse_feed_respects_max_items():
    assert len(parse_feed(GOOGLE_NEWS_RSS, "x", max_items=1)) == 1


def test_parse_feed_truncates_long_snippets():
    body = (
        b"<rss version='2.0'><channel><item><title>t</title><link>https://x</link>"
        b"<description>" + b"a" * 5000 + b"</description></item></channel></rss>"
    )
    assert len(parse_feed(body, "x")[0].snippet) == MAX_SNIPPET_CHARS


def test_parse_feed_raises_on_garbage():
    with pytest.raises(ValueError):
        parse_feed(b"<html>not a feed", "x")


# --- parse_hn_search ---------------------------------------------------------


def test_parse_hn_search_maps_hits_and_links_self_posts_to_discussion():
    articles = parse_hn_search(HN_JSON)

    assert [a.url for a in articles] == [
        "https://github.com/example/tiny-llm",
        "https://news.ycombinator.com/item?id=2",
    ]
    assert articles[1].snippet == "Curious & asking"
    assert all(a.source == "Hacker News" for a in articles)


def test_parse_hn_search_respects_max_items():
    assert len(parse_hn_search(HN_JSON, max_items=1)) == 1


@pytest.mark.parametrize("body", [b"not json", b"[]", b'{"hits": "nope"}', b"{}"])
def test_parse_hn_search_rejects_bad_payloads(body):
    with pytest.raises(ValueError):
        parse_hn_search(body)


# --- fetch_all ---------------------------------------------------------------


def test_fetch_all_sleeps_between_requests_not_before_first(mocker, no_sleep):
    mocker.patch.object(fetcher, "download_feed", return_value=GOOGLE_NEWS_RSS)

    fetch_all(["https://a", "https://b", "https://c"], delay=1.0)

    assert no_sleep.call_count == 2
    no_sleep.assert_called_with(1.0)


def test_fetch_all_default_backpressure_is_one_second():
    assert fetcher.REQUEST_DELAY_SECONDS == 1.0


@pytest.mark.parametrize(
    "error",
    [
        URLError("DNS failure"),
        HTTPError("https://bad", 503, "Service Unavailable", {}, None),
        TimeoutError("timed out"),
        ValueError("Unparseable feed"),
    ],
)
def test_fetch_all_skips_failing_feed_and_continues(mocker, no_sleep, error):
    mocker.patch.object(
        fetcher, "download_feed", side_effect=[GOOGLE_NEWS_RSS, error, HN_JSON]
    )

    articles, errors = fetch_all(
        ["https://news.google.com/1", "https://bad", HN_URL]
    )

    assert len(articles) == 4
    assert len(errors) == 1 and "https://bad" in errors[0]


def test_fetch_all_dispatches_parser_by_host(mocker, no_sleep):
    mocker.patch.object(fetcher, "download_feed", side_effect=[GOOGLE_NEWS_RSS, HN_JSON])

    articles, errors = fetch_all(["https://news.google.com/rss/search?q=a", HN_URL])

    assert errors == []
    assert {a.source for a in articles} == {"Tagesspiegel", "Google News", "Hacker News"}


def test_fetch_all_with_no_urls_makes_no_requests(mocker, no_sleep):
    download = mocker.patch.object(fetcher, "download_feed")

    assert fetch_all([]) == ([], [])
    download.assert_not_called()
    no_sleep.assert_not_called()


# --- fetcher_node ------------------------------------------------------------


def test_fetcher_node_fetches_google_urls_then_hn_keywords(mocker, no_sleep):
    mocker.patch.object(fetcher.time, "time", return_value=1_000_000)
    download = mocker.patch.object(
        fetcher, "download_feed", side_effect=[GOOGLE_NEWS_RSS, HN_JSON]
    )

    result = fetcher_node(
        {"rss_urls": ["https://news.google.com/rss/search?q=a"], "hn_queries": ["open LLM"]}
    )

    called_urls = [c.args[0] for c in download.call_args_list]
    assert called_urls == [
        "https://news.google.com/rss/search?q=a",
        hn_search_url("open LLM", now=1_000_000),
    ]
    assert len(result["raw_articles"]) == 4
    assert result["errors"] == []


def test_fetcher_node_survives_all_feeds_failing(mocker, no_sleep):
    mocker.patch.object(fetcher, "download_feed", side_effect=URLError("offline"))

    result = fetcher_node({"rss_urls": ["https://a", "https://b"], "hn_queries": []})

    assert result["raw_articles"] == []
    assert len(result["errors"]) == 2
