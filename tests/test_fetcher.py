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
    hn_rss_url,
    parse_feed,
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

HN_RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Hacker News: Newest</title>
<item>
  <title>Show HN: A tiny open-source LLM</title>
  <link>https://github.com/example/tiny-llm</link>
</item>
</channel></rss>"""


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


def test_hn_rss_url_encodes_keyword_and_filters():
    params = parse_qs(urlparse(hn_rss_url("C++ & LLMs", count=5)).query)

    assert params["q"] == ["C++ & LLMs"]
    assert params["points"] == [str(fetcher.HN_MIN_POINTS)]
    assert params["count"] == ["5"]
    assert params["description"] == ["0"]


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


# --- fetch_all ---------------------------------------------------------------


def test_fetch_all_sleeps_between_requests_not_before_first(mocker, no_sleep):
    mocker.patch.object(fetcher, "download_feed", return_value=HN_RSS)

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
        fetcher, "download_feed", side_effect=[GOOGLE_NEWS_RSS, error, HN_RSS]
    )

    articles, errors = fetch_all(
        ["https://news.google.com/1", "https://bad", "https://hnrss.org/newest?q=x"]
    )

    assert len(articles) == 3
    assert len(errors) == 1 and "https://bad" in errors[0]


def test_fetch_all_labels_sources_by_host(mocker, no_sleep):
    mocker.patch.object(fetcher, "download_feed", return_value=HN_RSS)

    articles, _ = fetch_all(["https://hnrss.org/newest?q=llm"])

    assert articles[0].source == "Hacker News"


def test_fetch_all_with_no_urls_makes_no_requests(mocker, no_sleep):
    download = mocker.patch.object(fetcher, "download_feed")

    assert fetch_all([]) == ([], [])
    download.assert_not_called()
    no_sleep.assert_not_called()


# --- fetcher_node ------------------------------------------------------------


def test_fetcher_node_fetches_google_urls_then_hn_keywords(mocker, no_sleep):
    download = mocker.patch.object(
        fetcher, "download_feed", side_effect=[GOOGLE_NEWS_RSS, HN_RSS]
    )

    result = fetcher_node(
        {"rss_urls": ["https://news.google.com/rss/search?q=a"], "hn_queries": ["open LLM"]}
    )

    called_urls = [c.args[0] for c in download.call_args_list]
    assert called_urls == ["https://news.google.com/rss/search?q=a", hn_rss_url("open LLM")]
    assert len(result["raw_articles"]) == 3
    assert result["errors"] == []


def test_fetcher_node_survives_all_feeds_failing(mocker, no_sleep):
    mocker.patch.object(fetcher, "download_feed", side_effect=URLError("offline"))

    result = fetcher_node({"rss_urls": ["https://a", "https://b"], "hn_queries": []})

    assert result["raw_articles"] == []
    assert len(result["errors"]) == 2
