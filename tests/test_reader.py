from email.message import Message
from unittest.mock import MagicMock
from urllib.error import HTTPError

import pytest

from src.graph.state import Article
from src.utils import reader
from src.utils.reader import (
    MAX_BODY_CHARS,
    extract_paragraphs,
    fetch_article_text,
    fetch_with_retry,
    read_articles,
    reader_node,
    resolve_url,
)

LONG = "The lab released its weights under Apache 2.0, letting companies run the model themselves."
GOOGLE = "https://news.google.com/rss/articles/CBMiabc?oc=5"


@pytest.fixture(autouse=True)
def no_sleep(mocker):
    return mocker.patch.object(reader.time, "sleep")


def fake_page(mocker, body: bytes, content_type: str = "text/html; charset=utf-8"):
    response = MagicMock()
    response.headers = {"Content-Type": content_type}
    response.read.return_value = body
    opener = mocker.patch.object(reader, "urlopen")
    opener.return_value.__enter__.return_value = response
    return opener


# --- paragraph extraction ----------------------------------------------------


def test_keeps_only_paragraphs_of_real_length():
    page = f"<html><body><nav>Home | News</nav><p>Too short.</p><p>{LONG}</p></body></html>"

    assert extract_paragraphs(page) == LONG


def test_ignores_script_and_style_inside_paragraphs():
    page = f"<p>{LONG}<script>var x = 'hidden tracking code that is long enough to pass';</script></p>"

    assert extract_paragraphs(page) == LONG


def test_handles_unclosed_paragraph_tags():
    page = f"<p>{LONG}<p>{LONG} Second."

    assert extract_paragraphs(page).split("\n\n") == [LONG, f"{LONG} Second."]


def test_paragraph_text_is_capped():
    page = "".join(f"<p>{LONG} {i}</p>" for i in range(200))

    assert len(extract_paragraphs(page)) <= MAX_BODY_CHARS


def test_prefers_paragraphs_inside_article_over_site_chrome():
    chrome = "Get a notification every time we publish a story, even when the app is closed."
    page = f"<body><div><p>{chrome}</p></div><article><p>{LONG}</p></article><p>{chrome}</p></body>"

    assert extract_paragraphs(page) == LONG


def test_uses_all_paragraphs_when_page_has_no_article_tag():
    page = f"<div><p>{LONG}</p></div><p>{LONG} Two.</p>"

    assert extract_paragraphs(page).split("\n\n") == [LONG, f"{LONG} Two."]


# --- link resolution ---------------------------------------------------------


def test_publisher_links_pass_through_unchanged():
    assert resolve_url("https://example.com/story") == "https://example.com/story"


def test_google_news_links_are_decoded(mocker):
    mocker.patch.object(
        reader, "gnewsdecoder", return_value={"success": True, "decoded_url": "https://pub.example/a"}
    )

    assert resolve_url(GOOGLE) == "https://pub.example/a"


def test_undecodable_google_links_resolve_to_none(mocker):
    mocker.patch.object(reader, "gnewsdecoder", return_value={"success": False})

    assert resolve_url(GOOGLE) is None


def test_decoder_exceptions_resolve_to_none(mocker):
    mocker.patch.object(reader, "gnewsdecoder", side_effect=OSError("timeout"))

    assert resolve_url(GOOGLE) is None


# --- fetching ----------------------------------------------------------------


def test_only_web_urls_are_fetched(mocker):
    opener = mocker.patch.object(reader, "urlopen")

    with pytest.raises(ValueError):
        fetch_article_text("file:///etc/passwd")
    opener.assert_not_called()


def test_fetch_extracts_paragraphs_from_html(mocker):
    fake_page(mocker, f"<html><p>{LONG}</p></html>".encode())

    assert fetch_article_text("https://pub.example/a") == LONG


def test_non_html_documents_yield_no_text(mocker):
    fake_page(mocker, b"%PDF-1.7 binary", content_type="application/pdf")

    assert fetch_article_text("https://gov.example/2026.pdf") == ""


def test_declared_charset_is_honoured(mocker):
    text = f"Café owners in München say {LONG}"
    fake_page(mocker, f"<p>{text}</p>".encode("latin-1"), content_type="text/html; charset=ISO-8859-1")

    assert fetch_article_text("https://pub.example/a") == text


def test_unknown_charset_falls_back_to_utf8(mocker):
    fake_page(mocker, f"<p>{LONG}</p>".encode(), content_type="text/html; charset=made-up")

    assert fetch_article_text("https://pub.example/a") == LONG


def _http_error(code: int, retry_after: str | None = None) -> HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return HTTPError("https://pub.example/a", code, "err", headers, None)


def test_rate_limited_fetch_is_retried_once_after_retry_after(mocker, no_sleep):
    fetch = mocker.patch.object(
        reader, "fetch_article_text", side_effect=[_http_error(429, "3"), LONG]
    )

    assert fetch_with_retry("https://pub.example/a") == LONG
    assert fetch.call_count == 2
    no_sleep.assert_called_once_with(3.0)


def test_retry_wait_is_capped_and_tolerates_http_dates(mocker, no_sleep):
    mocker.patch.object(
        reader, "fetch_article_text", side_effect=[_http_error(503, "Wed, 21 Oct 2026 07:28:00 GMT"), LONG]
    )
    fetch_with_retry("https://pub.example/a")
    no_sleep.assert_called_once_with(reader.DEFAULT_RETRY_WAIT_SECONDS)

    no_sleep.reset_mock()
    mocker.patch.object(reader, "fetch_article_text", side_effect=[_http_error(429, "600"), LONG])
    fetch_with_retry("https://pub.example/a")
    no_sleep.assert_called_once_with(reader.MAX_RETRY_WAIT_SECONDS)


def test_other_http_errors_are_not_retried(mocker):
    fetch = mocker.patch.object(reader, "fetch_article_text", side_effect=_http_error(403))

    with pytest.raises(HTTPError):
        fetch_with_retry("https://pub.example/a")
    assert fetch.call_count == 1


def test_errors_name_the_publisher_not_the_google_link(mocker):
    mocker.patch.object(
        reader, "gnewsdecoder", return_value={"success": True, "decoded_url": "https://pub.example/a"}
    )
    mocker.patch.object(reader, "fetch_article_text", side_effect=_http_error(403))

    _, errors = read_articles([Article(title="A", url=GOOGLE)], delay=0)

    assert errors[0].startswith("reader: https://pub.example/a:")


# --- read_articles and reader_node -------------------------------------------


def test_read_articles_attaches_body_and_drops_unreadable(mocker):
    articles = [
        Article(title="A", url="https://a.example/1"),
        Article(title="B", url="https://b.example/2"),
        Article(title="C", url="https://c.example/3"),
    ]

    def fake_fetch(url, *args, **kwargs):
        if "b.example" in url:
            return ""
        if "c.example" in url:
            raise OSError("connection reset")
        return LONG

    mocker.patch.object(reader, "fetch_article_text", side_effect=fake_fetch)

    kept, errors = read_articles(articles, delay=0)

    assert [a.url for a in kept] == ["https://a.example/1"]
    assert kept[0].title == "A" and kept[0].body == LONG
    assert kept[0].link == "https://a.example/1"
    assert len(errors) == 2


def test_decoded_publisher_url_is_kept_as_link_and_feed_url_is_untouched(mocker):
    mocker.patch.object(
        reader, "gnewsdecoder", return_value={"success": True, "decoded_url": "https://pub.example/a"}
    )
    mocker.patch.object(reader, "fetch_article_text", return_value=LONG)

    kept, _ = read_articles([Article(title="A", url=GOOGLE)], delay=0)

    assert kept[0].url == GOOGLE and kept[0].link == "https://pub.example/a"


def test_google_links_that_cannot_be_decoded_are_dropped(mocker):
    mocker.patch.object(reader, "gnewsdecoder", return_value={"success": False})
    fetch = mocker.patch.object(reader, "fetch_article_text")

    kept, errors = read_articles([Article(title="A", url=GOOGLE)], delay=0)

    assert kept == [] and len(errors) == 1
    fetch.assert_not_called()


def test_reader_node_returns_kept_articles_and_errors(mocker):
    mocker.patch.object(reader, "fetch_article_text", return_value=LONG)

    result = reader_node({"sifted_articles": [Article(title="A", url="https://a.example/1")]})

    assert result["sifted_articles"][0].body == LONG
    assert result["unreadable_count"] == 0
    assert result["errors"] == []


def test_reader_node_counts_unreadable_articles(mocker):
    mocker.patch.object(reader, "fetch_article_text", return_value="")

    result = reader_node({"sifted_articles": [Article(title="A", url="https://a.example/1")]})

    assert result["sifted_articles"] == [] and result["unreadable_count"] == 1


def test_reader_node_with_nothing_sifted():
    assert reader_node({}) == {"sifted_articles": [], "unreadable_count": 0, "errors": []}
