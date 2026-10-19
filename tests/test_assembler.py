from datetime import date
from html import escape

import pytest

from src.config import DigestSettings
from src.graph.state import Article
from src.utils.assembler import (
    DEGRADED_NOTICE,
    NOTHING_READABLE,
    QUIET_DAY,
    apply_tone_rules,
    assembler_node,
    build_subject,
    order_by_relevance,
    render_html,
    render_text,
)

DAY = date(2026, 10, 20)
SETTINGS = DigestSettings(hard_drop_tone=0.15, avoid_drop_tone=0.40, avoid_hit_similarity=0.60)


def art(name="a", similarity=0.7, avoid=0.5, tone=0.9, summary="A calm summary.", url=None) -> Article:
    return Article(
        title=f"Title {name}",
        url=url or f"https://example.com/{name}",
        source="Example News",
        similarity=similarity,
        avoid_similarity=avoid,
        tone_score=tone,
        summary=summary,
    )


# --- tone rules --------------------------------------------------------------


def test_hard_drop_removes_panic_content_regardless_of_topic():
    kept = apply_tone_rules([art("panic", tone=0.10, avoid=0.2), art("calm")], SETTINGS)

    assert [a.title for a in kept] == ["Title calm"]


def test_hard_drop_boundary_is_strictly_below():
    assert apply_tone_rules([art(tone=0.15)], SETTINGS) != []


def test_avoid_adjacent_and_dramatic_is_dropped():
    assert apply_tone_rules([art(avoid=0.62, tone=0.35)], SETTINGS) == []


def test_avoid_adjacent_but_calm_is_kept():
    assert len(apply_tone_rules([art(avoid=0.62, tone=0.50)], SETTINGS)) == 1


def test_dramatic_but_not_avoid_adjacent_is_kept():
    assert len(apply_tone_rules([art(avoid=0.55, tone=0.35)], SETTINGS)) == 1


def test_unscored_fallback_articles_pass_through():
    fallback = Article(title="t", url="https://x")

    assert apply_tone_rules([fallback], SETTINGS) == [fallback]


# --- ordering ----------------------------------------------------------------


def test_most_relevant_first_and_unscored_last():
    unscored = Article(title="u", url="https://u")
    ordered = order_by_relevance([art("low", similarity=0.63), unscored, art("high", similarity=0.69)])

    assert [a.title for a in ordered] == ["Title high", "Title low", "u"]


# --- subject -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "degraded", "expected"),
    [
        (3, False, "ZenSift · 3 calm reads · Tue 20 Oct"),
        (1, False, "ZenSift · 1 calm read · Tue 20 Oct"),
        (0, False, "ZenSift · a quiet day · Tue 20 Oct"),
        (3, True, "ZenSift · 3 stories (summaries unavailable) · Tue 20 Oct"),
    ],
)
def test_subject(n, degraded, expected):
    assert build_subject([art(str(i)) for i in range(n)], DAY, degraded) == expected


# --- rendering ---------------------------------------------------------------


def test_html_shows_source_summary_link_and_date_but_no_headline():
    html = render_html([art("x")], DAY, degraded=False)

    assert '<a href="https://example.com/x"' in html
    assert "Read at Example News" in html and "A calm summary." in html
    assert "Tuesday, 20 October 2026" in html
    assert "Title x" not in html


def test_link_label_without_source():
    html = render_html([art().model_copy(update={"source": ""})], DAY, degraded=False)

    assert "Read the full story" in html


def test_degraded_items_show_opening_lines_not_headline():
    body = "First sentence of the story is calm. " * 20
    fallback = Article(title="Fallback headline", url="https://x.example", source="Wire", body=body)

    html = render_html([fallback], DAY, degraded=True)
    text = render_text([fallback], DAY, degraded=True)

    assert "Fallback headline" not in html and "Fallback headline" not in text
    assert "First sentence of the story is calm." in html
    assert body.strip() not in text


def test_publisher_link_is_preferred_over_feed_url():
    article = art().model_copy(update={"link": "https://publisher.example/story"})

    assert 'href="https://publisher.example/story"' in render_html([article], DAY, degraded=False)
    assert "https://publisher.example/story" in render_text([article], DAY, degraded=False)


def test_unsafe_publisher_link_is_not_rendered():
    article = art().model_copy(update={"link": "javascript:alert(1)"})

    assert "href=" not in render_html([article], DAY, degraded=False)


def test_nothing_readable_has_its_own_message_and_subject():
    result = assembler_node({"digest_articles": [], "unreadable_count": 4, "digest_date": DAY})

    assert result["email_subject"] == "ZenSift · nothing readable today · Tue 20 Oct"
    assert escape(NOTHING_READABLE) in result["email_html"]
    assert NOTHING_READABLE in result["email_text"]


def test_tone_rules_emptying_the_digest_is_still_a_quiet_day():
    state = {"digest_articles": [art(tone=0.05)], "unreadable_count": 2, "digest_date": DAY}

    result = assembler_node(state)

    assert result["email_subject"].startswith("ZenSift · a quiet day")
    assert QUIET_DAY in result["email_text"]


def test_html_escapes_untrusted_text():
    evil = art(summary='<img src=x onerror="alert(1)">', url="https://example.com/?a=1&b=2")
    evil = evil.model_copy(update={"source": "<script>steal()</script>"})

    html = render_html([evil], DAY, degraded=False)

    assert "<script>" not in html and "<img" not in html
    assert "&lt;script&gt;" in html
    assert 'href="https://example.com/?a=1&amp;b=2"' in html


@pytest.mark.parametrize("url", ["javascript:alert(1)", "data:text/html,hi", "ftp://x"])
def test_html_never_links_non_http_urls(url):
    html = render_html([art(url=url)], DAY, degraded=False)

    assert "href=" not in html
    assert "A calm summary." in html


def test_degraded_notice_and_quiet_day():
    assert escape(DEGRADED_NOTICE) in render_html([art()], DAY, degraded=True)
    assert DEGRADED_NOTICE in render_text([art()], DAY, degraded=True)
    assert QUIET_DAY in render_html([], DAY, degraded=False)
    assert QUIET_DAY in render_text([], DAY, degraded=False)


def test_plain_text_version_has_source_summary_and_url_but_no_headline():
    text = render_text([art("x")], DAY, degraded=False)

    assert "EXAMPLE NEWS" in text and "A calm summary." in text
    assert "Read at Example News: https://example.com/x" in text
    assert "Title x" not in text


# --- node --------------------------------------------------------------------


def test_assembler_node_filters_orders_and_overwrites_digest_articles():
    state = {
        "digest_articles": [
            art("low", similarity=0.63),
            art("panic", tone=0.05, summary="Panic summary."),
            art("high", similarity=0.69),
        ],
        "digest": SETTINGS,
        "digest_date": DAY,
    }

    result = assembler_node(state)

    assert [a.title for a in result["digest_articles"]] == ["Title high", "Title low"]
    assert result["email_subject"] == "ZenSift · 2 calm reads · Tue 20 Oct"
    assert "Panic summary." not in result["email_html"]
    assert "Panic summary." not in result["email_text"]


def test_assembler_node_uses_default_settings_when_absent():
    result = assembler_node({"digest_articles": [art(tone=0.05)], "digest_date": DAY})

    assert result["digest_articles"] == []
