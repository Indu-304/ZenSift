from unittest.mock import MagicMock
from urllib.parse import parse_qs, urlparse

import pytest
from langchain_core.exceptions import ModelAPIError, ModelRateLimitError
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from src.agents import llm as llm_module
from src.agents import planner
from src.agents.planner import (
    PLANNER_SYSTEM_PROMPT,
    SearchPlan,
    fallback_query,
    generate_plan,
    google_news_rss_url,
    planner_node,
)

INTERESTS = [
    "AI master's programs at German universities",
    "Open-source LLM releases",
]

PLAN = SearchPlan(
    google_news_queries=[
        '"AI master" Germany university admission',
        '"open source LLM" release -politics',
        "Germany AI education policy",
    ],
    hackernews_keywords=["open source LLM", "Llama", "German AI"],
)


@pytest.fixture(autouse=True)
def no_retry_sleep(mocker):
    return mocker.patch.object(generate_plan.retry, "sleep")


def mock_llm(*, returns=None, raises=None) -> MagicMock:
    """Fake chat model whose structured runnable returns/raises in sequence."""
    llm = MagicMock()
    structured = llm.with_structured_output.return_value
    if raises is not None:
        structured.invoke.side_effect = raises
    else:
        structured.invoke.return_value = returns
    return llm


# --- SearchPlan schema -------------------------------------------------------


def test_schema_strips_whitespace():
    plan = SearchPlan(
        google_news_queries=["  a ", "b", "c"], hackernews_keywords=["x", "y ", "z"]
    )
    assert plan.google_news_queries[0] == "a"
    assert plan.hackernews_keywords[1] == "y"


@pytest.mark.parametrize("count", [2, 4])
def test_schema_requires_exactly_three_of_each(count):
    with pytest.raises(ValidationError):
        SearchPlan(google_news_queries=["q"] * count, hackernews_keywords=["k"] * 3)
    with pytest.raises(ValidationError):
        SearchPlan(google_news_queries=["q"] * 3, hackernews_keywords=["k"] * count)


def test_schema_rejects_blank_queries():
    with pytest.raises(ValidationError):
        SearchPlan(google_news_queries=["a", "  ", "c"], hackernews_keywords=["x", "y", "z"])


# --- URL building ------------------------------------------------------------


def test_google_news_rss_url_encodes_query_safely():
    url = google_news_rss_url('"open source" LLM & -politics')

    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == planner.GOOGLE_NEWS_RSS_SEARCH
    assert params["q"] == ['"open source" LLM & -politics']
    assert params["hl"] == ["en-US"] and params["ceid"] == ["US:en"]


# --- generate_plan -----------------------------------------------------------


def test_generate_plan_uses_structured_output_and_prompts():
    llm = mock_llm(returns=PLAN)

    result = generate_plan(INTERESTS, llm)

    assert result == PLAN
    llm.with_structured_output.assert_called_once_with(SearchPlan)
    messages = llm.with_structured_output.return_value.invoke.call_args.args[0]
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == PLANNER_SYSTEM_PROMPT
    assert isinstance(messages[1], HumanMessage)
    assert all(f"- {topic}" in messages[1].content for topic in INTERESTS)


def test_generate_plan_retries_on_rate_limit_then_succeeds(no_retry_sleep):
    llm = mock_llm(raises=[ModelRateLimitError("429"), ModelAPIError("503"), PLAN])

    assert generate_plan(INTERESTS, llm) == PLAN
    assert llm.with_structured_output.return_value.invoke.call_count == 3
    assert no_retry_sleep.call_count == 2


def test_generate_plan_gives_up_after_max_attempts():
    llm = mock_llm(raises=ModelRateLimitError("429"))

    with pytest.raises(ModelRateLimitError):
        generate_plan(INTERESTS, llm)
    assert llm.with_structured_output.return_value.invoke.call_count == llm_module.LLM_MAX_ATTEMPTS


def test_generate_plan_does_not_retry_non_transient_errors():
    llm = mock_llm(raises=ValueError("bad schema"))

    with pytest.raises(ValueError):
        generate_plan(INTERESTS, llm)
    assert llm.with_structured_output.return_value.invoke.call_count == 1


# --- planner_node ------------------------------------------------------------


def test_planner_node_maps_plan_to_state(mocker):
    build = mocker.patch.object(planner, "build_chat_model", return_value=mock_llm(returns=PLAN))

    result = planner_node({"interests": INTERESTS})

    build.assert_called_once()
    assert result["rss_urls"] == [google_news_rss_url(q) for q in PLAN.google_news_queries]
    assert result["hn_queries"] == PLAN.hackernews_keywords
    assert "errors" not in result


def test_planner_node_degrades_to_raw_interests_when_llm_is_down(mocker):
    mocker.patch.object(
        planner, "build_chat_model", return_value=mock_llm(raises=ModelAPIError("503"))
    )

    result = planner_node({"interests": INTERESTS})

    assert result["rss_urls"] == [google_news_rss_url(fallback_query(INTERESTS))]
    assert result["hn_queries"] == []
    assert any(e.startswith("planner:") for e in result["errors"])


def test_fallback_query_ors_all_interests():
    assert fallback_query(["a b", "c"]) == "(a b) OR (c)"


def test_planner_node_truncates_fallback_query(mocker):
    mocker.patch.object(planner, "build_chat_model", side_effect=ValueError("no API key"))

    result = planner_node({"interests": ["x" * 1000]})

    q = parse_qs(urlparse(result["rss_urls"][0]).query)["q"][0]
    assert len(q) == planner.MAX_FALLBACK_QUERY_CHARS


# --- build_chat_model --------------------------------------------------------


def test_build_chat_model_disables_sdk_retries_and_reads_model_from_env(mocker, monkeypatch):
    monkeypatch.setenv("GEMINI_CHAT_MODEL", "gemini-test")
    chat_cls = mocker.patch.object(llm_module, "ChatGoogleGenerativeAI")

    llm_module.build_chat_model()

    chat_cls.assert_called_once_with(model="gemini-test", max_retries=1)
