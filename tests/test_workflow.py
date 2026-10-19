import pytest

from src.graph import workflow
from src.graph.workflow import Article, compile_workflow
from src.utils.deduplicate import get_sent_urls, mark_sent

INPUT = {"email": "a@b.com", "interests": "open source LLMs", "errors": []}


@pytest.fixture(autouse=True)
def stub_planner(mocker):
    return mocker.patch.object(
        workflow, "planner_node", return_value={"rss_urls": [], "hn_queries": []}
    )


def _articles(n: int) -> list[Article]:
    return [Article(title=f"t{i}", url=f"https://x/{i}") for i in range(n)]


def _urls(articles: list[Article]) -> list[str]:
    return [a.url for a in articles]


def test_graph_runs_end_to_end_with_stub_nodes():
    result = compile_workflow().invoke(INPUT)

    assert result["email"] == "a@b.com"
    assert result["delivery_status"] == "failed"
    assert not result.get("degraded")


def test_deduplicate_filters_urls_already_in_sqlite(mocker, isolated_db):
    articles = _articles(4)
    mark_sent(["https://x/0", "https://x/1"], isolated_db)
    mocker.patch.object(workflow, "fetcher_node", return_value={"raw_articles": articles})
    sifter = mocker.patch.object(workflow, "sifter_node", return_value={})

    compile_workflow().invoke(INPUT)

    assert _urls(sifter.call_args.args[0]["fresh_articles"]) == ["https://x/2", "https://x/3"]


def test_deduplicate_drops_repeats_within_one_run():
    article = Article(title="t", url="https://x/dup")

    result = workflow.deduplicate_node({"raw_articles": [article, article]})

    assert _urls(result["fresh_articles"]) == ["https://x/dup"]


def test_deduplicate_passes_through_when_db_is_broken(mocker):
    mocker.patch.object(workflow, "get_sent_urls", side_effect=workflow.sqlite3.Error("locked"))
    articles = _articles(2)

    result = workflow.deduplicate_node({"raw_articles": articles})

    assert result["fresh_articles"] == articles
    assert result["errors"]


def test_editor_api_error_routes_to_fallback_with_top_3_raw_links(mocker):
    articles = _articles(5)
    mocker.patch.object(workflow, "fetcher_node", return_value={"raw_articles": articles})
    mocker.patch.object(workflow, "editor_node", side_effect=RuntimeError("429 after retries"))
    assembler = mocker.spy(workflow, "assembler_node")

    result = compile_workflow().invoke(INPUT)

    assert result["llm_failed"] is True
    assert result["degraded"] is True
    assert _urls(result["digest_articles"]) == _urls(articles[:3])
    assert any("editor" in e for e in result["errors"])
    assembler.assert_called_once()


def test_sifter_api_error_skips_sanitizer_and_editor(mocker):
    mocker.patch.object(workflow, "fetcher_node", return_value={"raw_articles": _articles(5)})
    mocker.patch.object(workflow, "sifter_node", side_effect=RuntimeError("503"))
    sanitizer = mocker.spy(workflow, "sanitizer_node")
    editor = mocker.spy(workflow, "editor_node")

    result = compile_workflow().invoke(INPUT)

    assert result["degraded"] is True
    assert len(result["digest_articles"]) == 3
    sanitizer.assert_not_called()
    editor.assert_not_called()


def test_fallback_never_resends_already_sent_articles(mocker, isolated_db):
    articles = _articles(5)
    mark_sent(["https://x/0"], isolated_db)
    mocker.patch.object(workflow, "fetcher_node", return_value={"raw_articles": articles})
    mocker.patch.object(workflow, "sifter_node", side_effect=RuntimeError("down"))

    result = compile_workflow().invoke(INPUT)

    assert _urls(result["digest_articles"]) == ["https://x/1", "https://x/2", "https://x/3"]


def test_successful_delivery_records_sent_urls(mocker, isolated_db):
    articles = _articles(2)
    mocker.patch.object(workflow, "fetcher_node", return_value={"raw_articles": articles})
    mocker.patch.object(workflow, "delivery_node", return_value={"delivery_status": "sent"})

    compile_workflow().invoke(INPUT)

    assert get_sent_urls(_urls(articles), isolated_db) == set(_urls(articles))


def test_failed_delivery_records_nothing(mocker, isolated_db):
    articles = _articles(2)
    mocker.patch.object(workflow, "fetcher_node", return_value={"raw_articles": articles})
    mocker.patch.object(workflow, "delivery_node", return_value={"delivery_status": "failed"})
    record = mocker.spy(workflow, "record_sent_node")

    compile_workflow().invoke(INPUT)

    record.assert_not_called()
    assert get_sent_urls(_urls(articles), isolated_db) == set()
