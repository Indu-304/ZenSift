import math
from unittest.mock import MagicMock

import chromadb
import pytest
from chromadb.config import Settings

from src.agents import sifter
from src.agents.sifter import (
    article_text,
    embed_texts,
    sift,
    sifter_node,
    similarity_matrix,
)
from src.graph.state import Article

INTERESTS = "open source LLMs"
QUERY = [1.0, 0.0]


def unit(degrees: float) -> list[float]:
    """2-D unit vector at an angle from QUERY; its cosine similarity to QUERY is cos(angle)."""
    r = math.radians(degrees)
    return [math.cos(r), math.sin(r)]


# cos(0)=1.0, cos(30)=0.866, cos(40)=0.766, cos(45)=0.707, cos(90)=0.0
VECTORS = [unit(45), unit(0), unit(90), unit(30), unit(40)]
ARTICLES = [Article(title=f"a{i}", url=f"https://x/{i}", snippet=f"s{i}") for i in range(5)]


class FakeTransientError(Exception):
    code = 429


def fake_embeddings(doc_vectors=VECTORS, avoid_vectors=(), errors=None) -> MagicMock:
    """Fake Gemini embeddings: queries -> [QUERY, *avoid_vectors]; documents -> doc_vectors."""
    pending = list(errors or [])

    def embed_documents(texts, task_type=None):
        if pending:
            raise pending.pop(0)
        if task_type == "RETRIEVAL_QUERY":
            return [QUERY, *avoid_vectors]
        return doc_vectors

    emb = MagicMock()
    emb.embed_documents.side_effect = embed_documents
    return emb


@pytest.fixture(autouse=True)
def no_retry_sleep(mocker):
    return mocker.patch.object(embed_texts.retry, "sleep")


# --- similarity_matrix (real in-memory ChromaDB) -----------------------------


def test_similarity_matrix_scores_every_query_against_every_document():
    scores = similarity_matrix([QUERY, unit(90)], VECTORS)

    assert scores[0] == pytest.approx([0.707, 1.0, 0.0, 0.866, 0.766], abs=1e-3)
    assert scores[1] == pytest.approx([0.707, 0.0, 1.0, 0.5, 0.643], abs=1e-3)


def test_similarity_matrix_empty_documents():
    assert similarity_matrix([QUERY], []) == [[]]


def test_similarity_matrix_cleans_up_its_collection():
    similarity_matrix([QUERY], VECTORS)

    client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    assert not [c for c in client.list_collections() if c.name.startswith("zensift-")]


# --- sift: interests only ----------------------------------------------------


def test_sift_keeps_only_articles_above_threshold_sorted_with_scores():
    kept = sift(INTERESTS, ARTICLES, fake_embeddings(), threshold=0.75)

    assert [a.url for a in kept] == ["https://x/1", "https://x/3", "https://x/4"]
    assert [a.similarity for a in kept] == pytest.approx([1.0, 0.866, 0.766], abs=1e-3)
    assert all(a.avoid_similarity is None for a in kept)


def test_sift_threshold_is_strictly_greater_than():
    # Article 1 is identical to the query (similarity exactly 1.0), so it must NOT pass 1.0.
    assert sift(INTERESTS, ARTICLES, fake_embeddings(), threshold=1.0) == []


def test_sift_caps_at_max_articles():
    kept = sift(INTERESTS, ARTICLES, fake_embeddings(), threshold=0.0, max_articles=2)

    assert [a.url for a in kept] == ["https://x/1", "https://x/3"]


def test_sift_does_not_mutate_input_articles():
    sift(INTERESTS, ARTICLES, fake_embeddings())

    assert all(a.similarity is None for a in ARTICLES)


def test_sift_with_no_articles_makes_no_api_call():
    emb = fake_embeddings()

    assert sift(INTERESTS, [], emb) == []
    emb.embed_documents.assert_not_called()


def test_sift_can_return_nothing_when_all_irrelevant():
    assert sift(INTERESTS, ARTICLES, fake_embeddings(doc_vectors=[unit(90)] * 5)) == []


# --- sift: interests vs avoid ------------------------------------------------
# Picture a compass: interests point east (0 deg), "political drama" points north (90 deg).

ON_TOPIC = Article(title="New open-weight LLM released", url="https://x/llm")
DRAMA = Article(title="Politicians trade insults over budget", url="https://x/drama")


def test_article_matching_interests_is_kept_and_article_matching_avoid_is_dropped():
    emb = fake_embeddings(doc_vectors=[unit(10), unit(80)], avoid_vectors=[unit(90)])

    kept = sift(INTERESTS, [ON_TOPIC, DRAMA], emb, avoid=["political drama"], threshold=0.75)

    assert [a.url for a in kept] == [ON_TOPIC.url]
    assert kept[0].similarity == pytest.approx(math.cos(math.radians(10)), abs=1e-3)
    assert kept[0].avoid_similarity == pytest.approx(math.cos(math.radians(80)), abs=1e-3)


def test_avoid_drops_article_even_if_it_scores_above_threshold():
    # "AI regulation fight" sits at 20 deg: close to the interests, but the article (15 deg) is closer still to it.
    emb = fake_embeddings(doc_vectors=[unit(15)], avoid_vectors=[unit(20)])

    kept = sift(INTERESTS, [ON_TOPIC], emb, avoid=["AI regulation fights"], threshold=0.75)

    assert kept == []


def test_avoid_score_is_the_closest_of_several_topics():
    # Far from topic 1 (north) but very close to topic 2 (at -80 deg).
    emb = fake_embeddings(doc_vectors=[unit(-70)], avoid_vectors=[unit(90), unit(-80)])

    kept = sift(INTERESTS, [DRAMA], emb, avoid=["politics", "gossip"], threshold=0.0)

    assert kept == []


def test_avoid_does_not_drop_article_closer_to_interests():
    emb = fake_embeddings(doc_vectors=[unit(35)], avoid_vectors=[unit(90)])

    kept = sift(INTERESTS, [ON_TOPIC], emb, avoid=["political drama"], threshold=0.75)

    assert [a.url for a in kept] == [ON_TOPIC.url]


def test_interests_and_avoid_embedded_together_as_queries_articles_as_documents():
    emb = fake_embeddings(avoid_vectors=[unit(90)])

    sift(INTERESTS, ARTICLES, emb, avoid=["political drama"])

    query_call, doc_call = emb.embed_documents.call_args_list
    assert query_call.args[0] == [INTERESTS, "political drama"]
    assert query_call.kwargs["task_type"] == "RETRIEVAL_QUERY"
    assert doc_call.args[0] == [article_text(a) for a in ARTICLES]
    assert doc_call.kwargs["task_type"] == "RETRIEVAL_DOCUMENT"
    assert article_text(ARTICLES[0]) == "a0\ns0"


def test_blank_avoid_topics_are_ignored():
    emb = fake_embeddings()

    sift(INTERESTS, ARTICLES, emb, avoid=["", "   "])

    assert emb.embed_documents.call_args_list[0].args[0] == [INTERESTS]


# --- retries -----------------------------------------------------------------


def wrapped(cause: Exception) -> Exception:
    """Mimic langchain-google-genai: a generic error whose __cause__ carries the HTTP code."""
    try:
        raise cause
    except Exception as inner:
        try:
            raise RuntimeError("Error embedding content") from inner
        except RuntimeError as outer:
            return outer


def test_embedding_rate_limit_is_retried_then_succeeds(no_retry_sleep):
    emb = fake_embeddings(errors=[wrapped(FakeTransientError())])

    kept = sift(INTERESTS, ARTICLES, emb)

    assert len(kept) == 3
    assert emb.embed_documents.call_count == 3  # failed attempt + (queries, documents)
    no_retry_sleep.assert_called_once()


def test_embedding_errors_propagate_after_max_attempts():
    emb = fake_embeddings(errors=[wrapped(FakeTransientError()) for _ in range(10)])

    with pytest.raises(RuntimeError):
        sift(INTERESTS, ARTICLES, emb)
    assert emb.embed_documents.call_count == 4


# --- sifter_node -------------------------------------------------------------


def test_sifter_node_passes_avoid_from_state(mocker):
    emb = fake_embeddings(doc_vectors=[unit(10), unit(80)], avoid_vectors=[unit(90)])
    mocker.patch.object(sifter, "build_embeddings", return_value=emb)

    result = sifter_node(
        {"interests": INTERESTS, "avoid": ["political drama"], "fresh_articles": [ON_TOPIC, DRAMA]}
    )

    assert [a.url for a in result["sifted_articles"]] == [ON_TOPIC.url]


def test_sifter_node_works_without_avoid(mocker):
    mocker.patch.object(sifter, "build_embeddings", return_value=fake_embeddings())

    result = sifter_node({"interests": INTERESTS, "fresh_articles": ARTICLES})

    assert [a.url for a in result["sifted_articles"]] == ["https://x/1", "https://x/3", "https://x/4"]


def test_sifter_node_skips_api_when_nothing_fresh(mocker):
    build = mocker.patch.object(sifter, "build_embeddings")

    assert sifter_node({"interests": INTERESTS, "fresh_articles": []}) == {"sifted_articles": []}
    build.assert_not_called()


def test_sifter_node_raises_so_workflow_can_fall_back(mocker):
    mocker.patch.object(
        sifter, "build_embeddings", return_value=fake_embeddings(errors=[ValueError("bad key")])
    )

    with pytest.raises(ValueError):
        sifter_node({"interests": INTERESTS, "fresh_articles": ARTICLES})
