"""Sifter: keep only articles whose meaning is close to the user's interests.

Embeds each interest, each avoid topic and each article, scores every article
against all of them in an in-memory ChromaDB collection, then keeps articles whose
best interest score is above the threshold and clearly beats every avoid topic.
"""

from __future__ import annotations

import logging
from uuid import uuid4

import chromadb
from chromadb.config import Settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.agents.llm import build_embeddings, llm_retry
from src.config import SifterSettings
from src.graph.state import Article, ZenSiftState

logger = logging.getLogger(__name__)

# Defaults live in SifterSettings (config.yaml overrides them); calibration notes are there too.
_DEFAULTS = SifterSettings()
SIMILARITY_THRESHOLD = _DEFAULTS.similarity_threshold
MIN_MARGIN = _DEFAULTS.min_margin
MAX_ARTICLES = _DEFAULTS.max_articles


def article_text(article: Article) -> str:
    return f"{article.title}\n{article.snippet}".strip()


@llm_retry
def embed_texts(
    embeddings: GoogleGenerativeAIEmbeddings, queries: list[str], documents: list[str]
) -> tuple[list[list[float]], list[list[float]]]:
    # Interests and avoid topics share one task type so their scores are comparable.
    query_vectors = embeddings.embed_documents(queries, task_type="RETRIEVAL_QUERY")
    doc_vectors = embeddings.embed_documents(documents, task_type="RETRIEVAL_DOCUMENT")
    return query_vectors, doc_vectors


def similarity_matrix(
    query_vectors: list[list[float]], doc_vectors: list[list[float]]
) -> list[list[float]]:
    """scores[q][d] = cosine similarity between query q and document d."""
    if not query_vectors or not doc_vectors:
        return [[] for _ in query_vectors]
    client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    # Ephemeral clients share one in-process store, so isolate each run by name.
    name = f"zensift-{uuid4().hex}"
    collection = client.create_collection(
        name, embedding_function=None, configuration={"hnsw": {"space": "cosine"}}
    )
    try:
        collection.add(ids=[str(i) for i in range(len(doc_vectors))], embeddings=doc_vectors)
        result = collection.query(
            query_embeddings=query_vectors, n_results=len(doc_vectors), include=["distances"]
        )
    finally:
        client.delete_collection(name)

    scores = [[0.0] * len(doc_vectors) for _ in query_vectors]
    for q, (ids, distances) in enumerate(zip(result["ids"], result["distances"])):
        for doc_id, distance in zip(ids, distances):
            # Chroma's cosine distance = 1 - cosine similarity.
            scores[q][int(doc_id)] = 1.0 - distance
    return scores


def sift(
    interests: list[str],
    articles: list[Article],
    embeddings: GoogleGenerativeAIEmbeddings,
    avoid: list[str] | None = None,
    threshold: float = SIMILARITY_THRESHOLD,
    min_margin: float = MIN_MARGIN,
    max_articles: int = MAX_ARTICLES,
) -> list[Article]:
    if not articles:
        return []
    interests = [topic for topic in interests if topic.strip()]
    if not interests:
        raise ValueError("At least one interest is required.")
    avoid = [topic for topic in (avoid or []) if topic.strip()]
    query_vectors, doc_vectors = embed_texts(
        embeddings, [*interests, *avoid], [article_text(a) for a in articles]
    )
    matrix = similarity_matrix(query_vectors, doc_vectors)
    interest_rows, avoid_rows = matrix[: len(interests)], matrix[len(interests) :]

    # An article only needs to match one interest well; scoring against the average would dilute it.
    best = [max(range(len(interests)), key=lambda q: interest_rows[q][i]) for i in range(len(articles))]
    scores = [interest_rows[best[i]][i] for i in range(len(articles))]

    kept: list[Article] = []
    for i in sorted(range(len(articles)), key=lambda i: scores[i], reverse=True):
        article, score = articles[i], scores[i]
        avoid_score = max((row[i] for row in avoid_rows), default=None)

        if avoid_score is not None and score - avoid_score <= min_margin:
            verdict = "drop (avoid)"
        elif score <= threshold:
            verdict = "drop (below threshold)"
        elif len(kept) >= max_articles:
            verdict = "drop (cap)"
        else:
            verdict = "KEEP"
            kept.append(
                article.model_copy(
                    update={
                        "similarity": round(score, 4),
                        "avoid_similarity": None if avoid_score is None else round(avoid_score, 4),
                    }
                )
            )
        avoid_text = "  -  " if avoid_score is None else f"{avoid_score:.3f}"
        logger.info(
            "%-22s interest=%.3f (#%d) avoid=%s  %s",
            verdict, score, best[i] + 1, avoid_text, article.title[:70],
        )

    logger.info("Sifter kept %d of %d articles.", len(kept), len(articles))
    return kept


def sifter_node(state: ZenSiftState) -> dict:
    articles = state.get("fresh_articles", [])
    if not articles:
        return {"sifted_articles": []}
    # Errors propagate on purpose: the workflow's llm_guarded wrapper routes them to fallback.
    settings = state.get("sifter") or _DEFAULTS
    kept = sift(
        state["interests"],
        articles,
        build_embeddings(),
        avoid=state.get("avoid"),
        threshold=settings.similarity_threshold,
        min_margin=settings.min_margin,
        max_articles=settings.max_articles,
    )
    return {"sifted_articles": kept}
