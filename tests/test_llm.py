import pytest
from langchain_core.exceptions import (
    ModelAPIError,
    ModelAuthenticationError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
)

from src.agents import llm


class CodedError(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.code = code


def chained(inner: Exception) -> Exception:
    outer = RuntimeError("wrapper")
    outer.__cause__ = inner
    return outer


@pytest.mark.parametrize(
    "exc",
    [
        ModelRateLimitError("429"),
        ModelAPIError("500"),
        CodedError(429),
        CodedError(503),
        chained(CodedError(429)),
        chained(chained(ModelAPIError("502"))),
    ],
)
def test_transient_errors(exc):
    assert llm.is_transient(exc)


@pytest.mark.parametrize(
    "exc",
    [ValueError("bad"), CodedError(400), CodedError(401), chained(CodedError(404)), CodedError("429")],
)
def test_non_transient_errors(exc):
    assert not llm.is_transient(exc)


@pytest.mark.parametrize(
    "exc",
    [
        ModelNotFoundError("404"),
        ModelAuthenticationError("401"),
        ModelPermissionDeniedError("403"),
        CodedError(404),
        chained(CodedError(401)),
    ],
)
def test_config_errors(exc):
    assert llm.is_config_error(exc)
    assert not llm.is_transient(exc)


@pytest.mark.parametrize("exc", [ValueError("bad"), CodedError(400), CodedError(429), ModelAPIError("500")])
def test_not_config_errors(exc):
    assert not llm.is_config_error(exc)


def test_build_embeddings_reads_model_from_env(mocker, monkeypatch):
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "models/test-embed")
    cls = mocker.patch.object(llm, "GoogleGenerativeAIEmbeddings")

    llm.build_embeddings()

    cls.assert_called_once_with(model="models/test-embed")
