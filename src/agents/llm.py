"""Shared Gemini model factories and retry policy (via LangChain only)."""

from __future__ import annotations

import logging
import os

from langchain_core.exceptions import (
    ModelAPIError,
    ModelAuthenticationError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# Pin a stable model, not a "-latest" alias, so behaviour only changes when we choose.
DEFAULT_CHAT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"

# 1 initial call + 3 retries.
LLM_MAX_ATTEMPTS = 4


def _status_codes(exc: BaseException | None):
    while exc is not None:
        yield exc, getattr(exc, "code", None)
        exc = exc.__cause__


def is_transient(exc: BaseException | None) -> bool:
    """True for rate limits (429) and server errors (5xx), even when wrapped in another error."""
    for err, code in _status_codes(exc):
        if isinstance(err, (ModelRateLimitError, ModelAPIError)):
            return True
        # Embeddings wrap the SDK error generically; its HTTP status survives on __cause__.code.
        if isinstance(code, int) and (code == 429 or 500 <= code < 600):
            return True
    return False


def is_config_error(exc: BaseException | None) -> bool:
    """True for bad key (401), no access (403) or unknown model (404): retrying can't help."""
    for err, code in _status_codes(exc):
        if isinstance(err, (ModelAuthenticationError, ModelPermissionDeniedError, ModelNotFoundError)):
            return True
        if code in (401, 403, 404):
            return True
    return False


llm_retry = retry(
    retry=retry_if_exception(is_transient),
    stop=stop_after_attempt(LLM_MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


def build_chat_model(temperature: float) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_CHAT_MODEL", DEFAULT_CHAT_MODEL),
        temperature=temperature,
        # Disable the SDK's built-in retries so tenacity is the single retry layer; 0 means "SDK default".
        max_retries=1,
    )


def build_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    )
