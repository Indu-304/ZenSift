"""Shared Gemini model factories and retry policy (via LangChain only)."""

from __future__ import annotations

import logging
import os

from langchain_core.exceptions import ModelAPIError, ModelRateLimitError
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

DEFAULT_CHAT_MODEL = "gemini-2.5-flash"
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"

# 1 initial call + 3 retries.
LLM_MAX_ATTEMPTS = 4


def is_transient(exc: BaseException | None) -> bool:
    """True for rate limits (429) and server errors (5xx), even when wrapped in another error."""
    while exc is not None:
        if isinstance(exc, (ModelRateLimitError, ModelAPIError)):
            return True
        # Embeddings wrap the SDK error generically; its HTTP status survives on __cause__.code.
        code = getattr(exc, "code", None)
        if isinstance(code, int) and (code == 429 or 500 <= code < 600):
            return True
        exc = exc.__cause__
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
