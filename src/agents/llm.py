"""Shared Gemini chat model factory and retry policy (via LangChain only)."""

from __future__ import annotations

import logging
import os

from langchain_core.exceptions import ModelAPIError, ModelRateLimitError
from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

DEFAULT_CHAT_MODEL = "gemini-2.5-flash"

# 1 initial call + 3 retries.
LLM_MAX_ATTEMPTS = 4

llm_retry = retry(
    retry=retry_if_exception_type((ModelRateLimitError, ModelAPIError)),
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
