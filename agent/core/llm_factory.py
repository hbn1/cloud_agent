"""Shared LLM factory — single source of truth for ChatOpenAI instances.

All Agents should use get_llm() instead of creating their own ChatOpenAI.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def get_llm(temperature: float = 0.1) -> ChatOpenAI:
    """Get a cached ChatOpenAI instance for the given temperature.

    Two temperature variants are cached: default (0.1) and creative (0.3).
    This avoids creating duplicate LLM instances across Agents.
    """
    settings = get_settings()
    api_key = settings.dashscope_api_key
    return ChatOpenAI(
        api_key=SecretStr(api_key) if api_key else None,
        model=settings.model,
        base_url=settings.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=temperature,
    )