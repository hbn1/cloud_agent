from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool

from core.memory.long_term import LongTermMemory

logger = logging.getLogger(__name__)

_long_term_memory = None


def set_long_term_memory(memory):
    global _long_term_memory
    _long_term_memory = memory


@tool
def get_user_profile(query: str) -> str:
    """Retrieve stored user preferences, background, and facts.

    Use when:
    - The user asks about their past preferences.
    - You need context about the user's technical background, budget, or habits.
    - The user mentions "my preferences" or "what you know about me".

    Args:
        query: Natural-language description of what user info to retrieve.
    """
    global _long_term_memory
    if _long_term_memory is None or not _long_term_memory.available:
        return "User profile store is currently unavailable."

    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    results = loop.run_until_complete(
        _long_term_memory.retrieve_relevant("_current_user_", query, top_k=5)
    )

    if not results:
        return "No relevant user profile information found."

    formatted = ["[User Profile / Background]"]
    for i, r in enumerate(results):
        formatted.append(f"{i+1}. {r}")
    return "\n".join(formatted)
