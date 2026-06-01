"""Reflection Worker — distill fragmented Milvus memories into structured tags.

Runs as a periodic background task outside the main conversation loop.
Reads recent Milvus preference vectors for a user, uses LLM to cluster and
abstract them into higher-level structured tags, then saves those tags as
new preference entries (tag:xxx format).

Design:
- Not tied to any single session — runs per user on a schedule.
- Idempotent: uses cosine dedup to avoid duplicate tag entries.
- Lightweight: only processes users with >= 5 recent unstructured preferences.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from core.llm_factory import get_llm
from core.memory.long_term import LongTermMemory

logger = logging.getLogger(__name__)

REFLECTION_MIN_PREFS = 5      # min unstructured prefs before reflection triggers
REFLECTION_INTERVAL_SEC = 300 # run every 5 minutes


class ReflectionWorker:
    """Periodically consolidates fragmented memories into structured tags."""

    def __init__(self, long_term_memory: LongTermMemory) -> None:
        self._lts = long_term_memory
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self, user_ids: list[str]) -> None:
        """Start periodic reflection for given user IDs."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(user_ids))
        logger.info("ReflectionWorker: started for %d users", len(user_ids))

    async def stop(self) -> None:
        """Gracefully stop the worker."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("ReflectionWorker: stopped")

    async def _loop(self, user_ids: list[str]) -> None:
        while self._running:
            for uid in user_ids:
                try:
                    await self._reflect_user(uid)
                except Exception as exc:
                    logger.warning("ReflectionWorker: failed for user %s: %s", uid, exc)
            await asyncio.sleep(REFLECTION_INTERVAL_SEC)

    async def _reflect_user(self, user_id: str) -> None:
        """Run one round of reflection for a single user."""
        if not self._lts.available:
            return

        # 1. Retrieve recent unstructured preferences
        raw_prefs = await self._lts.retrieve_relevant(user_id, "preferences background habits", top_k=20)
        unstructured = [p for p in raw_prefs if not p.startswith("tag:")]

        if len(unstructured) < REFLECTION_MIN_PREFS:
            return

        # 2. Use LLM to distill into structured tags
        tags = await self._extract_tags(unstructured)

        # 3. Save tags back to Milvus (dedup handled by LongTermMemory)
        for tag in tags:
            await self._lts.save_memory(user_id, tag, memory_type="preference")

        if tags:
            logger.info(
                "ReflectionWorker: user %s — %d raw prefs -> %d tags: %s",
                user_id, len(unstructured), len(tags), tags,
            )

    async def _extract_tags(self, preferences: list[str]) -> list[str]:
        """Use LLM to distill preference fragments into structured tags."""
        llm = get_llm(temperature=0.0)
        pref_text = "\n".join(f"- {p}" for p in preferences)

        prompt = f"""You are a knowledge distillation engine. Given a list of user preference fragments,
extract 3-5 key structured tags. Each tag must start with "tag:" followed by a category:value pair.

Rules:
- Combine similar fragments into a single tag.
- Use concise, machine-readable format: tag:category:subcategory=value
- Examples: tag:budget=sensitive, tag:product=GPU_preferred, tag:region=Beijing
- Only output tags that are clearly supported by the fragments.
- Output one tag per line, no explanation.

Fragments:
{pref_text}

Tags:"""

        response = await llm.ainvoke(prompt)
        content_str = (
            response.content
            if isinstance(response.content, str)
            else str(response.content[0]) if isinstance(response.content, list) and response.content
            else str(response.content)
        )
        tags = [line.strip() for line in content_str.split("\n") if line.strip().startswith("tag:")]
        return tags