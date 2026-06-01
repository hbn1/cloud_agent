"""? Redis ??????????

?????/???????? TTL ????? Redis ?? List ?? (RPUSH + LTRIM)
???????? get?append?set ??????
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

COMPRESSION_THRESHOLD = 10  # trim when messages exceed this count
DEFAULT_TTL = 1800          # 30 minutes in seconds


class ShortTermMemory:
    """?? Redis ???????????? List ?????

    ???
    - ???/?????? (`memory:short:{user_id}:{session_id}`)
    - ?? TTL ???????? 30 ???????
    - ?? COMPRESSION_THRESHOLD ???????
    - ?? RPUSH + LTRIM ??????????
    - ??????? Redis ????????????
    """

    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = DEFAULT_TTL) -> None:
        self._redis_url = redis_url
        self._ttl = ttl
        self._client: Any = None
        self._available: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Connect to Redis; sets _available=False on failure (no exception raised)."""
        try:
            import redis.asyncio as aioredis  # type: ignore[import]

            self._client = aioredis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
                health_check_interval=30,
                retry_on_timeout=True,
            )
            await self._client.ping()
            self._available = True
            logger.info("ShortTermMemory: Redis connected at %s", self._redis_url)
        except Exception as exc:
            logger.warning(
                "ShortTermMemory: Redis unavailable (%s) ? short-term memory disabled.", exc
            )
            self._available = False

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_messages(self, user_id: str, session_id: str) -> list[dict[str, Any]]:
        """Return stored messages for the given user/session.

        Uses LRANGE to read all entries from the Redis list.
        Returns an empty list when Redis is unavailable or the key is missing.
        """
        if not self._available:
            return []
        try:
            raw_list = await self._client.lrange(self._key(user_id, session_id), 0, -1)
            return [json.loads(item) for item in raw_list]
        except Exception as exc:
            logger.warning("ShortTermMemory.get_messages failed: %s", exc)
            self._available = False
            return []

    async def save_messages(
        self, user_id: str, session_id: str, messages: list[dict[str, Any]]
    ) -> None:
        """Atomically replace all messages using a pipeline with DEL + RPUSH.

        Args:
            user_id: Unique user identifier.
            session_id: Unique session identifier.
            messages: List of message dicts with ``role`` and ``content`` keys.
        """
        if not self._available:
            return
        try:
            if len(messages) > COMPRESSION_THRESHOLD:
                messages = self._trim(messages)
            key = self._key(user_id, session_id)
            pipe = self._client.pipeline()
            pipe.delete(key)
            if messages:
                serialized = [json.dumps(m, ensure_ascii=False) for m in messages]
                pipe.rpush(key, *serialized)
            pipe.expire(key, self._ttl)
            await pipe.execute()
            logger.debug(
                "ShortTermMemory: saved %d messages for %s:%s", len(messages), user_id, session_id
            )
        except Exception as exc:
            logger.warning("ShortTermMemory.save_messages failed: %s", exc)
            self._available = False

    async def append_message(
        self, user_id: str, session_id: str, role: str, content: str
    ) -> None:
        """Atomically append a single message using RPUSH + LTRIM + EXPIRE in a pipeline.

        Uses Redis pipeline to guarantee atomicity ? no get?append?set race condition.
        """
        if not self._available:
            return
        try:
            key = self._key(user_id, session_id)
            serialized = json.dumps({"role": role, "content": content}, ensure_ascii=False)
            # Atomic pipeline: RPUSH new message, trim old ones, refresh TTL
            pipe = self._client.pipeline()
            pipe.rpush(key, serialized)
            pipe.ltrim(key, -COMPRESSION_THRESHOLD, -1)  # keep last N
            pipe.expire(key, self._ttl)
            await pipe.execute()
        except Exception as exc:
            logger.warning("ShortTermMemory.append_message failed: %s", exc)
            self._available = False

    async def clear(self, user_id: str, session_id: str) -> None:
        """Delete all messages for a user/session."""
        if not self._available:
            return
        try:
            await self._client.delete(self._key(user_id, session_id))
        except Exception as exc:
            logger.error("ShortTermMemory.clear failed: %s", exc)

    @property
    def available(self) -> bool:
        """True if Redis is reachable."""
        return self._available

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(user_id: str, session_id: str) -> str:
        return f"memory:short:{user_id}:{session_id}"

    @staticmethod
    def _trim(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep system messages + the 6 most recent non-system messages."""
        system_msgs = [m for m in messages if m.get("role") == "system"]
        other_msgs = [m for m in messages if m.get("role") != "system"]
        return system_msgs + other_msgs[-6:]
    # ------------------------------------------------------------------
    # Summary Buffer
    # ------------------------------------------------------------------

    async def trim_with_summary(
        self,
        user_id: str,
        session_id: str,
        summary_fn=None,  # async callable(dropped_messages) -> summary_str
    ) -> str | None:
        """Trim messages and optionally generate a summary of dropped messages.

        Uses Redis pipeline to atomically read, trim, and write back.
        If summary_fn is provided, dropped messages are passed to it
        for compression into a summary string.

        Returns the summary string if generated, else None.
        """
        if not self._available:
            return None
        try:
            key = self._key(user_id, session_id)
            pipe = self._client.pipeline()
            # Read all + get length in one round-trip
            pipe.lrange(key, 0, -1)
            pipe.llen(key)
            results = await pipe.execute()
            raw_list = results[0]
            total = results[1]

            if total <= COMPRESSION_THRESHOLD:
                return None  # no trimming needed

            # Parse messages
            messages = [json.loads(item) for item in raw_list]
            system_msgs = [m for m in messages if m.get("role") == "system"]
            other_msgs = [m for m in messages if m.get("role") != "system"]

            # Messages to drop (oldest non-system)
            keep_count = 6
            dropped = other_msgs[: -keep_count] if len(other_msgs) > keep_count else []
            kept = system_msgs + other_msgs[-keep_count:]

            summary = None
            if dropped and summary_fn:
                try:
                    summary = await summary_fn(dropped)
                    if summary:
                        # Prepend summary as a system message
                        kept.insert(0, {"role": "system", "content": f"[Context Summary]\n{summary}"})
                except Exception as exc:
                    logger.warning("Summary generation failed: %s", exc)

            # Atomically replace with pipeline
            pipe2 = self._client.pipeline()
            pipe2.delete(key)
            if kept:
                serialized = [json.dumps(m, ensure_ascii=False) for m in kept]
                pipe2.rpush(key, *serialized)
            pipe2.expire(key, self._ttl)
            await pipe2.execute()

            logger.debug(
                "ShortTermMemory: trimmed %d messages, kept %d for %s:%s",
                len(dropped), len(kept), user_id, session_id,
            )
            return summary
        except Exception as exc:
            logger.warning("ShortTermMemory.trim_with_summary failed: %s", exc)
            return None
