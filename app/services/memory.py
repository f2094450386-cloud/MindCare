from __future__ import annotations

import json
import logging
from datetime import datetime
from importlib import import_module

from app.core.config import Settings
from app.models.entities import ChatMessage
from app.schemas.dtos import AiMessage
from app.services.privacy import PrivacySanitizer


logger = logging.getLogger(__name__)


class RedisShortTermMemoryStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.privacy = PrivacySanitizer()
        self.client = self._connect()

    def load_recent(self, session_public_id: str) -> list[AiMessage]:
        if self.client is None:
            return []
        try:
            return self._read(session_public_id, self.settings.redis_memory_max_messages)
        except Exception as exc:
            logger.warning("Redis memory read unavailable: %s", exc)
            return []

    def messages_from_rows(self, rows: list[ChatMessage]) -> list[AiMessage]:
        return [self._message_from_row(row) for row in rows]

    def append(self, session_public_id: str, role: str, content: str) -> None:
        if self.client is None:
            return
        key = self._key(session_public_id)
        payload = self._serialize(role, content)
        try:
            self.client.rpush(key, payload)
            self.client.ltrim(key, -self.settings.redis_memory_max_messages, -1)
            self.client.expire(key, self.settings.redis_memory_ttl_seconds)
        except Exception as exc:
            logger.warning("Redis memory append unavailable: %s", exc)

    def replace(self, session_public_id: str, messages: list[AiMessage]) -> None:
        if self.client is None:
            return
        key = self._key(session_public_id)
        pipe = self.client.pipeline()
        pipe.delete(key)
        if messages:
            pipe.rpush(key, *[self._serialize(message.role, message.content) for message in messages])
            pipe.ltrim(key, -self.settings.redis_memory_max_messages, -1)
            pipe.expire(key, self.settings.redis_memory_ttl_seconds)
        try:
            pipe.execute()
        except Exception as exc:
            logger.warning("Redis memory replace unavailable: %s", exc)

    def _read(self, session_public_id: str, limit: int) -> list[AiMessage]:
        raw_items = self.client.lrange(self._key(session_public_id), -limit, -1)
        messages = []
        for raw in raw_items:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            role = str(data.get("role", "")).lower()
            content = str(data.get("content", ""))
            if role and content:
                messages.append(AiMessage(role=role, content=self.privacy.sanitize(content)))
        return messages

    def _connect(self):
        try:
            redis_module = import_module("redis")
        except ModuleNotFoundError as exc:
            raise RuntimeError("请先安装 requirements.txt 中的 redis 依赖") from exc
        client = redis_module.Redis.from_url(
            self.settings.redis_url,
            decode_responses=True,
            socket_timeout=self.settings.redis_socket_timeout_seconds,
            socket_connect_timeout=self.settings.redis_socket_timeout_seconds,
        )
        try:
            client.ping()
        except Exception as exc:
            logger.warning("Redis memory disabled: %s", exc)
            return None
        return client

    def _message_from_row(self, row: ChatMessage) -> AiMessage:
        return AiMessage(role=row.role.lower(), content=self.privacy.sanitize(row.content))

    def _serialize(self, role: str, content: str) -> str:
        return json.dumps(
            {
                "role": role.lower(),
                "content": content,
                "createdAt": datetime.utcnow().isoformat(),
            },
            ensure_ascii=False,
        )

    def _key(self, session_public_id: str) -> str:
        return f"mindbridge:short-term-memory:{session_public_id}"
