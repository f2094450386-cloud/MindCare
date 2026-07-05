"""
MindBridge 短期记忆管理模块

使用 Redis 存储每个会话的最近 N 条消息作为短期上下文。
当 Redis 不可用时，静默降级（返回空列表），不影响主流程。

记忆策略：
- 每个会话独立存储，key 格式：mindbridge:short-term-memory:{session_public_id}
- 使用 Redis List 存储，rpush 追加，ltrim 保留最近 N 条
- 自动过期：TTL 默认 24 小时
- Redis 为空时从 MySQL 最近消息回填（由 MemoryAgent 调用）
- 所有读取和写入 Redis 的消息都经过隐私脱敏

数据流：
1. 用户发消息 → append() 写入 Redis（写入前脱敏）
2. Agent 启动 → load_recent() 读取 Redis
3. Redis 为空 → 从 MySQL 读取 → replace() 写入 Redis
4. 会话结束 → 助手消息也 append() 到 Redis

历史压缩（compact_history_for_prompt）：
MemoryAgent 注入 prompt 前，将较早消息折叠为一条内部摘要系统消息，
只保留最近 N 条原始消息，避免上下文过长；摘要同时用作 memory_brief。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from importlib import import_module
from typing import Protocol

from app.core.config import Settings
from app.models.entities import ChatMessage
from app.schemas.dtos import AiMessage
from app.services.privacy import PrivacySanitizer


logger = logging.getLogger(__name__)


class MemoryCompactionSettings(Protocol):
    """compact_history_for_prompt 所需的配置字段（结构化协议，便于测试传入 SimpleNamespace）。"""
    memory_compaction_enabled: bool
    memory_compaction_recent_messages: int
    memory_summary_max_chars: int


class RedisShortTermMemoryStore:
    """
    Redis 短期记忆存储。

    每个会话维护一个 Redis List，存储最近 N 条消息。
    消息格式：JSON {"role": "...", "content": "...", "createdAt": "..."}
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.privacy = PrivacySanitizer()
        self.client = self._connect()

    def load_recent(self, session_public_id: str) -> list[AiMessage]:
        """
        从 Redis 加载指定会话的最近消息。

        返回 AiMessage 列表，消息内容经过隐私脱敏。
        Redis 不可用或 key 不存在时返回空列表。
        """
        if self.client is None:
            return []
        try:
            return self._read(session_public_id, self.settings.redis_memory_max_messages)
        except Exception as exc:
            logger.warning("Redis memory read unavailable: %s", exc)
            return []

    def messages_from_rows(self, rows: list[ChatMessage]) -> list[AiMessage]:
        """将 MySQL ChatMessage 行转换为 AiMessage 列表（用于 Redis 为空时回填）。"""
        return [self._message_from_row(row) for row in rows]

    def append(self, session_public_id: str, role: str, content: str) -> None:
        """
        向会话的 Redis List 追加一条消息。

        操作步骤：
        1. rpush 追加到 List 尾部
        2. ltrim 只保留最近 N 条（防止 List 无限增长）
        3. expire 重置 TTL
        """
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
        """
        替换会话的整个短期记忆（用于从 MySQL 回填时）。

        使用 Redis pipeline 原子操作：
        1. 删除旧 key
        2. rpush 所有消息
        3. ltrim 保留最近 N 条
        4. expire 设置 TTL
        """
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
        """从 Redis List 读取最近 N 条消息，解析 JSON 并脱敏。"""
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
        """
        建立 Redis 连接。

        使用延迟导入 redis 模块，未安装时抛出明确错误。
        连接失败时返回 None（降级为无 Redis 模式）。
        """
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
        """将 MySQL ChatMessage 行转换为 AiMessage（脱敏后）。"""
        return AiMessage(role=row.role.lower(), content=self.privacy.sanitize(row.content))

    def _serialize(self, role: str, content: str) -> str:
        """
        将消息序列化为 JSON 字符串存储到 Redis。

        写入前对 content 做隐私脱敏，避免手机号/邮箱/身份证号等
        敏感信息以明文形式落入 Redis。
        """
        return json.dumps(
            {
                "role": role.lower(),
                "content": self.privacy.sanitize(content),
                "createdAt": datetime.utcnow().isoformat(),
            },
            ensure_ascii=False,
        )

    def _key(self, session_public_id: str) -> str:
        """生成 Redis key：mindbridge:short-term-memory:{session_public_id}"""
        return f"mindbridge:short-term-memory:{session_public_id}"


def compact_history_for_prompt(
    history: list[AiMessage],
    settings: MemoryCompactionSettings,
    current_input: str = "",
) -> tuple[list[AiMessage], str]:
    """
    压缩历史消息用于注入 prompt，同时返回学生不可见的记忆摘要。

    将较早的消息折叠为一条内部摘要系统消息，只保留最近 N 条原始消息，
    避免上下文过长；摘要是确定性生成的（非 LLM），不包含诊断性标签，
    既可用于 prompt 上下文，也可用于审计（AgentRunTrace.memory_brief）。
    """
    sanitized = [AiMessage(role=item.role, content=PrivacySanitizer().sanitize(item.content)) for item in history]
    if not sanitized:
        return [], "无相关历史记忆。"

    recent_count = max(2, int(getattr(settings, "memory_compaction_recent_messages", 8)))
    max_chars = max(120, int(getattr(settings, "memory_summary_max_chars", 500)))
    brief = summarize_history_for_memory(sanitized, current_input, max_chars)

    if not getattr(settings, "memory_compaction_enabled", True) or len(sanitized) <= recent_count:
        return sanitized, brief

    recent = sanitized[-recent_count:]
    summary_message = AiMessage(
        role="system",
        content=(
            "历史摘要（仅供 MindBridge 内部上下文使用；不要向学生展示；"
            "不要据此输出诊断、风险等级或后台标签）：\n" + brief
        ),
    )
    return [summary_message, *recent], brief


def summarize_history_for_memory(history: list[AiMessage], current_input: str = "", max_chars: int = 500) -> str:
    """
    确定性生成历史记忆摘要（不调用 LLM）。

    分别提取学生近期关注点和已给过的支持要点，拼接为简短摘要文本。
    """
    privacy = PrivacySanitizer()
    user_points = []
    assistant_points = []
    for message in history:
        content = " ".join(privacy.sanitize(message.content).split())
        if not content:
            continue
        if message.role == "user":
            user_points.append(content)
        elif message.role == "assistant":
            assistant_points.append(content)

    parts = []
    if user_points:
        parts.append("学生近期关注：" + "；".join(_clip(item, 80) for item in user_points[-4:]))
    if assistant_points:
        parts.append("已给过的支持：" + "；".join(_clip(item, 70) for item in assistant_points[-3:]))
    if current_input:
        parts.append("本轮输入关注：" + _clip(privacy.sanitize(current_input), 80))
    if not parts:
        return "无相关历史记忆。"
    return _clip("\n".join(parts), max_chars)


def _clip(text: str, limit: int) -> str:
    """将文本截断到指定长度，超出部分用 ... 代替。"""
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)] + "..."
