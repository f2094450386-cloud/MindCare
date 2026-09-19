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
import re
from dataclasses import dataclass
from datetime import datetime
from importlib import import_module
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Protocol

from app.core.config import Settings
from app.schemas.dtos import AiMessage
from app.services.privacy import PrivacySanitizer

if TYPE_CHECKING:
    from app.models.entities import ChatMessage


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptMemoryContext:
    """Exact memory fields consumed by the online ResponseAgent prompt."""

    memory_brief: str
    model_history: list[AiMessage]
    source_message_count: int
    compacted_message_count: int


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

    def messages_from_rows(self, rows: list["ChatMessage"]) -> list[AiMessage]:
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

    def _message_from_row(self, row: "ChatMessage") -> AiMessage:
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


def load_session_history(
    db,
    memory: RedisShortTermMemoryStore,
    session,
    settings: Settings,
) -> list[AiMessage]:
    """Load the same sanitized, storage-bounded history for online agents."""
    from app.models.entities import ChatMessage

    history = memory.load_recent(session.public_id)
    if not history:
        rows = (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == session.id)
            .order_by(ChatMessage.created_at.desc())
            .limit(settings.redis_memory_max_messages)
            .all()
        )
        rows.reverse()
        history = memory.messages_from_rows(rows)
        if history:
            memory.replace(session.public_id, history)
    return sanitize_and_limit_history(history, settings)


def sanitize_and_limit_history(
    history: list[AiMessage],
    settings: MemoryCompactionSettings,
) -> list[AiMessage]:
    """Apply online privacy and Redis message-count bounds to supplied history."""
    privacy = PrivacySanitizer()
    allowed_roles = {"system", "user", "assistant", "tool"}
    sanitized = [
        AiMessage(role=item.role.lower(), content=privacy.sanitize(item.content))
        for item in history
        if item.role.lower() in allowed_roles and item.content
    ]
    storage_limit = max(1, int(getattr(settings, "redis_memory_max_messages", len(sanitized) or 1)))
    return sanitized[-storage_limit:]


def assemble_memory_context(
    history: list[AiMessage],
    current_input: str,
    settings: Settings,
    *,
    compaction_enabled: bool | None = None,
    memory_brief: str | None = None,
    artifacts: list[dict[str, Any]] | None = None,
) -> PromptMemoryContext:
    """
    Build online memoryBrief and bounded modelHistory.

    Evaluation uses this exact function for both baselines. The only permitted
    baseline difference is ``compaction_enabled``.
    """
    sanitized = sanitize_and_limit_history(history, settings)
    sanitized_input = PrivacySanitizer().sanitize(current_input)
    effective_compaction = (
        settings.memory_compaction_enabled
        if compaction_enabled is None
        else bool(compaction_enabled)
    )
    compaction_settings = SimpleNamespace(
        memory_compaction_enabled=effective_compaction,
        memory_compaction_recent_messages=settings.memory_compaction_recent_messages,
        memory_summary_max_chars=settings.memory_summary_max_chars,
    )
    compacted, deterministic_brief = compact_history_for_prompt(
        sanitized,
        compaction_settings,
        sanitized_input,
        include_summary_message=False,
    )
    brief = memory_brief if memory_brief is not None else deterministic_brief
    artifact_message = _artifact_memory_message(artifacts or [])
    prompt_history = [
        *([artifact_message] if artifact_message is not None else []),
        *compacted,
        AiMessage(role="user", content=sanitized_input),
    ]
    bounded = bound_model_history(prompt_history, settings.chat_history_limit)
    return PromptMemoryContext(
        memory_brief=brief or "无相关历史记忆。",
        model_history=bounded,
        source_message_count=len(sanitized),
        compacted_message_count=len(compacted),
    )


def summarize_memory_for_prompt(
    history: list[AiMessage],
    current_input: str,
    settings: Settings,
    client,
    system_prompt: str,
) -> str:
    """Build the online memoryBrief with deterministic fallback and bounds."""
    sanitized = sanitize_and_limit_history(history, settings)
    max_chars = max(120, settings.memory_summary_max_chars)
    if not sanitized:
        return "无相关历史记忆。"
    fallback = summarize_history_for_memory(sanitized, current_input, max_chars)
    if str(getattr(getattr(client, "settings", None), "ai_provider", "")).lower() == "mock":
        return fallback
    try:
        rendered_history = "\n".join(
            f"{message.role}: {message.content}"
            for message in sanitized
        )
        summary = client.complete(
            [
                AiMessage(
                    role="system",
                    content=(
                        f"{system_prompt}\n"
                        "从完整可见历史中选择与本轮相关的稳定事实、最新事实更新、"
                        "未完成任务、已向用户公开的处理结果和安全历史。"
                        "旧事实被新事实替代时只保留新事实。"
                        "只输出简短中文记忆要点，不输出风险等级或诊断。"
                    ),
                ),
                AiMessage(
                    role="user",
                    content=(
                        f"当前输入：\n{current_input}\n\n"
                        f"完整可见历史（最多 Redis 配置上限）：\n{rendered_history}"
                    ),
                ),
            ]
        ).strip()
        return summary[:max_chars] or fallback
    except Exception:
        return fallback or "无相关历史记忆。"


def bound_model_history(history: list[AiMessage], chat_history_limit: int) -> list[AiMessage]:
    """Bound prompt history while retaining leading internal system context."""
    limit = max(2, int(chat_history_limit) * 2)
    if len(history) <= limit:
        return list(history)
    leading_system: list[AiMessage] = []
    for item in history:
        if item.role != "system":
            break
        leading_system.append(item)
    keep_system = leading_system[: max(0, limit - 1)]
    remaining = limit - len(keep_system)
    return [*keep_system, *history[-remaining:]]


def _artifact_memory_message(artifacts: list[dict[str, Any]]) -> AiMessage | None:
    """Render current production collaboration artifacts as internal context."""
    if not artifacts:
        return None
    privacy = PrivacySanitizer()
    normalized = []
    for artifact in artifacts:
        kind = str(artifact.get("kind") or "artifact")
        payload = artifact.get("payload", {})
        rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        normalized.append(f"- {kind}: {privacy.sanitize(rendered)}")
    return AiMessage(
        role="system",
        content="本轮协作 artifact（内部上下文，不向学生逐字展示）：\n" + "\n".join(normalized),
    )


def compact_history_for_prompt(
    history: list[AiMessage],
    settings: MemoryCompactionSettings,
    current_input: str = "",
    *,
    include_summary_message: bool = True,
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
    if include_summary_message:
        summary_message = AiMessage(
            role="system",
            content=(
                "历史摘要（仅供 MindBridge 内部上下文使用；不要向学生展示；"
                "不要据此输出诊断、风险等级或后台标签）：\n" + brief
            ),
        )
        return [summary_message, *recent], brief

    # 默认 event-driven ResponseAgent 已通过独立的 memoryBrief 字段注入摘要。
    # modelHistory 只保留近期原文，避免同一摘要在最终 prompt 中出现两次。
    return recent, brief


def summarize_history_for_memory(
    history: list[AiMessage],
    current_input: str = "",
    max_chars: int = 500,
) -> str:
    """
    确定性生成历史记忆摘要（不调用 LLM）。

    从完整可见历史中选择稳定事实、最新事实更新、未完成任务、
    已公开处理结果和安全历史。没有可复用长期事实时返回明确的空摘要；
    近期原文由 compact_history_for_prompt 独立保留，不在摘要中重复。
    """
    privacy = PrivacySanitizer()
    sanitized: list[AiMessage] = []
    for message in history:
        content = " ".join(privacy.sanitize(message.content).split())
        if not content:
            continue
        sanitized.append(AiMessage(role=message.role.lower(), content=content))

    selected = _select_memory_facts(
        sanitized,
        current_input,
        max_chars=max(40, max_chars - 20),
    )
    if selected:
        parts = ["相关历史事实："]
        parts.extend(f"- {item.content}" for item in selected)
        return _clip("\n".join(parts), max_chars)

    # 近期原文已经由 compact_history_for_prompt 单独保留。没有可复用长期事实时，
    # 不再把最后几轮无关闲聊复制进摘要，否则既增加 token，也制造虚假相关性。
    return "无可复用的长期事实。"


@dataclass(frozen=True)
class _MemoryFact:
    """One production-visible fact selected from conversation history."""

    index: int
    content: str
    kind: str
    slot: str
    replacement_scope: str | None = None
    asserted_update: bool = True
    source_content: str = ""
    source_unit_count: int = 1


_MEMORY_ACTION_PATTERN_TEXT = (
    r"(?:交|提交|完成|回复|回|发送|发给|发群|发|预约|整理|修改|改|"
    r"修订|修|打开|写|找|填|跟进|上传|处理|确认|联系|准备|"
    r"重做|补充|补|归还|参加|办理)"
)
_MEMORY_LABELED_ACTION_PATTERN = re.compile(
    rf"^[^：:\s]{{1,8}}[：:][^。！？!?]{{0,48}}{_MEMORY_ACTION_PATTERN_TEXT}",
    re.IGNORECASE,
)
_MEMORY_TASK_PATTERN = re.compile(
    r"(?:^|[，,；;。])[^，,；;。！？!?]{0,12}"
    r"(?:计划|打算|准备|需要|得|必须|"
    r"还要|尚未|还没|答应(?:了)?|别忘(?:了)?|记得|让我|要求我)"
    r"[^。！？!?]{0,48}"
    + _MEMORY_ACTION_PATTERN_TEXT,
    re.IGNORECASE,
)
_MEMORY_VISIBLE_OUTCOME_PATTERN = re.compile(
    r"(?:已经|已|目前|当前|正在|尚未|仍在|现已|结果|状态|进度|"
    r"提交|送达|通过|完成|更新|改约|导出|脱敏|发出|发送|排队|"
    r"检索到|等待|等候|没有空位|没有名额|失败|成功|生效|"
    r"建议(?:采用|使用)|后续(?:会|将))",
    re.IGNORECASE,
)
_MEMORY_OUTCOME_OBJECT_PATTERN = re.compile(
    r"(?:申请|预约|通知|邮件|导出|量表|工单|报修|处理|送达|学院|中心|"
    r"老师|辅导员|资源|记录|报告|文件|结果|状态|进度|安排|后续|"
    r"指引|方案|计划|服务|人工跟进)",
    re.IGNORECASE,
)
_MEMORY_STABLE_STATEMENT_PATTERN = re.compile(
    r"(?:"
    r"^(?:我|本人)(?:是|叫|住|来自|读|主修|辅修|喜欢|更喜欢|偏好|"
    r"习惯|怕|害怕|不吃|不能|希望|只想|通常|平时|每天|每周|周末|对)"
    r"|^我的[^，。！？]{1,20}(?:是|为|在)"
    r"|^[^，。！？]{1,20}(?:是|属于)(?:我|本人)"
    r"|^[^，。！？]{1,16}会让我"
    r"|^(?:我|本人)[^，。！？]{1,14}(?:更|比较|容易|适合|擅长|有效|不适)"
    r"|^(?:请|不要|别)(?:用|避免|在|把|给|催|提|说|回复|解释)"
    r"|^(?:饮食上|交流时|学习时|工作时|晚上|每周|周末|平时)"
    r")",
    re.IGNORECASE,
)
_MEMORY_FIRST_PERSON_DECLARATION_PATTERN = re.compile(
    r"^(?:我|本人)(?:[^？?]{2,})$",
    re.IGNORECASE,
)
_MEMORY_UPDATE_PATTERN = re.compile(
    r"(?:改为|改成|改到|变为|变成|延后|延期|推迟|提前|更新为|"
    r"搬到|不再|取消|决定|现在|目前|最近|这周|以后|改约|"
    r"原本|原来|此前|之前|截止|期限|门禁|目标(?:是|为)|"
    r"目标[^，。]{0,8}(?:是|为)|"
    r"超过\s*\d+|少于\s*\d+|每天\s*\d+)",
    re.IGNORECASE,
)
_MEMORY_EXPLICIT_REPLACEMENT_PATTERN = re.compile(
    r"(?:改为|改成|改到|变为|变成|调整为|更新为|延后|延期|推迟|提前|"
    r"搬到|不再|取消|改约|决定(?:改|先|不)|只想|只要|只能)",
    re.IGNORECASE,
)
_MEMORY_SAFETY_PATTERN = re.compile(
    r"(?:不想活|想死|自杀|自残|结束生命|伤害自己|吞药|遗书|做个了断"
    r"|安全计划|保护因素|有人陪|陪(?:着|同)?(?:我|本人)|留在这里对话|先不采取行动"
    r"|交给.{0,10}(?:保管|收走)|危险物品|可能伤害自己的东西"
    r"|(?:正在|还在).{0,10}(?:通话|视频)|紧急援助|可信任的人|当前安全|确认安全)",
    re.IGNORECASE,
)
_MEMORY_SAFETY_GUIDANCE_PATTERN = re.compile(
    r"(?:建议|提醒|要求|请).{0,18}"
    r"(?:联系.{0,10}(?:支持|援助|老师|辅导员|家人|朋友|医院|警方)"
    r"|让.{0,10}(?:陪同|陪着|过来)"
    r"|由.{0,10}(?:陪同|陪着|保管|收走)"
    r"|把.{0,12}(?:放远|移开|交给)"
    r"|离开危险位置|下楼|保持通话|继续陪同)",
    re.IGNORECASE,
)
_MEMORY_PROTECTIVE_FACT_PATTERN = re.compile(
    r"(?:药(?:物|片|盒)?|刀(?:具|片|子)?|绳子|危险物品|"
    r"可能伤害自己的东西).{0,12}(?:交给|交由|已由|已经由).{0,12}"
    r"(?:保管|收走|处理)",
    re.IGNORECASE,
)
_MEMORY_INTERROGATIVE_PATTERN = re.compile(
    r"(?:[？?]$|(?:吗|呢|什么|怎么|为何|为什么|多少|几点|哪里|哪儿|是否)[。！! ]*$)"
)
_MEMORY_SHORT_NON_FACT_PATTERN = re.compile(
    r"^(?:好的?|嗯+|哦+|行|可以|明白(?:了)?|收到|了解|知道了|"
    r"谢谢|多谢|你好|您好|再见|继续(?:说|聊)(?:吧)?|没事|算了)[。！! ]*$",
    re.IGNORECASE,
)
_MEMORY_GENERIC_SUPPORT_PATTERN = re.compile(
    r"^(?:我(?:明白|理解|听见)(?:了)?|我们(?:继续|一起)(?:说|聊|看看)|"
    r"你(?:已经)?(?:做得|应对得|处理得)(?:很好|不错)|"
    r"先(?:休息|缓一缓)|我会陪着你|谢谢你愿意(?:告诉|分享))"
    r"(?:[，,。！!].*)?$",
    re.IGNORECASE,
)
_MEMORY_ACK_PATTERN = re.compile(
    r"^(?:好的|好|明白|收到|了解|知道了|记住了|已记录|我会记住)",
    re.IGNORECASE,
)
_MEMORY_TEMPORAL_TASK_PATTERN = re.compile(
    rf"(?:今晚|明早|明天|后天|周[一二三四五六日天]|"
    rf"\d{{1,2}}[点号]|之前|以前|前).{{0,18}}{_MEMORY_ACTION_PATTERN_TEXT}"
    rf"|{_MEMORY_ACTION_PATTERN_TEXT}.{{0,18}}(?:今晚|明早|明天|后天|"
    rf"周[一二三四五六日天]|\d{{1,2}}[点号]|之前|以前|前)",
    re.IGNORECASE,
)
_MEMORY_EXPLICIT_PROPERTY_PATTERN = re.compile(
    r"^(?:我(?:的)?|本人(?:的)?|原来|原本|之前|此前|现在|目前|最近)?"
    r"(?P<property>[\u3400-\u9fffA-Za-z0-9_]{2,12}?)"
    r"(?:原来|原本|之前|此前|现在|目前|最近)?"
    r"(?:是|为|改为|改成|变为|变成|调整为|更新为|延后至|延期到|推迟到|提前到)",
    re.IGNORECASE,
)
_MEMORY_DEADLINE_OBJECT_PATTERN = re.compile(
    r"(?:要交|提交|完成|发出|交付)(?P<object>[^，。！？]{2,16})"
    r"|(?P<subject>[^，。！？]{2,16}?)(?:原本|原来|之前|此前)?"
    r"(?:在|于)?(?:周[一二三四五六日天]|下?周|今晚|明天|\d{1,2}[月号日])?"
    r"(?:截止|到期|改到|延后|延期|推迟|提前)",
    re.IGNORECASE,
)
_MEMORY_FOOD_REACTION_PATTERN = re.compile(
    r"(?:我|本人)?(?:以前|之前|原来|现在|目前|其实)?"
    r"对(?P<object>[\u3400-\u9fffA-Za-z0-9]{1,12}?)"
    r"(?:不再过敏|不过敏|没有过敏反应|过敏|不耐受)"
    r"|(?P<causal_object>[\u3400-\u9fffA-Za-z0-9]{1,12}?)"
    r"(?:会让我(?:身体)?不舒服|会引起|会导致)"
    r"[^，。！？]{0,6}(?:过敏|不适)?",
    re.IGNORECASE,
)
_MEMORY_FOOD_AVOID_PATTERN = re.compile(
    r"(?:我|本人)?(?:以前|之前|原来|过去|早先|现在|目前|平时)?"
    r"(?:一直|始终|通常|一向)?"
    r"(?:不吃|不能吃|不可以吃|不碰|避免吃|需要避开|避开)"
    r"(?P<object>[\u3400-\u9fffA-Za-z0-9]{1,12}?)(?:了)?"
    r"(?=$|[，。！？、；：,.!?:;])",
    re.IGNORECASE,
)
_MEMORY_FOOD_RECOVERY_PATTERN = re.compile(
    r"(?:我|本人)?(?:现在|目前|后来|如今)?(?:已经)?"
    r"(?<!不)(?:可以|能|能够|恢复)(?:正常|重新)?吃"
    r"(?P<object>[\u3400-\u9fffA-Za-z0-9]{1,12}?)(?:了)?"
    r"(?=$|[，。！？、；：,.!?:;])",
    re.IGNORECASE,
)
_MEMORY_FOOD_ALLERGY_CLEAR_PATTERN = re.compile(
    r"(?:我|本人)?(?:现在|目前|如今)?(?:其实|确实)?"
    r"(?:没有|并没有|不存在)"
    r"(?P<object>[\u3400-\u9fffA-Za-z0-9]{1,12}?)"
    r"(?:过敏|过敏反应)(?:了)?(?=$|[，。！？、；：,.!?:;])",
    re.IGNORECASE,
)
_MEMORY_UNCERTAIN_MODALITY_PATTERN = re.compile(
    r"(?:不确定|无法(?:确认|确定|证实|判断)|"
    r"(?:还|仍)?不能(?:确认|确定|证实|判断)|"
    r"(?:还|仍)?(?:没有|没能)(?:确认|确定|证实|判断)|"
    r"(?:尚未|未能)(?:确认|确定|证实|判断)|"
    r"(?:有待|等待|待)(?:确认|确定|证实|判断)|"
    r"是否|能否|可能|也许|或许|似乎|好像|如果|假如|若(?:是|能)?)",
    re.IGNORECASE,
)
_MEMORY_LATER_DOUBT_PATTERN = re.compile(
    r"(?:但是|不过|可是|然而|但|却)[^。！？]{0,12}"
    r"(?:不确定|不能确认|无法确认|仍需确认)"
    r"(?:[^。！？]{0,8}(?:结果|结论|说法|情况|状态|这件事|这一点))?"
    r"(?=$|[。！？；;])",
    re.IGNORECASE,
)
_MEMORY_PRECEDING_UNCERTAIN_GOVERNOR_PATTERN = re.compile(
    r"(?:不确定|无法(?:确认|确定|证实|判断)|"
    r"(?:还|仍)?不能(?:确认|确定|证实|判断)|"
    r"(?:还|仍)?(?:没有|没能)(?:确认|确定|证实|判断)|"
    r"(?:尚未|未能)(?:确认|确定|证实|判断))"
    r"[，,]\s*$",
    re.IGNORECASE,
)
_MEMORY_LOCAL_BOUNDARY_PATTERN = re.compile(
    r"[，,。！？!?；;\n]+|(?:但是|不过|可是|然而|但|却)"
)
_MEMORY_SUPPORT_SCOPE_PATTERN = re.compile(
    r"(?:只(?:能)?(?:和|找|联系)|不再(?:和|找|联系)|"
    r"(?:现在|目前)(?:也)?可以(?:找|联系)|改为(?:找|联系))",
    re.IGNORECASE,
)
_MEMORY_COMMUNICATION_MEDIUM_PATTERN = re.compile(
    r"(?:文字(?:交流|回复)?|语音(?:交流|回复)?|电话|视频|邮件|短信)",
    re.IGNORECASE,
)
_MEMORY_COMMUNICATION_SCOPE_REPLACEMENT_PATTERN = re.compile(
    r"(?:只想|只要|只能|只用|只接受|只采用|只选择|统一用|全部改用)",
    re.IGNORECASE,
)
_MEMORY_COLLECTION_SCOPE_REPLACEMENT_PATTERN = re.compile(
    r"(?:只考虑|只保留|只做|统一改为|全部改为|全部改做)",
    re.IGNORECASE,
)
_MEMORY_COLLECTION_MEMBER_RETRACTION_PREFIX = re.compile(
    r"(?:不|不再|不想|不做|不要|停止|放弃|取消|"
    r"不打算|不准备|不考虑)\s*$",
    re.IGNORECASE,
)
_MEMORY_COLLECTION_MEMBER_RETRACTION_SUFFIX = re.compile(
    r"^(?:了)?(?:又|后来|随后|现在|目前)?\s*"
    r"(?:改为|改成|换成|换做|改做|改练|转为|转去|改用|"
    r"调整为|调整成|变为|变成)",
    re.IGNORECASE,
)
_MEMORY_CONTEXTUAL_MEMBER_REPLACEMENT_PATTERN = re.compile(
    r"(?:改为|改成|换成|换做|改做|改练|转为|转去|改用|"
    r"调整为|调整成|变为|变成)",
    re.IGNORECASE,
)
_MEMORY_SLEEP_MEMBER_PATTERNS: dict[str, re.Pattern[str]] = {
    "onset": re.compile(r"(?:入睡|睡着|难以睡|很难睡|睡不着)", re.IGNORECASE),
    "early_waking": re.compile(
        r"(?:早醒|过早醒|清晨[^，。]{0,6}醒|天(?:没亮|亮前)[^，。]{0,4}醒)",
        re.IGNORECASE,
    ),
    "maintenance": re.compile(
        r"(?:半夜|夜里|夜间)[^，。]{0,8}(?:醒|睡不稳)|频繁醒",
        re.IGNORECASE,
    ),
    "quality": re.compile(
        r"(?:睡眠(?:质量|状态|[^，。]{0,4}(?:好|差|改善|变好|正常))|"
        r"睡得|失眠)",
        re.IGNORECASE,
    ),
    "schedule": re.compile(r"(?:作息|睡觉时间|起床时间|就寝)", re.IGNORECASE),
}
_MEMORY_EXERCISE_MEMBER_PATTERNS: dict[str, re.Pattern[str]] = {
    "run": re.compile(r"(?:跑步|慢跑)", re.IGNORECASE),
    "walk": re.compile(r"(?:快走|散步|健走)", re.IGNORECASE),
    "swim": re.compile(r"游泳", re.IGNORECASE),
    "cycle": re.compile(r"(?:骑行|骑车)", re.IGNORECASE),
    "yoga": re.compile(r"瑜伽", re.IGNORECASE),
    "strength": re.compile(r"(?:力量训练|健身|举铁)", re.IGNORECASE),
}
_MEMORY_CAREER_MEMBER_PATTERNS: dict[str, re.Pattern[str]] = {
    "graduate_study": re.compile(
        r"(?:考研|研究生考试|申请研究生|读研)",
        re.IGNORECASE,
    ),
    "internship": re.compile(
        r"(?:找[^，。]{0,4}实习|申请[^，。]{0,6}实习|"
        r"投递[^，。]{0,8}实习|实习岗位)",
        re.IGNORECASE,
    ),
    "job_search": re.compile(r"(?:求职|找工作|投递岗位)", re.IGNORECASE),
}
_MEMORY_CAREER_CONTEXTUAL_RETRACTION_PATTERN = re.compile(
    r"(?:不投|不再|不打算|不准备|停止|取消|放弃|决定不|暂时不|先不)"
    r"[^，。！？]{0,8}(?:投递|申请|考虑)?"
    r"(?:这批|这些|相关)?(?:岗位|职位|机会|申请)",
    re.IGNORECASE,
)
_MEMORY_COLLECTION_FACT_SEPARATOR_PATTERN = re.compile(
    r"(?:[，,、；;。]\s*(?:也|还|又|同时|并且|而且)?"
    r"|(?:和|与|以及|及|跟|并且|而且|同时|加上|还有|又)(?:也|会)?"
    r"|(?:还|也)(?:会|要|在)?"
    r"|一边)",
    re.IGNORECASE,
)
_MEMORY_COLLECTION_CONNECTOR_TAIL_PATTERN = re.compile(
    r"(?:会|可以|能|要|在|做|进行|用|采用|申请|准备|选择|打算|继续|保持|"
    r"仍然|依然)?",
    re.IGNORECASE,
)
_MEMORY_ADDITIVE_CORRELATIVE_PREFIX_PATTERN = re.compile(
    r"(?:既|不仅(?:仅)?|不但|不只(?:是)?|不光|一边)",
    re.IGNORECASE,
)
_MEMORY_RESPONSE_LENGTH_PATTERN = re.compile(
    r"(?:简短|精简|详细|展开|回复长度|篇幅|字数|长段落|短段落|分段|"
    r"(?:\d+|[一二三四五六七八九十]+)(?:句|行|段|字))",
    re.IGNORECASE,
)
_MEMORY_RESPONSE_TONE_PATTERN = re.compile(
    r"(?:措辞|语气|诊断式|命令式|温和|直接表达|先给结论)",
    re.IGNORECASE,
)

_MEMORY_KIND_QUERY_PATTERNS: dict[str, re.Pattern[str]] = {
    "stable_user_fact": re.compile(r"(?:偏好|约束|习惯|专业|身份|住|过敏|作息|记得|还有效)"),
    "generic_user_fact": re.compile(r"(?:记得|刚才|之前|提过|关于|相关|情况)"),
    "open_task": re.compile(r"(?:任务|没做完|未完成|还要|需要完成|约定|答应)"),
    "visible_outcome": re.compile(r"(?:结果|状态|进度|处理|提交|预约|申请|确认|送达)"),
    "safety": re.compile(r"(?:安全|保护|风险|冲动|自杀|自残|伤害|药|陪)"),
}

_MEMORY_DOMAIN_PATTERNS: dict[str, re.Pattern[str]] = {
    "education": re.compile(r"(?:专业|主修|辅修|年级|本科|研究生|交换生|课程)"),
    "identity": re.compile(r"(?:我叫|来自|方言|普通话|语言|中英)"),
    "residence": re.compile(r"(?:我住|住所|住址|宿舍|搬到|搬家)"),
    "access_time": re.compile(r"(?:门禁|开放时间|出入时间)"),
    "deadline": re.compile(r"(?:截止|期限|延期|延后|推迟|提前|改到|要交|提交)"),
    "career_decision": re.compile(
        r"(?:考研|研究生考试|申请研究生|读研|找.{0,4}实习|"
        r"申请.{0,6}实习|投递.{0,8}实习|实习岗位|不投|"
        r"求职方向|职业方向|职业安排)"
    ),
    "career_target": re.compile(r"(?:目标公司|目标岗位|意向单位)"),
    "sleep_state": re.compile(
        r"(?:睡眠|失眠|睡得|睡着|入睡|早醒|"
        r"清晨[^，。]{0,6}醒|天(?:没亮|亮前)[^，。]{0,4}醒|"
        r"(?:半夜|夜里|夜间)[^，。]{0,8}醒)"
    ),
    "nap_constraint": re.compile(r"(?:午睡|小睡)"),
    "support_availability": re.compile(
        r"(?:可以找|只能找|只和|支持者|支持资源|支持渠道|求助渠道|"
        r"辅导员|心理中心|咨询中心|心理热线|求助热线|舍友|家人)"
    ),
    "appointment": re.compile(r"(?:预约时间|改约|预约到|预约安排)"),
    "exercise_mode": re.compile(
        r"(?:跑步|慢跑|快走|散步|健走|游泳|骑行|骑车|瑜伽|"
        r"力量训练|健身|举铁|运动方式|锻炼)"
    ),
    "exercise_duration": re.compile(r"(?:每天|每次).{0,8}(?:分钟|小时)"),
    "food_constraint": re.compile(
        r"(?:过敏|不耐受|不吃|不能吃|不碰|吃东西|忌口|身体限制|"
        r"(?:可以|能|恢复)(?:重新)?吃|避开.{0,6}(?:食物|食品)|饮食|食物)"
    ),
    "communication_preference": re.compile(
        r"(?:文字(?:交流|回复)?|语音|电话|视频|邮件|短信|沟通|交流方式|"
        r"沟通方式|交流媒介|回复|回答|措辞|先给结论|简短段落|群里提|"
        r"篇幅|字数|[一二三四五六七八九十\d]+(?:句|行|段))"
    ),
    "task": re.compile(r"(?:任务|报告|邮件|量表|简历|答辩|报修|分工|数据图|文献)"),
    "outcome": re.compile(r"(?:结果|状态|进度|提交|申请|送达|确认|导出|通知)"),
    "safety": _MEMORY_SAFETY_PATTERN,
}


def _select_memory_facts(
    history: list[AiMessage],
    current_input: str,
    *,
    max_facts: int = 8,
    max_chars: int = 430,
) -> list[_MemoryFact]:
    """
    Select durable, task, outcome, and safety facts from production-visible history.

    Candidate extraction is schema-like rather than dataset-specific. The current
    input affects both semantic-domain and lexical relevance; relevance determines
    which facts survive when the summary budget cannot hold every candidate.
    """
    candidates: list[_MemoryFact] = []
    prior_user_contents: list[str] = []
    for index, message in enumerate(history):
        content = message.content
        if message.role == "assistant" and _is_assistant_echo(
            content,
            prior_user_contents,
        ):
            continue

        units = (
            _atomic_memory_fact_units(content)
            if message.role == "user"
            else [content]
        )
        for unit in units:
            kind = _memory_fact_kind(message.role, unit)
            if kind is None:
                continue
            slot = _memory_fact_slot(unit, kind)
            if (
                message.role == "user"
                and kind == "generic_user_fact"
                and slot.startswith("replace:")
            ):
                # Successfully parsed structured user state is durable even when its
                # wording is outside the stable-statement surface grammar.
                kind = "stable_user_fact"
            fact = _MemoryFact(
                index=index,
                content=unit,
                kind=kind,
                slot=slot,
                replacement_scope=_memory_replacement_scope(unit, slot),
                # A split unit inherits source-clause modality. Otherwise
                # ``还不确定是否不再跑步、游泳`` would make the tail look like
                # an asserted change merely because the uncertainty words
                # occurred in the first unit.
                asserted_update=_memory_update_is_asserted(
                    content if len(units) > 1 else unit,
                    slot,
                ),
                source_content=content if len(units) > 1 else "",
                source_unit_count=len(units),
            )
            candidates.append(fact)
        if message.role == "user":
            prior_user_contents.append(content)

    facts = _coalesce_complete_memory_fact_units(
        _resolve_memory_updates(candidates)
    )
    if not facts:
        return []

    current_domains = _memory_domains(current_input)
    current_features = _memory_lexical_features(current_input)
    ranked = sorted(
        facts,
        key=lambda item: (
            _memory_relevance_score(
                item,
                current_input,
                current_domains,
                current_features,
                len(history),
            ),
            item.index,
        ),
        reverse=True,
    )
    selected: list[_MemoryFact] = []
    selected_contents: set[str] = set()
    used_chars = 0
    for fact in ranked:
        normalized_content = _normalized_memory_content(fact.content)
        if normalized_content in selected_contents:
            continue
        if (
            fact.kind == "generic_user_fact"
            and not _generic_fact_is_relevant(
                fact,
                current_domains,
                current_features,
            )
        ):
            continue
        cost = len(fact.content) + 3
        if selected and (
            len(selected) >= max(1, max_facts)
            or used_chars + cost > max(40, max_chars)
        ):
            continue
        selected.append(fact)
        selected_contents.add(normalized_content)
        used_chars += cost
    return selected


def _memory_fact_kind(role: str, content: str) -> str | None:
    if role not in {"user", "assistant"}:
        return None
    if _MEMORY_SHORT_NON_FACT_PATTERN.fullmatch(content):
        return None
    if _MEMORY_INTERROGATIVE_PATTERN.search(content):
        return None

    if role == "assistant":
        if _MEMORY_GENERIC_SUPPORT_PATTERN.search(content):
            return None
        if (
            _MEMORY_SAFETY_GUIDANCE_PATTERN.search(content)
            or _MEMORY_PROTECTIVE_FACT_PATTERN.search(content)
        ):
            return "safety"
        if (
            _MEMORY_VISIBLE_OUTCOME_PATTERN.search(content)
            and _MEMORY_OUTCOME_OBJECT_PATTERN.search(content)
        ):
            return "visible_outcome"
        return None

    # Every substantive user declaration is a candidate. Classification changes
    # ranking/update behavior; it is not an admission gate.
    if (
        _MEMORY_SAFETY_PATTERN.search(content)
        or _MEMORY_PROTECTIVE_FACT_PATTERN.search(content)
    ):
        return "safety"
    if (
        _MEMORY_LABELED_ACTION_PATTERN.search(content)
        or _MEMORY_TASK_PATTERN.search(content)
        or _MEMORY_TEMPORAL_TASK_PATTERN.search(content)
    ):
        return "open_task"
    if (
        _MEMORY_STABLE_STATEMENT_PATTERN.search(content)
        or _MEMORY_FIRST_PERSON_DECLARATION_PATTERN.search(content)
        or _MEMORY_UPDATE_PATTERN.search(content)
    ):
        return "stable_user_fact"
    return "generic_user_fact"


def _memory_fact_slot(content: str, kind: str) -> str:
    """Map facts to narrow replace slots; broad domains never imply conflict."""
    if kind in {"visible_outcome", "safety"}:
        return f"unique:{kind}:{content}"

    food_slot = _food_constraint_slot(content)
    if food_slot:
        return food_slot

    communication_slot = _communication_medium_slot(content)
    if communication_slot:
        return communication_slot
    if _MEMORY_RESPONSE_LENGTH_PATTERN.search(content):
        return "replace:response_length"
    if _MEMORY_RESPONSE_TONE_PATTERN.search(content):
        return "replace:response_tone"
    domains = _memory_domains(content)
    if "deadline" in domains:
        deadline_slot = _deadline_fact_slot(content)
        if deadline_slot:
            return deadline_slot

    # A duration-bound exercise statement describes one replaceable plan.
    # Plain activity membership stays additive below (running and swimming
    # may both be current); an explicit old/new plan can still supersede.
    if "exercise_mode" in domains and "exercise_duration" in domains:
        return "replace:exercise_plan"

    collection_slot = _collection_member_slot(content, domains)
    if collection_slot:
        return collection_slot
    if (
        "career_decision" in domains
        and _MEMORY_CAREER_CONTEXTUAL_RETRACTION_PATTERN.search(content)
    ):
        return "replace:collection:career_decision:contextual_retraction"

    if _MEMORY_SUPPORT_SCOPE_PATTERN.search(content):
        return "replace:support_scope"

    property_match = _MEMORY_EXPLICIT_PROPERTY_PATTERN.search(content)
    if property_match:
        property_name = _normalize_property_name(property_match.group("property"))
        if property_name:
            return f"replace:property:{property_name}"
    for domain in (
        "residence",
        "access_time",
        "career_target",
        "nap_constraint",
        "appointment",
        "exercise_duration",
    ):
        if domain in domains:
            return f"replace:{domain}"

    if kind == "open_task":
        return f"unique:task:{content}"
    return f"unique:{kind}:{content}"


def _memory_replacement_scope(content: str, slot: str) -> str | None:
    """
    Return a collection scope only for an explicit whole-dimension replacement.

    Ordinary availability statements (for example text *and* video communication)
    are additive even when they share a broad domain. Wording such as ``只想`` or
    ``改用`` is different: it explicitly replaces the available medium set.
    """
    if (
        slot.startswith("replace:communication_medium:")
        and _MEMORY_COMMUNICATION_SCOPE_REPLACEMENT_PATTERN.search(content)
    ):
        return "replace:communication_medium:"
    collection_scope = _collection_slot_scope(slot)
    if (
        collection_scope
        and _MEMORY_COLLECTION_SCOPE_REPLACEMENT_PATTERN.search(content)
    ):
        return collection_scope
    return None


def _resolve_memory_updates(candidates: list[_MemoryFact]) -> list[_MemoryFact]:
    """
    Resolve stale facts in chronological order.

    Only facts mapped to the same narrow property/dimension replace each other.
    Weak temporal wording such as ``现在`` or ``最近`` is not replacement evidence;
    compatible facts in one broad domain therefore coexist.
    """
    resolved: list[_MemoryFact] = []
    for fact in candidates:
        if fact.asserted_update and fact.replacement_scope:
            resolved = [
                item
                for item in resolved
                if not item.slot.startswith(fact.replacement_scope)
            ]
        collection_domain = _collection_slot_domain(fact.slot)
        if collection_domain and fact.asserted_update:
            retracted_members = _collection_retracted_members(
                fact.content,
                collection_domain,
            )
            if retracted_members:
                resolved = [
                    item
                    for item in resolved
                    if not (
                        _collection_slot_domain(item.slot) == collection_domain
                        and _collection_slot_members(item.slot)
                        & retracted_members
                    )
                ]
            elif _MEMORY_CONTEXTUAL_MEMBER_REPLACEMENT_PATTERN.search(
                fact.content
            ):
                prior_collection_facts = [
                    item
                    for item in resolved
                    if _collection_slot_domain(item.slot) == collection_domain
                ]
                prior_members = set().union(
                    *(
                        _collection_slot_members(item.slot)
                        for item in prior_collection_facts
                    )
                ) if prior_collection_facts else set()
                # With exactly one prior member, an omitted "from" object is
                # unambiguous. Multiple concurrent members are preserved.
                if len(prior_members) == 1:
                    resolved = [
                        item
                        for item in resolved
                        if _collection_slot_domain(item.slot)
                        != collection_domain
                    ]
        if (
            fact.asserted_update
            and fact.slot
            == "replace:collection:career_decision:contextual_retraction"
        ):
            prior_career_slots = {
                item.slot
                for item in resolved
                if item.slot.startswith("replace:collection:career_decision:")
                and item.slot
                != "replace:collection:career_decision:contextual_retraction"
            }
            # An omitted object may retract the sole active career option, but
            # never guesses among several concurrent plans.
            if len(prior_career_slots) == 1:
                previous_slot = next(iter(prior_career_slots))
                resolved = [
                    item for item in resolved
                    if item.slot != previous_slot
                ]
        if fact.asserted_update and fact.slot == "replace:deadline:contextual":
            prior_deadline_slots = {
                item.slot
                for item in resolved
                if item.slot.startswith("replace:deadline:")
                and item.slot != "replace:deadline:contextual"
            }
            if len(prior_deadline_slots) == 1:
                previous_slot = next(iter(prior_deadline_slots))
                resolved = [
                    item for item in resolved
                    if item.slot != previous_slot
                ]
        if fact.asserted_update and fact.slot.startswith("replace:"):
            resolved = [
                item for item in resolved
                if item.slot != fact.slot
            ]
        resolved.append(fact)
    return resolved


def _atomic_memory_fact_units(content: str) -> list[str]:
    """
    Split additive collection statements into independently updateable facts.

    A source message such as ``我跑步，也会游泳`` contains two compatible
    members. Keeping it as one replaceable object makes a later ``不再跑步``
    remove the still-valid swimming fact, while retaining the whole source text
    would leak the stale running claim. We therefore split only at an explicit
    list/clause separator between recognized collection members. Old/new
    constructions without such a separator (for example ``跑步改成快走``)
    remain one ordered update statement.
    """
    occurrences = _collection_member_occurrences(content)
    distinct_members = {
        (domain, member) for _, _, domain, member in occurrences
    }
    if len(distinct_members) < 2:
        return [content]

    cuts: list[tuple[int, int]] = []
    for left, right in zip(occurrences, occurrences[1:]):
        left_end = left[1]
        right_start = right[0]
        if right_start <= left_end:
            continue
        gap = content[left_end:right_start]
        # ``跑步又改成游泳`` is one ordered replacement, not two compatible
        # collection members. A connector must describe the whole gap: merely
        # finding ``又``/``也`` inside ``受伤后又`` or ``不代表我也`` is not
        # sufficient evidence that both members remain current.
        separator = _additive_collection_separator(gap)
        if separator is None:
            continue
        cuts.append(
            (left_end + separator.start(), left_end + separator.end())
        )
    if not cuts:
        return [content]

    units: list[str] = []
    start = 0
    for cut_start, cut_end in cuts:
        unit = content[start:cut_start].strip(" ，,；;。")
        if unit:
            units.append(unit)
        start = cut_end
    tail = content[start:].strip(" ，,；;。")
    if tail:
        units.append(tail)
    if units and _MEMORY_ADDITIVE_CORRELATIVE_PREFIX_PATTERN.search(
        content[:occurrences[0][0]]
    ):
        # A surviving atomic member must remain grammatical after its paired
        # member is retracted: ``我既跑步`` becomes ``我跑步``. If every member
        # survives, source coalescing below restores the untouched original.
        units[0] = _MEMORY_ADDITIVE_CORRELATIVE_PREFIX_PATTERN.sub(
            "",
            units[0],
            count=1,
        ).strip()
    return units or [content]


def _additive_collection_separator(gap: str) -> re.Match[str] | None:
    """Return a connector only when it positively accounts for the whole gap."""
    if _MEMORY_CONTEXTUAL_MEMBER_REPLACEMENT_PATTERN.search(gap):
        return None
    for separator in reversed(
        list(_MEMORY_COLLECTION_FACT_SEPARATOR_PATTERN.finditer(gap))
    ):
        prefix = gap[:separator.start()].strip()
        suffix = gap[separator.end():].strip()
        if prefix:
            continue
        if not _MEMORY_COLLECTION_CONNECTOR_TAIL_PATTERN.fullmatch(suffix):
            continue
        return separator
    return None


def _collection_member_occurrences(
    content: str,
) -> list[tuple[int, int, str, str]]:
    """Return ordered concrete member spans across additive collection domains."""
    definitions = (
        ("sleep_state", _MEMORY_SLEEP_MEMBER_PATTERNS),
        ("exercise_mode", _MEMORY_EXERCISE_MEMBER_PATTERNS),
        ("career_decision", _MEMORY_CAREER_MEMBER_PATTERNS),
    )
    occurrences: set[tuple[int, int, str, str]] = set()
    for domain, patterns in definitions:
        for member, pattern in patterns.items():
            for match in pattern.finditer(content):
                occurrences.add(
                    (match.start(), match.end(), domain, member)
                )
    for match in _MEMORY_COMMUNICATION_MEDIUM_PATTERN.finditer(content):
        value = match.group(0)
        if value.startswith("文字"):
            member = "text"
        elif value.startswith("语音"):
            member = "voice"
        else:
            member = value
        occurrences.add(
            (
                match.start(),
                match.end(),
                "communication_medium",
                member,
            )
        )
    return sorted(occurrences)


def _memory_update_is_asserted(content: str, slot: str) -> bool:
    """
    Return whether this fact may mutate previously confirmed memory state.

    Uncertain language remains useful context and is still summarized, but it
    cannot retract or replace an asserted collection/property value. This
    applies to both targeted member changes and whole-set wording such as
    ``还不确定是否只改用视频``.
    """
    if not slot.startswith("replace:"):
        return True
    if _MEMORY_UNCERTAIN_MODALITY_PATTERN.search(content):
        return False
    return not bool(_MEMORY_LATER_DOUBT_PATTERN.search(content))


def _coalesce_complete_memory_fact_units(
    facts: list[_MemoryFact],
) -> list[_MemoryFact]:
    """
    Restore original wording only when every atomic member is still current.

    This keeps compound updates such as ``不再考研，先找实习`` readable and
    compatible with exact fact auditing. If a later event retracts only one
    member of ``跑步，也会游泳``, the source group is incomplete and only the
    safe surviving atomic span is rendered.
    """
    groups: dict[tuple[int, str], list[_MemoryFact]] = {}
    for fact in facts:
        if fact.source_content and fact.source_unit_count > 1:
            groups.setdefault(
                (fact.index, fact.source_content),
                [],
            ).append(fact)

    coalesced: list[_MemoryFact] = []
    emitted: set[tuple[int, str]] = set()
    kind_priority = {
        "safety": 4,
        "open_task": 3,
        "stable_user_fact": 2,
        "generic_user_fact": 1,
    }
    for fact in facts:
        if not fact.source_content or fact.source_unit_count <= 1:
            coalesced.append(fact)
            continue
        key = (fact.index, fact.source_content)
        group = groups[key]
        if len(group) != fact.source_unit_count:
            coalesced.append(fact)
            continue
        if key in emitted:
            continue
        emitted.add(key)
        representative = max(
            group,
            key=lambda item: kind_priority.get(item.kind, 0),
        )
        coalesced.append(
            _MemoryFact(
                index=fact.index,
                content=fact.source_content,
                kind=representative.kind,
                slot=f"unique:coalesced:{fact.index}",
            )
        )
    return coalesced


def _collection_member_slot(content: str, domains: set[str]) -> str | None:
    """Map multi-valued domains to concrete members rather than one broad slot."""
    definitions = (
        ("sleep_state", _MEMORY_SLEEP_MEMBER_PATTERNS),
        ("exercise_mode", _MEMORY_EXERCISE_MEMBER_PATTERNS),
        ("career_decision", _MEMORY_CAREER_MEMBER_PATTERNS),
    )
    for domain, patterns in definitions:
        if domain not in domains:
            continue
        members = sorted(
            name for name, pattern in patterns.items()
            if pattern.search(content)
        )
        if members:
            return f"replace:collection:{domain}:{'+'.join(members)}"
    return None


def _collection_slot_scope(slot: str) -> str | None:
    domain = _collection_slot_domain(slot)
    if not domain:
        return None
    return f"replace:collection:{domain}:"


def _collection_slot_domain(slot: str) -> str | None:
    match = re.match(r"^replace:collection:([^:]+):", slot)
    return match.group(1) if match else None


def _collection_slot_members(slot: str) -> set[str]:
    match = re.match(r"^replace:collection:[^:]+:(.+)$", slot)
    if not match:
        return set()
    member_text = match.group(1)
    if member_text == "contextual_retraction":
        return set()
    return set(member_text.split("+"))


def _collection_retracted_members(content: str, domain: str) -> set[str]:
    """Return only collection members explicitly negated or replaced in text."""
    definitions = {
        "sleep_state": _MEMORY_SLEEP_MEMBER_PATTERNS,
        "exercise_mode": _MEMORY_EXERCISE_MEMBER_PATTERNS,
        "career_decision": _MEMORY_CAREER_MEMBER_PATTERNS,
    }
    patterns = definitions.get(domain, {})
    retracted: set[str] = set()
    for member, pattern in patterns.items():
        for match in pattern.finditer(content):
            if _memory_match_is_uncertain(content, match):
                continue
            prefix = content[max(0, match.start() - 10):match.start()]
            suffix = content[match.end():match.end() + 10]
            if (
                _MEMORY_COLLECTION_MEMBER_RETRACTION_PREFIX.search(prefix)
                or _MEMORY_COLLECTION_MEMBER_RETRACTION_SUFFIX.search(suffix)
            ):
                retracted.add(member)
                break
    return retracted


def _food_constraint_slot(content: str) -> str | None:
    """Keep compatible dietary dimensions while allowing same-object corrections."""
    reaction = _MEMORY_FOOD_REACTION_PATTERN.search(content)
    if reaction and not _memory_match_is_uncertain(content, reaction):
        item = _normalize_property_name(
            reaction.group("object") or reaction.group("causal_object")
        )
        if item:
            return f"replace:food_reaction:{item}"
    clear = _MEMORY_FOOD_ALLERGY_CLEAR_PATTERN.search(content)
    if clear and not _memory_match_is_uncertain(content, clear):
        item = _normalize_property_name(clear.group("object"))
        if item:
            return f"replace:food_reaction:{item}"
    recovery = _MEMORY_FOOD_RECOVERY_PATTERN.search(content)
    if recovery and not _memory_match_is_uncertain(content, recovery):
        item = _normalize_property_name(recovery.group("object"))
        if item:
            return f"replace:food_consumption:{item}"
    avoid = _MEMORY_FOOD_AVOID_PATTERN.search(content)
    if avoid:
        item = _normalize_property_name(avoid.group("object"))
        if item:
            return f"replace:food_consumption:{item}"
    return None


def _memory_match_is_uncertain(content: str, match: re.Match[str]) -> bool:
    """Allow stale replacement only for a clause-local asserted state."""
    if _MEMORY_PRECEDING_UNCERTAIN_GOVERNOR_PATTERN.search(
        content[:match.start()]
    ):
        return True
    clause_start = 0
    clause_end = len(content)
    for boundary in _MEMORY_LOCAL_BOUNDARY_PATTERN.finditer(content):
        if boundary.end() <= match.start():
            clause_start = boundary.end()
            continue
        if boundary.start() >= match.end():
            clause_end = boundary.start()
            break
    clause = content[clause_start:clause_end]
    if _MEMORY_UNCERTAIN_MODALITY_PATTERN.search(clause):
        return True
    # A later explicit doubt about the just-stated result conservatively keeps
    # the previous safety constraint active.
    return bool(_MEMORY_LATER_DOUBT_PATTERN.search(content[match.end():]))


def _communication_medium_slot(content: str) -> str | None:
    """Identify the concrete medium member instead of replacing the whole domain."""
    media: set[str] = set()
    for match in _MEMORY_COMMUNICATION_MEDIUM_PATTERN.finditer(content):
        value = match.group(0)
        if value.startswith("文字"):
            media.add("text")
        elif value.startswith("语音"):
            media.add("voice")
        elif value in {"电话", "视频", "邮件", "短信"}:
            media.add(value)
    if not media:
        return None
    return f"replace:communication_medium:{'+'.join(sorted(media))}"


def _deadline_fact_slot(content: str) -> str | None:
    """
    Bind deadlines to their task when the task is stated.

    A task-less explicit reschedule may update the sole preceding deadline, but
    cannot safely erase several unrelated tasks. Such contextual reconciliation
    is handled in ``_resolve_memory_updates``.
    """
    match = _MEMORY_DEADLINE_OBJECT_PATTERN.search(content)
    if match:
        raw = match.group("object") or match.group("subject") or ""
        task = _normalize_deadline_subject(raw)
        if task:
            return f"replace:deadline:{task}"
    if _MEMORY_EXPLICIT_REPLACEMENT_PATTERN.search(content):
        return "replace:deadline:contextual"
    return None


def _normalize_deadline_subject(value: str) -> str:
    normalized = re.sub(
        r"^(?:我|本人|这个|这份|下周|本周|今晚|明天|原来|原本|之前|此前)",
        "",
        value,
    )
    normalized = re.sub(r"(?:报告|项目|作业|申请|材料|任务)$", "", normalized)
    normalized = normalized.strip(" 的：:,，。")
    if normalized in {"截止", "截止日期", "截止时间", "期限", "日期", "时间"}:
        return ""
    return normalized


def _normalize_property_name(value: str) -> str:
    normalized = re.sub(
        r"^(?:我|本人|我的|本人的|原来|原本|之前|此前|现在|目前|最近)",
        "",
        value,
    )
    return normalized.strip(" 的：:,，。")


def _normalized_memory_content(content: str) -> str:
    return re.sub(r"[\s，。！？、；：,.!?:;]+", "", content).lower()


def _generic_fact_is_relevant(
    fact: _MemoryFact,
    current_domains: set[str],
    current_features: set[str],
) -> bool:
    """Keep untyped statements only when the current request makes them useful."""
    fact_domains = _memory_domains(fact.content)
    if fact_domains & current_domains:
        return True
    fact_features = _memory_lexical_features(fact.content)
    return bool(fact_features & current_features)


def _is_assistant_echo(content: str, prior_user_contents: list[str]) -> bool:
    """Drop acknowledgements that merely repeat a user fact verbatim."""
    if not prior_user_contents:
        return False
    recent = prior_user_contents[-4:]
    for user_content in recent:
        normalized = user_content.strip("。！？!? ")
        if len(normalized) >= 4 and normalized in content:
            return True
    return bool(_MEMORY_ACK_PATTERN.search(content) and len(content) <= 24)


def _memory_domains(text: str) -> set[str]:
    return {
        name
        for name, pattern in _MEMORY_DOMAIN_PATTERNS.items()
        if pattern.search(text or "")
    }


def _memory_lexical_features(text: str) -> set[str]:
    """Create deterministic word/bigram features without an external tokenizer."""
    normalized = re.sub(
        r"(?:我|你|他|她|的|了|是|还|仍|现在|目前|最近|一下|什么|怎么|"
        r"有点|让我|继续|请|吗|呢|吧|\s|[，。！？、；：,.!?:;])",
        "",
        (text or "").lower(),
    )
    features = set(re.findall(r"[a-z0-9_]{2,}", normalized))
    cjk_runs = re.findall(r"[\u3400-\u9fff]{2,}", normalized)
    for run in cjk_runs:
        features.update(run[index:index + 2] for index in range(len(run) - 1))
    return features


def _memory_relevance_score(
    fact: _MemoryFact,
    current_input: str,
    current_domains: set[str],
    current_features: set[str],
    history_size: int,
) -> float:
    base = {
        "safety": 80.0,
        "open_task": 62.0,
        "visible_outcome": 58.0,
        "stable_user_fact": 50.0,
        "generic_user_fact": 30.0,
    }[fact.kind]
    fact_domains = _memory_domains(fact.content)
    domain_overlap = len(fact_domains & current_domains)
    fact_features = _memory_lexical_features(fact.content)
    lexical_overlap = len(fact_features & current_features)
    kind_query = _MEMORY_KIND_QUERY_PATTERNS[fact.kind]
    kind_relevance = 18.0 if kind_query.search(current_input or "") else 0.0
    recency = 4.0 * (fact.index + 1) / max(1, history_size)
    return base + domain_overlap * 22.0 + min(lexical_overlap, 6) * 5.0 + kind_relevance + recency


def _clip(text: str, limit: int) -> str:
    """将文本截断到指定长度，超出部分用 ... 代替。"""
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)] + "..."
