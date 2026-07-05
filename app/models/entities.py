"""
MindBridge 数据库 ORM 实体定义

定义所有 SQLAlchemy ORM 模型，映射到 MySQL 数据库表。

实体关系概览：
- UserAccount (1) ──→ (N) ChatSession ──→ (N) ChatMessage
- PsychologicalReport (1) ──→ (0..1) RiskCase ──→ (N) CaseNote
- PsychologicalReport (1) ──→ (N) AlertRecord / ExcelRecord / ToolJob
- ToolJob 失败超限 ──→ DeadLetterRecord（死信记录）
- KnowledgeChunk: 知识库切块，支持 embedding_json 缓存
- AgentRunTrace: 每轮 Agent 运行的完整执行轨迹（用于审计和调试）
- ToolAuditRecord: 工具治理审计记录（ToolGovernanceService 写入）

数据流：
1. 学生发送消息 → ChatMessage 写入
2. Agent 处理后生成 PsychologicalReport（心理评估报告）
3. 高风险报告触发 ToolJob 队列：Excel 台账 → 个案创建 → 预警发送
4. 失败任务重试，超限进入 DeadLetterRecord
5. 每轮 Agent 运行落库一条 AgentRunTrace，供后台审计和排查
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


def now() -> datetime:
    """返回当前 UTC 时间，作为默认时间戳。"""
    return datetime.utcnow()


class UserAccount(Base):
    """
    用户账号表。

    存储学生和管理员的基本信息。
    roles 通过 CSV 字符串存储（如 "ROLE_ADMIN,ROLE_USER"），
    通过 Python property 提供 list 接口。
    """
    __tablename__ = "user_accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(128))
    password_hash: Mapped[str] = mapped_column(String(128))
    roles_csv: Mapped[str] = mapped_column(String(256), default="ROLE_USER")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)

    sessions: Mapped[list["ChatSession"]] = relationship(back_populates="user")

    @property
    def roles(self) -> list[str]:
        """将 CSV 格式的角色字符串解析为列表。"""
        return [role for role in self.roles_csv.split(",") if role]

    @roles.setter
    def roles(self, value: list[str] | set[str]) -> None:
        """将角色列表/集合序列化为 CSV 字符串存储。"""
        self.roles_csv = ",".join(sorted(value))


class ChatSession(Base):
    """
    聊天会话表。

    每次新建对话创建一个会话，public_id 用于前端标识。
    touch() 方法更新 updated_at 时间戳。
    """
    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    public_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(160))
    user_id: Mapped[int] = mapped_column(ForeignKey("user_accounts.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now)

    user: Mapped[UserAccount] = relationship(back_populates="sessions")
    messages: Mapped[list["ChatMessage"]] = relationship(back_populates="session", cascade="all, delete-orphan")

    def touch(self) -> None:
        """更新会话的最后活动时间。"""
        self.updated_at = now()


class ChatMessage(Base):
    """
    聊天消息表。

    存储每条用户和助手的消息。
    完整消息写入 MySQL，短期上下文同时写入 Redis。
    """
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("user_accounts.id"))
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id"))
    role: Mapped[str] = mapped_column(String(32))    # USER / ASSISTANT / SYSTEM
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)

    session: Mapped[ChatSession] = relationship(back_populates="messages")


class KnowledgeChunk(Base):
    """
    知识库切块表。

    存储从 Markdown/PDF 文档切分后的知识片段。
    embedding_json 字段缓存 OpenAI embedding 向量（JSON 数组），
    避免重复调用 embedding API。

    source: 来源文件名（如 "anxiety-panic-grounding.md"）
    source_index: 在同一来源中的切块序号
    content: 切块文本内容
    embedding_json: 缓存的 embedding 向量（可为 None，首次检索时补建）
    """
    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(256), index=True)
    source_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    embedding_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class PsychologicalReport(Base):
    """
    心理评估报告表。

    当对话意图为 CONSULT 或 RISK 时生成。
    记录情绪标签、情绪分数、风险等级、置信度和评估摘要。
    学生端不展示这些后台评估结果。

    字段说明：
    - intent: CHAT/CONSULT/RISK
    - emotion: NORMAL/ANXIETY/DEPRESSED/HIGH_RISK
    - emotion_score: 情绪分数（0.0-4.0）
    - risk_level: LOW/MEDIUM/HIGH
    - confidence: 评估置信度（0.0-1.0）
    - summary: 评估摘要文本
    """
    __tablename__ = "psychological_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("user_accounts.id"))
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id"))
    content: Mapped[str] = mapped_column(Text)
    intent: Mapped[str] = mapped_column(String(32))
    emotion: Mapped[str] = mapped_column(String(32))
    emotion_score: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[float] = mapped_column(Float)
    summary: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class RiskCase(Base):
    """
    风险个案表。

    当心理报告为 MEDIUM/HIGH 风险时创建。
    状态流转：OPEN → ALERT_SENT → ACKNOWLEDGED

    handoff_summary: 由 counselor_handoff_summary skill 生成的个案交接摘要
    acknowledged_by: 确认接手的辅导员/管理员用户名
    """
    __tablename__ = "risk_cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    report_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    risk_level: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    owner: Mapped[str] = mapped_column(String(128), default="unassigned")
    summary: Mapped[str] = mapped_column(Text)
    handoff_summary: Mapped[str] = mapped_column(Text, default="")
    acknowledged_by: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class CaseNote(Base):
    """
    个案备注表。

    辅导员/管理员对风险个案的跟进记录。
    """
    __tablename__ = "case_notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_id: Mapped[int] = mapped_column(Integer, index=True)
    actor: Mapped[str] = mapped_column(String(128))
    note: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class AlertRecord(Base):
    """
    预警记录表。

    记录每次预警通知的发送结果。
    channel: 通知渠道（"email"）
    recipient: 收件人
    status: SUCCESS/FAILED
    message: 详细信息（成功原因或失败原因）
    """
    __tablename__ = "alert_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    report_id: Mapped[int] = mapped_column(Integer, index=True)
    channel: Mapped[str] = mapped_column(String(64))
    recipient: Mapped[str] = mapped_column(String(256))
    status: Mapped[str] = mapped_column(String(32))
    message: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class ExcelRecord(Base):
    """
    Excel 台账写入记录表。

    记录每次 Excel 台账写入的结果，用于幂等检查。
    同一 report_id 只写入一次。
    """
    __tablename__ = "excel_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    report_id: Mapped[int] = mapped_column(Integer, index=True)
    file_path: Mapped[str] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(32))
    message: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class ToolJob(Base):
    """
    异步工具任务表。

    心理报告生成后，后处理任务（Excel写入、个案创建、预警发送）
    不阻塞学生端回复，而是写入此队列表异步执行。

    任务依赖链：EXCEL_REPORT（独立）→ CASE_CREATE（独立）→ ALERT_SEND（依赖 CASE_CREATE）
    depends_on_job_id: 依赖的前置任务 ID（ALERT_SEND 依赖 CASE_CREATE）

    生命周期：PENDING → RUNNING → SUCCESS / DEAD
    重试策略：失败后按 attempts * retry_delay 延迟重试
    超过 max_attempts 后进入 dead_letter_records。
    """
    __tablename__ = "tool_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    report_id: Mapped[int] = mapped_column(Integer, index=True)
    kind: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    depends_on_job_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    run_after: Mapped[datetime] = mapped_column(DateTime, default=now, index=True)
    last_error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class DeadLetterRecord(Base):
    """
    死信记录表。

    当 ToolJob 超过最大重试次数后，将任务信息转入此表。
    用于人工排查和问题诊断。

    job_id: 原始任务 ID（可能已被标记为 DEAD）
    payload: 任务上下文的 JSON 快照
    """
    __tablename__ = "dead_letter_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    report_id: Mapped[int] = mapped_column(Integer, index=True)
    kind: Mapped[str] = mapped_column(String(64), index=True)
    reason: Mapped[str] = mapped_column(Text)
    payload: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class AgentRunTrace(Base):
    """
    Agent 运行轨迹表。

    每轮 Agent 运行结束后落库一条完整记录，供管理员审计和问题排查。
    与 PsychologicalReport 不同：trace 记录的是执行过程（原始输入、
    脱敏输入、记忆摘要、Agent 步骤、检索结果、发送给 LLM 的消息），
    而不是评估结论。

    agent_steps_json / retrieved_knowledge_json / response_messages_json /
    assessment_json 均为 JSON 序列化字符串，查询时反序列化为结构化数据。
    """
    __tablename__ = "agent_run_traces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("user_accounts.id"), index=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id"), index=True)
    report_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    intent: Mapped[str] = mapped_column(String(32), index=True)
    risk_level: Mapped[str] = mapped_column(String(32), default="LOW", index=True)
    original_input: Mapped[str] = mapped_column(Text)
    sanitized_input: Mapped[str] = mapped_column(Text)
    memory_brief: Mapped[str] = mapped_column(Text, default="")
    agent_steps_json: Mapped[str] = mapped_column(Text, default="[]")
    retrieved_knowledge_json: Mapped[str] = mapped_column(Text, default="[]")
    response_messages_json: Mapped[str] = mapped_column(Text, default="[]")
    assessment_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class ToolAuditRecord(Base):
    """
    工具治理审计记录表。

    由 ToolGovernanceService 在工具执行前写入，记录该次调用是否被
    ToolPolicyRegistry 允许、依据的策略和拒绝原因，用于事后审计。

    allowed=False 表示该次工具调用被策略拦截（如风险等级不匹配）。
    """
    __tablename__ = "tool_audit_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    report_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    tool_name: Mapped[str] = mapped_column(String(64), index=True)
    policy: Mapped[str] = mapped_column(String(128), default="")
    allowed: Mapped[bool] = mapped_column(Boolean, default=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    reason: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now)
