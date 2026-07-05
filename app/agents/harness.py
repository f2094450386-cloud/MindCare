"""
MindBridge Agent 运行时 Harness 模块

Harness 是单轮 Agent 运行的外层编排器，负责 HTTP/SSE 层和 Agent runtime 之间的桥接。

职责：
1. 输入准备：隐私脱敏、会话解析
2. Agent 调用：委托给 AgentRuntimeService（LangGraph 或自研）
3. 持久化：用户消息、助手消息、心理报告写入数据库
4. 记忆管理：消息写入 Redis 短期记忆
5. 工具规划：生成 AgentToolPlan 供后续异步处理
6. Trace 输出：返回 AgentHarnessOutcome 包含完整的执行信息

数据流：
  ChatRequest → harness.run() → AgentHarnessOutcome
                                      ↓
                              ChatService.stream_chat()
                                      ↓
                              SSE 流式输出 + dispatch_tools()
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.agents.factory import create_agent_runtime
from app.agents.runtime import AgentStep
from app.core.config import Settings
from app.core.enums import IntentType, MessageRole
from app.models.entities import ChatMessage, ChatSession, PsychologicalReport, UserAccount
from app.schemas.dtos import AiMessage, ChatRequest
from app.services.assessment import PsychologyAssessment
from app.services.knowledge import SearchResult
from app.services.mcp_client import MindBridgeMcpToolClient
from app.services.memory import RedisShortTermMemoryStore
from app.services.privacy import PrivacySanitizer
from app.services.tool_queue import ToolQueueService
from app.services.trace import AgentTraceService


@dataclass
class AgentToolPlan:
    """
    工具执行计划。

    包含 report_id 和 risk_level，供 ChatService 决定触发哪些后处理工具。
    requires_tools 为 True 时表示需要执行工具后处理。
    """
    report_id: int | None
    risk_level: str | None

    @property
    def requires_tools(self) -> bool:
        return self.report_id is not None


@dataclass
class AgentHarnessOutcome:
    """
    Agent 运行结果。

    包含一轮 Agent 运行的完整信息：
    - session: 聊天会话
    - original_input / model_input: 原始输入和脱敏输入
    - intent: 意图类型
    - risk_level: 风险等级
    - assessment: 心理评估结果
    - response_messages: 发送给 LLM 的消息列表
    - agent_steps: Agent 执行步骤（用于 trace）
    - retrieved_knowledge: RAG 检索结果
    - report_id: 心理报告 ID（如有）
    - tool_plan: 工具执行计划
    - trace_id: 本轮运行落库的 AgentRunTrace ID
    """
    session: ChatSession
    original_input: str
    model_input: str
    intent: IntentType
    risk_level: str | None
    assessment: PsychologyAssessment | None
    response_messages: list[AiMessage]
    agent_steps: list[AgentStep]
    retrieved_knowledge: list[SearchResult]
    report_id: int | None
    tool_plan: AgentToolPlan
    trace_id: int | None


class MindBridgeAgentHarness:
    """
    Agent 运行时 Harness。

    外层编排器，管理一轮 Agent 运行的完整生命周期。
    HTTP/SSE 层只负责认证和流式输出，所有业务逻辑集中在 harness 内。
    """

    def __init__(self, db: Session, settings: Settings):
        self.db = db
        self.settings = settings
        self.privacy = PrivacySanitizer()
        self.memory = RedisShortTermMemoryStore(settings)

    def run(self, user: UserAccount, request: ChatRequest) -> AgentHarnessOutcome:
        """
        执行一轮 Agent 运行。

        流程：
        1. 输入脱敏
        2. 会话解析（复用或新建）
        3. 委托 Agent runtime 执行多 Agent 工作流
        4. 保存用户消息
        5. 创建心理报告（如有需要）
        6. 落库 AgentRunTrace（审计用）
        7. 生成工具计划
        """
        original_input = request.message.strip()
        model_input = self.privacy.sanitize(original_input)
        session = self._resolve_session(user, request.sessionId, original_input)

        # 委托给 Agent runtime 执行
        agent_run = create_agent_runtime(self.db, self.settings).run(user, session, original_input, model_input)

        # 保存用户消息
        self.save_message(user, session, MessageRole.USER, original_input)

        # 创建心理报告
        report = self._create_report(user, session, original_input, agent_run)
        risk_level = report.risk_level if report is not None else None

        # 落库本轮运行的完整轨迹，供管理员审计和问题排查
        trace = AgentTraceService(self.db).save_run(
            user=user,
            session=session,
            original_input=original_input,
            sanitized_input=model_input,
            memory_brief=agent_run.memory_brief,
            agent_run=agent_run,
            report_id=report.id if report is not None else None,
        )

        tool_plan = AgentToolPlan(report_id=report.id if report is not None else None, risk_level=risk_level)

        return AgentHarnessOutcome(
            session=session,
            original_input=original_input,
            model_input=model_input,
            intent=agent_run.intent,
            risk_level=risk_level,
            assessment=agent_run.assessment,
            response_messages=agent_run.response_messages,
            agent_steps=agent_run.steps,
            retrieved_knowledge=agent_run.retrieved_knowledge,
            report_id=report.id if report is not None else None,
            tool_plan=tool_plan,
            trace_id=trace.id,
        )

    def save_assistant_message(self, user: UserAccount, session: ChatSession, content: str) -> None:
        """保存助手消息到数据库和 Redis。"""
        self.save_message(user, session, MessageRole.ASSISTANT, content)

    async def dispatch_tools(self, tool_plan: AgentToolPlan) -> list[str]:
        """
        触发工具后处理。

        根据配置选择执行方式：
        - tool_queue_enabled=true: 写入 tool_jobs 队列异步执行（默认）
        - tool_queue_enabled=false: 通过 MCP client 同步调用（备用方案）
        """
        if tool_plan.report_id is None:
            return []
        if self.settings.tool_queue_enabled:
            ToolQueueService(self.db, self.settings).enqueue_report(tool_plan.report_id, tool_plan.risk_level)
            return ["queued"]
        return await MindBridgeMcpToolClient(self.settings).handle_report(tool_plan.report_id, tool_plan.risk_level)

    def save_message(self, user: UserAccount, session: ChatSession, role: MessageRole, content: str) -> None:
        """保存消息到数据库和 Redis 短期记忆。"""
        self.db.add(ChatMessage(user_id=user.id, session_id=session.id, role=role.value, content=content))
        session.touch()
        self.db.add(session)
        self.db.commit()
        self.memory.append(session.public_id, role.value, content)

    def _resolve_session(self, user: UserAccount, public_id: str | None, text: str) -> ChatSession:
        """
        解析会话。

        如果提供了 sessionId 且存在，复用该会话。
        否则创建新会话（标题取输入文本前 36 字符）。
        """
        if public_id:
            session = self.db.query(ChatSession).filter(ChatSession.public_id == public_id, ChatSession.user_id == user.id).first()
            if session is None:
                raise ValueError("Session not found")
            return session
        session = ChatSession(public_id=uuid.uuid4().hex, user_id=user.id, title=text[:36])
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session

    def _create_report(self, user: UserAccount, session: ChatSession, text: str, agent_run) -> PsychologicalReport | None:
        """
        创建心理评估报告。

        仅 CONSULT/RISK 意图创建报告。
        报告内容包含意图、情绪、风险等级、置信度和摘要。
        """
        if not agent_run.requires_report or agent_run.assessment is None:
            return None
        report = PsychologicalReport(
            user_id=user.id,
            session_id=session.id,
            content=text,
            intent=agent_run.intent.value,
            emotion=agent_run.assessment.emotion.value,
            emotion_score=agent_run.assessment.emotion_score,
            risk_level=agent_run.assessment.risk.value,
            confidence=agent_run.assessment.confidence,
            summary=agent_run.assessment.summary,
        )
        self.db.add(report)
        self.db.commit()
        self.db.refresh(report)
        return report
