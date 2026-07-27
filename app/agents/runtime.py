"""
MindBridge 自研 Agent 运行时模块

实现有限循环的多 Agent 协作 runtime，作为 LangGraph 的兜底方案。

Agent 执行顺序（每轮对话固定）：
1. MemoryAgent → 加载短期记忆（Redis/MySQL）
2. SupervisorAgent → 意图分类（CHAT/CONSULT/RISK）
3. KnowledgeAgent → RAG 知识检索（仅 CONSULT/RISK）
4. RiskGuardianAgent → 心理风险评估（仅 CONSULT/RISK）
5. CompanionAgent → 普通陪伴回复（仅 CHAT）
6. CounselorAgent → 心理咨询回复（仅 CONSULT/RISK）

执行机制：
- max_steps=8 防止无限循环
- 每个 Agent 检查前置条件，条件不满足时跳过
- Agent 返回 True 表示已处理，False 表示跳过
- context.finished 标志位控制提前退出

意图路由逻辑：
- 高风险关键词 → 直接 RISK（跳过 LLM 分类）
- 通用任务关键词 → 直接 CHAT
- 其他 → LLM 分类
- 分类失败 → 关键词兜底

数据流：
- AgentContext 在所有 Agent 之间共享状态
- AgentRunResult 返回给 harness 做后续处理
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.enums import IntentType, MessageRole, RiskLevel
from app.models.entities import ChatMessage, ChatSession, UserAccount
from app.schemas.dtos import AiMessage
from app.services.ai import AiClient, PromptTemplates, has_consult_signal, has_high_risk_signal
from app.services.assessment import PsychologicalAssessmentService, PsychologyAssessment
from app.services.knowledge import KnowledgeService, SearchResult
from app.services.memory import RedisShortTermMemoryStore, compact_history_for_prompt
from app.services.skills import MindBridgeSkillLibrary


# 通用任务关键词列表：匹配到这些词时直接判定为 CHAT 意图
GENERAL_TASK_WORDS = [
    "java", "python", "javascript", "代码", "编程", "程序", "算法", "数据库", "spring", "maven",
    "前端", "后端", "项目", "接口", "bug", "报错", "作业", "论文", "翻译", "总结", "解释",
    "怎么写", "如何", "是什么", "为什么", "给我", "帮我", "推荐", "查询", "天气", "路线",
]


@dataclass
class AgentStep:
    """Agent 执行步骤记录（用于 trace 和调试）。"""
    step: int          # 步骤序号
    agent: str         # Agent 名称
    action: str        # 执行动作
    observation: str   # 执行结果描述


@dataclass
class AgentContext:
    """
    Agent 共享上下文。

    在所有 Agent 之间传递状态。
    每个 Agent 读取前置标志位，处理后设置自己的标志位。
    """
    user: UserAccount
    session: ChatSession
    original_input: str     # 用户原始输入（可能包含敏感信息）
    model_input: str        # 脱敏后的输入（传给 LLM）

    # Agent 执行标志位
    memory_loaded: bool = False
    intent_routed: bool = False
    knowledge_handled: bool = False
    risk_assessed: bool = False
    response_planned: bool = False
    finished: bool = False

    # Agent 处理结果
    memory_brief: str = "无相关历史记忆。"
    intent: IntentType | None = None
    risk_level: RiskLevel = RiskLevel.LOW
    assessment: PsychologyAssessment | None = None
    knowledge_query: str = ""
    retrieved_knowledge: list[SearchResult] = field(default_factory=list)
    model_history: list[AiMessage] = field(default_factory=list)
    response_messages: list[AiMessage] = field(default_factory=list)
    response_agent: str = ""
    response_plan: str = ""
    steps: list[AgentStep] = field(default_factory=list)


@dataclass
class AgentRunResult:
    """Agent 运行结果。"""
    intent: IntentType
    risk_level: RiskLevel
    assessment: PsychologyAssessment | None
    retrieved_knowledge: list[SearchResult]
    response_messages: list[AiMessage]
    steps: list[AgentStep]
    memory_brief: str    # 记忆摘要，供 harness 落库 AgentRunTrace 使用
    collaboration_events: list[Any] = field(default_factory=list)
    collaboration_tasks: list[Any] = field(default_factory=list)
    collaboration_artifacts: list[Any] = field(default_factory=list)

    @property
    def requires_report(self) -> bool:
        """是否需要生成心理评估报告（CHAT 意图不需要）。"""
        return self.intent != IntentType.CHAT


class AgentRuntimeService:
    """
    自研 Agent 运行时服务。

    使用有限循环 + 标志位实现多 Agent 协作。
    每个 Agent 是一个方法，检查前置条件后执行。
    """

    max_steps = 8

    def __init__(self, db: Session, settings: Settings):
        self.db = db
        self.settings = settings
        self.ai = AiClient(settings)
        self.knowledge = KnowledgeService(db, settings)
        self.memory = RedisShortTermMemoryStore(settings)
        self.assessment = PsychologicalAssessmentService(self.ai)

    def run(self, user: UserAccount, session: ChatSession, original_input: str, model_input: str) -> AgentRunResult:
        """
        执行一轮 Agent 工作流。

        循环执行所有 Agent，直到 context.finished 或达到 max_steps。
        每轮循环中，按顺序尝试每个 Agent，第一个返回 True 的 Agent 处理后跳出内循环。
        """
        context = AgentContext(user=user, session=session, original_input=original_input, model_input=model_input)
        agents = [
            self.memory_agent,
            self.supervisor_agent,
            self.knowledge_agent,
            self.risk_guardian_agent,
            self.companion_agent,
            self.counselor_agent,
        ]
        for step in range(1, self.max_steps + 1):
            if context.finished:
                break
            for agent in agents:
                if agent(step, context):
                    break
        return AgentRunResult(
            intent=context.intent or IntentType.CHAT,
            risk_level=context.risk_level,
            assessment=context.assessment,
            retrieved_knowledge=context.retrieved_knowledge,
            response_messages=context.response_messages,
            steps=context.steps,
            memory_brief=context.memory_brief,
        )

    # ── Agent 实现 ────────────────────────────────────────────────

    def memory_agent(self, step: int, context: AgentContext) -> bool:
        """
        MemoryAgent: 加载短期记忆。

        优先从 Redis 读取；Redis 为空时从 MySQL 最近消息回填。
        通过 compact_history_for_prompt 将较早消息压缩为一条内部摘要，
        只保留最近 N 条原始消息注入 prompt，并生成记忆摘要供后续 Agent 使用。
        """
        if context.memory_loaded:
            return False

        history = self.memory.load_recent(context.session.public_id)
        source = "redis"
        if not history:
            # Redis 为空，从 MySQL 回填
            rows = (
                self.db.query(ChatMessage)
                .filter(ChatMessage.session_id == context.session.id)
                .order_by(ChatMessage.created_at.desc())
                .limit(self.settings.redis_memory_max_messages)
                .all()
            )
            rows.reverse()
            history = self.memory.messages_from_rows(rows)
            if history:
                self.memory.replace(context.session.public_id, history)
                source = "mysql_seeded"

        # 历史压缩：较早消息折叠为一条内部摘要系统消息，只保留最近 N 条原始消息
        compacted_history, deterministic_brief = compact_history_for_prompt(history, self.settings, context.model_input)
        context.model_history = self._bounded_model_history(
            [*compacted_history, AiMessage(role="user", content=context.model_input)]
        )
        context.memory_brief = self._summarize_memory(history, context.model_input, deterministic_brief)
        context.memory_loaded = True
        context.steps.append(AgentStep(step, "MemoryAgent", "READ_MEMORY", f"loaded {len(history)} messages from {source}"))
        return True

    def supervisor_agent(self, step: int, context: AgentContext) -> bool:
        """
        SupervisorAgent: 意图分类与路由。

        判断用户输入属于 CHAT/CONSULT/RISK。
        CHAT 意图跳过 KnowledgeAgent 和 RiskGuardianAgent。
        """
        if not context.memory_loaded or context.intent_routed:
            return False

        context.intent = self._classify(context.model_input, context.model_history)
        context.intent_routed = True

        # CHAT 意图跳过知识检索和风险评估
        if context.intent == IntentType.CHAT:
            context.knowledge_handled = True
            context.risk_assessed = True

        context.steps.append(AgentStep(step, "SupervisorAgent", "ROUTE_INTENT", f"intent={context.intent.value}"))
        return True

    def knowledge_agent(self, step: int, context: AgentContext) -> bool:
        """
        KnowledgeAgent: RAG 知识检索。

        将用户输入改写为知识库查询词，执行混合检索。
        仅 CONSULT/RISK 意图执行。
        """
        if not context.intent_routed or context.knowledge_handled or context.intent == IntentType.CHAT:
            return False

        query = self._rewrite_query(context)
        retrieved = self.knowledge.retrieve(query, self.settings.knowledge_top_k)
        context.knowledge_query = query
        context.retrieved_knowledge = retrieved
        context.knowledge_handled = True
        context.steps.append(AgentStep(step, "KnowledgeAgent", "RETRIEVE_KNOWLEDGE", f"query={query}; retrieved={len(retrieved)}"))
        return True

    def risk_guardian_agent(self, step: int, context: AgentContext) -> bool:
        """
        RiskGuardianAgent: 心理风险评估。

        执行三级防线评估：
        1. 高风险词典硬兜底
        2. LLM JSON 评估
        3. 关键词启发式兜底

        如果 SupervisorAgent 判定为 RISK 但评估结果不是 HIGH，
        强制提升为 HIGH（宁可误报不可漏报）。
        """
        if not context.knowledge_handled or context.risk_assessed or context.intent == IntentType.CHAT:
            return False

        assessment = self.assessment.assess(context.model_input, context.model_history)

        # RISK 意图强制提升为 HIGH
        if context.intent == IntentType.RISK and assessment.risk != RiskLevel.HIGH:
            assessment.risk = RiskLevel.HIGH
            assessment.emotion_score = max(assessment.emotion_score, 4.0)

        context.assessment = assessment
        context.risk_level = assessment.risk
        context.risk_assessed = True
        context.steps.append(AgentStep(step, "RiskGuardianAgent", "ASSESS_RISK", f"risk={assessment.risk.value}, emotion={assessment.emotion.value}"))
        return True

    def companion_agent(self, step: int, context: AgentContext) -> bool:
        """
        CompanionAgent: 普通陪伴回复。

        处理 CHAT 意图：学习、编程、校园事务、闲聊。
        不做心理测评，不查知识库。
        """
        if not context.intent_routed or context.intent != IntentType.CHAT or context.response_planned:
            return False

        context.risk_level = RiskLevel.LOW
        context.response_agent = "CompanionAgent"
        context.response_plan = "围绕用户当前问题直接、自然地回答。"
        context.response_messages = [
            PromptTemplates.answer_system_prompt(IntentType.CHAT, RiskLevel.LOW, "", context.user.display_name),
            AiMessage(role="system", content=f"当前由 CompanionAgent 负责回复。\n记忆摘要：\n{context.memory_brief}\n回复策略：\n{context.response_plan}"),
            *context.model_history,
        ]
        context.response_planned = True
        context.finished = True
        context.steps.append(AgentStep(step, "CompanionAgent", "PLAN_RESPONSE", "normal companion response planned"))
        return True

    def counselor_agent(self, step: int, context: AgentContext) -> bool:
        """
        CounselorAgent: 心理咨询回复。

        处理 CONSULT/RISK 意图：
        结合记忆、RAG 知识、风险评估和 Skill 指引生成回复 prompt。
        """
        if not context.risk_assessed or context.intent == IntentType.CHAT or context.response_planned:
            return False

        context.response_agent = "CounselorAgent"
        context.response_plan = "先共情，再给出具体支持步骤；高风险时优先安全。"

        knowledge_context = "\n\n".join(f"- [{item.source}] {item.content}" for item in context.retrieved_knowledge)
        skill_context = MindBridgeSkillLibrary.response_skill_context(
            context.intent or IntentType.CONSULT,
            context.risk_level,
            context.original_input,
        )

        context.response_messages = [
            PromptTemplates.answer_system_prompt(
                context.intent or IntentType.CONSULT,
                context.risk_level,
                knowledge_context,
                context.user.display_name,
                skill_context,
            ),
            AiMessage(role="system", content=(
                f"当前由 CounselorAgent 负责回复。\n记忆摘要：\n{context.memory_brief}\n"
                f"KnowledgeAgent 检索 query：\n{context.knowledge_query}\n回复策略：\n{context.response_plan}"
            )),
            *context.model_history,
        ]
        context.response_planned = True
        context.finished = True
        context.steps.append(AgentStep(step, "CounselorAgent", "PLAN_RESPONSE", f"support response planned with risk={context.risk_level.value}"))
        return True

    # ── 辅助方法 ──────────────────────────────────────────────────

    def _classify(self, text: str, history: list[AiMessage]) -> IntentType:
        """
        意图分类。

        三级策略：
        1. 高风险关键词 → 直接 RISK
        2. 通用任务关键词且无咨询信号 → 直接 CHAT
        3. LLM 分类
        4. 分类失败 → 关键词兜底
        """
        lowered = text.lower()
        if has_high_risk_signal(lowered):
            return IntentType.RISK
        if not has_consult_signal(lowered) and any(word in lowered for word in GENERAL_TASK_WORDS):
            return IntentType.CHAT
        try:
            label = self.ai.complete(PromptTemplates.intent_prompt(history, text)).upper()
            if "RISK" in label:
                return IntentType.RISK
            if "CONSULT" in label:
                return IntentType.CONSULT
            if "CHAT" in label:
                return IntentType.CHAT
        except Exception:
            pass
        return IntentType.CONSULT if has_consult_signal(lowered) else IntentType.CHAT

    def _rewrite_query(self, context: AgentContext) -> str:
        """
        知识查询改写。

        将用户输入改写为适合检索校园心理知识库的中文查询词。
        改写失败时使用原始输入。
        """
        try:
            query = self.ai.complete([
                AiMessage(role="system", content="你是 MindBridge 的 KnowledgeAgent。把学生输入改写成适合检索校园心理知识库的中文查询词，只输出查询词。"),
                AiMessage(role="user", content=f"记忆摘要：\n{context.memory_brief}\n\n当前输入：\n{context.model_input}"),
            ]).strip()
            return (query or context.model_input)[:60]
        except Exception:
            return context.model_input

    def _bounded_model_history(self, history: list[AiMessage]) -> list[AiMessage]:
        """
        限制注入 prompt 的历史消息条数。

        限制为 chat_history_limit * 2 条（每轮=user+assistant）。
        如果第一条是压缩摘要（role=system），始终保留它，只截断其余部分。
        """
        limit = max(2, self.settings.chat_history_limit * 2)
        if len(history) <= limit:
            return history
        if history[0].role == "system":
            return [history[0], *history[-(limit - 1):]]
        return history[-limit:]

    def _summarize_memory(self, history: list[AiMessage], current_input: str, fallback: str) -> str:
        """
        生成记忆摘要（用于 CompanionAgent/CounselorAgent 的 prompt 注入和 AgentRunTrace 审计）。

        优先调用 LLM 提取 1-3 条关键记忆要点；
        LLM 调用失败或结果为空时，回退到 compact_history_for_prompt 生成的
        确定性摘要（fallback），避免记忆摘要完全丢失。
        """
        max_chars = max(120, self.settings.memory_summary_max_chars)
        if not history:
            return "无相关历史记忆。"
        try:
            summary = self.ai.complete([
                AiMessage(role="system", content="你是 MindBridge 的 MemoryAgent。只输出 1-3 条中文记忆要点，不输出风险等级或诊断。"),
                AiMessage(role="user", content=f"当前输入：\n{current_input}\n\n最近历史：\n{history[-12:]}"),
            ]).strip()
            return summary[:max_chars] or fallback
        except Exception:
            return fallback or "无相关历史记忆。"
