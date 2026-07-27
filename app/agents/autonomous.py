from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy.orm import Session

from app.agents.events import (
    AgentArtifact,
    AgentEvent,
    AgentEventType,
    AgentMessage,
    AgentTask,
    AgentTurnResult,
    CollaborationBlackboard,
)
from app.agents.decision import (
    GENERAL_TASK_WORDS,
    assess_safety,
    classify_intent,
    prepare_decision_history,
)
from app.agents.prompt_assembly import build_response_messages
from app.agents.registry import AgentCapability, AgentDecision, AgentProfile
from app.core.config import Settings
from app.core.enums import IntentType, RiskLevel
from app.schemas.dtos import AiMessage
from app.services.agent_models import AgentModelRegistry
from app.services.ai import AiClient, PromptTemplates, has_consult_signal, has_high_risk_signal

if TYPE_CHECKING:
    from app.models.entities import ChatSession, UserAccount
    from app.services.knowledge import KnowledgeService, SearchResult
    from app.services.memory import RedisShortTermMemoryStore


RESPONSE_PLAN_PROMPT_REVIEW_SCOPE = "response_plan_prompt"
_HIGH_RISK_PLAN_CONSTRAINT_MARKERS = {
    "high_risk_rule": ("高风险处理规则",),
    "current_safety_check": ("当前安全", "当前是否安全"),
    "trusted_human_contact": ("可信任的人",),
    "urgent_support_options": ("心理中心", "辅导员", "紧急救助", "紧急援助"),
    "no_dangerous_details": ("不提供任何危险操作细节", "不提供危险操作细节"),
}


@dataclass(frozen=True)
class ResponsePlanSafetyReview:
    """确定性审查生成前的回复 plan/prompt，不代表最终模型输出。"""

    approved: bool
    required: bool
    reason: str
    constraint_checks: dict[str, bool]
    missing_constraints: tuple[str, ...]


def review_response_plan_safety(
    messages: list[AiMessage],
    risk: RiskLevel,
) -> ResponsePlanSafetyReview:
    """只审查 SSE 生成开始前被采纳的 prompt/plan 安全约束。"""
    combined = "\n".join(
        getattr(message, "content", str(message))
        for message in messages
    )
    required = risk == RiskLevel.HIGH
    checks = {
        name: any(marker in combined for marker in markers)
        for name, markers in _HIGH_RISK_PLAN_CONSTRAINT_MARKERS.items()
    }
    missing = tuple(name for name, present in checks.items() if not present)
    approved = not required or not missing
    if not required:
        reason = "response generation plan does not require high-risk constraints"
    elif approved:
        reason = "high-risk response generation plan includes all required safety constraints"
    else:
        reason = (
            "high-risk response generation plan is missing required constraints: "
            + ", ".join(missing)
        )
    return ResponsePlanSafetyReview(
        approved=approved,
        required=required,
        reason=reason,
        constraint_checks=checks,
        missing_constraints=missing,
    )


@dataclass
class AgentRuntimeServices:
    db: Session
    settings: Settings
    user: UserAccount
    session: ChatSession
    ai: AiClient
    model_registry: AgentModelRegistry
    memory: RedisShortTermMemoryStore
    private_memory: "AgentPrivateMemory"
    knowledge: KnowledgeService


class AgentPrivateMemory:
    """按 Agent 名称隔离 Redis key 的私有记忆外观。"""

    def __init__(self, settings: Settings):
        from app.services.memory import RedisShortTermMemoryStore

        self.store = RedisShortTermMemoryStore(settings)

    def load(self, agent_name: str, session_public_id: str) -> list[AiMessage]:
        return self.store.load_recent(self._key(agent_name, session_public_id))

    def append(self, agent_name: str, session_public_id: str, content: str) -> None:
        self.store.append(self._key(agent_name, session_public_id), "system", content)

    def _key(self, agent_name: str, session_public_id: str) -> str:
        return f"agent:{agent_name}:{session_public_id}"


class BaseAutonomousAgent:
    profile: AgentProfile

    def __init__(self, services: AgentRuntimeServices):
        self.services = services

    @property
    def name(self) -> str:
        return self.profile.name

    def client(self) -> AiClient:
        return self.services.model_registry.client_for(self.name)

    def private_memory(self) -> list[AiMessage]:
        return self.services.private_memory.load(self.name, self.services.session.public_id)

    def remember(self, content: str) -> None:
        self.services.private_memory.append(self.name, self.services.session.public_id, content)

    def _artifact(
        self,
        kind: str,
        payload: dict[str, Any],
        task: AgentTask,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> AgentArtifact:
        return AgentArtifact(
            id=f"{self.name}:{kind}:{uuid.uuid4().hex[:10]}",
            owner=self.name,
            kind=kind,
            payload=payload,
            confidence=confidence,
            task_id=task.id,
            metadata=metadata or {},
        )


class UnderstandingAgent(BaseAutonomousAgent):
    profile = AgentProfile(
        name="UnderstandingAgent",
        capabilities=frozenset({AgentCapability.UNDERSTANDING}),
        system_prompt=(
            "你是 UnderstandingAgent。你只负责理解用户当前请求，输出意图、主题、置信度和理由，"
            "不生成最终回复，不做风险处置。"
        ),
        memory_policy="private_intent_history",
        model_profile="understanding",
        tool_permissions=frozenset({"llm.intent"}),
    )

    def decide(self, task: AgentTask, board: CollaborationBlackboard) -> AgentDecision:
        if board.latest_artifact("intent"):
            return AgentDecision(False, reason="intent artifact already exists")
        if self._is_directed(task, board):
            return AgentDecision(True, 0.82, "open user-turn task needs understanding")
        return AgentDecision(False, reason="task does not need understanding")

    def act(self, task: AgentTask, board: CollaborationBlackboard) -> AgentTurnResult:
        decision = self._classify(
            board.model_input or board.user_input,
            list(board.conversation_history),
        )
        intent = decision.intent
        confidence = 0.92 if intent == IntentType.RISK else 0.78
        payload = {
            "intent": intent.value,
            "topic": self._topic(board.model_input or board.user_input),
            "reason": decision.reason,
            "historyParticipated": bool(board.conversation_history),
            "privateMemoryKey": self.services.private_memory._key(self.name, self.services.session.public_id),
        }
        self.remember(f"intent={intent.value}; topic={payload['topic']}")
        return AgentTurnResult(
            artifacts=(self._artifact("intent", payload, task, confidence),),
            messages=(
                AgentMessage(
                    id=f"msg:{uuid.uuid4().hex[:10]}",
                    sender=self.name,
                    recipient="*",
                    task_id=task.id,
                    kind="PROPOSAL",
                    content=f"我判断本轮意图是 {intent.value}",
                ),
            ),
        )

    def _is_directed(self, task: AgentTask, board: CollaborationBlackboard) -> bool:
        if AgentCapability.UNDERSTANDING.value in task.required_capabilities:
            return True
        return bool(board.user_input and task.metadata.get("kind") in {"root", "understanding"})

    def _classify(self, text: str, history: list[AiMessage]):
        decision_history = prepare_decision_history(history, self.services.settings)
        return classify_intent(
            text,
            decision_history.prompt_history,
            self.client(),
            risk_history=decision_history.risk_scan_history,
        )

    def _topic(self, text: str) -> str:
        lowered = text.lower()
        if has_high_risk_signal(lowered):
            return "safety"
        if has_consult_signal(lowered):
            return "mental_health_support"
        if any(word in lowered for word in GENERAL_TASK_WORDS):
            return "general_task"
        return "conversation"


class SafetyAgent(BaseAutonomousAgent):
    profile = AgentProfile(
        name="SafetyAgent",
        capabilities=frozenset({AgentCapability.SAFETY}),
        system_prompt=(
            "你是 SafetyAgent。你独立评估风险，并审查候选回复生成 plan/prompt 的安全约束。"
            "你可以发布 SAFETY_OVERRIDE；你不生成最终回复，也不接触 SSE 阶段的最终模型文本。"
        ),
        memory_policy="private_safety_ledger",
        model_profile="safety",
        tool_permissions=frozenset({"llm.risk", "rules.high_risk", "response.review"}),
    )

    def decide(self, task: AgentTask, board: CollaborationBlackboard) -> AgentDecision:
        latest_response = board.latest_artifact("response_proposal")
        latest_review = board.latest_artifact("safety_review")
        if latest_response and (
            latest_review is None
            or latest_review.metadata.get("responseArtifactId") != latest_response.id
        ):
            return AgentDecision(
                True,
                0.95,
                "candidate response generation plan needs safety constraint review",
            )
        if not board.latest_artifact("risk") and board.user_input:
            confidence = 0.98 if has_high_risk_signal(board.user_input) else 0.84
            return AgentDecision(True, confidence, "user input needs independent risk assessment")
        if AgentCapability.SAFETY.value in task.required_capabilities:
            return AgentDecision(True, 0.8, "task explicitly asks for safety")
        return AgentDecision(False, reason="no safety work needed")

    def act(self, task: AgentTask, board: CollaborationBlackboard) -> AgentTurnResult:
        response = board.latest_artifact("response_proposal")
        review = board.latest_artifact("safety_review")
        if response and (review is None or review.metadata.get("responseArtifactId") != response.id):
            return self._review_response_plan(task, board, response)
        return self._assess_risk(task, board)

    def _assess_risk(self, task: AgentTask, board: CollaborationBlackboard) -> AgentTurnResult:
        decision_history = prepare_decision_history(
            list(board.conversation_history),
            self.services.settings,
        )
        assessment, safety_reason = assess_safety(
            board.model_input or board.user_input,
            decision_history.prompt_history,
            _intent(board),
            self.client(),
            risk_history=decision_history.risk_scan_history,
        )
        payload = {
            "risk": assessment.risk.value,
            "emotion": assessment.emotion.value,
            "emotionScore": assessment.emotion_score,
            "confidence": assessment.confidence,
            "summary": assessment.summary,
            "reason": safety_reason,
            "historyParticipated": bool(board.conversation_history),
            "assessment": assessment,
            "privateMemoryKey": self.services.private_memory._key(self.name, self.services.session.public_id),
        }
        events: tuple[AgentEvent, ...] = ()
        if assessment.risk == RiskLevel.HIGH:
            events = (
                AgentEvent(
                    type=AgentEventType.SAFETY_OVERRIDE,
                    actor=self.name,
                    task_id=task.id,
                    message="Independent safety assessment raised this turn to HIGH",
                    metadata={"risk": RiskLevel.HIGH.value},
                ),
            )
        self.remember(f"risk={assessment.risk.value}; summary={assessment.summary}")
        return AgentTurnResult(
            artifacts=(self._artifact("risk", payload, task, assessment.confidence),),
            events=events,
            messages=(
                AgentMessage(
                    id=f"msg:{uuid.uuid4().hex[:10]}",
                    sender=self.name,
                    recipient="CoordinatorAgent",
                    task_id=task.id,
                    kind="SAFETY_ASSESSMENT",
                    content=f"risk={assessment.risk.value}",
                ),
            ),
        )

    def _review_response_plan(
        self,
        task: AgentTask,
        board: CollaborationBlackboard,
        response: AgentArtifact,
    ) -> AgentTurnResult:
        risk = _risk_level(board)
        messages = response.payload.get("messages", [])
        plan_review = review_response_plan_safety(messages, risk)
        approved = plan_review.approved
        reason = plan_review.reason
        payload = {
            "approved": approved,
            "reason": reason,
            "responseArtifactId": response.id,
            "risk": risk.value,
            "reviewScope": RESPONSE_PLAN_PROMPT_REVIEW_SCOPE,
            "reviewedArtifactStage": response.payload.get(
                "artifactStage",
                RESPONSE_PLAN_PROMPT_REVIEW_SCOPE,
            ),
            "reviewedGeneratedText": False,
            "validatesFinalResponseSafety": False,
            "constraintChecks": plan_review.constraint_checks,
            "missingConstraints": list(plan_review.missing_constraints),
            "privateMemoryKey": self.services.private_memory._key(self.name, self.services.session.public_id),
        }
        kind = "safety_review" if approved else "critique"
        events: tuple[AgentEvent, ...] = ()
        if not approved:
            events = (
                AgentEvent(
                    type=AgentEventType.REVISION_REQUESTED,
                    actor=self.name,
                    task_id=task.id,
                    artifact_id=response.id,
                    message=reason,
                ),
            )
        self.remember(f"review approved={approved}; reason={reason}")
        return AgentTurnResult(
            artifacts=(
                self._artifact(
                    kind,
                    payload,
                    task,
                    0.95,
                    {"responseArtifactId": response.id},
                ),
            ),
            events=events,
        )


class ContextAgent(BaseAutonomousAgent):
    profile = AgentProfile(
        name="ContextAgent",
        capabilities=frozenset({AgentCapability.CONTEXT}),
        system_prompt=(
            "你是 ContextAgent。你只负责为本轮协作提供上下文，包括私有记忆、会话摘要、RAG 证据和 skill 约束。"
            "你不判断最终答案是否可采纳。"
        ),
        memory_policy="private_context_memory",
        model_profile="context",
        tool_permissions=frozenset({"redis.memory", "mysql.messages", "rag.retrieve", "skills.read"}),
    )

    def decide(self, task: AgentTask, board: CollaborationBlackboard) -> AgentDecision:
        if board.latest_artifact("context"):
            return AgentDecision(False, reason="context artifact already exists")
        risk = _risk_level(board)
        intent = _intent(board)
        if AgentCapability.CONTEXT.value in task.required_capabilities:
            return AgentDecision(True, 0.86, "task explicitly asks for context")
        if risk in {RiskLevel.MEDIUM, RiskLevel.HIGH} or intent in {IntentType.CONSULT, IntentType.RISK}:
            return AgentDecision(True, 0.82, "support path needs memory, RAG, and skill context")
        return AgentDecision(False, reason="context not necessary for current artifacts")

    def act(self, task: AgentTask, board: CollaborationBlackboard) -> AgentTurnResult:
        from app.services.memory import assemble_memory_context, summarize_history_for_memory
        from app.services.skills import MindBridgeSkillLibrary

        history = list(board.conversation_history) or self._load_history()
        deterministic_brief = summarize_history_for_memory(
            history,
            board.model_input,
            self.services.settings.memory_summary_max_chars,
        )
        memory_brief = self._summarize_memory(history, board.model_input, deterministic_brief)
        memory_artifacts = []
        for kind in ("intent", "risk"):
            artifact = board.latest_artifact(kind)
            if artifact is not None:
                memory_artifacts.append(
                    {
                        "kind": kind,
                        "payload": {
                            key: value
                            for key, value in artifact.payload.items()
                            if key in {"intent", "risk", "summary", "reason"}
                        },
                    }
                )
        assembled = assemble_memory_context(
            history,
            board.model_input,
            self.services.settings,
            memory_brief=memory_brief,
            artifacts=memory_artifacts,
        )
        model_history = assembled.model_history
        intent = _intent(board)
        risk = _risk_level(board)

        retrieved: list[SearchResult] = []
        query = ""
        skill_context = ""
        if intent != IntentType.CHAT or risk != RiskLevel.LOW:
            query = self._rewrite_query(memory_brief, board.model_input)
            retrieved = self.services.knowledge.retrieve(query, self.services.settings.knowledge_top_k)
            skill_context = MindBridgeSkillLibrary.response_skill_context(intent, risk, board.user_input)
        payload = {
            "memoryBrief": memory_brief,
            "modelHistory": model_history,
            "knowledgeQuery": query,
            "retrievedKnowledge": retrieved,
            "skillContext": skill_context,
            "privateMemoryKey": self.services.private_memory._key(self.name, self.services.session.public_id),
        }
        self.remember(f"context intent={intent.value}; risk={risk.value}; retrieved={len(retrieved)}")
        return AgentTurnResult(
            artifacts=(self._artifact("context", payload, task, 0.88),),
            messages=(
                AgentMessage(
                    id=f"msg:{uuid.uuid4().hex[:10]}",
                    sender=self.name,
                    recipient="ResponseAgent",
                    task_id=task.id,
                    kind="CONTEXT_READY",
                    content=f"context ready; retrieved={len(retrieved)}",
                ),
            ),
        )

    def _load_history(self) -> list[AiMessage]:
        from app.models.entities import ChatMessage

        history = self.services.memory.load_recent(self.services.session.public_id)
        if history:
            return history
        rows = (
            self.services.db.query(ChatMessage)
            .filter(ChatMessage.session_id == self.services.session.id)
            .order_by(ChatMessage.created_at.desc())
            .limit(self.services.settings.redis_memory_max_messages)
            .all()
        )
        rows.reverse()
        history = self.services.memory.messages_from_rows(rows)
        if history:
            self.services.memory.replace(self.services.session.public_id, history)
        return history

    def _rewrite_query(self, memory_brief: str, model_input: str) -> str:
        try:
            query = self.client().complete([
                AiMessage(
                    role="system",
                    content=f"{self.profile.system_prompt}\n把学生输入改写成适合检索校园心理知识库的中文查询词，只输出查询词。",
                ),
                AiMessage(role="user", content=f"记忆摘要：\n{memory_brief}\n\n当前输入：\n{model_input}"),
            ]).strip()
            return (query or model_input)[:60]
        except Exception:
            return model_input[:60]

    def _summarize_memory(self, history: list[AiMessage], current_input: str, fallback: str) -> str:
        from app.services.memory import summarize_memory_for_prompt

        return summarize_memory_for_prompt(
            history,
            current_input,
            self.services.settings,
            self.client(),
            self.profile.system_prompt,
        )

    def _bounded_model_history(self, history: list[AiMessage]) -> list[AiMessage]:
        from app.services.memory import bound_model_history

        return bound_model_history(history, self.services.settings.chat_history_limit)


class ResponseAgent(BaseAutonomousAgent):
    profile = AgentProfile(
        name="ResponseAgent",
        capabilities=frozenset({AgentCapability.RESPONSE}),
        system_prompt=(
            "你是 ResponseAgent。你根据黑板上的意图、风险、上下文和安全约束提出候选回复生成 plan/prompt。"
            "该 artifact 不包含最终模型回复文本；是否采纳此生成计划由 CoordinatorAgent 决定。"
        ),
        memory_policy="private_response_strategy",
        model_profile="response",
        tool_permissions=frozenset({"llm.response_plan"}),
    )

    def decide(self, task: AgentTask, board: CollaborationBlackboard) -> AgentDecision:
        if board.latest_artifact("response_proposal") and "revisionOf" not in task.metadata:
            return AgentDecision(False, reason="response proposal already exists")
        if not board.latest_artifact("intent") or not board.latest_artifact("risk"):
            return AgentDecision(False, reason="response needs intent and risk artifacts")
        intent = _intent(board)
        risk = _risk_level(board)
        if intent == IntentType.CHAT and risk == RiskLevel.LOW:
            return AgentDecision(True, 0.78, "normal chat response can be proposed")
        if board.latest_artifact("context") or risk == RiskLevel.HIGH:
            return AgentDecision(True, 0.84, "support response has enough artifacts")
        if AgentCapability.RESPONSE.value in task.required_capabilities:
            return AgentDecision(True, 0.65, "explicit response task")
        return AgentDecision(False, reason="waiting for context")

    def act(self, task: AgentTask, board: CollaborationBlackboard) -> AgentTurnResult:
        intent = _intent(board)
        risk = _risk_level(board)
        revision_of = str(task.metadata.get("revisionOf") or "").strip()
        critique = None
        if revision_of:
            critique = next(
                (
                    artifact
                    for artifact in reversed(board.artifacts_by_kind("critique"))
                    if artifact.payload.get("responseArtifactId") == revision_of
                ),
                None,
            )
        context = board.latest_artifact("context")
        context_payload = context.payload if context else {}
        model_history = context_payload.get("modelHistory") or [AiMessage(role="user", content=board.model_input)]
        memory_brief = context_payload.get("memoryBrief") or "无相关历史记忆。"
        knowledge = context_payload.get("retrievedKnowledge") or []
        skill_context = context_payload.get("skillContext") or ""
        knowledge_context = "\n\n".join(f"- [{item.source}] {item.content}" for item in knowledge)
        messages, mode = build_response_messages(
            intent=intent,
            risk=risk,
            display_name=self.services.user.display_name,
            model_history=model_history,
            memory_brief=memory_brief,
            response_agent_system_prompt=self.profile.system_prompt,
            private_memory_text=_format_private_memory(self.private_memory()),
            knowledge_context=knowledge_context,
            skill_context=skill_context,
        )
        if revision_of:
            critique_reason = str(
                critique.payload.get("reason")
                if critique is not None
                else "SafetyAgent requested a safer response revision."
            )
            revision_lines = [
                "安全修订指令：不要复用旧候选回复，必须根据 SafetyAgent critique 修改回复规划。",
                f"被修订的 response artifact：{revision_of}",
                f"critique 原因：{critique_reason}",
            ]
            if risk == RiskLevel.HIGH:
                revision_lines.append(
                    "高风险修订要求：最终回复必须先确认用户当前是否安全；"
                    "明确建议联系身边可信任的人；联系学校心理中心、辅导员或当地紧急援助；"
                    "不提供危险操作细节。"
                )
            messages.insert(2, AiMessage(role="system", content="\n".join(revision_lines)))
        payload = {
            "messages": messages,
            "mode": mode,
            "intent": intent.value,
            "risk": risk.value,
            "artifactStage": RESPONSE_PLAN_PROMPT_REVIEW_SCOPE,
            "containsGeneratedText": False,
            "responseAgent": self.name,
            "privateMemoryKey": self.services.private_memory._key(self.name, self.services.session.public_id),
        }
        if revision_of:
            payload.update(
                {
                    "revisionOf": revision_of,
                    "critiqueArtifactId": critique.id if critique is not None else "",
                    "critiqueReason": critique_reason,
                }
            )
        self.remember(f"response mode={mode}; intent={intent.value}; risk={risk.value}")
        return AgentTurnResult(
            artifacts=(self._artifact("response_proposal", payload, task, 0.86),),
            messages=(
                AgentMessage(
                    id=f"msg:{uuid.uuid4().hex[:10]}",
                    sender=self.name,
                    recipient="SafetyAgent",
                    task_id=task.id,
                    kind="REVIEW_REQUEST",
                    content="请审查候选回复方案。",
                ),
            ),
        )


class CoordinatorAgent(BaseAutonomousAgent):
    profile = AgentProfile(
        name="CoordinatorAgent",
        capabilities=frozenset({AgentCapability.COORDINATION}),
        system_prompt=(
            "你是 CoordinatorAgent。你不规定固定 Agent 顺序；你只维护任务板、预算、安全门槛、冲突仲裁和最终采纳。"
        ),
        memory_policy="private_coordination_trace",
        model_profile="coordinator",
        tool_permissions=frozenset({"taskboard.write", "blackboard.accept"}),
    )

    def decide(self, task: AgentTask, board: CollaborationBlackboard) -> AgentDecision:
        return AgentDecision(False, reason="CoordinatorAgent is driven by the event loop")

    def act(self, task: AgentTask, board: CollaborationBlackboard) -> AgentTurnResult:
        return AgentTurnResult(close_task=False)

    def root_task(self, board: CollaborationBlackboard) -> AgentTask:
        from app.agents.events import TaskPriority

        return AgentTask(
            id="task:root",
            title="Resolve user turn",
            description=board.user_input,
            priority=TaskPriority.CRITICAL if has_high_risk_signal(board.user_input) else TaskPriority.NORMAL,
            created_by=self.name,
            metadata={"kind": "root"},
        )

    def remember_acceptance(self, artifact_id: str, reason: str) -> None:
        self.remember(f"accepted={artifact_id}; reason={reason}")


def _intent(board: CollaborationBlackboard) -> IntentType:
    artifact = board.latest_artifact("intent")
    if artifact:
        try:
            return IntentType(str(artifact.payload.get("intent", IntentType.CHAT.value)).upper())
        except ValueError:
            return IntentType.CHAT
    if has_high_risk_signal(board.user_input):
        return IntentType.RISK
    if has_consult_signal(board.user_input):
        return IntentType.CONSULT
    return IntentType.CHAT


def _risk_level(board: CollaborationBlackboard) -> RiskLevel:
    highest = RiskLevel.LOW
    order = {RiskLevel.LOW: 1, RiskLevel.MEDIUM: 2, RiskLevel.HIGH: 3}
    for artifact in board.artifacts_by_kind("risk"):
        try:
            risk = RiskLevel(str(artifact.payload.get("risk", RiskLevel.LOW.value)).upper())
        except ValueError:
            risk = RiskLevel.LOW
        if order[risk] > order[highest]:
            highest = risk
    if any(event.type == AgentEventType.SAFETY_OVERRIDE for event in board.events):
        return RiskLevel.HIGH
    return highest


def _context_history(board: CollaborationBlackboard) -> list[AiMessage]:
    context = board.latest_artifact("context")
    if not context:
        return [AiMessage(role="user", content=board.model_input or board.user_input)]
    return context.payload.get("modelHistory") or [
        AiMessage(role="user", content=board.model_input or board.user_input)
    ]


def _format_private_memory(items: list[AiMessage]) -> str:
    if not items:
        return "无"
    return "\n".join(f"- {item.content}" for item in items[-5:])
