"""Shared Understanding + Safety decision unit.

The default event-driven runtime and route evaluation both call this module.
Conversation history is sanitized once into a storage-bounded risk view and a
smaller LLM-prompt view. An unresolved HIGH-risk fact cannot disappear merely
because it falls outside the prompt window. Semantic intent and final safety
routing remain separate outputs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.config import Settings
from app.core.enums import IntentType, RiskLevel
from app.schemas.dtos import AiMessage
from app.services.ai import (
    AiClient,
    PromptTemplates,
    has_consult_signal,
    has_high_risk_signal,
    has_medium_risk_signal,
    has_resolved_risk_transition,
    has_self_high_risk_signal,
    has_third_party_immediate_risk,
    risk_state_after_texts,
)
from app.services.assessment import (
    PsychologicalAssessmentService,
    PsychologyAssessment,
    risk_order,
)
from app.services.privacy import PrivacySanitizer


GENERAL_TASK_WORDS = [
    "java",
    "python",
    "javascript",
    "代码",
    "编程",
    "程序",
    "算法",
    "数据库",
    "spring",
    "maven",
    "前端",
    "后端",
    "项目",
    "接口",
    "bug",
    "报错",
    "作业",
    "论文",
    "翻译",
    "总结",
    "解释",
    "怎么写",
    "如何",
    "是什么",
    "为什么",
    "给我",
    "帮我",
    "推荐",
    "查询",
    "天气",
    "路线",
]
_INFORMATIONAL_TASK_PATTERN = re.compile(
    r"(?:心理学|量表|测验|问卷|论文|课程|课本|研究|统计|定义|概念|电影|小说)"
    r"[^。！？]{0,28}(?:解释|说明|翻译|总结|分析|改写|提纲|含义)"
    r"|(?:解释|说明|翻译|总结|分析|改写)[^。！？]{0,24}"
    r"(?:量表|测验|问卷|论文|课程|课本|定义|概念|台词)",
    re.IGNORECASE,
)
_PERSONAL_DISTRESS_ASSERTION_PATTERN = re.compile(
    r"(?:我|本人)(?:现在|最近|这几天|一直)?[^。！？]{0,10}"
    r"(?:焦虑|抑郁|难过|痛苦|无助|崩溃|失眠|害怕|担心|撑不住)",
    re.IGNORECASE,
)

@dataclass(frozen=True)
class IntentDecision:
    intent: IntentType
    reason: str


@dataclass(frozen=True)
class RouteSafetyDecision:
    """One online-compatible route and safety decision."""

    understanding_intent: IntentType
    final_intent: IntentType
    assessment: PsychologyAssessment
    safety_override: bool
    intent_reason: str
    safety_reason: str
    history_participated: bool


@dataclass(frozen=True)
class DecisionHistory:
    """Two online views derived from one sanitized session history."""

    risk_scan_history: list[AiMessage]
    prompt_history: list[AiMessage]


def prepare_decision_history(
    history: list[AiMessage],
    settings: Settings,
) -> DecisionHistory:
    """Apply online privacy/storage bounds, then derive the LLM prompt window."""
    privacy = PrivacySanitizer()
    sanitized = [
        AiMessage(role=item.role.lower(), content=privacy.sanitize(item.content))
        for item in history
        if item.role.lower() in {"system", "user", "assistant"} and item.content
    ]
    storage_limit = max(1, int(settings.redis_memory_max_messages))
    prompt_limit = max(2, int(settings.chat_history_limit) * 2)
    risk_scan_history = sanitized[-storage_limit:]
    return DecisionHistory(
        risk_scan_history=risk_scan_history,
        prompt_history=risk_scan_history[-prompt_limit:],
    )


def classify_intent(
    text: str,
    prompt_history: list[AiMessage],
    client: AiClient,
    *,
    risk_history: list[AiMessage] | None = None,
) -> IntentDecision:
    """Run the production Understanding decision without duplicating rules."""
    risk_history = prompt_history if risk_history is None else risk_history
    lowered = (text or "").lower()
    if has_self_high_risk_signal(lowered):
        return IntentDecision(IntentType.RISK, "self_high_risk_hard_signal")
    if has_third_party_immediate_risk(lowered):
        return IntentDecision(IntentType.CONSULT, "third_party_crisis_support")
    if has_resolved_risk_transition(lowered):
        return IntentDecision(IntentType.CONSULT, "current_risk_explicitly_resolved")
    combined_history = [*risk_history, AiMessage(role="user", content=text)]
    history_kind = unresolved_history_risk_kind(combined_history)
    if history_kind == "self":
        return IntentDecision(IntentType.RISK, "unresolved_self_risk_history")
    if (
        _INFORMATIONAL_TASK_PATTERN.search(lowered)
        and not _PERSONAL_DISTRESS_ASSERTION_PATTERN.search(lowered)
    ):
        return IntentDecision(IntentType.CHAT, "non_personal_informational_task")
    if has_medium_risk_signal(lowered):
        return IntentDecision(IntentType.CONSULT, "medium_risk_support")
    if not has_consult_signal(lowered) and any(word in lowered for word in GENERAL_TASK_WORDS):
        return IntentDecision(IntentType.CHAT, "general_task_words")
    if not has_consult_signal(lowered) and history_kind == "third_party":
        return IntentDecision(IntentType.CONSULT, "unresolved_third_party_risk_history")
    prior_risk_state = _history_risk_state(risk_history)
    if (
        history_kind is None
        and prior_risk_state in {"self", "third_party", "resolved"}
        and _history_risk_state(combined_history) == "resolved"
    ):
        return IntentDecision(IntentType.CONSULT, "resolved_risk_followup")

    try:
        label = client.complete(PromptTemplates.intent_prompt(prompt_history, text)).upper()
        if "RISK" in label:
            return IntentDecision(IntentType.RISK, "llm_intent_risk")
        if "CONSULT" in label:
            return IntentDecision(IntentType.CONSULT, "llm_intent_consult")
        if "CHAT" in label:
            return IntentDecision(IntentType.CHAT, "llm_intent_chat")
    except Exception:
        pass

    if has_consult_signal(lowered):
        return IntentDecision(IntentType.CONSULT, "consult_keyword_fallback")

    if history_kind == "third_party":
        return IntentDecision(IntentType.CONSULT, "unresolved_third_party_risk_history")
    return IntentDecision(IntentType.CHAT, "default_chat_fallback")


def assess_safety(
    text: str,
    prompt_history: list[AiMessage],
    understanding_intent: IntentType,
    client: AiClient,
    *,
    risk_history: list[AiMessage] | None = None,
) -> tuple[PsychologyAssessment, str]:
    """Run Safety and apply the online escalation policy."""
    risk_history = prompt_history if risk_history is None else risk_history
    assessment = PsychologicalAssessmentService(client).assess(text, prompt_history)
    reason = "current_turn_assessment"

    history_kind = unresolved_history_risk_kind(
        [*risk_history, AiMessage(role="user", content=text)]
    )
    if history_kind is not None and risk_order(assessment.risk) < risk_order(RiskLevel.HIGH):
        assessment.risk = RiskLevel.HIGH
        assessment.emotion_score = max(assessment.emotion_score, 4.0)
        assessment.confidence = max(assessment.confidence, 0.9)
        assessment.summary = "最近会话包含尚未解除的高风险事实"
        reason = f"unresolved_{history_kind}_risk_history"
    elif has_medium_risk_signal(text) and risk_order(assessment.risk) < risk_order(RiskLevel.MEDIUM):
        assessment.risk = RiskLevel.MEDIUM
        assessment.emotion_score = max(assessment.emotion_score, 3.0)
        assessment.confidence = max(assessment.confidence, 0.8)
        assessment.summary = "检测到持续或明显影响功能的风险信号"
        reason = "medium_risk_hard_signal"

    if understanding_intent == IntentType.RISK and assessment.risk != RiskLevel.HIGH:
        assessment.risk = RiskLevel.HIGH
        assessment.emotion_score = max(assessment.emotion_score, 4.0)
        assessment.confidence = max(assessment.confidence, 0.9)
        reason = "risk_intent_forced_high"
    return assessment, reason


def decide_route(
    text: str,
    history: list[AiMessage],
    settings: Settings,
    *,
    understanding_client: AiClient,
    safety_client: AiClient,
) -> RouteSafetyDecision:
    """Execute Understanding, Safety, and final SAFETY_OVERRIDE semantics."""
    decision_history = prepare_decision_history(history, settings)
    intent_decision = classify_intent(
        text,
        decision_history.prompt_history,
        understanding_client,
        risk_history=decision_history.risk_scan_history,
    )
    assessment, safety_reason = assess_safety(
        text,
        decision_history.prompt_history,
        intent_decision.intent,
        safety_client,
        risk_history=decision_history.risk_scan_history,
    )
    safety_override = assessment.risk == RiskLevel.HIGH
    final_intent = IntentType.RISK if safety_override else intent_decision.intent
    return RouteSafetyDecision(
        understanding_intent=intent_decision.intent,
        final_intent=final_intent,
        assessment=assessment,
        safety_override=safety_override,
        intent_reason=intent_decision.reason,
        safety_reason=safety_reason,
        history_participated=bool(decision_history.risk_scan_history),
    )


def unresolved_history_risk_kind(history: list[AiMessage]) -> str | None:
    """Replay ordered user-message risk events and return the unresolved state."""
    state = _history_risk_state(history)
    return state if state in {"self", "third_party"} else None


def _history_risk_state(history: list[AiMessage]) -> str | None:
    """Return the ordered self/third-party/resolved state for user messages."""
    return risk_state_after_texts(
        message.content or ""
        for message in history
        if message.role.lower() == "user"
    )
