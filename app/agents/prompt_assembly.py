"""Shared online ResponseAgent prompt assembly."""
from __future__ import annotations

from app.core.enums import IntentType, RiskLevel
from app.schemas.dtos import AiMessage
from app.services.ai import PromptTemplates


def build_response_messages(
    *,
    intent: IntentType,
    risk: RiskLevel,
    display_name: str,
    model_history: list[AiMessage],
    memory_brief: str,
    response_agent_system_prompt: str,
    private_memory_text: str = "无",
    knowledge_context: str = "",
    skill_context: str = "",
) -> tuple[list[AiMessage], str]:
    """Build the exact normal-chat/support prompt consumed by ResponseAgent."""
    if intent == IntentType.CHAT and risk == RiskLevel.LOW:
        effective_intent = IntentType.CHAT
        mode = "normal_chat"
        mode_description = "当前由 ResponseAgent 以 normal_chat mode 提出回复方案。"
    else:
        effective_intent = intent if intent != IntentType.CHAT else IntentType.CONSULT
        mode = "support"
        mode_description = "当前由 ResponseAgent 以 support mode 提出回复方案。"
    return (
        [
            PromptTemplates.answer_system_prompt(
                effective_intent,
                risk,
                knowledge_context,
                display_name,
                skill_context,
            ),
            AiMessage(
                role="system",
                content=(
                    f"{response_agent_system_prompt}\n"
                    f"{mode_description}\n"
                    f"私有记忆：\n{private_memory_text or '无'}\n"
                    f"记忆摘要：\n{memory_brief or '无相关历史记忆。'}"
                ),
            ),
            *model_history,
        ],
        mode,
    )
