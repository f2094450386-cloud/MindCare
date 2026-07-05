"""
MindBridge Agent 运行轨迹落库模块

将一轮 Agent 运行的完整执行过程（原始输入、脱敏输入、记忆摘要、
Agent 步骤、检索结果、发送给 LLM 的消息、评估结论）序列化落库为
一条 AgentRunTrace 记录，供管理员审计和问题排查。

与 PsychologicalReport 的区别：
- PsychologicalReport 只在 CONSULT/RISK 意图下生成，记录评估结论
- AgentRunTrace 每轮 Agent 运行都生成，记录完整执行过程

调用时机：由 MindBridgeAgentHarness.run() 在每轮 Agent 运行结束后调用。
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

from sqlalchemy.orm import Session

from app.agents.runtime import AgentRunResult
from app.models.entities import AgentRunTrace, ChatSession, UserAccount


class AgentTraceService:
    """Agent 运行轨迹落库服务。"""

    def __init__(self, db: Session):
        self.db = db

    def save_run(
        self,
        user: UserAccount,
        session: ChatSession,
        original_input: str,
        sanitized_input: str,
        memory_brief: str,
        agent_run: AgentRunResult,
        report_id: int | None,
    ) -> AgentRunTrace:
        """
        将一轮 Agent 运行结果落库为 AgentRunTrace 记录。

        agent_steps / retrieved_knowledge / response_messages / assessment
        均序列化为 JSON 字符串存储，查询时（ReportService.agent_run_traces）
        再反序列化为结构化数据返回。
        """
        trace = AgentRunTrace(
            user_id=user.id,
            session_id=session.id,
            report_id=report_id,
            intent=agent_run.intent.value,
            risk_level=agent_run.risk_level.value,
            original_input=original_input,
            sanitized_input=sanitized_input,
            memory_brief=memory_brief,
            agent_steps_json=_json(agent_run.steps),
            retrieved_knowledge_json=_json(agent_run.retrieved_knowledge),
            response_messages_json=_json(agent_run.response_messages),
            assessment_json=_json(agent_run.assessment or {}),
        )
        self.db.add(trace)
        self.db.commit()
        self.db.refresh(trace)
        return trace


def _json(value: Any) -> str:
    """将任意值序列化为 JSON 字符串（含 dataclass/Enum/pydantic 模型的兼容处理）。"""
    return json.dumps(_to_jsonable(value), ensure_ascii=False, default=str)


def _to_jsonable(value: Any) -> Any:
    """递归地将 dataclass/Enum/pydantic 模型/容器转换为可 JSON 序列化的结构。"""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump())
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return value
