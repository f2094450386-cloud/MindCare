"""
MindBridge Agent 运行时工厂模块

根据配置选择 Agent 运行时实现：
- agent_framework="event_driven_multi_agent" → EventDrivenAgentRuntimeService
- agent_framework="langgraph" 且 langgraph 已安装 → LangGraphAgentRuntimeService
- 否则 → AgentRuntimeService（自研有限循环 runtime）

LangGraph 实现使用有向图编排多 Agent，支持条件分支。
自研 runtime 使用简单的 for 循环 + 标志位控制 Agent 执行顺序。
"""
from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

from app.core.config import Settings

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from app.agents.runtime import AgentRuntimeService


def create_agent_runtime(db: "Session", settings: Settings) -> "AgentRuntimeService":
    """
    创建 Agent 运行时实例。

    默认使用事件驱动多 Agent Runtime，并保留 LangGraph/custom 回退。
    """
    if wants_event_driven(settings):
        from app.agents.event_driven_runtime import EventDrivenAgentRuntimeService

        return EventDrivenAgentRuntimeService(db, settings)
    if wants_langgraph(settings) and langgraph_available():
        from app.agents.langgraph_runtime import LangGraphAgentRuntimeService

        return LangGraphAgentRuntimeService(db, settings)
    from app.agents.runtime import AgentRuntimeService

    return AgentRuntimeService(db, settings)


def agent_framework_status(settings: Settings) -> dict:
    """
    返回 Agent 框架状态信息（用于 /api/agent/status）。

    包含：请求的框架、实际激活的框架、LangGraph 是否可用、是否发生了回退。
    """
    requested = settings.agent_framework.lower()
    available = langgraph_available()
    if wants_event_driven(settings):
        active = "event_driven_multi_agent"
    elif requested == "langgraph" and available:
        active = "langgraph"
    else:
        active = "custom"
    requested_active = "event_driven_multi_agent" if wants_event_driven(settings) else requested
    return {
        "requested": requested,
        "active": active,
        "langgraphAvailable": available,
        "fallback": active != requested_active,
    }


def wants_event_driven(settings: Settings) -> bool:
    """检查配置是否请求事件驱动多 Agent Runtime。"""
    return settings.agent_framework.lower() in {"event_driven_multi_agent", "multi_agent", "actors"}


def wants_langgraph(settings: Settings) -> bool:
    """检查配置是否请求使用 LangGraph。"""
    return settings.agent_framework.lower() == "langgraph"


def langgraph_available() -> bool:
    """检查 langgraph 包是否已安装。"""
    return find_spec("langgraph") is not None
