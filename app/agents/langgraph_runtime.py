"""
MindBridge LangGraph Agent 运行时模块

使用 LangGraph 构建多 Agent 有向图工作流，替代自研 runtime 的有限循环。

图结构（controller 循环模式）：
  controller → [路由到下一个待执行 Agent] → memory/supervisor/knowledge/
               risk_guardian/companion/counselor → controller → ... → END

节点说明：
- controller: 空转节点，仅作为条件路由的锚点，不修改状态
- memory: 加载短期记忆（Redis/MySQL）
- supervisor: 意图分类，决定是否需要 knowledge/risk_guardian
- knowledge: RAG 知识检索（CONSULT/RISK 意图）
- risk_guardian: 心理风险评估（CONSULT/RISK 意图）
- companion: 普通陪伴回复（CHAT 意图）
- counselor: 心理咨询回复（CONSULT/RISK 意图）

路由逻辑（_select_next_agent）：
每次回到 controller 后，根据 AgentContext 的标志位（memory_loaded/
intent_routed/knowledge_handled/risk_assessed/response_planned/finished）
决定下一个要执行的 Agent，直到 response_planned 或达到 max_steps → "end"。
这与自研 runtime（AgentRuntimeService.run 的 for 循环）遵循同一套路由规则，
只是用 LangGraph 有向图 + 条件边表达，而不是 Python for 循环。

优势：
- 图结构清晰，分支逻辑显式
- 支持可视化和调试
- 易于扩展新的 Agent 节点
"""
from __future__ import annotations

from typing import TypedDict

from sqlalchemy.orm import Session

from app.agents.runtime import AgentContext, AgentRunResult, AgentRuntimeService
from app.core.config import Settings
from app.core.enums import IntentType
from app.models.entities import ChatSession, UserAccount


class GraphState(TypedDict):
    """LangGraph 状态类型定义。"""
    context: AgentContext


class LangGraphAgentRuntimeService(AgentRuntimeService):
    """
    LangGraph 实现的 MindBridge 多 Agent 工作流。

    继承 AgentRuntimeService，复用所有 Agent 方法实现。
    只是将执行顺序从有限循环改为 LangGraph 有向图。
    """

    framework_name = "langgraph"

    def __init__(self, db: Session, settings: Settings):
        super().__init__(db, settings)
        self.graph = self._build_graph()

    def run(self, user: UserAccount, session: ChatSession, original_input: str, model_input: str) -> AgentRunResult:
        """
        通过 LangGraph 图执行 Agent 工作流。

        将 AgentContext 包装为 GraphState，调用图编译后的 invoke 方法。
        recursion_limit 按 max_steps 放大，为 controller 循环留出往返裕量
        （每一轮业务 Agent 执行都会经过 controller 节点一次）。
        """
        context = AgentContext(user=user, session=session, original_input=original_input, model_input=model_input)
        graph_limit = self.max_steps * 3 + 2
        state = self.graph.invoke({"context": context}, {"recursion_limit": graph_limit})
        result_context = state["context"]
        return AgentRunResult(
            intent=result_context.intent or IntentType.CHAT,
            risk_level=result_context.risk_level,
            assessment=result_context.assessment,
            retrieved_knowledge=result_context.retrieved_knowledge,
            response_messages=result_context.response_messages,
            steps=result_context.steps,
            memory_brief=result_context.memory_brief,
        )

    def _build_graph(self):
        """
        构建 LangGraph 有向图（controller 循环模式）。

        所有业务 Agent 节点执行完毕后都回到 controller，
        由 _select_next_agent 决定下一步走向，直到返回 "end"。
        这与自研 runtime 的 for 循环在路由语义上完全一致。
        """
        from langgraph.graph import END, StateGraph

        graph = StateGraph(GraphState)

        # 添加节点：controller 是循环路由锚点，其余为业务 Agent 节点
        graph.add_node("controller", self._controller_node)
        graph.add_node("memory", self._memory_node)
        graph.add_node("supervisor", self._supervisor_node)
        graph.add_node("knowledge", self._knowledge_node)
        graph.add_node("risk_guardian", self._risk_guardian_node)
        graph.add_node("companion", self._companion_node)
        graph.add_node("counselor", self._counselor_node)

        # 入口和条件路由：每个业务节点执行完都回到 controller 重新路由
        graph.set_entry_point("controller")
        graph.add_conditional_edges(
            "controller",
            self._select_next_agent,
            {
                "memory": "memory",
                "supervisor": "supervisor",
                "knowledge": "knowledge",
                "risk_guardian": "risk_guardian",
                "companion": "companion",
                "counselor": "counselor",
                "end": END,
            },
        )
        graph.add_edge("memory", "controller")
        graph.add_edge("supervisor", "controller")
        graph.add_edge("knowledge", "controller")
        graph.add_edge("risk_guardian", "controller")
        graph.add_edge("companion", "controller")
        graph.add_edge("counselor", "controller")
        return graph.compile()

    # ── 图节点包装 ────────────────────────────────────────────────
    # 每个节点调用父类的 Agent 方法，保持逻辑一致性

    def _controller_node(self, state: GraphState) -> GraphState:
        """controller 空转节点：不修改状态，仅作为条件路由的锚点。"""
        return state

    def _memory_node(self, state: GraphState) -> GraphState:
        self._run_agent(state, self.memory_agent)
        return state

    def _supervisor_node(self, state: GraphState) -> GraphState:
        self._run_agent(state, self.supervisor_agent)
        return state

    def _knowledge_node(self, state: GraphState) -> GraphState:
        self._run_agent(state, self.knowledge_agent)
        return state

    def _risk_guardian_node(self, state: GraphState) -> GraphState:
        self._run_agent(state, self.risk_guardian_agent)
        return state

    def _companion_node(self, state: GraphState) -> GraphState:
        self._run_agent(state, self.companion_agent)
        return state

    def _counselor_node(self, state: GraphState) -> GraphState:
        self._run_agent(state, self.counselor_agent)
        return state

    def _run_agent(self, state: GraphState, agent) -> None:
        """
        执行单个 Agent，并在无法继续推进时提前结束。

        - 已 finished 或达到 max_steps → 标记 finished，交给路由返回 "end"
        - Agent 返回 False 且没有新增步骤（说明本轮无 Agent 可执行）→ 标记 finished，
          防止条件不满足时在 controller 和某个节点之间死循环
        """
        context = state["context"]
        if context.finished or len(context.steps) >= self.max_steps:
            context.finished = True
            return
        before = len(context.steps)
        ran = agent(before + 1, context)
        if not ran and len(context.steps) == before:
            context.finished = True

    def _select_next_agent(self, state: GraphState) -> str:
        """
        controller 的条件路由：根据 AgentContext 标志位决定下一个 Agent。

        路由顺序与自研 runtime 的 Agent 列表顺序一致：
        memory → supervisor → (companion | knowledge → risk_guardian → counselor) → end
        """
        context = state["context"]
        if context.finished or len(context.steps) >= self.max_steps:
            return "end"
        if not context.memory_loaded:
            return "memory"
        if not context.intent_routed:
            return "supervisor"
        if context.intent == IntentType.CHAT:
            return "companion" if not context.response_planned else "end"
        if not context.knowledge_handled:
            return "knowledge"
        if not context.risk_assessed:
            return "risk_guardian"
        if not context.response_planned:
            return "counselor"
        return "end"
