from __future__ import annotations

from typing import TypedDict

from sqlalchemy.orm import Session

from app.agents.runtime import AgentContext, AgentRunResult, AgentRuntimeService
from app.core.config import Settings
from app.core.enums import IntentType
from app.models.entities import ChatSession, UserAccount


class GraphState(TypedDict):
    context: AgentContext


class LangGraphAgentRuntimeService(AgentRuntimeService):
    """LangGraph implementation of the MindBridge multi-agent workflow."""

    framework_name = "langgraph"

    def __init__(self, db: Session, settings: Settings):
        super().__init__(db, settings)
        self.graph = self._build_graph()

    def run(self, user: UserAccount, session: ChatSession, original_input: str, model_input: str) -> AgentRunResult:
        context = AgentContext(user=user, session=session, original_input=original_input, model_input=model_input)
        state = self.graph.invoke({"context": context})
        result_context = state["context"]
        return AgentRunResult(
            intent=result_context.intent or IntentType.CHAT,
            risk_level=result_context.risk_level,
            assessment=result_context.assessment,
            retrieved_knowledge=result_context.retrieved_knowledge,
            response_messages=result_context.response_messages,
            steps=result_context.steps,
        )

    def _build_graph(self):
        from langgraph.graph import END, StateGraph

        graph = StateGraph(GraphState)
        graph.add_node("memory", self._memory_node)
        graph.add_node("supervisor", self._supervisor_node)
        graph.add_node("knowledge", self._knowledge_node)
        graph.add_node("risk_guardian", self._risk_guardian_node)
        graph.add_node("companion", self._companion_node)
        graph.add_node("counselor", self._counselor_node)

        graph.set_entry_point("memory")
        graph.add_edge("memory", "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            self._route_after_supervisor,
            {"chat": "companion", "support": "knowledge"},
        )
        graph.add_edge("knowledge", "risk_guardian")
        graph.add_edge("risk_guardian", "counselor")
        graph.add_edge("companion", END)
        graph.add_edge("counselor", END)
        return graph.compile()

    def _memory_node(self, state: GraphState) -> GraphState:
        self.memory_agent(1, state["context"])
        return state

    def _supervisor_node(self, state: GraphState) -> GraphState:
        self.supervisor_agent(2, state["context"])
        return state

    def _knowledge_node(self, state: GraphState) -> GraphState:
        self.knowledge_agent(3, state["context"])
        return state

    def _risk_guardian_node(self, state: GraphState) -> GraphState:
        self.risk_guardian_agent(4, state["context"])
        return state

    def _companion_node(self, state: GraphState) -> GraphState:
        self.companion_agent(3, state["context"])
        return state

    def _counselor_node(self, state: GraphState) -> GraphState:
        self.counselor_agent(5, state["context"])
        return state

    def _route_after_supervisor(self, state: GraphState) -> str:
        return "chat" if state["context"].intent == IntentType.CHAT else "support"
