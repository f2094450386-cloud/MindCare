"""
MindBridge 聊天服务模块

实现学生端 SSE 流式聊天的核心流程：
1. 调用 MindBridgeAgentHarness.run() 执行 Agent 工作流
2. 发送 meta 事件（包含 sessionId）
3. 流式调用 LLM，逐 token 发送 token 事件
4. 保存助手消息到数据库和 Redis
5. 异步触发工具后处理（Excel/个案/预警）
6. 发送 done 事件

SSE 事件格式：
- event: meta  → {"sessionId": "...", "type": "meta"}
- event: token → {"content": "...", "type": "token"}
- event: done  → {"sessionId": "...", "type": "done"}

工具后处理不阻塞流式回复：
- 工具队列启用时：写入 tool_jobs 队列异步执行
- 工具队列关闭时：通过 MCP client 同步调用（备用方案）
"""
from __future__ import annotations

import json
import logging

from sqlalchemy.orm import Session

from app.agents.harness import MindBridgeAgentHarness
from app.core.config import Settings
from app.models.entities import UserAccount
from app.schemas.dtos import ChatRequest, ChatStreamEvent
from app.services.ai import AiClient


logger = logging.getLogger(__name__)


class ChatService:
    """
    聊天服务。

    负责协调 Agent 工作流、LLM 流式调用和工具后处理。
    """

    def __init__(self, db: Session, settings: Settings):
        self.db = db
        self.settings = settings
        self.ai = AiClient(settings)
        self.agent_harness = MindBridgeAgentHarness(db, settings)

    async def stream_chat(self, user: UserAccount, request: ChatRequest):
        """
        SSE 流式聊天主流程。

        使用 async generator 逐块返回 SSE 格式的事件文本。
        FastAPI StreamingResponse 会将这些块推送给客户端。

        流程：
        1. Agent harness 执行完整的多 Agent 工作流
        2. 发送 meta 事件
        3. 流式调用 LLM，逐 token 发送
        4. 保存助手消息
        5. 触发工具后处理（不阻塞）
        6. 发送 done 事件
        """
        # 执行 Agent 工作流（获取意图、风险评估、回复 prompt 等）
        outcome = self.agent_harness.run(user, request)

        # 发送元数据事件
        yield sse("meta", ChatStreamEvent(type="meta", sessionId=outcome.session.public_id).model_dump(by_alias=True))

        # 流式调用 LLM
        assistant = []
        async for token in self.ai.stream(outcome.response_messages):
            assistant.append(token)
            yield sse("token", ChatStreamEvent(type="token", sessionId=outcome.session.public_id, content=token).model_dump())

        # 保存助手消息
        if assistant:
            self.agent_harness.save_assistant_message(user, outcome.session, "".join(assistant))

        # 异步触发工具后处理（不阻塞 SSE 流）
        try:
            await self.agent_harness.dispatch_tools(outcome.tool_plan)
        except Exception as exc:
            logger.warning(
                "Post-response tool dispatch failed for session=%s report_id=%s: %s",
                outcome.session.public_id,
                outcome.report_id,
                exc,
                exc_info=True,
            )

        # 发送完成事件
        yield sse("done", ChatStreamEvent(type="done", sessionId=outcome.session.public_id).model_dump())


def sse(event: str, data: dict) -> str:
    """
    将事件数据格式化为 SSE 文本格式。

    格式：event: {event}\ndata: {json}\n\n
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
