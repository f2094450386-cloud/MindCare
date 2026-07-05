"""
MindBridge Pydantic 数据传输对象（DTO）定义

定义 API 请求和响应的数据结构。
所有 DTO 使用 camelCase 别名，与前端 JavaScript 命名习惯一致。

DTO 分组：
- 聊天相关：ChatRequest、ChatStreamEvent
- 知识库相关：KnowledgeIngestRequest、KnowledgeIngestResponse
- 报告相关：ReportResponse
- 会话相关：ConversationMessageResponse、ConversationResponse
- 工具记录：ToolRecordResponse、ToolJobResponse、DeadLetterResponse
- 个案管理：RiskCaseResponse、CaseNoteResponse
- 审计追踪：AgentRunTraceResponse、ToolAuditResponse
- AI 内部：AiMessage（LLM 对话消息格式）
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """学生聊天请求。message 为必填，sessionId 可选（为空时新建会话）。"""
    message: str = Field(min_length=1)
    sessionId: Optional[str] = None


class ChatStreamEvent(BaseModel):
    """
    SSE 流式事件结构。

    type 字段区分事件类型：
    - "meta": 元数据事件，包含 sessionId
    - "token": 流式文本片段，包含 content
    - "done": 流式结束标记
    """
    sessionId: Optional[str] = None
    content: Optional[str] = None
    message: Optional[str] = None
    type: str


class KnowledgeIngestRequest(BaseModel):
    """知识库入库请求。source 为来源标识，content 为原始文本。"""
    source: str
    content: str


class KnowledgeIngestResponse(BaseModel):
    """知识库入库响应。返回来源标识和切块数量。"""
    source: str
    chunks: int


class ReportResponse(BaseModel):
    """
    心理评估报告响应。

    仅管理员可查看。包含完整的评估信息：
    - intent: 意图类型
    - emotion: 情绪标签
    - emotionScore: 情绪分数
    - riskLevel: 风险等级
    - confidence: 置信度
    - summary: 评估摘要
    """
    id: int
    sessionId: str
    username: str
    displayName: str
    content: str
    intent: str
    emotion: str
    emotionScore: float
    riskLevel: str
    confidence: float
    summary: str
    createdAt: datetime


class ConversationMessageResponse(BaseModel):
    """会话中的单条消息响应。"""
    role: str
    content: str
    createdAt: datetime


class ConversationResponse(BaseModel):
    """完整会话响应，包含会话标题和所有消息。"""
    sessionId: str
    title: str
    messages: list[ConversationMessageResponse]


class ToolRecordResponse(BaseModel):
    """
    工具记录响应（Excel/Alert 通用）。

    channel 和 recipient 仅 AlertRecord 有值。
    filePath 仅 ExcelRecord 有值。
    """
    id: int
    reportId: int
    status: str
    message: str
    createdAt: datetime
    channel: Optional[str] = None
    recipient: Optional[str] = None
    filePath: Optional[str] = None


class RiskCaseResponse(BaseModel):
    """
    风险个案响应。

    handoffSummary: 由 counselor_handoff_summary skill 生成的交接摘要，
    包含学生信息、风险评估和建议的下一步行动。
    """
    id: int
    reportId: int
    riskLevel: str
    status: str
    owner: str
    summary: str
    handoffSummary: str
    acknowledgedBy: Optional[str] = None
    acknowledgedAt: Optional[datetime] = None
    createdAt: datetime
    updatedAt: datetime


class CaseNoteResponse(BaseModel):
    """个案备注响应。"""
    id: int
    caseId: int
    actor: str
    note: str
    createdAt: datetime


class ToolJobResponse(BaseModel):
    """
    异步工具任务响应。

    dependsOnJobId: 依赖的前置任务 ID（ALERT_SEND 依赖 CASE_CREATE）
    runAfter: 最早可执行时间（用于延迟重试）
    lastError: 最近一次失败原因
    """
    id: int
    reportId: int
    kind: str
    status: str
    attempts: int
    maxAttempts: int
    dependsOnJobId: Optional[int] = None
    runAfter: datetime
    lastError: str
    createdAt: datetime
    updatedAt: datetime


class DeadLetterResponse(BaseModel):
    """死信记录响应。任务超过最大重试次数后转入死信。"""
    id: int
    jobId: Optional[int] = None
    reportId: int
    kind: str
    reason: str
    payload: str
    createdAt: datetime


class AgentRunTraceResponse(BaseModel):
    """
    Agent 运行轨迹响应。

    仅管理员可查看。记录一轮 Agent 运行的完整执行过程，
    包含原始输入、脱敏输入、记忆摘要、Agent 步骤、检索结果和评估结论，
    用于审计和问题排查。
    """
    id: int
    sessionId: str
    reportId: Optional[int] = None
    username: str
    intent: str
    riskLevel: str
    originalInput: str
    sanitizedInput: str
    memoryBrief: str
    agentSteps: list[dict[str, Any]]
    retrievedKnowledge: list[dict[str, Any]]
    responseMessages: list[dict[str, Any]]
    assessment: dict[str, Any]
    createdAt: datetime


class ToolAuditResponse(BaseModel):
    """
    工具治理审计响应。

    记录 ToolGovernanceService 对每次工具调用的授权判定，
    allowed=False 表示该次调用被策略拦截。
    """
    id: int
    jobId: Optional[int] = None
    reportId: Optional[int] = None
    toolName: str
    policy: str
    allowed: bool
    status: str
    reason: str
    payload: dict[str, Any]
    createdAt: datetime
    updatedAt: datetime


class AiMessage(BaseModel):
    """
    LLM 对话消息格式。

    用于构造发送给 Ollama/OpenAI 的 messages 数组。
    role: "system" / "user" / "assistant"
    content: 消息文本
    """
    role: str
    content: str


def authority(role: str) -> dict[str, Any]:
    """将角色字符串包装为前端 authority 格式。"""
    return {"authority": role}
