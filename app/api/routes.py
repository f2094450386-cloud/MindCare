"""
MindBridge API 路由定义

定义所有 HTTP 接口，分为学生端和管理员端。

学生端接口（需要 ROLE_USER）：
- GET  /actuator/health          → 健康检查
- GET  /api/profile              → 当前用户信息
- POST /api/chat/stream          → SSE 流式聊天（仅学生可用）
- GET  /api/agent/status         → Agent 状态信息
- GET  /api/reports/me           → 我的心理报告

管理员端接口（需要 ROLE_ADMIN）：
- GET  /api/admin/reports        → 所有心理报告
- GET  /api/admin/excel-records  → Excel 台账记录
- GET  /api/admin/alerts         → 预警记录
- GET  /api/admin/cases          → 风险个案列表
- GET  /api/admin/cases/{id}/notes → 个案备注
- GET  /api/admin/tool-jobs      → 工具任务列表
- GET  /api/admin/dead-letters   → 死信记录
- GET  /api/admin/agent-traces   → Agent 运行轨迹（审计）
- GET  /api/admin/tool-audits    → 工具治理审计记录
- GET  /api/admin/conversations/{id} → 完整会话
- POST /api/admin/knowledge      → 入库知识文本
- GET  /api/admin/knowledge/status → 知识库状态
- POST /api/admin/knowledge/rebuild-vector → 重建向量索引
- POST /api/admin/knowledge/backup → 向量库快照
- POST /api/admin/knowledge/file → 上传文件入库

认证方式：HTTP Basic Auth
- 学生：student / student123
- 管理员：admin / admin123
"""
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.agents.factory import agent_framework_status
from app.agents.runtime import AgentRuntimeService
from app.core.config import get_settings
from app.core.database import get_db
from app.core.security import current_user, require_admin
from app.models.entities import UserAccount
from app.schemas.dtos import KnowledgeIngestRequest, KnowledgeIngestResponse, ChatRequest, authority
from app.services.chat import ChatService
from app.services.knowledge import KnowledgeService
from app.services.model_assets import finetuned_model_status
from app.services.report import ReportService
from app.services.skills import MindBridgeSkillLibrary

router = APIRouter()


# ── 公共接口 ──────────────────────────────────────────────────────

@router.get("/actuator/health")
def health():
    """健康检查端点。"""
    return {"status": "UP"}


@router.get("/api/profile")
def profile(user: Annotated[UserAccount, Depends(current_user)]):
    """返回当前登录用户的基本信息和角色。"""
    return {
        "id": user.id,
        "username": user.username,
        "displayName": user.display_name,
        "roles": [authority(role) for role in user.roles],
    }


@router.post("/api/chat/stream")
async def chat_stream(
    request: ChatRequest,
    user: Annotated[UserAccount, Depends(current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    """
    学生端 SSE 流式聊天接口。

    管理员禁止发起学生对话。
    返回 text/event-stream 格式的 SSE 流。
    """
    if "ROLE_ADMIN" in user.roles:
        raise HTTPException(403, "管理员账号只能查看后台记录，不能发起学生对话。")
    service = ChatService(db, get_settings())
    return StreamingResponse(service.stream_chat(user, request), media_type="text/event-stream")


@router.get("/api/agent/status")
def agent_status(user: Annotated[UserAccount, Depends(current_user)]):
    """
    返回 Agent 系统状态信息。

    包含：AI provider、模型信息、Agent 框架状态、Skill 列表、微调模型资产状态。
    """
    settings = get_settings()
    provider = settings.ai_provider.lower()
    model = settings.ollama_model if provider == "ollama" else settings.openai_model if provider == "openai" else "mock"
    framework = agent_framework_status(settings)
    return {
        "provider": provider,
        "model": model,
        "realModelEnabled": provider in {"ollama", "openai"},
        "agentFramework": framework,
        "finetunedModel": finetuned_model_status(settings),
        "agents": [
            {"name": "MemoryAgent", "status": "READY", "description": "短期上下文与长期记忆摘要"},
            {"name": "SupervisorAgent", "status": "READY", "description": "意图识别与路由"},
            {"name": "KnowledgeAgent", "status": "READY", "description": "RAG 检索与证据补充"},
            {"name": "RiskGuardianAgent", "status": "READY", "description": "心理风险识别与分级"},
            {"name": "CompanionAgent", "status": "READY", "description": "普通陪伴式回复"},
            {"name": "CounselorAgent", "status": "READY", "description": "咨询式支持回复"},
        ],
        "skills": MindBridgeSkillLibrary.status_items(),
        "runtimeHarness": {
            "name": "MindBridgeAgentHarness",
            "status": "READY",
            "description": "统一管理单轮 Agent run 的输入脱敏、上下文注入、风险报告、工具计划和 trace 输出",
        },
        "loop": {
            "type": "bounded-agent-loop",
            "maxSteps": AgentRuntimeService.max_steps,
            "scheduler": "langgraph-controller" if framework["active"] == "langgraph" else "custom-runtime",
        },
    }


@router.get("/api/reports/me")
def my_reports(user: Annotated[UserAccount, Depends(current_user)], db: Annotated[Session, Depends(get_db)]):
    """学生查看自己的心理报告。"""
    return ReportService(db).latest_reports(user.id)


# ── 管理员接口 ────────────────────────────────────────────────────

@router.get("/api/admin/reports")
def admin_reports(_: Annotated[UserAccount, Depends(require_admin)], db: Annotated[Session, Depends(get_db)]):
    """管理员查看所有心理报告。"""
    return ReportService(db).latest_reports()


@router.get("/api/admin/excel-records")
def admin_excel(_: Annotated[UserAccount, Depends(require_admin)], db: Annotated[Session, Depends(get_db)]):
    """管理员查看 Excel 台账写入记录。"""
    return ReportService(db).excel_records()


@router.get("/api/admin/alerts")
def admin_alerts(_: Annotated[UserAccount, Depends(require_admin)], db: Annotated[Session, Depends(get_db)]):
    """管理员查看预警发送记录。"""
    return ReportService(db).alert_records()


@router.get("/api/admin/cases")
def admin_cases(_: Annotated[UserAccount, Depends(require_admin)], db: Annotated[Session, Depends(get_db)]):
    """管理员查看风险个案列表。"""
    return ReportService(db).risk_cases()


@router.get("/api/admin/cases/{case_id}/notes")
def admin_case_notes(case_id: int, _: Annotated[UserAccount, Depends(require_admin)], db: Annotated[Session, Depends(get_db)]):
    """管理员查看指定个案的跟进备注。"""
    return ReportService(db).case_notes(case_id)


@router.get("/api/admin/tool-jobs")
def admin_tool_jobs(_: Annotated[UserAccount, Depends(require_admin)], db: Annotated[Session, Depends(get_db)]):
    """管理员查看异步工具任务列表。"""
    return ReportService(db).tool_jobs()


@router.get("/api/admin/dead-letters")
def admin_dead_letters(_: Annotated[UserAccount, Depends(require_admin)], db: Annotated[Session, Depends(get_db)]):
    """管理员查看死信记录。"""
    return ReportService(db).dead_letters()


@router.get("/api/admin/agent-traces")
def admin_agent_traces(_: Annotated[UserAccount, Depends(require_admin)], db: Annotated[Session, Depends(get_db)]):
    """管理员查看 Agent 运行轨迹（审计用）。"""
    return ReportService(db).agent_run_traces()


@router.get("/api/admin/tool-audits")
def admin_tool_audits(_: Annotated[UserAccount, Depends(require_admin)], db: Annotated[Session, Depends(get_db)]):
    """管理员查看工具治理审计记录。"""
    return ReportService(db).tool_audits()


@router.get("/api/admin/conversations/{session_id}")
def admin_conversation(session_id: str, _: Annotated[UserAccount, Depends(require_admin)], db: Annotated[Session, Depends(get_db)]):
    """管理员查看完整会话消息。"""
    try:
        return ReportService(db).conversation(session_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/api/admin/knowledge")
def ingest_knowledge(
    request: KnowledgeIngestRequest,
    _: Annotated[UserAccount, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
):
    """管理员手动入库知识文本。"""
    chunks = KnowledgeService(db, get_settings()).ingest(request.source, request.content)
    return KnowledgeIngestResponse(source=request.source, chunks=chunks)


@router.get("/api/admin/knowledge/status")
def knowledge_status(_: Annotated[UserAccount, Depends(require_admin)], db: Annotated[Session, Depends(get_db)]):
    """管理员查看知识库状态。"""
    return KnowledgeService(db, get_settings()).status()


@router.post("/api/admin/knowledge/rebuild-vector")
def rebuild_knowledge_vector(_: Annotated[UserAccount, Depends(require_admin)], db: Annotated[Session, Depends(get_db)]):
    """管理员重建向量索引。"""
    try:
        indexed = KnowledgeService(db, get_settings()).rebuild_vector_index()
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"indexedChunks": indexed}


@router.post("/api/admin/knowledge/backup")
def backup_knowledge_vector(_: Annotated[UserAccount, Depends(require_admin)], db: Annotated[Session, Depends(get_db)]):
    """管理员创建向量库快照备份。"""
    try:
        snapshot = KnowledgeService(db, get_settings()).backup_vector_index()
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"snapshot": snapshot}


@router.post("/api/admin/knowledge/file")
async def ingest_file(
    _: Annotated[UserAccount, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
    file: UploadFile = File(...),
):
    """管理员上传文件（Markdown/PDF）入库。"""
    chunks = KnowledgeService(db, get_settings()).ingest_file(file.filename or "uploaded-file", await file.read())
    return KnowledgeIngestResponse(source=file.filename or "uploaded-file", chunks=chunks)
