"""
MindBridge 工程 Harness 测试运行器

一键验证核心链路的集成测试框架。
使用 mock AI、临时 SQLite、内存短期记忆，无需外部服务。

测试套件：
1. Risk Safety Harness: 高风险识别、报告生成、工具队列入队
2. Agent Routing Harness: CHAT/CONSULT/RISK 路由和多 Agent 步骤验证
3. Standard Skills Harness: Skill 加载、选择逻辑和交接摘要模板渲染
4. RAG Harness: 基于评测集验证 Recall@K、MRR、NDCG、HitRate
5. API Harness: 健康检查、认证授权、SSE 聊天、管理员接口
6. Tool Queue Harness: Excel/Case/Alert 依赖、幂等、限流和死信

使用方式：
  python3 -m app.harness.runner
  python3 -m app.harness.runner --suite risk --suite routing
  python3 -m app.harness.runner --json

报告输出：
  target/harness/harness-report.json
  target/harness/rag-eval-report.json
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable


class HarnessFailure(AssertionError):
    """Harness 断言失败异常。"""
    pass


@dataclass
class CheckResult:
    """单个测试套件的结果。"""
    name: str
    passed: bool
    details: dict = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)


@dataclass
class HarnessContext:
    """Harness 运行上下文。"""
    root: Path
    target_dir: Path
    settings: object
    database: object

    def session(self):
        return self.database.SessionLocal()


class InMemoryShortTermMemoryStore:
    """
    内存短期记忆存储（替代 Redis）。

    用于 harness 测试，避免依赖外部 Redis 服务。
    所有数据存储在类变量 _messages 中。
    写入前对内容做隐私脱敏，行为与 RedisShortTermMemoryStore 保持一致。
    """
    _messages: dict[str, list[object]] = {}

    def __init__(self, settings):
        self.settings = settings

    def load_recent(self, session_public_id: str) -> list[object]:
        limit = self.settings.redis_memory_max_messages
        return list(self._messages.get(session_public_id, []))[-limit:]

    def messages_from_rows(self, rows: list[object]) -> list[object]:
        from app.schemas.dtos import AiMessage

        return [AiMessage(role=row.role.lower(), content=row.content) for row in rows]

    def append(self, session_public_id: str, role: str, content: str) -> None:
        from app.schemas.dtos import AiMessage
        from app.services.privacy import PrivacySanitizer

        values = self._messages.setdefault(session_public_id, [])
        values.append(AiMessage(role=role.lower(), content=PrivacySanitizer().sanitize(content)))
        del values[:-self.settings.redis_memory_max_messages]

    def replace(self, session_public_id: str, messages: list[object]) -> None:
        from app.schemas.dtos import AiMessage
        from app.services.privacy import PrivacySanitizer

        privacy = PrivacySanitizer()
        self._messages[session_public_id] = [
            AiMessage(role=message.role, content=privacy.sanitize(message.content))
            for message in list(messages)[-self.settings.redis_memory_max_messages:]
        ]

    @classmethod
    def reset(cls) -> None:
        cls._messages.clear()


def main(argv: list[str] | None = None) -> int:
    """Harness 主入口。"""
    parser = argparse.ArgumentParser(description="Run MindBridge engineering harness checks.")
    parser.add_argument(
        "--suite",
        action="append",
        choices=["risk", "routing", "skills", "rag", "api", "tool-queue", "all"],
        default=None,
        help="Harness suite to run. Can be supplied multiple times.",
    )
    parser.add_argument("--json", action="store_true", help="Print only JSON output.")
    args = parser.parse_args(argv)

    configure_environment()
    context = build_context()
    install_harness_patches()
    reset_database(context)

    suites = resolve_suites(args.suite)
    results: list[CheckResult] = []
    for name, fn in suites:
        reset_database(context)
        InMemoryShortTermMemoryStore.reset()
        results.append(run_check(name, fn, context))

    report = write_report(context, results)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)
    return 0 if all(result.passed for result in results) else 1


def configure_environment() -> None:
    """
    配置 harness 运行环境。

    使用临时 SQLite 数据库、mock AI、事件驱动 runtime、禁用向量库和工具队列。
    """
    root = Path(__file__).resolve().parents[2]
    target_dir = root / "target" / "harness"
    target_dir.mkdir(parents=True, exist_ok=True)
    db_path = target_dir / "mindbridge-harness.sqlite3"
    for suffix in ["", "-wal", "-shm"]:
        candidate = Path(f"{db_path}{suffix}")
        if candidate.exists():
            candidate.unlink()

    os.environ["DATABASE_URL"] = f"sqlite:///{db_path.as_posix()}"
    os.environ["AI_PROVIDER"] = "mock"
    os.environ["AGENT_FRAMEWORK"] = "event_driven_multi_agent"
    os.environ["KNOWLEDGE_VECTOR_ENABLED"] = "false"
    os.environ["KNOWLEDGE_VECTOR_REQUIRED"] = "false"
    os.environ["TOOL_QUEUE_ENABLED"] = "false"
    os.environ["ALERT_EMAIL_DELIVERY_MODE"] = "log"
    os.environ["EXCEL_PATH"] = str((target_dir / "mindbridge-risk-ledger.xlsx").as_posix())
    os.environ["RAG_EVAL_OUTPUT"] = str((target_dir / "rag-eval-report.json").as_posix())


def build_context() -> HarnessContext:
    """构建 harness 运行上下文。"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.core.config import get_settings
    import app.core.database as database

    get_settings.cache_clear()
    settings = get_settings()
    if getattr(database, "engine", None) is not None:
        database.engine.dispose()
    database.engine = create_engine(settings.database_url, connect_args={"check_same_thread": False}, pool_pre_ping=True)
    database.SessionLocal = sessionmaker(bind=database.engine, autoflush=False, autocommit=False)
    return HarnessContext(
        root=Path(__file__).resolve().parents[2],
        target_dir=Path(__file__).resolve().parents[2] / "target" / "harness",
        settings=settings,
        database=database,
    )


def install_harness_patches() -> None:
    """
    安装 harness 补丁。

    将 RedisShortTermMemoryStore 替换为 InMemoryShortTermMemoryStore，
    避免测试依赖外部 Redis 服务。
    """
    import app.agents.event_driven_runtime as event_driven_runtime_module
    import app.agents.harness as harness_module
    import app.agents.runtime as runtime_module
    import app.services.memory as memory_module

    event_driven_runtime_module.RedisShortTermMemoryStore = InMemoryShortTermMemoryStore
    harness_module.RedisShortTermMemoryStore = InMemoryShortTermMemoryStore
    runtime_module.RedisShortTermMemoryStore = InMemoryShortTermMemoryStore
    memory_module.RedisShortTermMemoryStore = InMemoryShortTermMemoryStore


def reset_database(context: HarnessContext) -> None:
    """重置数据库：删除所有表 → 重建 → 初始化默认数据。"""
    from app.core.bootstrap import seed_data

    context.database.Base.metadata.drop_all(bind=context.database.engine)
    context.database.Base.metadata.create_all(bind=context.database.engine)
    db = context.session()
    try:
        seed_data(db)
    finally:
        db.close()


def resolve_suites(requested: list[str] | None) -> list[tuple[str, Callable[[HarnessContext], dict]]]:
    """解析要运行的测试套件。"""
    all_suites: list[tuple[str, Callable[[HarnessContext], dict]]] = [
        ("Risk Safety Harness", run_risk_safety_harness),
        ("Agent Routing Harness", run_agent_routing_harness),
        ("Standard Skills Harness", run_standard_skills_harness),
        ("RAG Harness", run_rag_harness),
        ("API Harness", run_api_harness),
        ("Tool Queue Harness", run_tool_queue_harness),
    ]
    if not requested or "all" in requested:
        return all_suites
    selected = set(requested)
    aliases = {
        "risk": "Risk Safety Harness",
        "routing": "Agent Routing Harness",
        "skills": "Standard Skills Harness",
        "rag": "RAG Harness",
        "api": "API Harness",
        "tool-queue": "Tool Queue Harness",
    }
    names = {aliases[item] for item in selected}
    return [suite for suite in all_suites if suite[0] in names]


def run_check(name: str, fn: Callable[[HarnessContext], dict], context: HarnessContext) -> CheckResult:
    """运行单个测试套件。"""
    try:
        return CheckResult(name=name, passed=True, details=fn(context))
    except HarnessFailure as exc:
        return CheckResult(name=name, passed=False, failures=[str(exc)])
    except Exception as exc:
        return CheckResult(
            name=name,
            passed=False,
            failures=[f"{type(exc).__name__}: {exc}", traceback.format_exc()],
        )


# ── 测试套件实现 ──────────────────────────────────────────────────

def run_risk_safety_harness(context: HarnessContext) -> dict:
    """
    Risk Safety Harness: 高风险安全测试。

    验证：
    - 高风险消息（中/英文）→ 生成报告 + 工具队列
    - 咨询消息 → 生成报告但不触发预警
    - 普通消息 → 不生成报告
    - 后台风险元数据不暴露给学生端
    """
    from app.core.enums import RiskLevel, ToolJobKind
    from app.models.entities import PsychologicalReport, ToolJob, UserAccount
    from app.schemas.dtos import ChatRequest
    from app.services.chat import ChatService

    context.settings.tool_queue_enabled = True
    db = context.session()
    observed = []
    try:
        user = db.query(UserAccount).filter(UserAccount.username == "student").one()
        cases = [
            {
                "id": "high-risk-cn",
                "message": "我不想活了，想结束生命。",
                "expects_report": True,
                "expects_risk": RiskLevel.HIGH.value,
                "expects_alert": True,
            },
            {
                "id": "high-risk-en",
                "message": "I want to kill myself and I feel unsafe tonight.",
                "expects_report": True,
                "expects_risk": RiskLevel.HIGH.value,
                "expects_alert": True,
            },
            {
                "id": "consult-low",
                "message": "我最近压力很大，连续几天失眠，想找人聊聊。",
                "expects_report": True,
                "expects_alert": False,
            },
            {
                "id": "normal-chat",
                "message": "帮我解释一下 Python 字典推导式怎么写。",
                "expects_report": False,
                "expects_alert": False,
            },
        ]

        for case in cases:
            service = ChatService(db, context.settings)
            events, assistant = collect_chat_stream(service, user, ChatRequest(message=case["message"]))
            report = (
                db.query(PsychologicalReport)
                .filter(PsychologicalReport.content == case["message"])
                .order_by(PsychologicalReport.id.desc())
                .first()
            )
            token_text = assistant.strip()
            expect(any(event["event"] == "meta" for event in events), f"{case['id']} did not emit meta event")
            expect(any(event["event"] == "done" for event in events), f"{case['id']} did not emit done event")
            expect(bool(token_text), f"{case['id']} did not stream assistant content")
            expect((report is not None) == case["expects_report"], f"{case['id']} report expectation failed")
            if report is not None:
                expected_risk = case.get("expects_risk")
                if expected_risk:
                    expect(report.risk_level == expected_risk, f"{case['id']} expected {expected_risk}, got {report.risk_level}")
                jobs = db.query(ToolJob).filter(ToolJob.report_id == report.id).all()
                has_alert = any(job.kind == ToolJobKind.ALERT_SEND.value for job in jobs)
                expect(has_alert == case["expects_alert"], f"{case['id']} alert job expectation failed")
                expect(
                    any(job.kind == ToolJobKind.EXCEL_REPORT.value for job in jobs),
                    f"{case['id']} did not enqueue Excel report job",
                )
                if case["expects_alert"]:
                    expect(
                        any(job.kind == ToolJobKind.CASE_CREATE.value for job in jobs),
                        f"{case['id']} did not enqueue case creation job",
                    )
            # 验证后台风险元数据不暴露给学生
            forbidden = ["风险等级", "报告ID", "emotionScore", "HIGH_RISK"]
            expect(not any(term in token_text for term in forbidden), f"{case['id']} exposed backend risk metadata")
            observed.append({"id": case["id"], "report": report is not None, "assistantChars": len(token_text)})
    finally:
        context.settings.tool_queue_enabled = False
        db.close()
    return {"cases": observed}


def run_agent_routing_harness(context: HarnessContext) -> dict:
    """
    Agent Routing Harness: Agent 路由测试。

    验证：
    - 默认运行真正的 event_driven_multi_agent，而不是强制 custom
    - CHAT 不运行 ContextAgent/RAG，但仍经过 SafetyAgent 审查
    - CONSULT/RISK 运行 ContextAgent、RAG 和 Skill；RISK 触发 SAFETY_OVERRIDE
    - custom/langgraph 兼容链路仍可运行
    """
    from app.agents.harness import MindBridgeAgentHarness
    from app.core.enums import IntentType, RiskLevel
    from app.models.entities import ChatSession, UserAccount
    from app.schemas.dtos import ChatRequest

    context.settings.agent_framework = "event_driven_multi_agent"
    db = context.session()
    observed = []
    try:
        user = db.query(UserAccount).filter(UserAccount.username == "student").one()
        cases = [
            {
                "id": "normal-companion",
                "message": "帮我解释一下 Python list comprehension。",
                "intent": IntentType.CHAT.value,
                "must_steps": ["UnderstandingAgent", "SafetyAgent", "ResponseAgent", "CoordinatorAgent"],
                "must_not_steps": ["ContextAgent"],
            },
            {
                "id": "consult-counselor",
                "message": "我最近压力很大，睡不着，白天也很焦虑。",
                "intent": IntentType.CONSULT.value,
                "must_steps": ["UnderstandingAgent", "SafetyAgent", "ContextAgent", "ResponseAgent", "CoordinatorAgent"],
            },
            {
                "id": "risk-counselor",
                "message": "我不想活了，觉得撑不下去了。",
                "intent": IntentType.RISK.value,
                "risk": RiskLevel.HIGH.value,
                "must_steps": ["UnderstandingAgent", "SafetyAgent", "ContextAgent", "ResponseAgent", "CoordinatorAgent"],
            },
        ]
        for case in cases:
            session = ChatSession(public_id=uuid.uuid4().hex, user_id=user.id, title=case["id"])
            db.add(session)
            db.commit()
            db.refresh(session)
            result = MindBridgeAgentHarness(db, context.settings).run(
                user,
                ChatRequest(message=case["message"], sessionId=session.public_id),
            )
            step_agents = [step.agent for step in result.agent_steps]
            expect(result.intent.value == case["intent"], f"{case['id']} expected intent {case['intent']}, got {result.intent.value}")
            if "risk" in case:
                expect(result.risk_level == case["risk"], f"{case['id']} expected risk {case['risk']}, got {result.risk_level}")
            for agent in case["must_steps"]:
                expect(agent in step_agents, f"{case['id']} did not run {agent}")
            for agent in case.get("must_not_steps", []):
                expect(agent not in step_agents, f"{case['id']} should not run {agent}")
            actions = [step.action for step in result.agent_steps]
            expect("FINAL_ACCEPTED" in actions, f"{case['id']} response was not accepted")
            if case["id"] == "risk-counselor":
                expect("SAFETY_OVERRIDE" in actions, "risk case did not publish SAFETY_OVERRIDE")
            if case["intent"] != IntentType.CHAT.value:
                expect(len(result.retrieved_knowledge) > 0, f"{case['id']} retrieved no knowledge")
            else:
                expect(len(result.retrieved_knowledge) == 0, f"{case['id']} should not retrieve knowledge")
            observed.append({"id": case["id"], "intent": result.intent.value, "risk": result.risk_level, "steps": step_agents})

        # 回退链路保持同一 AgentRunResult/Harness 契约。
        for framework in ["custom", "langgraph"]:
            context.settings.agent_framework = framework
            session = ChatSession(public_id=uuid.uuid4().hex, user_id=user.id, title=f"fallback-{framework}")
            db.add(session)
            db.commit()
            db.refresh(session)
            result = MindBridgeAgentHarness(db, context.settings).run(
                user,
                ChatRequest(message="帮我解释一下 Python 元组。", sessionId=session.public_id),
            )
            expect(result.intent == IntentType.CHAT, f"{framework} fallback did not preserve CHAT routing")
            expect(bool(result.response_messages), f"{framework} fallback produced no response messages")
            observed.append({"id": f"fallback-{framework}", "intent": result.intent.value})
    finally:
        context.settings.agent_framework = "event_driven_multi_agent"
        db.close()
    return {"cases": observed}


def run_standard_skills_harness(context: HarnessContext) -> dict:
    """
    Standard Skills Harness: Skill 系统测试。

    验证：
    - 所有 7 个标准 Skill 都已加载
    - Skill 选择逻辑正确（关键词触发）
    - 交接摘要模板渲染正确
    """
    from app.core.enums import EmotionLabel, IntentType, RiskLevel
    from app.models.entities import PsychologicalReport, UserAccount
    from app.services.skills import MindBridgeSkillLibrary

    expected = {
        "supportive_response_baseline",
        "high_risk_safety_plan",
        "anxiety_grounding_support",
        "sleep_routine_support",
        "academic_stress_planning",
        "referral_resource_guidance",
        "counselor_handoff_summary",
    }
    skills = MindBridgeSkillLibrary.list_skills()
    names = {skill.name for skill in skills}
    missing = sorted(expected - names)
    expect(not missing, f"missing standard skills: {missing}")

    statuses = MindBridgeSkillLibrary.status_items()
    failed = [item for item in statuses if item["status"] != "READY"]
    expect(not failed, f"standard skill load failures: {failed}")
    expect(all(Path(item["path"]).name == "SKILL.md" for item in statuses), "skill status did not expose SKILL.md paths")

    # 验证 CONSULT 意图的 Skill 选择
    selected_names = MindBridgeSkillLibrary.response_skill_names(
        IntentType.CONSULT,
        RiskLevel.LOW,
        "我最近焦虑、失眠，考试压力也很大。",
    )
    for name in [
        "supportive_response_baseline",
        "referral_resource_guidance",
        "anxiety_grounding_support",
        "sleep_routine_support",
        "academic_stress_planning",
    ]:
        expect(name in selected_names, f"consult response did not select {name}")

    # 验证 Skill 上下文注入
    context_text = MindBridgeSkillLibrary.response_skill_context(
        IntentType.CONSULT,
        RiskLevel.LOW,
        "我最近焦虑、失眠，考试压力也很大。",
    )
    expect("应用 skill: anxiety_grounding_support" in context_text, "response context did not include standard skill body")

    # 验证高风险 Skill 选择
    high_risk_names = MindBridgeSkillLibrary.response_skill_names(
        IntentType.RISK,
        RiskLevel.HIGH,
        "我不想活了。",
    )
    expect(high_risk_names == ["supportive_response_baseline", "high_risk_safety_plan"], "high-risk skill selection changed")

    # 验证交接摘要模板渲染
    report = PsychologicalReport(
        id=7,
        user_id=42,
        session_id=1,
        content="我不想活了，觉得撑不下去。",
        intent=IntentType.RISK.value,
        emotion=EmotionLabel.HIGH_RISK.value,
        emotion_score=4.0,
        risk_level=RiskLevel.HIGH.value,
        confidence=0.95,
        summary="检测到明确高风险表达",
    )
    user = UserAccount(
        id=42,
        username="student",
        display_name="测试学生",
        password_hash="unused",
        roles_csv="ROLE_USER",
    )
    handoff = MindBridgeSkillLibrary.counselor_handoff_summary(report, user)
    for term in ["应用 skill: counselor_handoff_summary", "报告ID：7", "测试学生 (student)", "立即跟进"]:
        expect(term in handoff, f"handoff summary missing {term}")

    return {
        "skills": sorted(names),
        "selectedConsultSkills": selected_names,
        "selectedHighRiskSkills": high_risk_names,
        "handoffChars": len(handoff),
    }


def run_rag_harness(context: HarnessContext) -> dict:
    """
    RAG Harness: 知识检索质量测试。

    验证：
    - 评测集至少 50 个用例
    - HitRate >= 0.95
    - Recall@K >= 0.95
    - MRR >= 0.75
    - NDCG@K >= 0.75
    """
    from app.rag_eval.runner import evaluate_case
    from app.services.knowledge import KnowledgeService

    db = context.session()
    try:
        service = KnowledgeService(db, context.settings)
        dataset_path = context.root / context.settings.rag_eval_dataset
        cases = json.loads(dataset_path.read_text(encoding="utf-8"))
        results = [evaluate_case(service, case, context.settings.knowledge_top_k) for case in cases]
        total = max(1, len(results))
        hits = [item for item in results if item["hit"]]
        metrics = {
            "totalCases": len(results),
            "topK": context.settings.knowledge_top_k,
            "recallAtK": sum(item["recallAtK"] for item in results) / total,
            "precisionAtK": sum(item["precisionAtK"] for item in results) / total,
            "mrr": sum(item["reciprocalRank"] for item in results) / total,
            "ndcgAtK": sum(item["ndcgAtK"] for item in results) / total,
            "hitRate": len(hits) / total,
        }
        expect(metrics["totalCases"] >= 50, f"RAG dataset is too small: {metrics['totalCases']}")
        expect(metrics["hitRate"] >= 0.95, f"RAG hitRate below threshold: {metrics['hitRate']:.3f}")
        expect(metrics["recallAtK"] >= 0.95, f"RAG recallAtK below threshold: {metrics['recallAtK']:.3f}")
        expect(metrics["mrr"] >= 0.75, f"RAG MRR below threshold: {metrics['mrr']:.3f}")
        expect(metrics["ndcgAtK"] >= 0.75, f"RAG NDCG below threshold: {metrics['ndcgAtK']:.3f}")
        report = {"createdAt": datetime.utcnow().isoformat(), "metrics": metrics, "results": results}
        output = context.target_dir / "rag-eval-report.json"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return metrics | {"report": str(output)}
    finally:
        db.close()


def run_api_harness(context: HarnessContext) -> dict:
    """
    API Harness: HTTP 接口测试。

    验证：
    - 健康检查端点
    - 学生认证和 profile
    - Agent 状态接口
    - SSE 聊天流
    - 管理员权限隔离
    - 知识库入库接口
    """
    from fastapi.testclient import TestClient

    from app.main import create_app

    context.settings.tool_queue_enabled = False
    app = create_app()
    student_auth = basic_auth("student", "student123")
    admin_auth = basic_auth("admin", "admin123")
    observed = {}
    with TestClient(app) as client:
        health = client.get("/actuator/health")
        expect(health.status_code == 200 and health.json()["status"] == "UP", "health endpoint failed")
        observed["health"] = health.json()

        profile = client.get("/api/profile", headers=student_auth)
        expect(profile.status_code == 200, f"student profile failed: {profile.status_code}")
        expect(profile.json()["username"] == "student", "student profile returned wrong user")

        agent_status = client.get("/api/agent/status", headers=student_auth)
        expect(agent_status.status_code == 200, f"agent status failed: {agent_status.status_code}")
        status_skills = agent_status.json()["skills"]
        expect(len(status_skills) >= 7, f"agent status exposed too few standard skills: {len(status_skills)}")
        expect(all(Path(skill["path"]).name == "SKILL.md" for skill in status_skills), "agent status did not expose standard skill paths")
        expect(
            agent_status.json()["agentFramework"]["active"] == "event_driven_multi_agent",
            "agent status did not expose event-driven runtime",
        )
        expect(
            "not runtime-enforced" in agent_status.json()["collaboration"]["agentIsolation"]["tools"],
            "agent status overstated tool_permissions enforcement",
        )

        # 管理员禁止发起学生对话
        admin_chat = client.post("/api/chat/stream", headers=admin_auth, json={"message": "hello"})
        expect(admin_chat.status_code == 403, f"admin chat should be forbidden, got {admin_chat.status_code}")

        # 学生聊天流
        chat = client.post("/api/chat/stream", headers=student_auth, json={"message": "帮我解释一下 Python 函数。"})
        expect(chat.status_code == 200, f"student chat stream failed: {chat.status_code}")
        expect("event: meta" in chat.text and "event: done" in chat.text, "chat stream missing meta/done events")
        observed["chatStreamChars"] = len(chat.text)

        # 权限隔离
        student_reports = client.get("/api/admin/reports", headers=student_auth)
        expect(student_reports.status_code == 403, f"student should not read admin reports: {student_reports.status_code}")

        admin_reports = client.get("/api/admin/reports", headers=admin_auth)
        expect(admin_reports.status_code == 200, f"admin reports failed: {admin_reports.status_code}")

        admin_traces = client.get("/api/admin/agent-traces", headers=admin_auth)
        expect(admin_traces.status_code == 200, f"admin agent traces failed: {admin_traces.status_code}")
        expect(bool(admin_traces.json()), "admin agent traces returned no event-driven trace")
        trace_steps = admin_traces.json()[0]["agentSteps"]
        trace_kinds = {item.get("kind") for item in trace_steps if isinstance(item, dict)}
        expect(
            {"agent_event", "agent_task", "agent_artifact"}.issubset(trace_kinds),
            f"event-driven trace omitted collaboration entries: {trace_kinds}",
        )

        admin_audits = client.get("/api/admin/tool-audits", headers=admin_auth)
        expect(admin_audits.status_code == 200, f"admin tool audits failed: {admin_audits.status_code}")
        expect(isinstance(admin_audits.json(), list), "admin tool audits did not return a list")

        # 知识库入库
        ingest = client.post(
            "/api/admin/knowledge",
            headers=admin_auth,
            json={"source": "harness-note", "content": "考试焦虑时可以先做呼吸练习，并联系辅导员获得支持。"},
        )
        expect(ingest.status_code == 200, f"knowledge ingest failed: {ingest.status_code} {ingest.text}")
        expect(ingest.json()["chunks"] >= 1, "knowledge ingest did not create chunks")

        status = client.get("/api/admin/knowledge/status", headers=admin_auth)
        expect(status.status_code == 200, f"knowledge status failed: {status.status_code}")
        expect(status.json()["databaseChunks"] >= 1, "knowledge status returned no chunks")
        observed["knowledgeStatus"] = {
            "databaseChunks": status.json()["databaseChunks"],
            "vectorAvailable": status.json()["vectorAvailable"],
        }
    return observed


def run_tool_queue_harness(context: HarnessContext) -> dict:
    """
    Tool Queue Harness: 工具队列测试。

    验证：
    - 高风险报告创建 3 个任务（Excel + Case + Alert）
    - Alert 任务依赖 Case 任务
    - Excel 写入幂等性
    - 个案创建幂等性
    - 预警发送后个案状态更新
    - 限流器行为
    - 死信记录生成
    """
    from app.core.enums import EmotionLabel, IntentType, RiskCaseStatus, RiskLevel, ToolJobKind, ToolJobStatus, ToolStatus
    from app.models.entities import DeadLetterRecord, PsychologicalReport, ToolAuditRecord, ToolJob, ChatSession, UserAccount
    from app.services.tool_queue import RateLimiter, ToolQueueService, ToolQueueWorker
    from app.services.tools import ToolOrchestrationService

    context.settings.tool_queue_enabled = True
    db = context.session()
    worker = ToolQueueWorker(context.settings)
    try:
        user = db.query(UserAccount).filter(UserAccount.username == "student").one()
        session = ChatSession(public_id=uuid.uuid4().hex, user_id=user.id, title="tool-queue-harness")
        db.add(session)
        db.commit()
        db.refresh(session)
        report = PsychologicalReport(
            user_id=user.id,
            session_id=session.id,
            content="我不想活了，想结束生命。",
            intent=IntentType.RISK.value,
            emotion=EmotionLabel.HIGH_RISK.value,
            emotion_score=4.0,
            risk_level=RiskLevel.HIGH.value,
            confidence=0.95,
            summary="harness high risk case",
        )
        db.add(report)
        db.commit()
        db.refresh(report)

        # 验证任务创建
        jobs = ToolQueueService(db, context.settings).enqueue_report(report.id, report.risk_level)
        expect(len(jobs) == 3, f"expected 3 jobs for high risk report, got {len(jobs)}")
        excel_job = next(job for job in jobs if job.kind == ToolJobKind.EXCEL_REPORT.value)
        case_job = next(job for job in jobs if job.kind == ToolJobKind.CASE_CREATE.value)
        alert_job = next(job for job in jobs if job.kind == ToolJobKind.ALERT_SEND.value)
        expect(alert_job.depends_on_job_id == case_job.id, "alert job does not depend on case creation job")
        expect(not worker._dependency_ready(db, alert_job), "alert dependency should not be ready before case creation success")

        # 通过真实 Worker 路径执行三类任务，并验证每次都产生治理审计。
        for job in [excel_job, case_job, alert_job]:
            if job.kind == ToolJobKind.ALERT_SEND.value:
                db.expire_all()
                alert_job = db.get(ToolJob, alert_job.id)
                expect(worker._dependency_ready(db, alert_job), "alert dependency was not ready after case worker success")
                job = alert_job
            job.status = ToolJobStatus.RUNNING.value
            db.add(job)
            db.commit()
            worker._run_job(job.id)
            db.expire_all()
            finished = db.get(ToolJob, job.id)
            expect(finished.status == ToolJobStatus.SUCCESS.value, f"worker did not complete {job.kind}")

        audits = db.query(ToolAuditRecord).filter(ToolAuditRecord.report_id == report.id).all()
        expect(len(audits) == 3, f"expected 3 tool audit records, got {len(audits)}")
        expect(all(audit.allowed and audit.status == "SUCCESS" for audit in audits), "tool audits did not record successful authorization")

        # 验证幂等性
        tools = ToolOrchestrationService(db, context.settings)
        excel_record = tools.write_excel(report)
        expect(excel_record.status == ToolStatus.SUCCESS.value, f"Excel write failed: {excel_record.message}")
        second_excel_record = tools.write_excel(report)
        expect(second_excel_record.id == excel_record.id, "Excel write is not idempotent")

        case_record = tools.create_case(report)
        second_case_record = tools.create_case(report)
        expect(second_case_record.id == case_record.id, "case creation is not idempotent")

        # 验证依赖就绪
        db.expire_all()
        alert_job = db.get(ToolJob, alert_job.id)
        expect(worker._dependency_ready(db, alert_job), "alert dependency was not ready after case creation success")

        # 验证预警发送
        alert_record = tools.send_case_alert(case_record)
        expect(alert_record.status == ToolStatus.SUCCESS.value, f"alert notify failed: {alert_record.message}")
        db.refresh(case_record)
        expect(case_record.status == RiskCaseStatus.ALERT_SENT.value, "case did not move to ALERT_SENT after alert")

        # 不匹配的任务必须在治理层被拦截，不能执行工具。
        low_report = PsychologicalReport(
            user_id=user.id,
            session_id=session.id,
            content="普通低风险内容",
            intent=IntentType.CONSULT.value,
            emotion=EmotionLabel.NORMAL.value,
            emotion_score=0.0,
            risk_level=RiskLevel.LOW.value,
            confidence=0.9,
            summary="low risk governance case",
        )
        db.add(low_report)
        db.commit()
        db.refresh(low_report)
        blocked_job = ToolJob(
            report_id=low_report.id,
            kind=ToolJobKind.CASE_CREATE.value,
            status=ToolJobStatus.RUNNING.value,
            attempts=0,
            max_attempts=1,
        )
        db.add(blocked_job)
        db.commit()
        db.refresh(blocked_job)
        worker._run_job(blocked_job.id)
        blocked_audit = (
            db.query(ToolAuditRecord)
            .filter(ToolAuditRecord.job_id == blocked_job.id)
            .order_by(ToolAuditRecord.id.desc())
            .first()
        )
        expect(blocked_audit is not None, "blocked worker task produced no audit")
        expect(not blocked_audit.allowed and blocked_audit.status == "BLOCKED", "governance did not block low-risk alert")

        # 验证限流器
        limiter = RateLimiter(1)
        first_allowed, _ = limiter.allow()
        second_allowed, retry_after = limiter.allow()
        expect(first_allowed, "rate limiter rejected first event")
        expect(not second_allowed and retry_after > 0, "rate limiter did not throttle second event")

        # 验证死信
        dead_job = ToolJob(
            report_id=report.id,
            kind=ToolJobKind.EXCEL_REPORT.value,
            status=ToolJobStatus.RUNNING.value,
            attempts=3,
            max_attempts=3,
        )
        db.add(dead_job)
        db.commit()
        db.refresh(dead_job)
        worker._fail_or_dead_letter(db, dead_job.id, RuntimeError("harness failure"))
        db.refresh(dead_job)
        dead_letter = db.query(DeadLetterRecord).filter(DeadLetterRecord.job_id == dead_job.id).first()
        expect(dead_job.status == ToolJobStatus.DEAD.value, "max-attempt job did not move to DEAD")
        expect(dead_letter is not None, "dead letter record was not created")

        return {
            "reportId": report.id,
            "excelJobId": excel_job.id,
            "caseJobId": case_job.id,
            "alertJobId": alert_job.id,
            "caseId": case_record.id,
            "excelPath": excel_record.file_path,
            "deadLetterId": dead_letter.id,
        }
    finally:
        worker.stop()
        context.settings.tool_queue_enabled = False
        db.close()


# ── 辅助函数 ──────────────────────────────────────────────────────

def collect_chat_stream(service, user, request) -> tuple[list[dict], str]:
    """收集 SSE 流式聊天的所有事件和助手文本。"""
    async def collect() -> list[dict]:
        events = []
        async for chunk in service.stream_chat(user, request):
            events.extend(parse_sse(chunk))
        return events

    events = asyncio.run(collect())
    assistant = "".join(event["data"].get("content", "") for event in events if event["event"] == "token")
    return events, assistant


def parse_sse(chunk: str) -> list[dict]:
    """解析 SSE 文本为事件列表。"""
    events = []
    for block in chunk.strip().split("\n\n"):
        if not block:
            continue
        event_name = ""
        data = {}
        for line in block.splitlines():
            if line.startswith("event: "):
                event_name = line.removeprefix("event: ").strip()
            elif line.startswith("data: "):
                data = json.loads(line.removeprefix("data: ").strip())
        events.append({"event": event_name, "data": data})
    return events


def basic_auth(username: str, password: str) -> dict[str, str]:
    """生成 Basic Auth 请求头。"""
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def expect(condition: bool, message: str) -> None:
    """断言辅助函数。"""
    if not condition:
        raise HarnessFailure(message)


def write_report(context: HarnessContext, results: list[CheckResult]) -> dict:
    """写入测试报告。"""
    report = {
        "createdAt": datetime.utcnow().isoformat(),
        "environment": {
            "databaseUrl": context.settings.database_url,
            "aiProvider": context.settings.ai_provider,
            "agentFramework": context.settings.agent_framework,
            "knowledgeVectorEnabled": context.settings.knowledge_vector_enabled,
        },
        "passed": all(result.passed for result in results),
        "results": [
            {
                "name": result.name,
                "passed": result.passed,
                "details": result.details,
                "failures": result.failures,
            }
            for result in results
        ],
    }
    output = context.target_dir / "harness-report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    report["reportPath"] = str(output)
    return report


def print_report(report: dict) -> None:
    """打印测试报告到控制台。"""
    print("MindBridge Engineering Harness")
    print(f"Report: {report['reportPath']}")
    print("")
    for result in report["results"]:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {result['name']}")
        if result["passed"] and result["details"]:
            compact = json.dumps(result["details"], ensure_ascii=False, default=str)
            print(f"       {compact[:900]}")
        for failure in result["failures"]:
            print(f"       {failure}")
    print("")
    print("Overall: PASS" if report["passed"] else "Overall: FAIL")


if __name__ == "__main__":
    sys.exit(main())
