"""
MindBridge MCP 工具服务

通过 MCP (Model Context Protocol) 协议暴露心理报告后处理工具。
使用 FastMCP 框架，支持 stdio 传输。

暴露工具：
- mindbridge_excel_report: 写入 Excel 台账
- mindbridge_case_create: 创建风险个案
- mindbridge_alert_send: 发送预警通知
- mindbridge_alert_ack: 确认接手个案
- mindbridge_case_note_add: 添加个案备注
- mindbridge_alert_notify: 发送预警邮件并记录

使用方式：
  python -m app.mcp_tools.server

注意：默认情况下系统使用异步工具队列（tool_queue）处理后处理任务，
MCP server 仅在 tool_queue_enabled=false 时作为备用方案。
MCP Python 包建议使用 Python 3.10 或 3.11。
"""
from app.core.bootstrap import create_schema
from app.core.config import get_settings
from app.core.database import SessionLocal
from app.models.entities import PsychologicalReport
from app.services.tools import ToolOrchestrationService

try:
    from mcp.server.fastmcp import FastMCP
except Exception as exc:  # pragma: no cover
    raise RuntimeError("请先安装 requirements.txt 中的 mcp 依赖") from exc


# 创建 MCP server 实例
mcp = FastMCP("mindbridge-python-tools")


@mcp.tool()
def mindbridge_excel_report(report_id: int) -> str:
    """Write one psychological risk report into the MindBridge Excel ledger."""
    create_schema()
    db = SessionLocal()
    try:
        report = db.get(PsychologicalReport, report_id)
        if report is None:
            return f"report {report_id} not found"
        record = ToolOrchestrationService(db, get_settings()).write_excel(report)
        return f"success: {record.file_path}"
    finally:
        db.close()


@mcp.tool()
def mindbridge_case_create(report_id: int) -> str:
    """Create or return the active MindBridge risk case for one psychological report."""
    create_schema()
    db = SessionLocal()
    try:
        report = db.get(PsychologicalReport, report_id)
        if report is None:
            return f"report {report_id} not found"
        case = ToolOrchestrationService(db, get_settings()).create_case(report)
        return f"success: caseId={case.id}, reportId={case.report_id}, status={case.status}"
    finally:
        db.close()


@mcp.tool()
def mindbridge_alert_send(case_id: int) -> str:
    """Send or record the counselor alert for one MindBridge risk case."""
    create_schema()
    db = SessionLocal()
    try:
        from app.models.entities import RiskCase

        case = db.get(RiskCase, case_id)
        if case is None:
            return f"case {case_id} not found"
        record = ToolOrchestrationService(db, get_settings()).send_case_alert(case)
        return f"{record.status}: caseId={case_id}, {record.channel} -> {record.recipient}: {record.message}"
    finally:
        db.close()


@mcp.tool()
def mindbridge_alert_ack(case_id: int, actor: str, note: str = "") -> str:
    """Mark a MindBridge risk case as acknowledged by a counselor or administrator."""
    create_schema()
    db = SessionLocal()
    try:
        case = ToolOrchestrationService(db, get_settings()).acknowledge_case(case_id, actor, note)
        return f"success: caseId={case.id}, status={case.status}, acknowledgedBy={case.acknowledged_by}"
    except RuntimeError as exc:
        return str(exc)
    finally:
        db.close()


@mcp.tool()
def mindbridge_case_note_add(case_id: int, actor: str, note: str) -> str:
    """Append a follow-up note to a MindBridge risk case."""
    create_schema()
    db = SessionLocal()
    try:
        record = ToolOrchestrationService(db, get_settings()).add_case_note(case_id, actor, note)
        return f"success: noteId={record.id}, caseId={record.case_id}"
    except RuntimeError as exc:
        return str(exc)
    finally:
        db.close()


@mcp.tool()
def mindbridge_alert_notify(report_id: int) -> str:
    """Send a high-risk alert email and record the notification result for one psychological report."""
    create_schema()
    db = SessionLocal()
    try:
        report = db.get(PsychologicalReport, report_id)
        if report is None:
            return f"report {report_id} not found"
        record = ToolOrchestrationService(db, get_settings()).notify(report)
        return f"{record.status}: {record.channel} -> {record.recipient}: {record.message}"
    finally:
        db.close()


if __name__ == "__main__":
    mcp.run()
