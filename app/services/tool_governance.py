"""
MindBridge 工具治理模块

在工具后处理任务（Excel 写入、个案创建、预警发送）执行前，
校验该任务是否被允许对当前风险等级的报告执行，并落库审计记录。

策略表（ToolPolicyRegistry.POLICIES）：
- EXCEL_REPORT: 所有风险等级（LOW/MEDIUM/HIGH）均允许
- CASE_CREATE:  仅 MEDIUM/HIGH 风险允许
- ALERT_SEND:   仅 HIGH 风险允许
- RISK_ALERT:   仅 HIGH 风险允许（兼容旧版工具类型）

审计记录（ToolAuditRecord）由 ToolGovernanceService 写入，
记录每次授权判定的结果（allowed/blocked）和原因，供后台查询。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from app.core.enums import RiskLevel, ToolJobKind
from app.models.entities import PsychologicalReport, ToolAuditRecord, ToolJob


@dataclass(frozen=True)
class ToolPolicy:
    """单个工具的治理策略：允许哪些风险等级调用、是否要求关联报告。"""
    name: str
    description: str
    allowed_risks: tuple[str, ...]
    requires_report: bool = True


class ToolPolicyRegistry:
    """工具治理策略表：定义每个工具允许在哪些风险等级下执行。"""

    POLICIES: dict[str, ToolPolicy] = {
        ToolJobKind.EXCEL_REPORT.value: ToolPolicy(
            name=ToolJobKind.EXCEL_REPORT.value,
            description="将心理报告写入面向辅导员的 Excel 台账。",
            allowed_risks=(RiskLevel.LOW.value, RiskLevel.MEDIUM.value, RiskLevel.HIGH.value),
        ),
        ToolJobKind.CASE_CREATE.value: ToolPolicy(
            name=ToolJobKind.CASE_CREATE.value,
            description="为中/高风险报告创建或复用辅导员可见的风险个案。",
            allowed_risks=(RiskLevel.MEDIUM.value, RiskLevel.HIGH.value),
        ),
        ToolJobKind.ALERT_SEND.value: ToolPolicy(
            name=ToolJobKind.ALERT_SEND.value,
            description="为高风险报告发送或记录紧急辅导员预警。",
            allowed_risks=(RiskLevel.HIGH.value,),
        ),
        ToolJobKind.RISK_ALERT.value: ToolPolicy(
            name=ToolJobKind.RISK_ALERT.value,
            description="旧版高风险预警动作，仅为兼容保留。",
            allowed_risks=(RiskLevel.HIGH.value,),
        ),
    }

    @classmethod
    def policy_for(cls, tool_name: str) -> ToolPolicy | None:
        """按工具名查找策略，不存在时返回 None。"""
        return cls.POLICIES.get(tool_name)

    @classmethod
    def authorize(cls, tool_name: str, report: PsychologicalReport | None) -> tuple[bool, str, ToolPolicy | None]:
        """
        判断某次工具调用是否被允许。

        返回 (allowed, reason, policy)：
        - 工具未知 → 拒绝
        - 策略要求关联报告但未提供 → 拒绝
        - 报告风险等级不在允许范围内 → 拒绝
        - 其余情况 → 允许
        """
        policy = cls.policy_for(tool_name)
        if policy is None:
            return False, f"未知工具：{tool_name}", None
        if policy.requires_report and report is None:
            return False, "工具执行需要心理报告，但未找到 report", policy
        risk = report.risk_level if report is not None else ""
        if risk not in policy.allowed_risks:
            return False, f"工具 {tool_name} 不允许处理风险等级 {risk}", policy
        return True, "允许执行", policy


class ToolGovernanceService:
    """工具治理服务：在任务执行前后落库审计记录。"""

    def __init__(self, db: Session):
        self.db = db

    def start_job(self, job: ToolJob, report: PsychologicalReport | None) -> ToolAuditRecord:
        """
        在任务开始执行前进行授权判定，并落库一条 ToolAuditRecord。

        无论授权是否通过都会落库，allowed=False 表示该次调用被策略拦截。
        """
        allowed, reason, policy = ToolPolicyRegistry.authorize(job.kind, report)
        record = ToolAuditRecord(
            job_id=job.id,
            report_id=job.report_id,
            tool_name=job.kind,
            policy=policy.name if policy else "unknown",
            allowed=allowed,
            status="AUTHORIZED" if allowed else "BLOCKED",
            reason=reason,
            payload=_json(
                {
                    "jobId": job.id,
                    "kind": job.kind,
                    "attempts": job.attempts,
                    "riskLevel": report.risk_level if report is not None else None,
                    "policy": asdict(policy) if policy else None,
                }
            ),
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def require_allowed(self, job: ToolJob, report: PsychologicalReport | None) -> None:
        """校验任务是否被允许执行，不允许时抛出异常。"""
        allowed, reason, _ = ToolPolicyRegistry.authorize(job.kind, report)
        if not allowed:
            raise RuntimeError(reason)

    def finish(self, record: ToolAuditRecord, status: str, reason: str = "", payload: dict[str, Any] | None = None) -> ToolAuditRecord:
        """更新审计记录的最终状态（如任务执行完成或失败）。"""
        record.status = status
        record.reason = reason or record.reason
        if payload is not None:
            record.payload = _json(payload)
        record.updated_at = datetime.utcnow()
        self.db.add(record)
        self.db.commit()
        return record


def _json(value: Any) -> str:
    """将任意值序列化为 JSON 字符串。"""
    return json.dumps(value, ensure_ascii=False, default=str)
