"""
MindBridge 业务枚举定义

定义系统中所有状态、类型和标签的枚举值。
所有枚举都继承 str + Enum，可以直接用于 JSON 序列化和数据库存储。

枚举分组：
- 消息相关：MessageRole（消息角色）
- 意图路由：IntentType（对话意图三分类）
- 风险评估：RiskLevel（风险等级）、EmotionLabel（情绪标签）
- 工具状态：ToolStatus（工具执行结果）、ToolJobKind（工具任务类型）、ToolJobStatus（任务生命周期）
- 个案管理：RiskCaseStatus（风险个案状态）
"""
from enum import Enum


class MessageRole(str, Enum):
    """消息角色：用户、助手、系统。"""
    USER = "USER"
    ASSISTANT = "ASSISTANT"
    SYSTEM = "SYSTEM"


class IntentType(str, Enum):
    """
    对话意图三分类（由 SupervisorAgent 决定）。

    CHAT:    普通闲聊、学习、编程、校园事务 → 走 CompanionAgent，不查知识库
    CONSULT: 压力、焦虑、低落、失眠等心理倾诉 → 走 CounselorAgent + RAG
    RISK:    自杀、自残、伤人等高风险信号 → 走 CounselorAgent + RAG + 风险报告
    """
    CHAT = "CHAT"
    CONSULT = "CONSULT"
    RISK = "RISK"


class RiskLevel(str, Enum):
    """
    心理风险等级（由 RiskGuardianAgent 评估）。

    LOW:    无明显风险
    MEDIUM: 中等风险，需要关注
    HIGH:   高风险，需要立即干预和预警
    """
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class EmotionLabel(str, Enum):
    """
    情绪标签（由 RiskGuardianAgent 评估）。

    NORMAL:    正常情绪
    ANXIETY:   焦虑/紧张
    DEPRESSED: 低落/抑郁
    HIGH_RISK: 高危情绪（自杀意念等）
    """
    NORMAL = "NORMAL"
    ANXIETY = "ANXIETY"
    DEPRESSED = "DEPRESSED"
    HIGH_RISK = "HIGH_RISK"


class ToolStatus(str, Enum):
    """工具执行结果状态。"""
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class ToolJobKind(str, Enum):
    """
    异步工具任务类型。

    EXCEL_REPORT: 将心理报告写入 Excel 台账
    CASE_CREATE:  创建风险个案记录
    ALERT_SEND:   发送预警邮件/日志（依赖 CASE_CREATE 完成）
    RISK_ALERT:   风险预警通知（独立于个案创建的备用类型）
    """
    EXCEL_REPORT = "EXCEL_REPORT"
    CASE_CREATE = "CASE_CREATE"
    ALERT_SEND = "ALERT_SEND"
    RISK_ALERT = "RISK_ALERT"


class ToolJobStatus(str, Enum):
    """
    工具任务生命周期状态。

    PENDING: 等待执行（包括等待依赖任务完成）
    RUNNING: 正在执行
    SUCCESS: 执行成功
    DEAD:    超过最大重试次数，进入死信队列
    """
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    DEAD = "DEAD"


class RiskCaseStatus(str, Enum):
    """
    风险个案状态流转。

    OPEN:          已创建，等待辅导员确认
    ALERT_SENT:    预警已发送给辅导员
    ACKNOWLEDGED:  辅导员已确认接手
    """
    OPEN = "OPEN"
    ALERT_SENT = "ALERT_SENT"
    ACKNOWLEDGED = "ACKNOWLEDGED"
