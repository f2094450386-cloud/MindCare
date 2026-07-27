"""
MindBridge 心理风险评估模块

实现三级防线的风险评估策略：
1. 高风险规则硬兜底：结合局部主体、否定和引用语境确认当前风险 → 判定 HIGH_RISK
2. LLM JSON 评估：通过 prompt 让模型输出结构化评估结果
3. 关键词启发式兜底：LLM 调用失败时，基于咨询关键词做简单分类

评估结果包含：
- emotion: 情绪标签（NORMAL/ANXIETY/DEPRESSED/HIGH_RISK）
- emotion_score: 情绪分数（0.0-4.0，越高越严重）
- risk: 风险等级（LOW/MEDIUM/HIGH）
- confidence: 评估置信度（0.0-1.0）
- summary: 评估摘要

风险等级映射：
- emotion_score >= 4.0 → HIGH
- emotion_score >= 3.0 → MEDIUM
- emotion_score < 3.0  → LOW
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from app.core.enums import EmotionLabel, RiskLevel
from app.schemas.dtos import AiMessage
from app.services.ai import (
    AiClient,
    PromptTemplates,
    has_consult_signal,
    has_high_risk_signal,
    has_medium_risk_signal,
)


@dataclass
class PsychologyAssessment:
    """心理评估结果数据类。"""
    emotion: EmotionLabel
    emotion_score: float
    risk: RiskLevel
    confidence: float
    summary: str


class PsychologicalAssessmentService:
    """
    心理评估服务。

    调用 AiClient 进行 LLM 评估，包含三层防御：
    1. 高风险规则快速拦截
    2. LLM JSON 结构化评估
    3. 关键词启发式兜底
    """

    def __init__(self, ai: AiClient):
        self.ai = ai

    def assess(self, text: str, history: list[AiMessage] | None = None) -> PsychologyAssessment:
        """
        执行心理风险评估。

        流程：
        1. 检查经局部语境确认的高风险表达 → 命中则直接返回 HIGH_RISK
        2. 调用 LLM psychology_prompt 获取 JSON 评估
        3. 解析 JSON，校验字段
        4. 如果情绪分数推算的风险等级更高，取更高的
        5. HIGH_RISK 情绪强制设为 HIGH 风险
        6. LLM 调用失败 → 降级到关键词启发式
        """
        # 第一道防线：带主体、否定和引用语境判断的高风险硬兜底
        if has_high_risk_signal(text):
            return PsychologyAssessment(EmotionLabel.HIGH_RISK, 4.0, RiskLevel.HIGH, 0.95, "检测到明确高风险表达")
        if has_medium_risk_signal(text):
            return PsychologyAssessment(
                EmotionLabel.DEPRESSED,
                3.2,
                RiskLevel.MEDIUM,
                0.84,
                "检测到持续或明显影响功能的风险信号",
            )

        try:
            # 第二道防线：LLM JSON 评估
            raw = self.ai.complete(PromptTemplates.psychology_prompt(history or [], text))
            # 从 LLM 响应中提取 JSON（可能包含额外文本）
            start = raw.find("{")
            end = raw.rfind("}")
            data = json.loads(raw[start:end + 1] if start >= 0 and end > start else raw)

            emotion = EmotionLabel(data.get("emotion", "NORMAL").upper())
            score = float(data.get("emotionScore", score_for_emotion(emotion)))
            risk = RiskLevel(data.get("risk", risk_from_score(score).value).upper())
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.75))))

            # 风险等级校正：情绪分数推算的风险等级不能低于 LLM 直接输出的
            score_risk = risk_from_score(score)
            if risk_order(score_risk) > risk_order(risk):
                risk = score_risk
            # HIGH_RISK 情绪强制为 HIGH 风险
            if emotion == EmotionLabel.HIGH_RISK:
                risk = RiskLevel.HIGH

            return PsychologyAssessment(emotion, score, risk, confidence, data.get("summary", "模型评估结果"))
        except Exception:
            # 第三道防线：关键词启发式兜底
            return heuristic(text)


def heuristic(text: str) -> PsychologyAssessment:
    """
    关键词启发式评估（LLM 调用失败时的兜底方案）。

    基于咨询关键词词典做简单分类：
    - 包含"抑郁""低落"等 → DEPRESSED / MEDIUM
    - 包含"焦虑""压力"等 → ANXIETY / LOW
    - 其他 → NORMAL / LOW
    """
    if has_consult_signal(text):
        if any(word in text.lower() for word in ["抑郁", "低落", "崩溃", "难过", "depress", "hopeless"]):
            return PsychologyAssessment(EmotionLabel.DEPRESSED, 3.1, RiskLevel.MEDIUM, 0.75, "检测到低落或抑郁相关表达")
        return PsychologyAssessment(EmotionLabel.ANXIETY, 2.2, RiskLevel.LOW, 0.72, "检测到焦虑或压力相关表达")
    return PsychologyAssessment(EmotionLabel.NORMAL, 0.0, RiskLevel.LOW, 0.66, "未检测到明显风险信号")


def score_for_emotion(emotion: EmotionLabel) -> float:
    """根据情绪标签返回默认情绪分数。"""
    return {
        EmotionLabel.HIGH_RISK: 4.0,
        EmotionLabel.DEPRESSED: 3.0,
        EmotionLabel.ANXIETY: 2.0,
        EmotionLabel.NORMAL: 0.0,
    }[emotion]


def risk_from_score(score: float) -> RiskLevel:
    """根据情绪分数推算风险等级。"""
    if score >= 4:
        return RiskLevel.HIGH
    if score >= 3:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def risk_order(risk: RiskLevel) -> int:
    """返回风险等级的数值排序（用于比较）。"""
    return {RiskLevel.LOW: 1, RiskLevel.MEDIUM: 2, RiskLevel.HIGH: 3}[risk]
