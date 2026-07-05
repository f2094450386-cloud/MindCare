"""
MindBridge AI 客户端与 Prompt 模板模块

本模块是整个系统的 AI 推理核心，负责：
1. PromptTemplates: 构造发送给 LLM 的各类 prompt
2. AiClient: 统一的 AI 调用客户端（支持 Ollama/OpenAI/Mock）
3. 风险关键词检测：高风险词典和咨询词典的快速匹配

AiClient 支持三种 provider：
- ollama: 本地 Ollama 服务，使用微调 GGUF 模型
- openai: OpenAI-compatible API（GPT-4o-mini 等）
- mock: 离线模拟，根据关键词返回预设响应，用于开发测试

Prompt 设计要点：
- 意图分类 prompt: 只输出 CHAT/CONSULT/RISK，不做回答
- 心理评估 prompt: 输出严格 JSON，包含情绪、风险、置信度
- 回复系统 prompt: 根据意图和风险等级动态组装，包含知识上下文和 skill 指引
- 高风险规则: 先回应情绪，再关注安全，鼓励联系可信任的人
"""
from __future__ import annotations

import json
from typing import Iterable

import httpx

from app.core.config import Settings
from app.core.enums import IntentType, RiskLevel
from app.schemas.dtos import AiMessage


class PromptTemplates:
    """
    LLM Prompt 模板工厂。

    所有 prompt 都返回 AiMessage 列表，直接作为 LLM 的 messages 参数。
    """

    @staticmethod
    def intent_prompt(history: list[AiMessage], user_input: str) -> list[AiMessage]:
        """
        意图分类 prompt。

        让 LLM 判断用户输入属于 CHAT/CONSULT/RISK 中的哪一个。
        系统提示明确限定输出格式，不做任何回答。
        """
        return [
            AiMessage(role="system", content=(
                "你是一个用户意图分类器，只做意图识别，不回答问题。"
                "只输出 CHAT、CONSULT、RISK 之一。CHAT 包含普通闲聊、学习、编程、作业、校园事务；"
                "CONSULT 包含压力、焦虑、低落、失眠、情绪倾诉；RISK 包含自杀、自残、伤人或即时危险信号。"
            )),
            AiMessage(role="user", content=f"最近上下文：\n{format_history(history)}\n\n当前输入：\n{user_input}"),
        ]

    @staticmethod
    def psychology_prompt(history: list[AiMessage], user_input: str) -> list[AiMessage]:
        """
        心理评估 prompt。

        让 LLM 输出严格 JSON 格式的评估结果：
        - emotion: 情绪标签（NORMAL/ANXIETY/DEPRESSED/HIGH_RISK）
        - emotionScore: 情绪分数（0.0-4.0）
        - risk: 风险等级（LOW/MEDIUM/HIGH）
        - confidence: 置信度（0.0-1.0）
        - summary: 简短评估原因
        """
        return [
            AiMessage(role="system", content=(
                "你负责分析校园心理健康消息。只返回严格 JSON："
                '{"emotion":"NORMAL|ANXIETY|DEPRESSED|HIGH_RISK","emotionScore":0.0,'
                '"risk":"LOW|MEDIUM|HIGH","confidence":0.0,"summary":"short reason"}'
            )),
            AiMessage(role="user", content=f"最近上下文：\n{format_history(history)}\n\n当前输入：\n{user_input}"),
        ]

    @staticmethod
    def answer_system_prompt(intent: IntentType, risk: RiskLevel, context: str, display_name: str, skill_context: str = "") -> AiMessage:
        """
        回复系统 prompt（根据意图和风险等级动态组装）。

        CHAT 意图：普通助手角色，不做心理测评
        CONSULT/RISK 意图：心理关怀角色，包含：
          - 基础行为规则（共情、非评判、不诊断）
          - 学生显示名
          - RAG 检索到的知识上下文
          - Skill 指引（如焦虑 grounding、睡眠建议等）
          - 高风险处理规则（仅 HIGH 风险时注入）

        参数：
        - intent: 对话意图
        - risk: 风险等级
        - context: RAG 检索到的知识文本
        - display_name: 学生显示名
        - skill_context: 动态选择的 skill 内容
        """
        if intent == IntentType.CHAT:
            content = (
                "你是 MindBridge，一个面向学生的日常陪伴与校园生活助手。"
                "普通学习、编程、校园事务和通用问题请自然、准确、直接地回答。"
                "不要主动做心理测评，不要输出风险等级、心理标签、诊断结论或报告口吻。"
                f"学生显示名：{display_name}"
            )
            return AiMessage(role="system", content=content)

        # 高风险时注入危机处理规则
        crisis_rule = ""
        if risk == RiskLevel.HIGH:
            crisis_rule = (
                "\n高风险处理规则：先回应情绪，再关注当前安全；鼓励用户立刻联系身边可信任的人、"
                "学校辅导员/心理中心或当地紧急救助；不提供任何危险操作细节。"
            )
        content = (
            "你是 MindBridge，一个面向学生的校园心理关怀智能体。"
            "回答要共情、谨慎、非评判，不诊断疾病，不开药，不替代持证心理咨询师。"
            "不要向学生输出风险等级、报告分数或后台标签。"
            "优先基于检索知识回答；知识不足时明确说明并给出安全通用建议。"
            f"\n学生显示名：{display_name}\n检索知识：\n{context}\n\n可用 skill 指引：\n{skill_context or '无'}{crisis_rule}"
        )
        return AiMessage(role="system", content=content)


class AiClient:
    """
    统一的 AI 调用客户端。

    根据 settings.ai_provider 自动选择后端：
    - ollama: 调用本地 Ollama /api/chat 接口
    - openai: 调用 OpenAI-compatible /chat/completions 接口
    - mock:   根据关键词返回预设响应

    提供两种调用方式：
    - complete(): 同步一次性返回完整结果
    - stream(): 异步生成器，逐 token 流式返回（用于 SSE）
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def complete(self, messages: list[AiMessage]) -> str:
        """同步调用 LLM，返回完整文本结果。"""
        provider = self.settings.ai_provider.lower()
        if provider == "ollama":
            return self._ollama(messages, stream=False)
        if provider == "openai":
            return self._openai(messages, stream=False)
        return self._mock(messages)

    async def stream(self, messages: list[AiMessage]):
        """
        异步流式调用 LLM，逐 token 返回。

        mock 模式下将完整文本按 12 字符一组切分模拟流式输出。
        """
        provider = self.settings.ai_provider.lower()
        if provider == "ollama":
            async for token in self._ollama_stream(messages):
                yield token
            return
        if provider == "openai":
            async for token in self._openai_stream(messages):
                yield token
            return
        text = self._mock(messages)
        for chunk in split_text(text, 12):
            yield chunk

    def _ollama(self, messages: list[AiMessage], stream: bool) -> str:
        """调用本地 Ollama /api/chat 接口（同步）。"""
        payload = {
            "model": self.settings.ollama_model,
            "messages": [m.model_dump() for m in messages],
            "stream": stream,
            "options": {"temperature": self.settings.ai_temperature, "num_predict": self.settings.ai_max_tokens},
        }
        response = httpx.post(f"{self.settings.ollama_base_url}/api/chat", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["message"]["content"]

    async def _ollama_stream(self, messages: list[AiMessage]):
        """
        调用本地 Ollama /api/chat 接口（异步流式）。

        Ollama 流式返回 NDJSON 格式，每行一个 JSON 对象。
        从每个 JSON 中提取 message.content 字段作为 token。
        """
        payload = {
            "model": self.settings.ollama_model,
            "messages": [m.model_dump() for m in messages],
            "stream": True,
            "options": {"temperature": self.settings.ai_temperature, "num_predict": self.settings.ai_max_tokens},
        }
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream("POST", f"{self.settings.ollama_base_url}/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token

    def _openai(self, messages: list[AiMessage], stream: bool) -> str:
        """调用 OpenAI-compatible /chat/completions 接口（同步）。"""
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        payload = {
            "model": self.settings.openai_model,
            "messages": [m.model_dump() for m in messages],
            "temperature": self.settings.ai_temperature,
            "max_tokens": self.settings.ai_max_tokens,
            "stream": stream,
        }
        response = httpx.post(f"{self.settings.openai_base_url}/chat/completions", headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    async def _openai_stream(self, messages: list[AiMessage]):
        """
        调用 OpenAI-compatible /chat/completions 接口（异步流式）。

        OpenAI 流式返回 SSE 格式：
        - 每行以 "data: " 开头
        - 数据为 JSON 对象
        - 最后一行 "data: [DONE]" 表示流式结束
        - token 在 choices[0].delta.content 中
        """
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        payload = {
            "model": self.settings.openai_model,
            "messages": [m.model_dump() for m in messages],
            "temperature": self.settings.ai_temperature,
            "max_tokens": self.settings.ai_max_tokens,
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream("POST", f"{self.settings.openai_base_url}/chat/completions", headers=headers, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line.removeprefix("data: ").strip()
                    if raw == "[DONE]":
                        break
                    data = json.loads(raw)
                    token = data["choices"][0].get("delta", {}).get("content", "")
                    if token:
                        yield token

    def _mock(self, messages: list[AiMessage]) -> str:
        """
        离线模拟 LLM 响应。

        根据系统提示中的关键词和用户输入，返回预设的模拟响应。
        用于开发测试，无需真实 AI 服务。

        判断逻辑：
        1. 系统提示含"严格 JSON" → 心理评估 prompt → 返回模拟 JSON
        2. 系统提示含"意图分类器" → 意图分类 prompt → 返回 CHAT/CONSULT/RISK
        3. 系统提示含"high_risk_safety_plan" → 高危安全计划响应
        4. 系统提示含"CounselorAgent" → 心理咨询响应
        5. 系统提示含"CompanionAgent" → 普通陪伴响应
        6. 系统提示含"KnowledgeAgent" → 知识查询改写
        7. 兜底：通用心理关怀响应
        """
        last = next((m.content for m in reversed(messages) if m.role == "user"), "")
        system = " ".join(m.content for m in messages if m.role == "system")

        # 心理评估 mock
        if "严格 JSON" in system:
            if has_high_risk_signal(last):
                return '{"emotion":"HIGH_RISK","emotionScore":4.0,"risk":"HIGH","confidence":0.95,"summary":"检测到明确高风险表达"}'
            if has_consult_signal(last):
                return '{"emotion":"ANXIETY","emotionScore":2.5,"risk":"LOW","confidence":0.72,"summary":"检测到压力或情绪求助表达"}'
            return '{"emotion":"NORMAL","emotionScore":0.0,"risk":"LOW","confidence":0.66,"summary":"未检测到明显风险信号"}'

        # 意图分类 mock
        if "意图分类器" in system:
            if has_high_risk_signal(last):
                return "RISK"
            if has_consult_signal(last):
                return "CONSULT"
            return "CHAT"

        # 高危安全计划 mock
        if "high_risk_safety_plan" in system and has_high_risk_signal(last):
            return "我听到你现在已经痛苦到觉得撑不下去了。现在最重要的是先让你不要一个人扛：请马上联系身边可信任的人，或者直接联系辅导员、学校心理中心、校园保卫/当地紧急服务。接下来 10 分钟，请先把自己移到有人在的地方，并把可能伤害自己的东西放远一点。如果可以，回我一句：你现在身边有没有可以马上联系或走过去找的人？"

        # 心理咨询 mock
        if "当前由 CounselorAgent" in system:
            return "我听到你最近压力很大，还影响到了睡眠，这种状态确实会让人很消耗。你可以先做两件小事：今晚把最担心的事情写成清单，先只选一个最小步骤处理；睡前 30 分钟把手机和学习任务放远一点，用缓慢呼吸或热水澡帮身体降下来。如果这种失眠持续一周以上，建议联系学校心理中心或辅导员一起看一看。"

        # 普通陪伴 mock
        if "当前由 CompanionAgent" in system:
            return "我在。这个问题可以直接拆开来看，我们先从你最想解决的那一部分开始。"

        # 知识查询改写 mock
        if "KnowledgeAgent" in system and "SUFFICIENT" in system:
            return "SUFFICIENT"
        if "KnowledgeAgent" in system:
            return last[:40] or "校园心理支持"

        # 兜底响应
        return "我在。先把你现在最具体的困扰说出来，我们可以一步一步拆开。如果情况已经影响安全，请马上联系身边可信任的人或学校心理中心。"


def format_history(history: list[AiMessage]) -> str:
    """将对话历史格式化为可读文本，最多取最近 20 条。"""
    if not history:
        return "无"
    return "\n".join(f"{m.role}: {m.content}" for m in history[-20:])


# ── 风险关键词词典 ────────────────────────────────────────────────
# 高风险词典：匹配到任意一个即判定为高风险信号
HIGH_RISK_WORDS = ["自杀", "自残", "不想活", "结束生命", "伤害自己", "轻生", "suicide", "kill myself", "self harm"]

# 咨询词典：匹配到任意一个即判定为心理咨询相关
CONSULT_WORDS = ["焦虑", "抑郁", "压力", "失眠", "难过", "崩溃", "痛苦", "无助", "心理", "咨询", "anxious", "depress", "stress"]


def has_high_risk_signal(text: str) -> bool:
    """
    检测文本是否包含高风险关键词。

    这是风险评估的第一道防线（硬兜底）。
    即使 LLM 评估为低风险，只要包含高风险关键词，
    RiskGuardianAgent 也会强制提升为 HIGH 风险。
    """
    normalized = text.lower()
    return any(word in normalized for word in HIGH_RISK_WORDS)


def has_consult_signal(text: str) -> bool:
    """检测文本是否包含心理咨询相关关键词。"""
    normalized = text.lower()
    return any(word in normalized for word in CONSULT_WORDS)


def split_text(text: str, size: int) -> Iterable[str]:
    """将文本按指定大小切分为片段，用于 mock 模式模拟流式输出。"""
    for index in range(0, len(text), size):
        yield text[index:index + size]
