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
import re
from dataclasses import dataclass
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
        current = _current_input_from_prompt(last)
        system = " ".join(m.content for m in messages if m.role == "system")

        # 心理评估 mock
        if "严格 JSON" in system:
            if has_high_risk_signal(current):
                return '{"emotion":"HIGH_RISK","emotionScore":4.0,"risk":"HIGH","confidence":0.95,"summary":"检测到明确高风险表达"}'
            if has_medium_risk_signal(current):
                return '{"emotion":"DEPRESSED","emotionScore":3.2,"risk":"MEDIUM","confidence":0.84,"summary":"检测到持续或明显功能受损"}'
            if has_consult_signal(current):
                return '{"emotion":"ANXIETY","emotionScore":2.5,"risk":"LOW","confidence":0.72,"summary":"检测到压力或情绪求助表达"}'
            return '{"emotion":"NORMAL","emotionScore":0.0,"risk":"LOW","confidence":0.66,"summary":"未检测到明显风险信号"}'

        # 意图分类 mock
        if "意图分类器" in system:
            if has_self_high_risk_signal(current):
                return "RISK"
            if has_third_party_immediate_risk(current) or has_consult_signal(current):
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
# 高风险表达词典：候选命中还需结合局部否定、引用语境和表达主体判断。
HIGH_RISK_WORDS = [
    "自杀",
    "自残",
    "不想活",
    "结束生命",
    "伤害自己",
    "轻生",
    "遗书",
    "永久睡过去",
    "一觉不醒",
    "今晚了断",
    "做个了断",
    "suicide",
    "kill myself",
    "self harm",
    "self-harm",
]

# 咨询词典：匹配到任意一个即判定为心理咨询相关
CONSULT_WORDS = ["焦虑", "抑郁", "压力", "失眠", "难过", "崩溃", "痛苦", "无助", "心理", "咨询", "anxious", "depress", "stress"]

MEDIUM_RISK_WORDS = [
    "连续两周无法",
    "连续两周都",
    "几乎吃不下",
    "彻夜失眠",
    "无法上课",
    "不能正常上课",
    "强烈绝望",
    "完全无法学习",
    "多天没吃",
]

NON_LITERAL_RISK_MARKERS = [
    "论文",
    "报告",
    "作业标题",
    "翻译",
    "电影",
    "小说",
    "新闻",
    "课本",
    "课堂",
    "讲座",
    "预防",
    "干预政策",
    "舆情",
    "敏感词",
    "公益海报",
    "宣传资料",
    "量表",
    "统计",
    "引用",
    "志愿者",
    "正式表述",
]

IMMEDIATE_MARKERS = ["现在", "今晚", "刚刚", "又说", "准备", "已经", "正在", "站在", "手里", "联系不上"]

_RISK_WORD_PATTERN = re.compile(
    "|".join(re.escape(word) for word in sorted(HIGH_RISK_WORDS, key=len, reverse=True)),
    re.IGNORECASE,
)
_CLAUSE_BOUNDARY_PATTERN = re.compile(
    r"[。！？!?；;，,\n]+|(?:但现在|但是现在|不过现在|可是现在|但|但是|不过|然而|可是|却|只是|其实|而是)"
)
_THIRD_PERSON_SUBJECT_PATTERN = re.compile(
    r"(?:室友|朋友|同学|舍友|家人|父母|亲属|哥哥|姐姐|弟弟|妹妹|"
    r"主角|角色|有人|别人|(?<!其)他(?!人)|她)"
)
_SELF_INTENT_PATTERN = re.compile(
    r"(?:我(?:现在|今晚|真的|只|好|总|还|又|也|就|一直|仍然|突然|最近|已经|准备|打算|开始|觉得|感觉){0,3})?"
    r"(?:只想|总想|好想|想要|想|要|准备|打算|会|就要)"
    r".{0,3}(?:自杀|自残|伤害自己|结束生命|轻生|不想活|永久睡过去|一觉不醒|做个了断|今晚了断|"
    r"kill myself|self harm|self-harm|suicide)",
    re.IGNORECASE,
)
_DIRECT_SELF_PATTERN = re.compile(
    r"我(?!妈|爸|朋友|室友|同学|舍友|家人)"
    r"(?:现在|今晚|真的|只|好|总|还|又|也|就|一直|仍然|突然|最近|已经|准备|打算|开始|觉得|感觉|很|太|特别|非常|有点|之前|从前|曾经|再|不|想|要|\s){0,8}"
    r"(?:不想活|想死|结束自己的生命|伤害自己|自杀|自残|轻生|遗书|永久睡过去|一觉不醒|"
    r"kill myself|self harm|self-harm|suicide)",
    re.IGNORECASE,
)
_NEGATED_RISK_PATTERN = re.compile(
    r"(?:没有|没想|不想|不是想|并不想|否认(?:自己)?(?:有)?|不再有|现在没有|已经没有)"
    r".{0,8}(?:自杀|自残|伤害自己|轻生|结束生命|不想活|kill myself|self harm|self-harm|suicide)",
    re.IGNORECASE,
)
_META_RISK_PATTERN = re.compile(
    r"(?:讨论|翻译|解释|引用|搜索|过滤|润色|改写|列出|了解|研究|报道|整理)"
    r".{0,24}(?:自杀|自残|自伤|伤害自己|轻生|结束生命|不想活|kill myself|self harm|self-harm|suicide)"
    r"|(?:自杀|自残|自伤|伤害自己|轻生|结束生命|不想活|kill myself|self harm|self-harm|suicide)"
    r".{0,20}(?:综述|统计|量表|政策|流程|提纲|台词|报道|主题|标题|引用|资料|敏感词|倡议书|辟谣|预防|干预|讲座|ppt)"
    r"|(?:论文|报告|作业|课本|课堂|讲座|电影|小说|新闻|海报|宣传资料)"
    r".{0,30}(?:综述|统计|量表|政策|流程|提纲|台词|剧情|阅读笔记|引用|资料|预防|干预)",
    re.IGNORECASE,
)
_REPORTED_SELF_PATTERN = re.compile(
    r"(?:我妈|我爸|父母|家人|亲属|朋友|同学|室友|舍友|老师|辅导员|医生|同事|网友|"
    r"别人|有人|他们|她们|(?<!其)他(?!人)|她|[\u3400-\u9fff]{1,6}(?:老师|医生|辅导员))"
    r".{0,8}(?:说|觉得|认为|声称|怀疑|表示|提到|判断|看起来)"
    r".{0,16}我.{0,8}(?:自杀|自残|伤害自己|轻生|结束生命|不想活)",
    re.IGNORECASE,
)
_SELF_HUMAN_TEXT = r"(?:我本人|我自己|本人|自己|我)"
_THIRD_PARTY_HUMAN_TEXT = (
    r"(?:室友|朋友|同学|舍友|家人|父母|亲属|老师|辅导员|医生|"
    r"家属|哥哥|姐姐|弟弟|妹妹|他|她)"
)
_HUMAN_SUBJECT_TEXT = rf"(?:{_SELF_HUMAN_TEXT}|{_THIRD_PARTY_HUMAN_TEXT})"
_TRUSTED_SUPPORT_TEXT = (
    r"(?:老师|辅导员|心理中心|咨询中心|家人|父母|亲属|室友|朋友|"
    r"同学|舍友|可信任的人|紧急援助|急救人员|警方|保卫处|医院|医生)"
)
_DANGEROUS_ITEM_TEXT = (
    r"(?:药(?:物|片|盒)?|刀(?:具|片|子)?|绳子|危险物品|"
    r"可能伤害自己的东西)"
)
_SPECIFIC_RISK_TEXT = r"(?:自杀|自残|轻生|伤害自己|结束生命)"

_DIRECT_HUMAN_SAFETY_PATTERN = re.compile(
    rf"^(?P<subject>{_HUMAN_SUBJECT_TEXT})"
    r"(?:现在|目前|当前|已经|确实|真的|很){0,2}安全(?:了)?$",
    re.IGNORECASE,
)
_CONFIRMED_HUMAN_SAFETY_PATTERN = re.compile(
    rf"^(?:(?:{_SELF_HUMAN_TEXT})(?:已经|已)?|(?:已经|已))?"
    rf"确认(?P<subject>{_HUMAN_SUBJECT_TEXT})"
    r"(?:现在|目前|当前|已经)?安全(?:了)?$",
    re.IGNORECASE,
)
_SPECIFIC_RISK_DENIAL_PATTERN = re.compile(
    rf"^(?P<subject>{_HUMAN_SUBJECT_TEXT})(?:现在|目前|当前|已经)?"
    rf"(?:没有|并无|不再有|不想|不是想|否认(?:自己)?(?:有)?)"
    rf"(?:任何)?(?P<risks>{_SPECIFIC_RISK_TEXT}"
    rf"(?:(?:或|和|以及|、){_SPECIFIC_RISK_TEXT})*)"
    r"(?:的)?(?:想法|念头|冲动|意图|打算)?(?:了)?$",
    re.IGNORECASE,
)
_ANTECEDENT_RISK_DENIAL_PATTERN = re.compile(
    rf"^(?P<subject>{_HUMAN_SUBJECT_TEXT})(?:现在|目前|当前)?"
    r"(?:没有|不再有)(?:这个|这种|那种|这样的)"
    r"(?:想法|念头|冲动)(?:了)?$",
    re.IGNORECASE,
)
_COMPLETE_SELF_NO_RISK_PATTERN = re.compile(
    rf"^(?P<subject>{_SELF_HUMAN_TEXT})(?:现在|目前|当前)?"
    r"(?:没有|不存在)(?:任何)?风险(?:了)?$",
    re.IGNORECASE,
)
_STANDALONE_CONTEXTUAL_RESOLUTION_PATTERN = re.compile(
    r"^(?:(?:现在|目前)(?:已经)?|已经)没有(?:这些|这种|那个)?"
    r"(?:想法|念头|冲动)?了?$",
    re.IGNORECASE,
)
_DANGEROUS_ITEM_TRANSFER_PATTERN = re.compile(
    rf"^(?:(?:{_SELF_HUMAN_TEXT})(?:已经|已)?把)?"
    rf"(?P<item>{_DANGEROUS_ITEM_TEXT})(?:已经|已)?"
    rf"(?:交给|交由)(?P<support>{_TRUSTED_SUPPORT_TEXT})"
    r"(?:保管|收走|处理)?(?:了)?$"
    rf"|^(?P<item_by>{_DANGEROUS_ITEM_TEXT})(?:已经|已)"
    rf"由(?P<support_by>{_TRUSTED_SUPPORT_TEXT})(?:保管|收走|处理)(?:了)?$",
    re.IGNORECASE,
)
_HUMAN_ACCOMPANIMENT_PATTERN = re.compile(
    rf"^(?:现在|目前|当前)?(?P<support>{_TRUSTED_SUPPORT_TEXT}|有人)"
    r"(?:现在|目前|当前|正在|还在|今晚)?[^，。；]{0,10}"
    rf"(?:陪着|陪同|陪伴)(?P<subject>{_HUMAN_SUBJECT_TEXT})"
    r"(?:下楼|留在这里|待着)?$",
    re.IGNORECASE,
)
_CONTACT_AND_SAFETY_PATTERN = re.compile(
    rf"^(?:(?:{_SELF_HUMAN_TEXT}))?(?:已经|已)?"
    rf"联系(?:上)?(?P<support>{_TRUSTED_SUPPORT_TEXT})"
    r"(?:并|且|而且)(?:已经|已)?(?:确认)?"
    rf"(?P<subject>{_HUMAN_SUBJECT_TEXT})?"
    r"(?:现在|目前|当前)?安全(?:了)?$",
    re.IGNORECASE,
)
_SELF_IMPORTANT_POSSESSION_TRANSFER_PATTERN = re.compile(
    r"(?:^|[。！？；;，,\n“\"‘「『【`])(?P<subject>我|本人)"
    r"(?:现在|已经|都|亲手|正|正在){0,3}(?:把|将)"
    r"[^。！？]{0,16}"
    r"(?:最喜欢(?:的)?|最重要(?:的)?|重要(?:的)?|珍藏(?:的)?|"
    r"珍贵(?:的)?|收藏(?:的)?|有纪念意义(?:的)?)"
    r"[^。！？]{0,16}(?:东西|物品|收藏|纪念品|书|礼物|纪念物|财物)?"
    r"[^。！？]{0,8}(?:分给|送给|留给|赠给|赠出|送人|送走|分掉)",
    re.IGNORECASE,
)
_OMITTED_SELF_POSSESSION_TRANSFER_PATTERN = re.compile(
    r"(?:^|[。！？；;，,“\"‘「『【`])"
    r"(?:最喜欢(?:的)?|最重要(?:的)?|重要(?:的)?物品|珍藏(?:的)?|"
    r"珍贵(?:的)?|收藏(?:的)?|有纪念意义(?:的)?)"
    r"[^。！？]{0,18}(?:分给|送给|留给|赠给|赠出|送人|送走|分掉)",
    re.IGNORECASE,
)
_SELF_VALUED_POSSESSION_DISPOSAL_PATTERN = re.compile(
    r"(?:^|[。！？；;，,\n“\"‘「『【`])(?P<subject>我|本人)"
    r"(?:现在|已经|都|亲手|正|正在){0,3}(?:把|将)"
    r"(?=[^。！？]{0,32}(?:都|全|全部|悉数|清空))"
    r"[^。！？]{0,16}(?:最喜欢(?:的)?|最重要(?:的)?|重要(?:的)?|"
    r"珍藏(?:的)?|珍贵(?:的)?|收藏(?:的)?|有纪念意义(?:的)?)"
    r"[^。！？]{0,18}(?:清空|(?:处理|清理|丢|扔|处置)(?:掉|完|了)?)",
    re.IGNORECASE,
)
_OMITTED_VALUED_POSSESSION_DISPOSAL_PATTERN = re.compile(
    r"(?:^|[。！？；;，,\n“\"‘「『【`])"
    r"(?=[^。！？]{0,28}(?:都|全|全部|悉数|清空))"
    r"(?:最喜欢(?:的)?|最重要(?:的)?|重要(?:的)?物品|珍藏(?:的)?|"
    r"珍贵(?:的)?|收藏(?:的)?|有纪念意义(?:的)?)"
    r"[^。！？]{0,18}(?:清空(?:送走)?|"
    r"(?:处理|清理|丢|扔|处置)(?:掉|完|了)?)",
    re.IGNORECASE,
)
_POSTHUMOUS_HANDOFF_MODIFIERS_TEXT = (
    r"(?:(?:也|已经|已|都|逐项|分别|亲手|正式|妥善(?:地)?|一一|\s)){0,5}"
)
_CLOSE_SUPPORT_PERSON_TEXT = (
    r"(?:家人|家里人|父母|亲人|亲属|伴侣|配偶|爱人|"
    r"哥哥|姐姐|弟弟|妹妹|兄弟|姐妹|朋友|可信任的人)"
)
_ACCESS_CREDENTIAL_TEXT = (
    r"(?:(?:银行卡|银行账户|个人账户|支付账户|网银|账号|设备)"
    r"(?:密码|口令|取用凭据|取用信息|登录凭据|登录信息|访问凭据)?"
    r"|(?:账户|账号)(?:取用|登录|访问)(?:凭据|信息))"
)
_POSTHUMOUS_RECORD_TEXT = (
    r"(?:遗物(?:清单|目录|清册)?|遗嘱|"
    r"身后(?:物品|物件|财物|资料|目录|清单|清册)|"
    r"后事(?:清单|安排|目录)?)"
)
_SELF_COMPLETED_FINAL_HANDOFF_PATTERN = re.compile(
    r"(?:^|[。！？；;，,\n“\"‘「『【`])(?P<subject>我|本人)"
    r"(?:(?:已经|已|都|亲手|正式|逐项|一一|\s)){0,4}(?:把|将)"
    rf"(?P<objects>(?=[^。！？；;\n]{{0,64}}{_ACCESS_CREDENTIAL_TEXT})"
    rf"(?=[^。！？；;\n]{{0,64}}{_POSTHUMOUS_RECORD_TEXT})"
    r"[^。！？；;\n]{1,64}?)"
    r"(?:(?:都|全部|悉数|逐项|一并|一起|分别|也|已经|已|\s)){0,5}"
    rf"(?:留给|交给|托付给|分给|转交给){_CLOSE_SUPPORT_PERSON_TEXT}"
    r"(?:保管|收好|保存)?(?:了)?",
    re.IGNORECASE,
)
_SELF_OWNED_COMPLETED_FINAL_HANDOFF_PATTERN = re.compile(
    r"(?:^|[。！？；;\n“\"‘「『【`])"
    rf"(?=[^。！？；;\n]{{0,120}}(?:我的|属于我的|本人(?:自己)?的)"
    rf"{_ACCESS_CREDENTIAL_TEXT})"
    rf"(?=[^。！？；;\n]{{0,120}}(?:我的|属于我的|本人(?:自己)?的)"
    rf"{_POSTHUMOUS_RECORD_TEXT})"
    rf"(?=[^。！？；;\n]{{0,120}}(?:"
    rf"{_CLOSE_SUPPORT_PERSON_TEXT}[^。！？；;\n]{{0,12}}"
    r"(?:收下|接手|接收|保管|保存)"
    r"|(?:留给|交给|托付给|分给|转交给)"
    rf"(?:{_CLOSE_SUPPORT_PERSON_TEXT}|他|她)[^。！？；;\n]{{0,6}}了))"
    r"(?P<segment>[^。！？；;\n]{1,120})",
    re.IGNORECASE,
)
_SELF_POSTHUMOUS_HANDOFF_PATTERN = re.compile(
    r"(?:^|[。！？；;，,\n“\"‘「『【`])(?:"
    r"(?P<subject>我|本人)(?:已经|已|都)?(?:把|将)"
    r"(?:我(?:自己)?的|自己的)?"
    r"(?:遗物(?:清单|目录)?|遗嘱|身后(?:物品|财物))"
    + _POSTHUMOUS_HANDOFF_MODIFIERS_TEXT
    + r"(?:留给|交给|托付给|分给)"
    r"(?:家人|家里人|父母|亲人|亲属|伴侣|朋友|可信任的人)"
    r"|(?P<possessive>我的|本人自己的|属于我的)"
    r"(?:遗物(?:清单|目录)?|遗嘱|身后(?:物品|财物))"
    + _POSTHUMOUS_HANDOFF_MODIFIERS_TEXT
    + r"(?:留给|交给|托付给|分给)"
    r"(?:家人|家里人|父母|亲人|亲属|伴侣|朋友|可信任的人)"
    r")",
    re.IGNORECASE,
)
_IMPLICIT_POSTHUMOUS_HANDOFF_PATTERN = re.compile(
    r"(?:^|[。！？；;，,\n“\"‘「『【`])"
    r"(?:遗物(?:清单|目录)?|遗嘱|身后(?:物品|财物))"
    + _POSTHUMOUS_HANDOFF_MODIFIERS_TEXT
    + r"(?:留给|交给|托付给|分给)"
    r"(?:家人|家里人|父母|亲人|亲属|伴侣|朋友|可信任的人)"
    r"(?=$|[。！？；;，,\n])",
    re.IGNORECASE,
)
_POSTHUMOUS_AFFAIRS_COMPLETED_PATTERN = re.compile(
    r"(?:^|[。！？；;，,\n“\"‘「『【`])"
    r"(?P<subject>(?:我|本人)(?:已经|已|都)?(?:把|将)?)?"
    r"(?:我(?:自己)?的|自己的|该安排的|该交代的|所有|全部)?"
    r"(?:身后(?:的)?事(?:情)?|身后安排|后事)"
    r"(?:都|已经|已)?[^。！？]{0,6}"
    r"(?:安排|交代|处理)(?:好|妥|完|好了|妥了|完了)",
    re.IGNORECASE,
)
_SELF_POSTHUMOUS_AFFAIRS_RECEIVED_PATTERN = re.compile(
    r"(?:^|[。！？；;，,\n“\"‘「『【`])"
    rf"{_CLOSE_SUPPORT_PERSON_TEXT}(?:已经|已)?"
    r"(?:接手|收下|接收|保管|保存)"
    r"(?:了)?(?:由)?(?:我|本人)(?:亲自)?"
    r"(?:安排|交代|处理)(?:好|妥|完|好了|妥了|完了)的"
    r"(?:身后(?:的)?事(?:情)?|身后安排|后事)",
    re.IGNORECASE,
)
_IRREVERSIBLE_FAREWELL_PATTERN = re.compile(
    r"(?:^|[。！？；;，,\n“\"‘「『【`])(?:"
    r"(?P<bare>(?:(?:这次|这回)(?:是)?(?:真的|确实)?|"
    r"真的|确实|正式)(?:要)?(?:永别|永久(?:的)?道别))"
    r"|(?P<people>(?:我|本人)?(?:这次|这回)?(?:是)?"
    r"(?:真的|确实)?(?:准备|打算|要)?"
    r"(?:和|向)(?:你们|大家|所有人|身边的人|家人)"
    r"(?:永别|永久(?:地|的)?道别))"
    r"|(?P<no_reunion>(?:往后|今后|以后|从今往后)"
    r"(?:不会|不可能)(?:再)?有"
    r"(?:和|同)(?:你们|大家|所有人|身边的人|家人)"
    r"(?:再)?见面(?:的)?机会)"
    r")(?=$|[。！？；;，,\n])",
    re.IGNORECASE,
)
# Natural spoken Chinese may place one or two certainty/linking adverbs between
# the audience and a terminal predicate. Keep this slot closed: uncertainty
# markers such as ``可能``/``也许`` must not turn ordinary separation into HIGH.
_TERMINAL_LINK_ADVERB_TEXT = r"(?:(?:就|也|真的|确实|仍然|还是)\s*){0,2}"
_BURDEN_CONTINUATION_TEXT = r"(?:(?:再)?继续|再|一直)?"
_TONIGHT_TERMINAL_BURDEN_PATTERN = re.compile(
    r"(?:^|[。！？；;，,\n“\"‘「『【`])"
    r"(?:今晚|今夜|今天晚上)(?:过后|之后|以后|起)"
    r"\s*[，,]?\s*"
    r"(?:(?:你们|大家|所有人|身边的人|别人|他人|家人)\s*)?"
    rf"{_TERMINAL_LINK_ADVERB_TEXT}(?:不用|不必|无需|用不着)"
    rf"{_BURDEN_CONTINUATION_TEXT}"
    r"(?:为我(?:操心|担心)|替我(?:操心|担心)|担心我)"
    r"(?:了)?(?=$|[。！？；;，,\n])",
    re.IGNORECASE,
)
_ROUTINE_ESTATE_PLANNING_PATTERN = re.compile(
    r"(?:律师|公证员|法律顾问)[^。！？；]{0,32}"
    r"(?:建议|提醒|要求|常规|长期|遗嘱|遗产规划|财产规划)"
    r"|(?:常规|长期|年度|每年|定期|例行|应急)[^。！？；]{0,24}"
    r"(?:遗嘱|遗产规划|财产规划|遗物清单|身后资料)"
    r"|(?:明年|未来|以后|下次|下一年度)[^。！？；]{0,18}"
    r"(?:修改|更新|核对|维护|整理|规划)[^。！？；]{0,12}"
    r"(?:遗嘱|遗产|遗物清单|身后资料)"
    r"|(?:计划|打算)[^。！？；]{0,18}"
    r"(?:修改|更新|核对|维护|整理|规划)[^。！？；]{0,12}"
    r"(?:遗嘱|遗产|遗物清单|身后资料)"
    r"|(?:遗嘱|遗产规划|财产规划|遗物清单|身后资料)"
    r"[^。！？；]{0,28}(?:计划|打算)?"
    r"(?:明年|未来|以后|下次|下一年度|下季度)"
    r"[^。！？；]{0,12}(?:修改|更新|核对|维护|整理|复核)",
    re.IGNORECASE,
)
_THIRD_PARTY_POSTHUMOUS_OWNER_PATTERN = re.compile(
    rf"(?:{_THIRD_PARTY_HUMAN_TEXT}|祖父|祖母|爷爷|奶奶|外公|外婆|"
    r"逝者|死者|已故者|客户)(?:留下(?:来)?的|的)"
    r"[^。！？；]{0,24}(?:遗物|遗嘱|遗产|身后)",
    re.IGNORECASE,
)
_STRONG_SELF_ABSENCE_PATTERN = re.compile(
    r"(?:^|[。！？；;，,\n“\"‘「『【`])"
    r"(?:今晚|今夜|今天晚上)(?:过后|之后|以后|起)"
    r"\s*[，,]?\s*"
    r"(?:(?:你们|大家|所有人|任何人)\s*)?"
    rf"{_TERMINAL_LINK_ADVERB_TEXT}"
    r"(?:再也|永远|不会再|不可能再)"
    r"[^。！？]{0,5}"
    r"(?:见不到|看不到|找不到|找不着|联系不到|见到|看到|找到|联系到)"
    r"(?:我|本人)(?:了)?(?=$|[。！？；;，,\n])"
    r"|(?:^|[。！？；;，,\n“\"‘「『【`])"
    r"(?:从今往后|从今夜起|从今晚起)"
    r"\s*[，,]?\s*(?:谁也|没人)"
    rf"{_TERMINAL_LINK_ADVERB_TEXT}"
    r"[^。！？]{0,5}(?:不可能再|不会再|再也|永远)?"
    r"[^。！？]{0,4}(?:见到|看到|找到|联系到)"
    r"(?:我|本人)(?:了)?(?=$|[。！？；;，,\n])",
    re.IGNORECASE,
)
_FAREWELL_FINALITY_PATTERN = re.compile(
    r"(?P<bare>(?:准备|打算|最后一次|正式)?(?:告别|道别)(?:了)?"
    r"(?=$|[。！？；;，,\n”\"’」』】`]))"
    r"|(?P<people>(?:准备|打算)?(?:和|向)"
    r"(?:你们|大家|所有人|身边的人)(?:告别|道别))"
    r"|(?P<last>(?:这|那)?是(?:我)?最后一次(?:再)?"
    r"(?:(?:和|同|向)(?:你们|大家|所有人|身边的人))?"
    r"(?:说话|交谈|联系|见面))",
    re.IGNORECASE,
)
_UNBOUNDED_SELF_ABSENCE_PATTERN = re.compile(
    r"(?:以后|今后|之后|往后|从今往后|今晚之后)?[^。！？]{0,8}"
    r"(?:"
    r"(?:你们|大家|别人|所有人|任何人|谁也|没人)[^。！？]{0,8}"
    r"(?:见不到|不会再见到|看不到|不会再看到|"
    r"找不到|找不着|联系不到|无法找到|无法联系|不能找到)"
    r"|没人[^。！？]{0,8}"
    r"(?:能找到|找得到|联系得到|能联系到|能看到|能见到)"
    r")"
    r"我(?:了)?(?=$|[。！？；;，,\n])",
    re.IGNORECASE,
)
_ORDINARY_CONTINUING_TRANSITION_PATTERN = re.compile(
    r"(?:毕业|结业|离职|离校|搬家|调岗|交换结束|项目结束|课题收尾)"
    r"[^。！？]{0,28}"
    r"(?:去|到|将|会|准备|计划|随后|之后|下周|下月|明年|秋天|暑假后)"
    r"[^。！？]{0,12}"
    r"(?:入职|上班|工作|读研|入学|报到|继续学习|旅行|回国|搬到)",
    re.IGNORECASE,
)
_BURDEN_RELEASE_PATTERN = re.compile(
    rf"(?:不用|不必|不会|无需|用不着){_BURDEN_CONTINUATION_TEXT}"
    r"[^。！？]{0,8}(?:烦|麻烦|打扰|拖累)"
    r"[^。！？]{0,6}(?:你们|大家|别人|身边的人)"
    r"|(?:你们|大家|别人|身边的人)[^。！？]{0,6}"
    rf"(?:不用|不必|不会|无需|用不着){_BURDEN_CONTINUATION_TEXT}"
    r"[^。！？]{0,6}(?:为我(?:操心|担心)|替我(?:操心|担心)|担心我)"
    r"(?:了)?(?=$|[。！？；;，,\n])"
    r"|(?:不用|不必|不会|无需|用不着)"
    rf"{_BURDEN_CONTINUATION_TEXT}"
    r"(?:为我(?:操心|担心)|替我(?:操心|担心)|担心我)"
    r"(?:了)?(?=$|[。！？；;，,\n])",
    re.IGNORECASE,
)
_PERMANENT_DEPARTURE_PATTERN = re.compile(
    r"(?:^|[。！？；;，,\n“\"‘「『【`])(?P<subject>我|本人)"
    r"(?:今晚之后|以后|今后|之后|往后|可能|也许|大概|就|再)?"
    r"(?:不会|不再|再也不)[^。！？]{0,4}(?:回来|出现)"
    r"|(?:^|[。！？；;，,\n“\"‘「『【`])(?P<contact_subject>我|本人)"
    r"(?:不会|不再|再也不)[^。！？]{0,4}"
    r"联系(?:任何人|所有人|你们|大家)"
    r"|(?:^|[。！？；;，,\n“\"‘「『【`])(?P<leave_subject>我|本人)"
    r"(?:要|准备|打算)?(?:永远|永久)(?:离开|消失)",
    re.IGNORECASE,
)
_IMMEDIATE_SELF_DISAPPEARANCE_PATTERN = re.compile(
    r"(?:^|[。！？；;，,\n“\"‘「『【`])(?:"
    r"(?:今晚|今夜|今天晚上)(?P<subject>我|本人)"
    r"(?:就|将|会|要|可能){0,2}(?:彻底|永远)?消失"
    r"|(?P<subject_after>我|本人)(?:今晚|今夜|今天晚上)"
    r"(?:就|将|会|要|可能){0,2}(?:彻底|永远)?消失"
    r")(?=$|[。！？；;，,\n])",
    re.IGNORECASE,
)
_OUTSIDE_PROTECTIVE_BOUNDARY_PATTERN = re.compile(
    r"(?:^|[。！？；;，,\n“\"‘「『【`])(?P<subject>我|本人)"
    r"(?:人|现在|今晚|已经|正|正在){0,3}[^。！？]{0,5}(?:"
    r"(?:站|坐|待|走到|来到|到了|到达|身处|位于|在)"
    r"[^。！？]{0,8}(?:桥|高架桥|天桥|楼顶|天台)"
    r"[^。！？]{0,5}(?:的)?(?:护栏|栏杆)(?:外|外侧|外面)"
    r"|(?:翻过|跨过|越过)[^。！？]{0,6}"
    r"(?:桥|高架桥|天桥|楼顶|天台)(?:的)?(?:护栏|栏杆)"
    r")",
    re.IGNORECASE,
)
_IMMEDIATE_DANGEROUS_LOCATION_PATTERN = re.compile(
    r"(?:^|[。！？；;，,\n“\"‘「『【`])(?P<subject>我|本人)"
    r"(?:人|现在|今晚|已经|正|正在){0,3}[^。！？]{0,5}"
    r"(?:站|坐|待|走到|来到|到了|到达|身处|位于|在)"
    r"[^。！？]{0,8}(?:"
    r"(?:天台|楼顶|桥面|桥边)[^。！？]{0,5}"
    r"(?:边缘|边上)"
    r"|悬崖(?:边|边缘|外侧)"
    r"|轨道(?:上|中间)"
    r")",
    re.IGNORECASE,
)
_REFUSAL_TO_LEAVE_DANGER_PATTERN = re.compile(
    r"(?:不打算|不准备|没打算|没准备|不想|不会|拒绝)[^。！？]{0,5}"
    r"(?:下来|下去|离开|回去|退回)",
    re.IGNORECASE,
)
_IMPLICIT_META_LEAD_PATTERN = re.compile(
    r"(?:电影|小说|新闻|论文|量表|代码|翻译|台词|剧情|题目|作业)"
    r"[^。！？]{0,18}(?:写着|写|说|提到|描述|讨论|内容是|原文是|翻成|：|:|“|\")",
    re.IGNORECASE,
)
_IMPLICIT_META_REQUEST_PATTERN = re.compile(
    r"(?:这句(?:话)?|这段(?:话)?|引号里|原文|句子)?[^。！？]{0,10}"
    r"(?:怎么翻译|如何翻译|翻译一下|是什么意思|什么意思|解释一下|改写一下)",
    re.IGNORECASE,
)
_IMPLICIT_SELF_CONFIRMATION_PATTERN = re.compile(
    r"(?:这就是|这是|表达的是)[^。！？]{0,8}(?:我|本人)"
    r"[^。！？]{0,8}(?:现在|当前|真实|真正|真的)"
    r"[^。！？]{0,8}(?:想法|打算|状态|感受)",
    re.IGNORECASE,
)
_SUPPORT_SEEKING_PATTERN = re.compile(
    r"(?:失恋|分手|被拒绝|关系破裂|吵架|冲突|低落|不好受|难受|担心|"
    r"害怕|不安|孤单|撑不住)[^。！？]{0,18}"
    r"(?:想聊|聊聊|倾诉|求助|帮帮|需要支持|怎么办)"
    r"|(?:我|本人)?(?:还是|仍然|依然|现在)?(?:有点|很|特别)?"
    r"(?:担心|害怕|不安|不好受)(?:[。！! ]|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _RiskSemanticEvent:
    """One ordered mutation of the current conversation risk state."""

    position: int
    action: str
    category: str = ""
    clause: str = ""
    subject: str = ""
    target: str = ""
    successful: bool = True


@dataclass(frozen=True)
class _ImplicitRiskEvidence:
    """Structured evidence used by the closed implicit-crisis contract."""

    start: int
    end: int
    kind: str
    subject: str
    target: str


def has_self_high_risk_signal(text: str) -> bool:
    """Detect whether ordered local semantics leave a current self risk active."""
    return risk_state_after_texts([text]) == "self"


def has_third_party_immediate_risk(text: str) -> bool:
    """Detect whether ordered local semantics leave an immediate third-party risk active."""
    return risk_state_after_texts([text]) == "third_party"


def has_high_risk_signal(text: str) -> bool:
    """
    检测文本是否包含当前有效的高风险表达。

    这是风险评估的第一道防线（硬兜底）：候选风险词会按局部子句判断
    主体、否定和引用语境；确认是当前自身风险或第三方即时危机后才升级。
    """
    return has_self_high_risk_signal(text) or has_third_party_immediate_risk(text)


def risk_state_after_texts(texts: Iterable[str]) -> str | None:
    """
    Apply ordered risk, negation, correction, and resolution events.

    Negating one risk category does not erase a different active category.
    Explicit safety/resolution statements clear prior state only at their own
    position, so a later new risk expression re-opens the state. Completed
    protective actions are retained as semantic events but do not, by
    themselves, assert that suicidal/self-harm thoughts have ended.
    """
    self_categories: set[str] = set()
    third_party_categories: set[str] = set()
    saw_resolution = False
    for text in texts:
        for event in _ordered_risk_events((text or "").lower()):
            if event.action == "self_risk":
                self_categories.add(event.category)
            elif event.action == "third_party_risk":
                third_party_categories.add(event.category)
            elif event.action == "negate_self":
                self_categories.discard(event.category)
                saw_resolution = True
            elif event.action == "negate_third_party":
                third_party_categories.discard(event.category)
                saw_resolution = True
            elif event.action == "resolution":
                if not event.successful or event.target == "irrelevant":
                    continue
                if event.target == "self":
                    self_categories.clear()
                    saw_resolution = True
                elif event.target == "third_party":
                    third_party_categories.clear()
                    saw_resolution = True
                elif event.target == "contextual":
                    # 省略主语的“现在没有了/已经联系并安全”等，只能解除
                    # 唯一活跃的风险主体；self 与 third-party 同时活跃时保守保留。
                    if self_categories and not third_party_categories:
                        self_categories.clear()
                        saw_resolution = True
                    elif third_party_categories and not self_categories:
                        third_party_categories.clear()
                        saw_resolution = True
                    elif not self_categories and not third_party_categories:
                        saw_resolution = True
    if self_categories:
        return "self"
    if third_party_categories:
        return "third_party"
    return "resolved" if saw_resolution else None


def has_resolved_risk_transition(text: str) -> bool:
    """Return true when a real risk was stated and then explicitly resolved."""
    events = _ordered_risk_events((text or "").lower())
    stated_risk = any(
        event.action in {"self_risk", "third_party_risk"}
        for event in events
    )
    return stated_risk and risk_state_after_texts([text]) == "resolved"


def _ordered_risk_events(text: str) -> list[_RiskSemanticEvent]:
    """Parse one message into position-ordered state mutations."""
    events: list[_RiskSemanticEvent] = _implicit_self_crisis_events(text)

    for clause, clause_offset in _iter_local_clauses(text):
        events.extend(_positive_resolution_events(clause, clause_offset))

        for match in _RISK_WORD_PATTERN.finditer(clause):
            category = _risk_category(match.group(0))
            position = clause_offset + match.start()
            if _is_reported_self_mention(clause, match):
                continue
            if _is_non_literal_risk_clause(clause):
                continue
            if _is_negated_risk_mention(clause, match):
                action = (
                    "negate_self"
                    if _has_first_person_reference(clause, match)
                    or not _THIRD_PERSON_SUBJECT_PATTERN.search(clause)
                    else "negate_third_party"
                )
                target = "self" if action == "negate_self" else "third_party"
                events.append(
                    _RiskSemanticEvent(
                        position,
                        action,
                        category,
                        clause=clause,
                        subject=target,
                        target=target,
                    )
                )
                continue
            if _has_first_person_intent(clause):
                events.append(
                    _RiskSemanticEvent(
                        position,
                        "self_risk",
                        category,
                        clause=clause,
                        subject="self",
                        target="self",
                    )
                )
                continue
            if _THIRD_PERSON_SUBJECT_PATTERN.search(clause):
                if any(marker in clause for marker in IMMEDIATE_MARKERS):
                    events.append(
                        _RiskSemanticEvent(
                            position,
                            "third_party_risk",
                            category,
                            clause=clause,
                            subject="third_party",
                            target="third_party",
                        )
                    )
                continue
            # 无明确引用、否定或第三人称主体时采取安全优先的当前说话者解释。
            events.append(
                _RiskSemanticEvent(
                    position,
                    "self_risk",
                    category,
                    clause=clause,
                    subject="self",
                    target="self",
                )
            )
    return sorted(events, key=lambda item: item.position)


def _implicit_self_crisis_events(text: str) -> list[_RiskSemanticEvent]:
    """
    Detect high-risk meaning that emerges from a combination of ordinary phrases.

    No single word such as ``告别``、``赠送`` or ``天台`` is sufficient. The
    contract requires a first-person action plus a separate finality/danger cue,
    while academic, quoted, fictional, and task descriptions are excluded.
    """
    evidence = _collect_implicit_risk_evidence(text)
    direct = next(
        (
            item
            for item in evidence
            if item.kind
            in {
                "outside_protective_boundary",
                "immediate_disappearance",
                "posthumous_self_preparation",
                "strong_permanent_absence",
                "strong_terminal_burden",
            }
            and item.subject == "self"
        ),
        None,
    )
    if direct is not None:
        return [
            _RiskSemanticEvent(
                position=direct.start,
                action="self_risk",
                category="implicit_crisis",
                clause=text,
                subject="self",
                target="self",
            )
        ]

    selected: tuple[_ImplicitRiskEvidence, _ImplicitRiskEvidence] | None = None

    transfers = [item for item in evidence if item.kind == "possession_disposal"]
    finalities = [
        item
        for item in evidence
        if item.kind in {"terminal_finality", "permanent_absence"}
    ]
    for transfer in transfers:
        for finality in finalities:
            # Omitted subjects may inherit only an explicit self-terminal cue;
            # two unknown subjects are never promoted to a self-risk event.
            if transfer.subject == "self" and finality.subject in {"self", "implicit"}:
                selected = (transfer, finality)
                break
            if transfer.subject == "implicit" and finality.subject == "self":
                selected = (transfer, finality)
                break
        if selected:
            break

    if selected is None:
        posthumous = [
            item
            for item in evidence
            if item.kind == "posthumous_preparation"
            and item.subject in {"self", "implicit"}
        ]
        irreversible_farewells = [
            item
            for item in evidence
            if item.kind == "irreversible_farewell"
        ]
        if posthumous and irreversible_farewells:
            selected = (posthumous[0], irreversible_farewells[0])

    if selected is None:
        burdens = [item for item in evidence if item.kind == "burden_release"]
        departures = [
            item
            for item in evidence
            if item.kind == "permanent_absence" and item.subject == "self"
        ]
        if burdens and departures:
            selected = (burdens[0], departures[0])

    if selected is None:
        positions = [
            item
            for item in evidence
            if item.kind == "danger_boundary" and item.subject == "self"
        ]
        refusals = [item for item in evidence if item.kind == "refusal_to_leave"]
        if positions and refusals:
            selected = (positions[0], refusals[0])

    if selected is None:
        return []
    start = min(item.start for item in selected)
    return [
        _RiskSemanticEvent(
            position=start,
            action="self_risk",
            category="implicit_crisis",
            clause=text,
            subject="self",
            target="self",
        )
    ]


def _local_semantic_sentence(text: str, start: int, end: int) -> str:
    """Return the major-punctuation-bounded sentence containing one evidence span."""
    left = max((text.rfind(mark, 0, start) for mark in "。！？!?；;\n"), default=-1)
    right_candidates = [
        position
        for mark in "。！？!?；;\n"
        if (position := text.find(mark, end)) >= 0
    ]
    right = min(right_candidates) if right_candidates else len(text)
    return text[left + 1:right]


def _estate_context_is_routine(
    text: str,
    start: int,
    end: int,
) -> bool:
    """
    Distinguish ordinary legal/future maintenance from current final preparation.

    Routine planning suppresses estate evidence only inside its own sentence.
    A co-located irreversible farewell, immediate disappearance, or strong
    unbounded absence is a current terminal assertion and therefore wins.
    """
    window = _local_semantic_sentence(text, start, end)
    if (
        _IRREVERSIBLE_FAREWELL_PATTERN.search(window)
        or _STRONG_SELF_ABSENCE_PATTERN.search(window)
        or _IMMEDIATE_SELF_DISAPPEARANCE_PATTERN.search(window)
        or _TONIGHT_TERMINAL_BURDEN_PATTERN.search(window)
    ):
        return False
    return bool(_ROUTINE_ESTATE_PLANNING_PATTERN.search(window))


def _handoff_has_third_party_owner(segment: str) -> bool:
    """Do not reinterpret another person's estate as the speaker's preparation."""
    return bool(_THIRD_PARTY_POSTHUMOUS_OWNER_PATTERN.search(segment))


def _handoff_is_completed(segment: str) -> bool:
    """Require an achieved handoff, not an unexecuted estate-planning intention."""
    return bool(
        re.search(
            r"(?:已经|已|都|全部|悉数|逐项|一并|一起|分别)"
            r"|(?:留给|交给|托付给|分给|转交给)[^。！？；]{0,16}了"
            r"(?:$|[。！？；;，,\n])",
            segment,
            re.IGNORECASE,
        )
    )


def _is_will_only_handoff(segment: str) -> bool:
    """A will by itself is ordinary estate planning without another final cue."""
    return (
        "遗嘱" in segment
        and not re.search(r"(?:遗物|身后|后事)", segment)
        and not re.search(_ACCESS_CREDENTIAL_TEXT, segment, re.IGNORECASE)
    )


def _collect_implicit_risk_evidence(text: str) -> list[_ImplicitRiskEvidence]:
    """Parse subject and target before combining any implicit-risk cues."""
    evidence: list[_ImplicitRiskEvidence] = []

    for pattern in (
        _SELF_IMPORTANT_POSSESSION_TRANSFER_PATTERN,
        _SELF_VALUED_POSSESSION_DISPOSAL_PATTERN,
    ):
        for match in pattern.finditer(text):
            if not _implicit_evidence_is_meta(text, match.start(), match.end()):
                evidence.append(
                    _ImplicitRiskEvidence(
                        match.start(),
                        match.end(),
                        "possession_disposal",
                        "self",
                        "valued_possession",
                    )
                )
    for pattern in (
        _OMITTED_SELF_POSSESSION_TRANSFER_PATTERN,
        _OMITTED_VALUED_POSSESSION_DISPOSAL_PATTERN,
    ):
        for match in pattern.finditer(text):
            if not _implicit_evidence_is_meta(text, match.start(), match.end()):
                evidence.append(
                    _ImplicitRiskEvidence(
                        match.start(),
                        match.end(),
                        "possession_disposal",
                        "implicit",
                        "valued_possession",
                    )
                )

    for pattern in (
        _SELF_COMPLETED_FINAL_HANDOFF_PATTERN,
        _SELF_OWNED_COMPLETED_FINAL_HANDOFF_PATTERN,
    ):
        for handoff in pattern.finditer(text):
            segment = handoff.group(0)
            if (
                _implicit_evidence_is_meta(text, handoff.start(), handoff.end())
                or _estate_context_is_routine(
                    text,
                    handoff.start(),
                    handoff.end(),
                )
                or _handoff_has_third_party_owner(segment)
                or not _handoff_is_completed(segment)
            ):
                continue
            evidence.append(
                _ImplicitRiskEvidence(
                    handoff.start(),
                    handoff.end(),
                    "posthumous_self_preparation",
                    "self",
                    "completed_multi_domain_handoff",
                )
            )

    for handoff in _SELF_POSTHUMOUS_HANDOFF_PATTERN.finditer(text):
        segment = handoff.group(0)
        if (
            _implicit_evidence_is_meta(text, handoff.start(), handoff.end())
            or _estate_context_is_routine(text, handoff.start(), handoff.end())
            or _handoff_has_third_party_owner(segment)
            or _is_will_only_handoff(segment)
        ):
            continue
        evidence.append(
            _ImplicitRiskEvidence(
                handoff.start(),
                handoff.end(),
                "posthumous_self_preparation",
                "self",
                "posthumous_property",
            )
        )

    for handoff in _IMPLICIT_POSTHUMOUS_HANDOFF_PATTERN.finditer(text):
        if (
            _implicit_evidence_is_meta(text, handoff.start(), handoff.end())
            or _estate_context_is_routine(text, handoff.start(), handoff.end())
            or _handoff_has_third_party_owner(handoff.group(0))
        ):
            continue
        evidence.append(
            _ImplicitRiskEvidence(
                handoff.start(),
                handoff.end(),
                "posthumous_self_preparation",
                "self",
                "posthumous_property",
            )
        )

    for affairs in _POSTHUMOUS_AFFAIRS_COMPLETED_PATTERN.finditer(text):
        if (
            _implicit_evidence_is_meta(text, affairs.start(), affairs.end())
            or _estate_context_is_routine(text, affairs.start(), affairs.end())
            or _handoff_has_third_party_owner(affairs.group(0))
        ):
            continue
        subject = "self" if affairs.group("subject") else "implicit"
        evidence.append(
            _ImplicitRiskEvidence(
                affairs.start(),
                affairs.end(),
                "posthumous_preparation",
                subject,
                "posthumous_affairs",
            )
        )
    for affairs in _SELF_POSTHUMOUS_AFFAIRS_RECEIVED_PATTERN.finditer(text):
        if (
            _implicit_evidence_is_meta(text, affairs.start(), affairs.end())
            or _estate_context_is_routine(text, affairs.start(), affairs.end())
            or _handoff_has_third_party_owner(affairs.group(0))
        ):
            continue
        evidence.append(
            _ImplicitRiskEvidence(
                affairs.start(),
                affairs.end(),
                "posthumous_preparation",
                "self",
                "posthumous_affairs",
            )
        )

    for farewell in _IRREVERSIBLE_FAREWELL_PATTERN.finditer(text):
        if _implicit_evidence_is_meta(text, farewell.start(), farewell.end()):
            continue
        evidence.append(
            _ImplicitRiskEvidence(
                farewell.start(),
                farewell.end(),
                "irreversible_farewell",
                "implicit",
                "human_audience",
            )
        )

    for absence in _STRONG_SELF_ABSENCE_PATTERN.finditer(text):
        if _implicit_evidence_is_meta(text, absence.start(), absence.end()):
            continue
        evidence.append(
            _ImplicitRiskEvidence(
                absence.start(),
                absence.end(),
                "strong_permanent_absence",
                "self",
                "unbounded",
            )
        )

    for burden in _TONIGHT_TERMINAL_BURDEN_PATTERN.finditer(text):
        if _implicit_evidence_is_meta(text, burden.start(), burden.end()):
            continue
        evidence.append(
            _ImplicitRiskEvidence(
                burden.start(),
                burden.end(),
                "strong_terminal_burden",
                "self",
                "unbounded_human_audience",
            )
        )

    for farewell in _FAREWELL_FINALITY_PATTERN.finditer(text):
        if _implicit_evidence_is_meta(text, farewell.start(), farewell.end()):
            continue
        weak_farewell = bool(farewell.group("bare") or farewell.group("people"))
        if weak_farewell and any(
            abs(transition.start() - farewell.start()) <= 80
            and not _implicit_evidence_is_meta(
                text,
                transition.start(),
                transition.end(),
            )
            for transition in _ORDINARY_CONTINUING_TRANSITION_PATTERN.finditer(text)
        ):
            continue
        subject = "self" if farewell.group("last") else "implicit"
        evidence.append(
            _ImplicitRiskEvidence(
                farewell.start(),
                farewell.end(),
                "terminal_finality",
                subject,
                "weak_farewell" if weak_farewell else "terminal_communication",
            )
        )

    for absence in _UNBOUNDED_SELF_ABSENCE_PATTERN.finditer(text):
        local = text[
            max(0, absence.start() - 24):
            min(len(text), absence.end() + 36)
        ]
        weak_or_bounded = bool(
            re.search(r"(?:可能|也许|或许|大概|不常|少见)", local)
        ) or bool(_ORDINARY_CONTINUING_TRANSITION_PATTERN.search(local))
        if (
            not weak_or_bounded
            and not _implicit_evidence_is_meta(
                text,
                absence.start(),
                absence.end(),
            )
        ):
            evidence.append(
                _ImplicitRiskEvidence(
                    absence.start(),
                    absence.end(),
                    "permanent_absence",
                    "self",
                    "unbounded",
                )
            )

    for burden in _BURDEN_RELEASE_PATTERN.finditer(text):
        if not _implicit_evidence_is_meta(text, burden.start(), burden.end()):
            evidence.append(
                _ImplicitRiskEvidence(
                    burden.start(),
                    burden.end(),
                    "burden_release",
                    "implicit",
                    "human_audience",
                )
            )
    for departure in _PERMANENT_DEPARTURE_PATTERN.finditer(text):
        if not _implicit_evidence_is_meta(text, departure.start(), departure.end()):
            evidence.append(
                _ImplicitRiskEvidence(
                    departure.start(),
                    departure.end(),
                    "permanent_absence",
                    "self",
                    "unbounded",
                )
            )
    for disappearance in _IMMEDIATE_SELF_DISAPPEARANCE_PATTERN.finditer(text):
        if not _implicit_evidence_is_meta(
            text,
            disappearance.start(),
            disappearance.end(),
        ):
            evidence.append(
                _ImplicitRiskEvidence(
                    disappearance.start(),
                    disappearance.end(),
                    "immediate_disappearance",
                    "self",
                    "unbounded",
                )
            )

    for outside in _OUTSIDE_PROTECTIVE_BOUNDARY_PATTERN.finditer(text):
        if not _implicit_evidence_is_meta(text, outside.start(), outside.end()):
            evidence.append(
                _ImplicitRiskEvidence(
                    outside.start(),
                    outside.end(),
                    "outside_protective_boundary",
                    "self",
                    "physical_boundary",
                )
            )
    for location in _IMMEDIATE_DANGEROUS_LOCATION_PATTERN.finditer(text):
        if not _implicit_evidence_is_meta(text, location.start(), location.end()):
            evidence.append(
                _ImplicitRiskEvidence(
                    location.start(),
                    location.end(),
                    "danger_boundary",
                    "self",
                    "physical_edge",
                )
            )
    for refusal in _REFUSAL_TO_LEAVE_DANGER_PATTERN.finditer(text):
        if not _implicit_evidence_is_meta(text, refusal.start(), refusal.end()):
            evidence.append(
                _ImplicitRiskEvidence(
                    refusal.start(),
                    refusal.end(),
                    "refusal_to_leave",
                    "implicit",
                    "danger_location",
                )
            )
    return sorted(evidence, key=lambda item: item.start)


def _implicit_evidence_is_meta(
    text: str,
    evidence_start: int,
    evidence_end: int,
) -> bool:
    """Keep quoted/media/task examples from becoming real conversation state."""
    prefix = text[max(0, evidence_start - 40):evidence_start]
    meta = _IMPLICIT_META_LEAD_PATTERN.search(prefix)
    if meta:
        tail = prefix[meta.end():]
        if not re.search(
            r"(?:但|但是|不过|可是|然而|却)(?:我|本人)?(?:现在|今晚)?$",
            tail,
        ):
            return True

    if _IMPLICIT_SELF_CONFIRMATION_PATTERN.search(text):
        return False
    for opening, closing in (
        ("“", "”"),
        ('"', '"'),
        ("‘", "’"),
        ("「", "」"),
        ("『", "』"),
        ("【", "】"),
        ("`", "`"),
    ):
        search_from = 0
        while True:
            quote_start = text.find(opening, search_from)
            if quote_start < 0:
                break
            quote_end = text.find(closing, quote_start + 1)
            if quote_end < 0:
                break
            if quote_start <= evidence_start and evidence_end <= quote_end:
                outside = text[:quote_start] + text[quote_end + 1:]
                if _IMPLICIT_META_REQUEST_PATTERN.search(outside):
                    return True
            search_from = quote_end + 1
    return False


def _positive_resolution_events(
    clause: str,
    clause_offset: int,
) -> list[_RiskSemanticEvent]:
    """
    Build resolution events only from closed, affirmative safety contracts.

    Unknown wording is intentionally ignored. A clause must identify a human
    beneficiary and a complete personal-safety predicate, deny a concrete risk
    state, or complete a recognized protection measure for that beneficiary.
    """
    for pattern in (
        _DIRECT_HUMAN_SAFETY_PATTERN,
        _CONFIRMED_HUMAN_SAFETY_PATTERN,
        _SPECIFIC_RISK_DENIAL_PATTERN,
        _ANTECEDENT_RISK_DENIAL_PATTERN,
        _COMPLETE_SELF_NO_RISK_PATTERN,
    ):
        match = pattern.fullmatch(clause)
        if match:
            subject = match.group("subject")
            target = _human_resolution_target(subject)
            return [
                _RiskSemanticEvent(
                    clause_offset + match.start(),
                    "resolution",
                    clause=clause,
                    subject=target,
                    target=target,
                    successful=True,
                )
            ]

    contextual = _STANDALONE_CONTEXTUAL_RESOLUTION_PATTERN.fullmatch(clause)
    if contextual:
        return [
            _RiskSemanticEvent(
                clause_offset,
                "resolution",
                clause=clause,
                subject="implicit",
                target="contextual",
                successful=True,
            )
        ]

    transfer = _DANGEROUS_ITEM_TRANSFER_PATTERN.fullmatch(clause)
    if transfer:
        return [
            _RiskSemanticEvent(
                clause_offset,
                "protection",
                clause=clause,
                subject="self",
                target="self",
                successful=True,
            )
        ]

    accompaniment = _HUMAN_ACCOMPANIMENT_PATTERN.fullmatch(clause)
    if accompaniment:
        target = _human_resolution_target(accompaniment.group("subject"))
        return [
            _RiskSemanticEvent(
                clause_offset,
                "protection",
                clause=clause,
                subject=target,
                target=target,
                successful=True,
            )
        ]

    contact = _CONTACT_AND_SAFETY_PATTERN.fullmatch(clause)
    if contact:
        subject = contact.group("subject")
        target = _human_resolution_target(subject) if subject else "contextual"
        return [
            _RiskSemanticEvent(
                clause_offset,
                "resolution",
                clause=clause,
                subject=_human_resolution_target(subject) if subject else "implicit",
                target=target,
                successful=True,
            )
        ]
    return []


def _human_resolution_target(subject: str) -> str:
    """Map a positively parsed human beneficiary to its independent risk state."""
    if re.fullmatch(_SELF_HUMAN_TEXT, subject or "", re.IGNORECASE):
        return "self"
    return "third_party"


def _iter_local_clauses(text: str) -> Iterable[tuple[str, int]]:
    """Yield local clauses with offsets so semantic events retain text order."""
    start = 0
    for boundary in _CLAUSE_BOUNDARY_PATTERN.finditer(text or ""):
        clause = text[start:boundary.start()].strip()
        if clause:
            leading = len(text[start:boundary.start()]) - len(text[start:boundary.start()].lstrip())
            yield clause, start + leading
        start = boundary.end()
    clause = (text or "")[start:].strip()
    if clause:
        leading = len((text or "")[start:]) - len((text or "")[start:].lstrip())
        yield clause, start + leading


def _risk_category(word: str) -> str:
    normalized = word.lower()
    if normalized in {"自残", "伤害自己", "self harm", "self-harm"}:
        return "self_harm"
    return "suicide"


def _is_reported_self_mention(clause: str, risk_match: re.Match[str]) -> bool:
    """Return true only when a reporting construction covers this risk mention."""
    return any(
        match.start() <= risk_match.start() and match.end() >= risk_match.end()
        for match in _REPORTED_SELF_PATTERN.finditer(clause)
    )


def _is_negated_risk_mention(clause: str, risk_match: re.Match[str]) -> bool:
    """Apply negation to the risk mention covered by that negation, not the clause."""
    return any(
        match.start() <= risk_match.start() and match.end() >= risk_match.end()
        for match in _NEGATED_RISK_PATTERN.finditer(clause)
    )


def _has_first_person_reference(clause: str, risk_match: re.Match[str]) -> bool:
    """Detect a local first-person subject before one risk mention."""
    prefix = clause[max(0, risk_match.start() - 18):risk_match.start()]
    return bool(re.search(r"我(?!妈|爸|朋友|室友|同学|舍友|家人)", prefix))


def _has_explicit_self_intent(clause: str) -> bool:
    """Identify a local first-person/current intent instead of any remote ``我``."""
    return bool(_SELF_INTENT_PATTERN.search(clause) or _DIRECT_SELF_PATTERN.search(clause))


def _has_first_person_intent(clause: str) -> bool:
    """Require an actual first-person risk subject when a third party is present."""
    return bool(_DIRECT_SELF_PATTERN.search(clause))


def _is_non_literal_risk_clause(clause: str) -> bool:
    """Recognize risk terms used as academic/media/task content in this clause."""
    if _META_RISK_PATTERN.search(clause):
        return True
    if not any(marker in clause for marker in NON_LITERAL_RISK_MARKERS):
        return False
    # A nearby explicit current intent wins over a remote word such as “报告”.
    return not _has_explicit_self_intent(clause)


def has_medium_risk_signal(text: str) -> bool:
    """Detect sustained distress or functional impairment without imminent danger."""
    normalized = (text or "").lower()
    return any(word in normalized for word in MEDIUM_RISK_WORDS)


def has_consult_signal(text: str) -> bool:
    """检测文本是否包含心理咨询相关关键词。"""
    normalized = text.lower()
    return (
        any(word in normalized for word in CONSULT_WORDS)
        or bool(_SUPPORT_SEEKING_PATTERN.search(normalized))
    )


def _current_input_from_prompt(text: str) -> str:
    """Extract PromptTemplates' current-input section for deterministic mock routing."""
    marker = "当前输入："
    if marker not in text:
        return text
    return text.rsplit(marker, 1)[-1].strip()


def split_text(text: str, size: int) -> Iterable[str]:
    """将文本按指定大小切分为片段，用于 mock 模式模拟流式输出。"""
    for index in range(0, len(text), size):
        yield text[index:index + size]
