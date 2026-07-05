"""
MindBridge 隐私数据脱敏模块

在将用户输入发送给 LLM 之前，对敏感个人信息进行脱敏处理。
防止手机号、邮箱、身份证号等隐私信息被传递到模型上下文中。

脱敏规则（正则匹配）：
- 手机号：1[3-9]XXXXXXXXX → [已脱敏]
- 邮箱：xxx@xxx.xxx → [已脱敏]
- 身份证号：18位数字（末位可为X）→ [已脱敏]

调用时机：
- ChatService 中对用户原始输入脱敏后再传给 Agent
- RedisShortTermMemoryStore 读取历史消息时脱敏
"""
import re


class PrivacySanitizer:
    """隐私数据脱敏器，使用正则表达式匹配并替换敏感信息。"""

    # 脱敏规则列表：
    # 1. 中国大陆手机号：1开头，第二位3-9，后面9位数字
    # 2. 邮箱地址：标准邮箱格式
    # 3. 身份证号：18位，最后一位可以是数字或X/x
    patterns = [
        re.compile(r"1[3-9]\d{9}"),                          # 手机号
        re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),         # 邮箱
        re.compile(r"\b\d{17}[\dXx]\b"),                     # 身份证号
    ]

    def sanitize(self, text: str) -> str:
        """
        对输入文本进行隐私脱敏。

        依次匹配所有敏感信息模式，将匹配到的内容替换为 [已脱敏]。
        返回脱敏后的文本。输入为空时返回空字符串。
        """
        sanitized = text or ""
        for pattern in self.patterns:
            sanitized = pattern.sub("[已脱敏]", sanitized)
        return sanitized
