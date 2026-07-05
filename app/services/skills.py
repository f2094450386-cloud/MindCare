"""
MindBridge Skill 注册与选择模块

Skill 是结构化的心理支持指引文档（Markdown 格式），
存储在 skills/*/SKILL.md 中，运行时由 MindBridgeSkillRegistry 加载。

每个 Skill 文件包含：
- YAML frontmatter: name、description
- body: 具体指引内容（可包含 ```text 模板）

Skill 选择逻辑（由 response_skill_names 决定）：
- CHAT 意图 → 不选任何 skill
- RISK 高风险 → supportive_response_baseline + high_risk_safety_plan
- CONSULT 意图 → 根据关键词动态选择：
  - 基础：supportive_response_baseline + referral_resource_guidance
  - 焦虑相关 → + anxiety_grounding_support
  - 睡眠相关 → + sleep_routine_support
  - 学业相关 → + academic_stress_planning

特殊功能：
- counselor_handoff_summary: 使用模板生成个案交接摘要
  （模板从 SKILL.md 中的 ```text 块提取，用 {{key}} 占位符渲染）
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from app.core.enums import IntentType, RiskLevel
from app.models.entities import PsychologicalReport, UserAccount


class SkillLoadError(RuntimeError):
    """Skill 加载异常。"""
    pass


@dataclass(frozen=True)
class MindBridgeSkill:
    """Skill 数据对象。"""
    name: str
    description: str
    body: str
    path: Path

    def prompt_context(self) -> str:
        """将 skill 内容格式化为 LLM prompt 上下文。"""
        return f"应用 skill: {self.name}\n{self.body.strip()}"


class MindBridgeSkillRegistry:
    """
    Skill 注册表。

    扫描 skills/*/SKILL.md 文件，解析 YAML frontmatter 和 body。
    """

    def __init__(self, root: Path | None = None):
        self.root = root or Path(__file__).resolve().parents[2] / "skills"

    def list_skills(self) -> list[MindBridgeSkill]:
        """列出所有已注册的 skill。"""
        if not self.root.exists():
            return []
        skills = []
        for skill_file in sorted(self.root.glob("*/SKILL.md")):
            skills.append(self._load_skill_file(skill_file))
        return skills

    def status_items(self) -> list[dict[str, str]]:
        """返回所有 skill 的状态信息（用于 /api/agent/status）。"""
        if not self.root.exists():
            return []
        items = []
        for skill_file in sorted(self.root.glob("*/SKILL.md")):
            try:
                skill = self._load_skill_file(skill_file)
            except SkillLoadError as exc:
                items.append(
                    {
                        "name": skill_file.parent.name,
                        "status": "FAILED",
                        "description": str(exc),
                        "path": str(skill_file.relative_to(self.root.parent)),
                    }
                )
                continue
            items.append(
                {
                    "name": skill.name,
                    "status": "READY",
                    "description": skill.description,
                    "path": str(skill.path.relative_to(self.root.parent)),
                }
            )
        return items

    def get_required(self, name: str) -> MindBridgeSkill:
        """按名称获取 skill，不存在时抛出异常。"""
        for skill in self.list_skills():
            if skill.name == name:
                return skill
        raise SkillLoadError(f"required standard skill not found: {name}")

    def template_for(self, name: str) -> str:
        """
        从 skill body 中提取 ```text 模板。

        用于 counselor_handoff_summary 等需要模板渲染的 skill。
        模板使用 {{key}} 作为占位符。
        """
        skill = self.get_required(name)
        match = re.search(r"```text\s*\n(?P<template>.*?)\n```", skill.body, re.DOTALL)
        if match is None:
            raise SkillLoadError(f"standard skill {name} does not define a text template")
        return match.group("template").strip()

    def _load_skill_file(self, path: Path) -> MindBridgeSkill:
        """解析单个 SKILL.md 文件。"""
        text = path.read_text(encoding="utf-8")
        metadata, body = _split_frontmatter(text, path)
        name = metadata.get("name") or path.parent.name
        description = metadata.get("description", "")
        if not name.strip():
            raise SkillLoadError(f"{path} is missing frontmatter name")
        if not description.strip():
            raise SkillLoadError(f"{path} is missing frontmatter description")
        if not body.strip():
            raise SkillLoadError(f"{path} is missing skill body")
        return MindBridgeSkill(name=name.strip(), description=description.strip(), body=body.strip(), path=path)


class MindBridgeSkillLibrary:
    """
    Skill 静态工具类。

    提供便捷的静态方法访问 skill 注册表。
    """

    @staticmethod
    def registry() -> MindBridgeSkillRegistry:
        return MindBridgeSkillRegistry()

    @staticmethod
    def list_skills() -> list[MindBridgeSkill]:
        return MindBridgeSkillLibrary.registry().list_skills()

    @staticmethod
    def status_items() -> list[dict[str, str]]:
        return MindBridgeSkillLibrary.registry().status_items()

    @staticmethod
    def response_skill_context(intent: IntentType, risk: RiskLevel, text: str) -> str:
        """
        获取回复时需要注入的 skill 上下文。

        将选中的所有 skill 内容拼接为一个字符串，
        注入到 CounselorAgent 的 system prompt 中。
        """
        names = MindBridgeSkillLibrary.response_skill_names(intent, risk, text)
        registry = MindBridgeSkillLibrary.registry()
        return "\n\n".join(registry.get_required(name).prompt_context() for name in names)

    @staticmethod
    def response_skill_names(intent: IntentType, risk: RiskLevel, text: str) -> list[str]:
        """
        根据意图、风险等级和文本内容选择需要的 skill。

        选择逻辑：
        - CHAT → 空列表
        - RISK HIGH → supportive_response_baseline + high_risk_safety_plan
        - CONSULT → 根据关键词动态组合
        """
        if intent == IntentType.CHAT:
            return []

        if risk == RiskLevel.HIGH:
            return ["supportive_response_baseline", "high_risk_safety_plan"]

        lowered = text.lower()
        names = ["supportive_response_baseline", "referral_resource_guidance"]
        if _contains_any(lowered, ["焦虑", "惊恐", "恐慌", "panic", "anxious", "崩溃", "呼吸"]):
            names.append("anxiety_grounding_support")
        if _contains_any(lowered, ["失眠", "睡不着", "睡眠", "熬夜", "sleep", "insomnia"]):
            names.append("sleep_routine_support")
        if _contains_any(lowered, ["考试", "挂科", "绩点", "论文", "作业", "学业", "学习", "academic", "exam"]):
            names.append("academic_stress_planning")
        return _dedupe(names)

    @staticmethod
    def high_risk_safety_plan_prompt() -> str:
        """获取高风险安全计划的 prompt 内容。"""
        return MindBridgeSkillLibrary.registry().get_required("high_risk_safety_plan").prompt_context()

    @staticmethod
    def counselor_handoff_summary(report: PsychologicalReport, user: UserAccount | None) -> str:
        """
        生成个案交接摘要。

        使用 counselor_handoff_summary skill 中的 ```text 模板，
        填充报告信息和建议的下一步行动。

        用于：
        1. RiskCase 创建时写入 handoff_summary 字段
        2. 预警邮件正文中包含
        """
        template = MindBridgeSkillLibrary.registry().template_for("counselor_handoff_summary")
        student = _student_label(user, report.user_id)
        urgency = "立即跟进" if report.risk_level == RiskLevel.HIGH.value else "尽快跟进"
        next_steps = [
            f"{urgency}，确认学生当前位置、身边是否有人陪伴，以及当前是否安全。",
            "联系学生本人或其可用的现实支持人，并记录已采取的联系方式。",
            "必要时联系校园保卫、心理中心值班老师或当地紧急救助。",
            "将后续安排、接手人和下一次复访时间写入个案备注。",
        ]
        return _render_template(
            template,
            {
                "report_id": str(report.id),
                "student": student,
                "risk_level": report.risk_level,
                "emotion": report.emotion,
                "confidence": f"{report.confidence:.2f}",
                "summary": report.summary,
                "next_steps": "\n".join(f"- {step}" for step in next_steps),
                "content_excerpt": _truncate(report.content, 700),
            },
        )


def _split_frontmatter(text: str, path: Path) -> tuple[dict[str, str], str]:
    """解析 YAML frontmatter，返回 (metadata_dict, body_text)。"""
    if not text.startswith("---\n"):
        raise SkillLoadError(f"{path} is missing YAML frontmatter")
    end = text.find("\n---", 4)
    if end == -1:
        raise SkillLoadError(f"{path} has unterminated YAML frontmatter")
    metadata = {}
    for line in text[4:end].splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise SkillLoadError(f"{path} has invalid frontmatter line: {line}")
        key, value = stripped.split(":", 1)
        metadata[key.strip()] = value.strip().strip("\"'")
    return metadata, text[end + len("\n---") :].strip()


def _contains_any(text: str, terms: list[str]) -> bool:
    """检查文本是否包含任意一个关键词。"""
    return any(term in text for term in terms)


def _dedupe(values: list[str]) -> list[str]:
    """列表去重（保持顺序）。"""
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _render_template(template: str, values: dict[str, str]) -> str:
    """模板渲染：将 {{key}} 替换为对应值。"""
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    return rendered


def _student_label(user: UserAccount | None, user_id: int) -> str:
    """生成学生标签文本。"""
    if user is None:
        return f"userId={user_id}"
    if user.display_name:
        return f"{user.display_name} ({user.username})"
    return user.username


def _truncate(text: str, limit: int) -> str:
    """截断文本到指定长度，超出部分用 ... 替代。"""
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit - 3]}..."
