import tempfile
import unittest
from pathlib import Path

from app.core.enums import IntentType, RiskLevel
from app.services.skills import MindBridgeSkillLibrary, MindBridgeSkillRegistry, SkillLoadError


def write_skill(root: Path, name: str, text: str) -> None:
    path = root / name / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")


class SkillRegistryTests(unittest.TestCase):
    def test_skill_registry_loads_valid_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_skill(
                root,
                "demo_skill",
                """---\nname: demo_skill\ndescription: Use for a clear and sufficiently described demo scenario.\n---\n\n# Demo\n\n## Workflow\n\n- Do one thing.\n""",
            )

            skill = MindBridgeSkillRegistry(root).get_required("demo_skill")

            self.assertEqual(skill.name, "demo_skill")
            self.assertEqual(skill.validation_issues(), [])

    def test_skill_status_reports_warnings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_skill(
                root,
                "demo_skill",
                """---\nname: demo_skill\ndescription: short\n---\n\n# Demo\n""",
            )

            status = MindBridgeSkillRegistry(root).status_items()[0]

            self.assertEqual(status["status"], "WARN")
            self.assertTrue(status["issues"])

    def test_skill_requires_frontmatter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_skill(root, "bad", "# Missing metadata")

            with self.assertRaises(SkillLoadError):
                MindBridgeSkillRegistry(root).get_required("bad")

    def test_high_risk_skill_contract_overrides_chat_intent(self):
        names = MindBridgeSkillLibrary.response_skill_names(
            IntentType.CHAT,
            RiskLevel.HIGH,
            "请解释一个普通技术概念。",
        )
        self.assertEqual(
            names,
            ["supportive_response_baseline", "high_risk_safety_plan"],
        )

        context = MindBridgeSkillLibrary.response_skill_context(
            IntentType.CHAT,
            RiskLevel.HIGH,
            "请解释一个普通技术概念。",
        )
        self.assertIn("应用 skill: high_risk_safety_plan", context)


if __name__ == "__main__":
    unittest.main()
