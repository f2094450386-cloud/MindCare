"""
长对话 Memory 压缩评测 CLI。

用法：
  AI_PROVIDER=mock python -m app.memory_eval.runner
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Memory 压缩评测入口。"""
    from app.core.config import Settings
    from app.memory_eval import run_memory_eval, run_memory_stress_eval

    parser = argparse.ArgumentParser(description="Run MindCare long-dialogue memory compression evaluation.")
    parser.add_argument(
        "--suite",
        choices=["standard", "stress"],
        default="standard",
        help="standard 保持正式门禁；stress 运行三档长历史 Pareto 评测",
    )
    parser.add_argument("--dataset", type=str, default=None, help="场景 JSON 路径")
    parser.add_argument("--output", type=str, default=None, help="报告输出路径")
    parser.add_argument(
        "--split",
        choices=["dev", "holdout"],
        default=None,
        help="只评测指定 semantic split；默认评测全部",
    )
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="只生成报告；质量门槛失败时仍返回 0",
    )
    args = parser.parse_args(argv)

    settings = Settings(
        ai_provider="mock",
        database_url="sqlite://",
        knowledge_vector_enabled=False,
        knowledge_vector_required=False,
        agent_framework="event_driven_multi_agent",
    )

    runner = (
        run_memory_stress_eval
        if args.suite == "stress"
        else run_memory_eval
    )
    report = runner(
        settings,
        dataset_path=Path(args.dataset) if args.dataset else None,
        output_path=Path(args.output) if args.output else None,
        split=args.split,
    )
    summary = {
        "evalName": report.get("evalName"),
        "appVersion": report.get("repro", {}).get("appVersion"),
        "git": report.get("repro", {}).get("git", {}).get("shortCommit"),
        "caseCount": report.get("caseCount"),
        "split": report.get("split"),
        "aggregates": (
            report.get("aggregates")
            or report.get("selectedProfileMetrics")
        ),
        "selectedProfile": report.get("selectedProfile"),
        "paretoProfiles": report.get("paretoProfiles"),
        "qualityGate": report.get("qualityGate"),
        "qualityGateEnforced": not args.no_gate,
        "output": report.get("outputPath"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return _gate_exit_code(report, no_gate=args.no_gate)


def _gate_exit_code(report: dict, *, no_gate: bool) -> int:
    """Default to a failing process when semantic Memory quality does not pass."""
    if no_gate:
        return 0
    return 0 if report.get("qualityGate", {}).get("passed") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
