"""
路由评测 CLI。

用法：
  AI_PROVIDER=mock python -m app.route_eval.runner
  AI_PROVIDER=mock python -m app.route_eval.runner --split holdout
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """路由评测入口。"""
    from app.core.config import Settings
    from app.route_eval import run_route_eval

    parser = argparse.ArgumentParser(description="Run MindCare CHAT/CONSULT/RISK route evaluation.")
    parser.add_argument("--dataset", type=str, default=None, help="JSONL 数据集路径")
    parser.add_argument("--output", type=str, default=None, help="报告输出路径")
    parser.add_argument("--split", type=str, default=None, help="仅评测指定 split，例如 holdout")
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

    report = run_route_eval(
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
        "intentAccuracy": report.get("intent", {}).get("accuracy"),
        "riskAccuracy": report.get("risk", {}).get("accuracy"),
        "riskRecall": report.get("riskRecall"),
        "riskFalsePositiveRate": report.get("riskFalsePositiveRate"),
        "qualityGate": report.get("qualityGate"),
        "qualityGateEnforced": not args.no_gate,
        "output": report.get("outputPath"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return _gate_exit_code(report, no_gate=args.no_gate)


def _gate_exit_code(report: dict, *, no_gate: bool) -> int:
    """Default to a failing process when Route safety quality does not pass."""
    if no_gate:
        return 0
    return 0 if report.get("qualityGate", {}).get("passed") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
