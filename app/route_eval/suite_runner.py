"""Run the regression, standard, and challenge route suites together.

The command merges execution and reporting only.  Metrics remain separated by
suite because the three datasets have different sampling policies and uses.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.eval import utc_now_iso, write_json_report
from app.route_eval import run_route_eval


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "target" / "eval" / "route-suite"
DEFAULT_OUTPUT = ROOT / "target" / "eval" / "route-suite-report.json"


@dataclass(frozen=True)
class RouteSuiteSpec:
    """One independently sampled route evaluation suite."""

    key: str
    dataset_name: str
    purpose: str
    metric_role: str
    enforce_regression_gate: bool = False


SUITE_SPECS = (
    RouteSuiteSpec(
        key="regression",
        dataset_name="route_eval.jsonl",
        purpose="发布稳定性门禁，验证已知规则与安全边界不回退。",
        metric_role="release_gate",
        enforce_regression_gate=True,
    ),
    RouteSuiteSpec(
        key="standard",
        dataset_name="route_standard_eval.jsonl",
        purpose="常见用户单轮话术的分层标准评测，作为主要对外指标集。",
        metric_role="primary_benchmark",
    ),
    RouteSuiteSpec(
        key="challenge",
        dataset_name="route_challenge_eval.jsonl",
        purpose="集中检查隐晦表达、否定和语义边界等困难样本。",
        metric_role="diagnostic",
    ),
)


def run_route_suite(
    settings: Any,
    *,
    dataset_dir: Path | None = None,
    output_dir: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run all route suites and write one manifest plus three full reports."""
    datasets = Path(dataset_dir or ROOT / "datasets")
    reports_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    manifest_path = Path(output_path or DEFAULT_OUTPUT)

    suite_summaries: list[dict[str, Any]] = []
    for spec in SUITE_SPECS:
        dataset_path = datasets / spec.dataset_name
        report_path = reports_dir / f"{spec.key}-report.json"
        report = run_route_eval(
            settings,
            dataset_path=dataset_path,
            output_path=report_path,
        )
        suite_summaries.append(_summarize_suite(spec, report))

    total_cases = sum(item["caseCount"] for item in suite_summaries)
    manifest = {
        "evalName": "route_eval_suite",
        "createdAt": utc_now_iso(),
        "caseCount": total_cases,
        "primaryMetricSuite": "standard",
        "releaseGateSuite": "regression",
        "diagnosticSuite": "challenge",
        "samplingPolicy": (
            "三套均为离线自建数据；standard 是分层覆盖集，不代表线上类别自然占比。"
        ),
        "aggregation": {
            "executionMerged": True,
            "blendedMetrics": None,
            "reason": "数据用途和难度分布不同，不计算跨套混合准确率。",
        },
        "suites": suite_summaries,
        "outputPath": str(manifest_path),
    }
    write_json_report(manifest_path, manifest)
    return manifest


def _summarize_suite(spec: RouteSuiteSpec, report: dict[str, Any]) -> dict[str, Any]:
    dataset_meta = report.get("repro", {}).get("datasets", [])
    dataset_sha256 = dataset_meta[0].get("sha256") if dataset_meta else None
    raw_gate = report.get("qualityGate", {})
    gate = {
        "enforced": spec.enforce_regression_gate,
        "passed": raw_gate.get("passed") if spec.enforce_regression_gate else None,
    }
    if not spec.enforce_regression_gate:
        gate["reason"] = "发布回归门槛不用于标准集或挑战集。"
    return {
        "key": spec.key,
        "purpose": spec.purpose,
        "metricRole": spec.metric_role,
        "caseCount": report.get("caseCount"),
        "intentAccuracy": report.get("intent", {}).get("accuracy"),
        "intentMacroF1": report.get("intent", {}).get("macroF1"),
        "riskAccuracy": report.get("risk", {}).get("accuracy"),
        "riskMacroF1": report.get("risk", {}).get("macroF1"),
        "highRiskRecall": report.get("riskRecall"),
        "highRiskFalsePositiveRate": report.get("riskFalsePositiveRate"),
        "highRiskSupport": report.get("riskSupport"),
        "safetyOverrideAccuracy": report.get("safetyOverrideAccuracy"),
        "badCaseCount": len(report.get("badCases", [])),
        "gate": gate,
        "datasetSha256": dataset_sha256,
        "reportPath": report.get("outputPath"),
    }


def _exit_code(manifest: dict[str, Any], *, no_gate: bool) -> int:
    """Only the release-regression suite controls the process gate."""
    if no_gate:
        return 0
    regression = next(
        (item for item in manifest.get("suites", []) if item.get("key") == "regression"),
        None,
    )
    return 0 if regression and regression.get("gate", {}).get("passed") is True else 2


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the merged route suite."""
    from app.core.config import Settings

    parser = argparse.ArgumentParser(description="Run all MindBridge route evaluation suites.")
    parser.add_argument("--dataset-dir", type=str, default=None, help="三套 JSONL 所在目录")
    parser.add_argument("--output-dir", type=str, default=None, help="三份完整子报告目录")
    parser.add_argument("--output", type=str, default=None, help="合并摘要报告路径")
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="只生成报告；发布回归门禁失败时仍返回 0",
    )
    args = parser.parse_args(argv)
    settings = Settings(
        ai_provider="mock",
        database_url="sqlite://",
        knowledge_vector_enabled=False,
        knowledge_vector_required=False,
        agent_framework="event_driven_multi_agent",
    )
    manifest = run_route_suite(
        settings,
        dataset_dir=Path(args.dataset_dir) if args.dataset_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        output_path=Path(args.output) if args.output else None,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return _exit_code(manifest, no_gate=args.no_gate)


if __name__ == "__main__":
    raise SystemExit(main())
