"""Offline evaluation of the production Memory and Response prompt assembly."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from app.agents.autonomous import (
    RESPONSE_PLAN_PROMPT_REVIEW_SCOPE,
    ContextAgent,
    ResponseAgent,
)
from app.agents.decision import decide_route
from app.agents.factory import agent_framework_status
from app.agents.prompt_assembly import build_response_messages
from app.core.config import Settings, get_settings
from app.core.enums import IntentType, RiskLevel
from app.eval import attach_repro_header, resolve_split_report_path, write_json_report
from app.schemas.dtos import AiMessage
from app.services.agent_models import AgentModelRegistry
from app.services.assessment import risk_order
from app.services.memory import assemble_memory_context, summarize_memory_for_prompt


VALID_ROLES = {"user", "assistant"}
VALID_RISKS = {item.value for item in RiskLevel}
VALID_SPLITS = {"dev", "holdout"}
TOKEN_ESTIMATE_METHOD = "cjk_chars_plus_ascii_wordpieces_v1"
MEMORY_STRESS_PROFILES = {
    "conservative": {
        "redis_memory_max_messages": 80,
        "chat_history_limit": 10,
        "memory_compaction_recent_messages": 8,
        "memory_summary_max_chars": 500,
    },
    "balanced": {
        "redis_memory_max_messages": 80,
        "chat_history_limit": 10,
        "memory_compaction_recent_messages": 6,
        "memory_summary_max_chars": 350,
    },
    "aggressive": {
        "redis_memory_max_messages": 80,
        "chat_history_limit": 10,
        "memory_compaction_recent_messages": 4,
        "memory_summary_max_chars": 250,
    },
}
MEMORY_STRESS_QUALITY_THRESHOLDS = {
    "currentAvgRetainRecallMin": 0.95,
    "currentMinRetainRecallMin": 0.8,
    "currentAvgFactCorrectnessMin": 0.9,
    "currentMaxForbiddenRetentionRateMax": 0.0,
    "zeroRetainCaseCountMax": 0,
    "safetyFactRecallMin": 1.0,
    "avgEstimatedTokenReductionRatioMin": 0.25,
    "avgEstimatedModelHistoryTokenReductionRatioMin": 0.4,
    "positiveEstimatedTokenReductionRateMin": 0.9,
}
MEMORY_QUALITY_THRESHOLDS = {
    "currentAvgRetainRecallMin": 0.9,
    "currentMinRetainRecallMin": 0.8,
    "currentAvgFactCorrectnessMin": 0.9,
    "currentAvgForbiddenRetentionRateMax": 0.1,
    "currentMaxForbiddenRetentionRateMax": 0.0,
    "zeroRetainCaseCountMax": 0,
    "avgEstimatedTokenReductionRatioMin": 0.1,
    "positiveEstimatedTokenReductionRateMin": 0.9,
    "onlineRouteMinimumRiskPassRateMin": 1.0,
    "requiredCrisisConstraintInjectionRateMin": 1.0,
    "semanticBoundarySliceSupportMin": 1,
    "semanticBoundarySliceMinRetainRecallMin": 1.0,
    "semanticBoundarySliceMaxForbiddenRetentionRateMax": 0.0,
}
MEMORY_SEMANTIC_BOUNDARY_CATEGORIES = {
    "compatibleFacts": frozenset({"compatible_property_dimensions"}),
    "collectionMembers": frozenset(
        {
            "communication_medium_collection",
            "sleep_member_collection",
            "exercise_member_collection",
            "career_option_collection",
        }
    ),
    "explicitStateCorrections": frozenset(
        {"dietary_recovery", "allergy_correction"}
    ),
    "uncertainStateUpdates": frozenset({"uncertain_state_update"}),
    "atomicMemberUpdates": frozenset(
        {
            "atomic_collection_member_update",
            "coordinated_collection_member_update",
        }
    ),
    "uncertainCollectionUpdates": frozenset(
        {
            "uncertain_collection_member_update",
            "coordinated_uncertain_collection_update",
        }
    ),
    "coordinatedCollectionSyntax": frozenset(
        {
            "coordinated_collection_member_update",
            "coordinated_uncertain_collection_update",
        }
    ),
}


def load_memory_cases(path: Path) -> list[dict[str, Any]]:
    """Load and validate long-dialogue Memory scenarios."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path} 未包含非空 cases")
    seen_ids: set[str] = set()
    group_splits: dict[str, str] = {}
    for index, case in enumerate(cases):
        _validate_memory_case(case, path, index)
        case_id = str(case["id"])
        if case_id in seen_ids:
            raise ValueError(f"{path} cases[{index}] 重复 id: {case_id}")
        seen_ids.add(case_id)
        group = str(case["group"])
        split = str(case["split"])
        previous = group_splits.setdefault(group, split)
        if previous != split:
            raise ValueError(
                f"{path} cases[{index}] group {group} 跨 split: {previous}/{split}"
            )
    return cases


def estimate_tokens(text: str) -> int:
    """Return an explicitly labelled deterministic token estimate."""
    if not text:
        return 0
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    ascii_words = re.findall(r"[A-Za-z0-9_]+", text)
    ascii_estimate = sum(max(1, math.ceil(len(word) / 4)) for word in ascii_words)
    punctuation = len(re.findall(r"[^\w\s\u3400-\u9fff]", text))
    return cjk + ascii_estimate + math.ceil(punctuation / 2)


def evaluate_memory_case(case: dict[str, Any], settings: Settings) -> dict[str, Any]:
    """Evaluate one case in the production order: route, then Context compaction."""
    history = [
        AiMessage(role=str(item["role"]).lower(), content=str(item["content"]))
        for item in case["messages"]
    ]
    current_input = str(case["current_input"])
    must_retain = [str(item) for item in case["must_retain"]]
    must_not_retain = [str(item) for item in case.get("must_not_retain", [])]
    registry = AgentModelRegistry(settings)
    online_decision = decide_route(
        current_input,
        history,
        settings,
        understanding_client=registry.client_for("UnderstandingAgent"),
        safety_client=registry.client_for("SafetyAgent"),
    )
    context_eligible = (
        online_decision.understanding_intent != IntentType.CHAT
        or online_decision.assessment.risk != RiskLevel.LOW
    )
    if not context_eligible:
        raise ValueError(
            f"memory case {case['id']} 不会触发线上 ContextAgent，不能作为压缩评测样本"
        )
    online_artifacts = _online_context_artifacts(online_decision)
    shared_brief = summarize_memory_for_prompt(
        history,
        current_input,
        settings,
        registry.client_for("ContextAgent"),
        ContextAgent.profile.system_prompt,
    )

    baselines = {
        "none": _assemble_baseline(
            "none",
            history,
            current_input,
            settings,
            shared_brief,
            online_artifacts,
            False,
            online_decision,
        ),
        "current": _assemble_baseline(
            "current",
            history,
            current_input,
            settings,
            shared_brief,
            online_artifacts,
            True,
            online_decision,
        ),
    }
    for baseline in baselines.values():
        baseline["facts"] = _fact_metrics(
            baseline.pop("_promptCorpus"),
            must_retain,
            must_not_retain,
        )

    expected_min_risk = str(case.get("minimum_expected_risk") or RiskLevel.LOW.value).upper()
    meets_minimum_risk = (
        risk_order(online_decision.assessment.risk)
        >= risk_order(RiskLevel(expected_min_risk))
    )
    return {
        "id": case["id"],
        "category": case["category"],
        "mustRetain": must_retain,
        "mustNotRetain": must_not_retain,
        "minimumExpectedRisk": expected_min_risk,
        "onlineRoute": {
            "inputStage": "pre_compaction_session_history",
            "historyMessageCount": len(history),
            "understandingIntent": online_decision.understanding_intent.value,
            "risk": online_decision.assessment.risk.value,
            "finalIntent": online_decision.final_intent.value,
            "safetyOverride": online_decision.safety_override,
            "productionContextEligible": context_eligible,
            "meetsMinimumExpectedRisk": meets_minimum_risk,
        },
        "baselines": baselines,
        "estimatedTokenReductionRatio": _reduction_ratio(
            baselines["none"]["estimatedTokens"],
            baselines["current"]["estimatedTokens"],
        ),
        "estimatedModelHistoryTokenReductionRatio": _reduction_ratio(
            baselines["none"]["estimatedModelHistoryTokens"],
            baselines["current"]["estimatedModelHistoryTokens"],
        ),
        "factCorrectnessDelta": round(
            baselines["current"]["facts"]["correctness"]
            - baselines["none"]["facts"]["correctness"],
            6,
        ),
    }


def evaluate_memory_dataset(
    cases: list[dict[str, Any]],
    settings: Settings | None = None,
    *,
    split: str | None = None,
) -> dict[str, Any]:
    """Evaluate all cases; empty input is an error, never a zero-sample success."""
    if not cases:
        raise ValueError("memory eval cases 不能为空")
    if split not in {None, *VALID_SPLITS}:
        raise ValueError(f"未知 memory split: {split}")
    selected_cases = (
        [case for case in cases if case.get("split") == split]
        if split
        else list(cases)
    )
    if not selected_cases:
        raise ValueError(f"memory eval split={split or 'all'} 没有样本")
    settings = settings or get_settings()
    _require_event_driven(settings)
    rows = [evaluate_memory_case(case, settings) for case in selected_cases]
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row)

    def average(values: list[float]) -> float:
        return round(sum(values) / len(values), 6)

    aggregates = {
        "none": _aggregate_baseline(rows, "none", average),
        "current": _aggregate_baseline(rows, "current", average),
        "avgEstimatedTokenReductionRatio": average(
            [row["estimatedTokenReductionRatio"] for row in rows]
        ),
        "positiveEstimatedTokenReductionRate": average(
            [
                1.0 if row["estimatedTokenReductionRatio"] > 0 else 0.0
                for row in rows
            ]
        ),
        "avgEstimatedModelHistoryTokenReductionRatio": average(
            [row["estimatedModelHistoryTokenReductionRatio"] for row in rows]
        ),
        "positiveEstimatedModelHistoryTokenReductionRate": average(
            [
                1.0
                if row["estimatedModelHistoryTokenReductionRatio"] > 0
                else 0.0
                for row in rows
            ]
        ),
        "onlineRouteMinimumRiskPassRate": average(
            [1.0 if row["onlineRoute"]["meetsMinimumExpectedRisk"] else 0.0 for row in rows]
        ),
        "requiredCrisisConstraintInjectionRate": (
            _required_crisis_constraint_metric(rows)
        ),
    }
    semantic_boundary_slices = _semantic_boundary_slices(rows)
    return {
        "split": split or "all",
        "caseCount": len(rows),
        "tokenMeasurement": {
            "kind": "estimate",
            "exactTokenizer": False,
            "method": TOKEN_ESTIMATE_METHOD,
        },
        "aggregates": aggregates,
        "semanticBoundarySlices": semantic_boundary_slices,
        "qualityGate": _memory_quality_gate(
            rows,
            aggregates,
            semantic_boundary_slices,
        ),
        "byCategory": {
            category: {
                "count": len(items),
                "avgRetainRecallCurrent": average(
                    [
                        item["baselines"]["current"]["facts"]["retainRecall"]
                        for item in items
                    ]
                ),
                "minRetainRecallCurrent": min(
                    item["baselines"]["current"]["facts"]["retainRecall"]
                    for item in items
                ),
                "avgForbiddenRetentionRateCurrent": average(
                    [
                        item["baselines"]["current"]["facts"][
                            "forbiddenRetentionRate"
                        ]
                        for item in items
                    ]
                ),
                "maxForbiddenRetentionRateCurrent": max(
                    item["baselines"]["current"]["facts"][
                        "forbiddenRetentionRate"
                    ]
                    for item in items
                ),
                "avgFactCorrectnessCurrent": average(
                    [item["baselines"]["current"]["facts"]["correctness"] for item in items]
                ),
                "avgEstimatedTokenReductionRatio": average(
                    [item["estimatedTokenReductionRatio"] for item in items]
                ),
                "avgEstimatedModelHistoryTokenReductionRatio": average(
                    [
                        item["estimatedModelHistoryTokenReductionRatio"]
                        for item in items
                    ]
                ),
                "requiredCrisisConstraintInjectionRate": (
                    _required_crisis_constraint_metric(items)
                ),
            }
            for category, items in sorted(by_category.items())
        },
        "results": rows,
    }


def run_memory_eval(
    settings: Settings | None = None,
    *,
    dataset_path: Path | None = None,
    output_path: Path | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    """Run Memory evaluation and write a reproducible report."""
    settings = settings or get_settings()
    framework = _require_event_driven(settings)
    dataset = Path(dataset_path or settings.memory_eval_dataset)
    output = (
        Path(output_path)
        if output_path is not None
        else resolve_split_report_path(Path(settings.memory_eval_output), split)
    )
    cases = load_memory_cases(dataset)
    body = evaluate_memory_dataset(cases, settings, split=split)
    report = attach_repro_header(
        body,
        dataset_paths=[dataset],
        config_snapshot=_config_snapshot(settings, framework),
        prompt_rules_version="memory-response-prompt-v1",
        rule_source_paths=[
            Path(__file__).resolve().parents[1] / "services" / "memory.py",
            Path(__file__).resolve().parents[1] / "agents" / "prompt_assembly.py",
            Path(__file__).resolve().parents[1] / "agents" / "decision.py",
            Path(__file__).resolve().parents[1] / "services" / "ai.py",
        ],
        eval_name="memory_compression_eval",
        app_version=getattr(settings, "app_version", ""),
    )
    report["outputPath"] = str(output)
    write_json_report(output, report)
    return report


def evaluate_memory_stress_profiles(
    cases: list[dict[str, Any]],
    settings: Settings | None = None,
    *,
    split: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate production compaction under three explicit long-history budgets.

    The standard dataset and its gate remain unchanged. Stress profiles expose a
    quality/cost frontier and select the highest full-prompt reduction that still
    satisfies the stricter stress semantic contract.
    """
    settings = settings or get_settings()
    _require_event_driven(settings)
    profile_reports: dict[str, dict[str, Any]] = {}
    for name, overrides in MEMORY_STRESS_PROFILES.items():
        profile_settings = settings.model_copy(update=overrides)
        report = evaluate_memory_dataset(
            cases,
            profile_settings,
            split=split,
        )
        report["standardQualityGate"] = report.pop("qualityGate")
        report["profileSettings"] = dict(overrides)
        report["qualityGate"] = _memory_stress_quality_gate(report)
        profile_reports[name] = report

    eligible = [
        name
        for name, report in profile_reports.items()
        if report["qualityGate"]["passed"]
    ]
    selected_profile = (
        max(
            eligible,
            key=lambda name: profile_reports[name]["aggregates"][
                "avgEstimatedTokenReductionRatio"
            ],
        )
        if eligible
        else None
    )
    selected_metrics = (
        profile_reports[selected_profile]["qualityGate"]["observed"]
        if selected_profile is not None
        else None
    )
    pareto_profiles = _memory_stress_pareto_profiles(profile_reports)
    return {
        "split": split or "all",
        "caseCount": (
            profile_reports[next(iter(profile_reports))]["caseCount"]
            if profile_reports
            else 0
        ),
        "profileOrder": list(MEMORY_STRESS_PROFILES),
        "stressThresholds": dict(MEMORY_STRESS_QUALITY_THRESHOLDS),
        "selectedProfile": selected_profile,
        "selectedProfileMetrics": selected_metrics,
        "paretoProfiles": pareto_profiles,
        "qualityGate": {
            "policy": "memory-stress-pareto-v1",
            "passed": selected_profile is not None,
            "status": "pass" if selected_profile is not None else "fail",
            "thresholds": dict(MEMORY_STRESS_QUALITY_THRESHOLDS),
            "eligibleProfiles": eligible,
            "selectedProfile": selected_profile,
            "observed": selected_metrics,
            "failures": (
                []
                if selected_profile is not None
                else [
                    {
                        "metric": "eligibleProfileCount",
                        "actual": 0,
                        "operator": ">=",
                        "threshold": 1,
                        "reason": "没有压缩档位同时满足压力集语义与成本门槛",
                    }
                ]
            ),
            "failedCaseIds": (
                []
                if selected_profile is not None
                else sorted(
                    {
                        case_id
                        for report in profile_reports.values()
                        for case_id in report["qualityGate"][
                            "failedCaseIds"
                        ]
                    }
                )
            ),
        },
        "profiles": profile_reports,
    }


def run_memory_stress_eval(
    settings: Settings | None = None,
    *,
    dataset_path: Path | None = None,
    output_path: Path | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    """Run the independent long-history stress suite and write one Pareto report."""
    settings = settings or get_settings()
    framework = _require_event_driven(settings)
    dataset = Path(
        dataset_path
        or "datasets/memory_compression_stress_eval.json"
    )
    default_output = Path(settings.memory_eval_output).with_name(
        "memory-stress-eval-report.json"
    )
    output = (
        Path(output_path)
        if output_path is not None
        else resolve_split_report_path(default_output, split)
    )
    cases = load_memory_cases(dataset)
    body = evaluate_memory_stress_profiles(cases, settings, split=split)
    config_snapshot = _config_snapshot(settings, framework)
    config_snapshot["stressProfiles"] = {
        name: dict(values)
        for name, values in MEMORY_STRESS_PROFILES.items()
    }
    report = attach_repro_header(
        body,
        dataset_paths=[dataset],
        config_snapshot=config_snapshot,
        prompt_rules_version="memory-response-prompt-stress-v1",
        rule_source_paths=[
            Path(__file__).resolve().parents[1] / "services" / "memory.py",
            Path(__file__).resolve().parents[1] / "agents" / "prompt_assembly.py",
            Path(__file__).resolve().parents[1] / "agents" / "decision.py",
            Path(__file__).resolve().parents[1] / "services" / "ai.py",
        ],
        eval_name="memory_compression_stress_eval",
        app_version=getattr(settings, "app_version", ""),
    )
    report["outputPath"] = str(output)
    write_json_report(output, report)
    return report


def _assemble_baseline(
    name: str,
    history: list[AiMessage],
    current_input: str,
    settings: Settings,
    memory_brief: str,
    artifacts: list[dict[str, Any]],
    compaction_enabled: bool,
    online_decision,
) -> dict[str, Any]:
    context = assemble_memory_context(
        history,
        current_input,
        settings,
        compaction_enabled=compaction_enabled,
        memory_brief=memory_brief,
        artifacts=artifacts,
    )
    messages, mode = build_response_messages(
        intent=online_decision.understanding_intent,
        risk=online_decision.assessment.risk,
        display_name="评测用户",
        model_history=context.model_history,
        memory_brief=context.memory_brief,
        response_agent_system_prompt=ResponseAgent.profile.system_prompt,
        private_memory_text="无",
    )
    prompt_corpus = "\n".join(f"{message.role}: {message.content}" for message in messages)
    model_history_corpus = "\n".join(
        f"{message.role}: {message.content}"
        for message in context.model_history
    )
    crisis_rule_required = online_decision.assessment.risk == RiskLevel.HIGH
    crisis_rule_present = "高风险处理规则" in prompt_corpus
    return {
        "name": name,
        "compactionEnabled": compaction_enabled,
        "sourceMessageCount": context.source_message_count,
        "modelHistoryCount": len(context.model_history),
        "memoryBrief": context.memory_brief,
        "promptMode": mode,
        "promptChars": len(prompt_corpus),
        "estimatedTokens": estimate_tokens(prompt_corpus),
        "estimatedModelHistoryTokens": estimate_tokens(model_history_corpus),
        "promptSha256": _sha256(prompt_corpus),
        "memoryBriefDuplicatedInModelHistory": any(
            context.memory_brief
            and context.memory_brief != "无相关历史记忆。"
            and context.memory_brief in message.content
            for message in context.model_history
        ),
        "crisisConstraint": {
            "source": "pre_compaction_online_route",
            "onlineRisk": online_decision.assessment.risk.value,
            "required": crisis_rule_required,
            "present": crisis_rule_present,
            "injected": crisis_rule_present if crisis_rule_required else None,
            "reviewScope": RESPONSE_PLAN_PROMPT_REVIEW_SCOPE,
            "validationScope": "pre_generation_prompt_constraint_presence_only",
            "reviewedGeneratedText": False,
            "validatesFinalResponseSafety": False,
        },
        "_promptCorpus": prompt_corpus,
    }


def _online_context_artifacts(online_decision) -> list[dict[str, Any]]:
    """Mirror the intent/risk artifacts ContextAgent receives in production."""
    return [
        {
            "kind": "intent",
            "payload": {
                "intent": online_decision.understanding_intent.value,
                "reason": online_decision.intent_reason,
            },
        },
        {
            "kind": "risk",
            "payload": {
                "risk": online_decision.assessment.risk.value,
                "summary": online_decision.assessment.summary,
                "reason": online_decision.safety_reason,
            },
        },
    ]


def _fact_metrics(
    corpus: str,
    must_retain: list[str],
    must_not_retain: list[str],
) -> dict[str, Any]:
    retained = [fact for fact in must_retain if fact and fact in corpus]
    forbidden = [fact for fact in must_not_retain if fact and fact in corpus]
    retain_recall = len(retained) / len(must_retain) if must_retain else 1.0
    forbidden_rate = len(forbidden) / len(must_not_retain) if must_not_retain else 0.0
    denominator = len(must_retain) + len(must_not_retain)
    correctness = (
        (len(retained) + len(must_not_retain) - len(forbidden)) / denominator
        if denominator
        else 1.0
    )
    return {
        "retainRecall": round(retain_recall, 6),
        "forbiddenRetentionRate": round(forbidden_rate, 6),
        "correctness": round(correctness, 6),
        "retained": retained,
        "missing": [fact for fact in must_retain if fact not in retained],
        "forbiddenRetained": forbidden,
    }


def _aggregate_baseline(rows, name, average):
    retain_recalls = [
        row["baselines"][name]["facts"]["retainRecall"]
        for row in rows
    ]
    forbidden_rates = [
        row["baselines"][name]["facts"]["forbiddenRetentionRate"]
        for row in rows
    ]
    return {
        "avgEstimatedTokens": average(
            [row["baselines"][name]["estimatedTokens"] for row in rows]
        ),
        "avgEstimatedModelHistoryTokens": average(
            [
                row["baselines"][name]["estimatedModelHistoryTokens"]
                for row in rows
            ]
        ),
        "avgRetainRecall": average(retain_recalls),
        "minRetainRecall": min(retain_recalls),
        "zeroRetainCaseCount": sum(value == 0 for value in retain_recalls),
        "avgForbiddenRetentionRate": average(forbidden_rates),
        "maxForbiddenRetentionRate": max(forbidden_rates),
        "avgFactCorrectness": average(
            [row["baselines"][name]["facts"]["correctness"] for row in rows]
        ),
    }


def _semantic_boundary_slices(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Report strict current-baseline quality for regression-critical semantics."""
    slices: dict[str, dict[str, Any]] = {}
    for name, categories in MEMORY_SEMANTIC_BOUNDARY_CATEGORIES.items():
        selected = [row for row in rows if row["category"] in categories]
        if not selected:
            slices[name] = {
                "support": 0,
                "minRetainRecall": None,
                "maxForbiddenRetentionRate": None,
                "avgFactCorrectness": None,
            }
            continue
        facts = [row["baselines"]["current"]["facts"] for row in selected]
        slices[name] = {
            "support": len(selected),
            "minRetainRecall": min(item["retainRecall"] for item in facts),
            "maxForbiddenRetentionRate": max(
                item["forbiddenRetentionRate"] for item in facts
            ),
            "avgFactCorrectness": round(
                sum(item["correctness"] for item in facts) / len(facts),
                6,
            ),
        }
    return slices


def _memory_quality_gate(
    rows: list[dict[str, Any]],
    aggregates: dict[str, Any],
    semantic_boundary_slices: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Apply semantic preservation and meaningful-compression release thresholds."""
    current = aggregates["current"]
    crisis = aggregates["requiredCrisisConstraintInjectionRate"]
    observed = {
        "currentAvgRetainRecall": current["avgRetainRecall"],
        "currentMinRetainRecall": current["minRetainRecall"],
        "currentAvgFactCorrectness": current["avgFactCorrectness"],
        "currentAvgForbiddenRetentionRate": current["avgForbiddenRetentionRate"],
        "currentMaxForbiddenRetentionRate": current["maxForbiddenRetentionRate"],
        "zeroRetainCaseCount": current["zeroRetainCaseCount"],
        "avgEstimatedTokenReductionRatio": aggregates[
            "avgEstimatedTokenReductionRatio"
        ],
        "positiveEstimatedTokenReductionRate": aggregates[
            "positiveEstimatedTokenReductionRate"
        ],
        "onlineRouteMinimumRiskPassRate": aggregates[
            "onlineRouteMinimumRiskPassRate"
        ],
        "requiredCrisisConstraintInjectionRate": crisis["rate"],
        "semanticBoundarySlices": semantic_boundary_slices,
    }
    checks = [
        (
            "currentAvgRetainRecall",
            observed["currentAvgRetainRecall"],
            ">=",
            MEMORY_QUALITY_THRESHOLDS["currentAvgRetainRecallMin"],
            "压缩后必须保留绝大多数标注事实",
        ),
        (
            "currentMinRetainRecall",
            observed["currentMinRetainRecall"],
            ">=",
            MEMORY_QUALITY_THRESHOLDS["currentMinRetainRecallMin"],
            "任何单个场景都不能大量丢失必须保留事实",
        ),
        (
            "currentAvgFactCorrectness",
            observed["currentAvgFactCorrectness"],
            ">=",
            MEMORY_QUALITY_THRESHOLDS["currentAvgFactCorrectnessMin"],
            "事实保留与 stale/forbidden 惩罚后的正确性不足",
        ),
        (
            "currentAvgForbiddenRetentionRate",
            observed["currentAvgForbiddenRetentionRate"],
            "<=",
            MEMORY_QUALITY_THRESHOLDS["currentAvgForbiddenRetentionRateMax"],
            "压缩后保留了过多 stale、矛盾或禁止事实",
        ),
        (
            "currentMaxForbiddenRetentionRate",
            observed["currentMaxForbiddenRetentionRate"],
            "<=",
            MEMORY_QUALITY_THRESHOLDS["currentMaxForbiddenRetentionRateMax"],
            "至少一个场景仍保留 stale、矛盾或禁止事实",
        ),
        (
            "zeroRetainCaseCount",
            observed["zeroRetainCaseCount"],
            "<=",
            MEMORY_QUALITY_THRESHOLDS["zeroRetainCaseCountMax"],
            "存在 must_retain 全部丢失的场景",
        ),
        (
            "avgEstimatedTokenReductionRatio",
            observed["avgEstimatedTokenReductionRatio"],
            ">=",
            MEMORY_QUALITY_THRESHOLDS["avgEstimatedTokenReductionRatioMin"],
            "估算 token 降幅不足 10%，压缩收益不具备实际意义",
        ),
        (
            "positiveEstimatedTokenReductionRate",
            observed["positiveEstimatedTokenReductionRate"],
            ">=",
            MEMORY_QUALITY_THRESHOLDS["positiveEstimatedTokenReductionRateMin"],
            "有过多场景没有获得任何估算 token 收益",
        ),
        (
            "onlineRouteMinimumRiskPassRate",
            observed["onlineRouteMinimumRiskPassRate"],
            ">=",
            MEMORY_QUALITY_THRESHOLDS["onlineRouteMinimumRiskPassRateMin"],
            "预压缩线上路由未达到标注的最低风险等级",
        ),
    ]
    if crisis["support"]:
        checks.append(
            (
                "requiredCrisisConstraintInjectionRate",
                observed["requiredCrisisConstraintInjectionRate"],
                ">=",
                MEMORY_QUALITY_THRESHOLDS[
                    "requiredCrisisConstraintInjectionRateMin"
                ],
                "需要危机约束的 HIGH 样本未全部注入 Prompt 约束",
            )
        )
    for name, metrics in semantic_boundary_slices.items():
        checks.append(
            (
                f"semanticBoundarySlices.{name}.support",
                metrics["support"],
                ">=",
                MEMORY_QUALITY_THRESHOLDS["semanticBoundarySliceSupportMin"],
                f"语义边界 slice {name} 缺少样本支持",
            )
        )
        if metrics["support"]:
            checks.extend(
                [
                    (
                        f"semanticBoundarySlices.{name}.minRetainRecall",
                        metrics["minRetainRecall"],
                        ">=",
                        MEMORY_QUALITY_THRESHOLDS[
                            "semanticBoundarySliceMinRetainRecallMin"
                        ],
                        f"语义边界 slice {name} 存在事实丢失",
                    ),
                    (
                        (
                            f"semanticBoundarySlices.{name}."
                            "maxForbiddenRetentionRate"
                        ),
                        metrics["maxForbiddenRetentionRate"],
                        "<=",
                        MEMORY_QUALITY_THRESHOLDS[
                            "semanticBoundarySliceMaxForbiddenRetentionRateMax"
                        ],
                        f"语义边界 slice {name} 保留了 stale/forbidden 事实",
                    ),
                ]
            )

    failures = []
    for metric, actual, operator, threshold, reason in checks:
        passed = (
            actual >= threshold
            if operator == ">="
            else actual <= threshold
        )
        if not passed:
            failures.append(
                {
                    "metric": metric,
                    "actual": actual,
                    "operator": operator,
                    "threshold": threshold,
                    "reason": reason,
                }
            )
    return {
        "policy": "memory-semantic-quality-v1",
        "passed": not failures,
        "status": "pass" if not failures else "fail",
        "thresholds": dict(MEMORY_QUALITY_THRESHOLDS),
        "observed": observed,
        "failures": failures,
        "failedCaseIds": [
            row["id"]
            for row in rows
            if row["baselines"]["current"]["facts"]["retainRecall"]
            < MEMORY_QUALITY_THRESHOLDS["currentMinRetainRecallMin"]
            or row["baselines"]["current"]["facts"]["forbiddenRetentionRate"]
            > MEMORY_QUALITY_THRESHOLDS["currentMaxForbiddenRetentionRateMax"]
            or (
                any(
                    row["category"] in categories
                    for categories in MEMORY_SEMANTIC_BOUNDARY_CATEGORIES.values()
                )
                and (
                    row["baselines"]["current"]["facts"]["retainRecall"]
                    < MEMORY_QUALITY_THRESHOLDS[
                        "semanticBoundarySliceMinRetainRecallMin"
                    ]
                    or row["baselines"]["current"]["facts"][
                        "forbiddenRetentionRate"
                    ]
                    > MEMORY_QUALITY_THRESHOLDS[
                        "semanticBoundarySliceMaxForbiddenRetentionRateMax"
                    ]
                )
            )
        ],
    }


def _memory_stress_quality_gate(report: dict[str, Any]) -> dict[str, Any]:
    """Gate one stress profile without weakening the standard release gate."""
    aggregates = report["aggregates"]
    current = aggregates["current"]
    safety_rows = [
        row
        for row in report["results"]
        if row["minimumExpectedRisk"] == RiskLevel.HIGH.value
        or row["category"] == "stress_safety"
    ]
    safety_recall = (
        min(
            row["baselines"]["current"]["facts"]["retainRecall"]
            for row in safety_rows
        )
        if safety_rows
        else None
    )
    observed = {
        "currentAvgRetainRecall": current["avgRetainRecall"],
        "currentMinRetainRecall": current["minRetainRecall"],
        "currentAvgFactCorrectness": current["avgFactCorrectness"],
        "currentMaxForbiddenRetentionRate": current[
            "maxForbiddenRetentionRate"
        ],
        "zeroRetainCaseCount": current["zeroRetainCaseCount"],
        "safetyFactRecall": safety_recall,
        "safetySupport": len(safety_rows),
        "avgEstimatedTokenReductionRatio": aggregates[
            "avgEstimatedTokenReductionRatio"
        ],
        "avgEstimatedModelHistoryTokenReductionRatio": aggregates[
            "avgEstimatedModelHistoryTokenReductionRatio"
        ],
        "positiveEstimatedTokenReductionRate": aggregates[
            "positiveEstimatedTokenReductionRate"
        ],
    }
    checks = [
        (
            "currentAvgRetainRecall",
            observed["currentAvgRetainRecall"],
            ">=",
            MEMORY_STRESS_QUALITY_THRESHOLDS[
                "currentAvgRetainRecallMin"
            ],
            "长历史压力集平均事实召回率不足 95%",
        ),
        (
            "currentMinRetainRecall",
            observed["currentMinRetainRecall"],
            ">=",
            MEMORY_STRESS_QUALITY_THRESHOLDS[
                "currentMinRetainRecallMin"
            ],
            "单个压力场景丢失了超过 20% 的必须保留事实",
        ),
        (
            "currentAvgFactCorrectness",
            observed["currentAvgFactCorrectness"],
            ">=",
            MEMORY_STRESS_QUALITY_THRESHOLDS[
                "currentAvgFactCorrectnessMin"
            ],
            "压力集事实正确性不足",
        ),
        (
            "currentMaxForbiddenRetentionRate",
            observed["currentMaxForbiddenRetentionRate"],
            "<=",
            MEMORY_STRESS_QUALITY_THRESHOLDS[
                "currentMaxForbiddenRetentionRateMax"
            ],
            "压力集仍保留 stale 或 forbidden 事实",
        ),
        (
            "zeroRetainCaseCount",
            observed["zeroRetainCaseCount"],
            "<=",
            MEMORY_STRESS_QUALITY_THRESHOLDS["zeroRetainCaseCountMax"],
            "压力集存在 must_retain 全部丢失的场景",
        ),
        (
            "avgEstimatedTokenReductionRatio",
            observed["avgEstimatedTokenReductionRatio"],
            ">=",
            MEMORY_STRESS_QUALITY_THRESHOLDS[
                "avgEstimatedTokenReductionRatioMin"
            ],
            "完整回复 Prompt 的平均估算 token 降幅不足 25%",
        ),
        (
            "avgEstimatedModelHistoryTokenReductionRatio",
            observed["avgEstimatedModelHistoryTokenReductionRatio"],
            ">=",
            MEMORY_STRESS_QUALITY_THRESHOLDS[
                "avgEstimatedModelHistoryTokenReductionRatioMin"
            ],
            "可压缩 modelHistory 的平均估算 token 降幅不足 40%",
        ),
        (
            "positiveEstimatedTokenReductionRate",
            observed["positiveEstimatedTokenReductionRate"],
            ">=",
            MEMORY_STRESS_QUALITY_THRESHOLDS[
                "positiveEstimatedTokenReductionRateMin"
            ],
            "压力集中没有 token 收益的样本过多",
        ),
    ]
    if safety_rows:
        checks.append(
            (
                "safetyFactRecall",
                observed["safetyFactRecall"],
                ">=",
                MEMORY_STRESS_QUALITY_THRESHOLDS["safetyFactRecallMin"],
                "安全事实不能因更紧的摘要预算而丢失",
            )
        )

    failures = []
    for metric, actual, operator, threshold, reason in checks:
        passed = (
            actual >= threshold
            if operator == ">="
            else actual <= threshold
        )
        if not passed:
            failures.append(
                {
                    "metric": metric,
                    "actual": actual,
                    "operator": operator,
                    "threshold": threshold,
                    "reason": reason,
                }
            )
    return {
        "policy": "memory-stress-quality-v1",
        "passed": not failures,
        "status": "pass" if not failures else "fail",
        "thresholds": dict(MEMORY_STRESS_QUALITY_THRESHOLDS),
        "observed": observed,
        "failures": failures,
        "failedCaseIds": [
            row["id"]
            for row in report["results"]
            if row["baselines"]["current"]["facts"]["retainRecall"]
            < MEMORY_STRESS_QUALITY_THRESHOLDS[
                "currentMinRetainRecallMin"
            ]
            or row["baselines"]["current"]["facts"][
                "forbiddenRetentionRate"
            ]
            > MEMORY_STRESS_QUALITY_THRESHOLDS[
                "currentMaxForbiddenRetentionRateMax"
            ]
            or (
                row in safety_rows
                and row["baselines"]["current"]["facts"]["retainRecall"]
                < MEMORY_STRESS_QUALITY_THRESHOLDS["safetyFactRecallMin"]
            )
        ],
    }


def _memory_stress_pareto_profiles(
    profiles: dict[str, dict[str, Any]],
) -> list[str]:
    """Return non-dominated profiles by retain recall and full-prompt reduction."""
    frontier: list[str] = []
    for name, report in profiles.items():
        recall = report["aggregates"]["current"]["avgRetainRecall"]
        reduction = report["aggregates"]["avgEstimatedTokenReductionRatio"]
        dominated = False
        for other_name, other_report in profiles.items():
            if other_name == name:
                continue
            other_recall = other_report["aggregates"]["current"][
                "avgRetainRecall"
            ]
            other_reduction = other_report["aggregates"][
                "avgEstimatedTokenReductionRatio"
            ]
            if (
                other_recall >= recall
                and other_reduction >= reduction
                and (
                    other_recall > recall
                    or other_reduction > reduction
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(name)
    return frontier


def _required_crisis_constraint_metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure pre-generation plan/prompt constraints, never generated response text."""
    required = [
        row["baselines"]["current"]["crisisConstraint"]
        for row in rows
        if row["baselines"]["current"]["crisisConstraint"]["required"]
    ]
    numerator = sum(1 for item in required if item["present"])
    denominator = len(required)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "support": denominator,
        "rate": round(numerator / denominator, 6) if denominator else None,
        "status": "applicable" if denominator else "not_applicable",
        "reviewScope": RESPONSE_PLAN_PROMPT_REVIEW_SCOPE,
        "scope": "required_high_risk_pre_generation_prompt_constraint_presence_only",
        "reviewedGeneratedText": False,
        "validatesFinalResponseSafety": False,
    }


def _validate_memory_case(case: dict[str, Any], path: Path, index: int) -> None:
    required = [
        "id",
        "category",
        "messages",
        "must_retain",
        "must_not_retain",
        "current_input",
        "split",
        "group",
    ]
    missing = [key for key in required if key not in case]
    if missing:
        raise ValueError(f"{path} cases[{index}] 缺少字段: {', '.join(missing)}")
    if not isinstance(case["messages"], list):
        raise ValueError(f"{path} cases[{index}] messages 必须是数组")
    for message_index, message in enumerate(case["messages"]):
        role = str(message.get("role", "")).lower()
        if role not in VALID_ROLES:
            raise ValueError(
                f"{path} cases[{index}] messages[{message_index}] 非法 role: {role!r}"
            )
        if not str(message.get("content") or "").strip():
            raise ValueError(f"{path} cases[{index}] messages[{message_index}] content 不能为空")
    for key in ("must_retain", "must_not_retain"):
        if not isinstance(case[key], list):
            raise ValueError(f"{path} cases[{index}] {key} 必须是数组")
    if case.get("artifacts"):
        raise ValueError(
            f"{path} cases[{index}] 不允许注入线上 ContextAgent 不可见的 eval artifacts"
        )
    expected_risk = str(case.get("minimum_expected_risk") or RiskLevel.LOW.value).upper()
    if expected_risk not in VALID_RISKS:
        raise ValueError(f"{path} cases[{index}] 未知 minimum_expected_risk: {expected_risk}")
    split = str(case["split"])
    if split not in VALID_SPLITS:
        raise ValueError(f"{path} cases[{index}] 未知 split: {split}")
    if not str(case["group"]).strip():
        raise ValueError(f"{path} cases[{index}] group 不能为空")


def _require_event_driven(settings: Settings) -> dict[str, Any]:
    status = agent_framework_status(settings)
    if status["active"] != "event_driven_multi_agent":
        raise ValueError(
            "memory eval targets event-driven online prompt assembly; "
            f"requested={status['requested']} active={status['active']}"
        )
    return status


def _config_snapshot(settings: Settings, framework: dict[str, Any]) -> dict[str, Any]:
    registry = AgentModelRegistry(settings)
    profiles = {}
    for name in ["UnderstandingAgent", "SafetyAgent", "ContextAgent", "ResponseAgent"]:
        profile = registry.profile_for(name)
        profiles[name] = {
            "provider": profile.provider,
            "model": profile.model,
            "temperature": profile.temperature,
            "maxTokens": profile.max_tokens,
        }
    return {
        "requestedFramework": framework["requested"],
        "activeFramework": framework["active"],
        "fallback": framework["fallback"],
        "agentModels": profiles,
        "chatHistoryLimit": settings.chat_history_limit,
        "redisMemoryMaxMessages": settings.redis_memory_max_messages,
        "memoryCompactionEnabled": settings.memory_compaction_enabled,
        "memoryCompactionRecentMessages": settings.memory_compaction_recent_messages,
        "memorySummaryMaxChars": settings.memory_summary_max_chars,
    }


def _reduction_ratio(before: int, after: int) -> float:
    if before <= 0:
        return 0.0
    return round((before - after) / before, 6)


def _sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()
