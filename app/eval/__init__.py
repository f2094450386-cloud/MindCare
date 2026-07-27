"""
可复现评测公共工具。

统一：
- 报告头部元数据（版本 / git / 数据集 hash / 配置快照）
- 分类指标（accuracy、macro/per-class P·R·F1、混淆矩阵）
- JSON 报告写出
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.version import build_repro_metadata


def utc_now_iso() -> str:
    """UTC ISO8601 时间戳。"""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json_report(path: Path, payload: dict[str, Any]) -> Path:
    """写出 UTF-8 JSON 报告，自动创建父目录。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def resolve_split_report_path(base: Path, split: str | None) -> Path:
    """
    Derive a non-overwriting default path for a semantic split.

    The all-cases report keeps the configured base path for backward
    compatibility. Explicit caller paths are handled by callers and are never
    rewritten.
    """
    path = Path(base)
    if split in {None, "", "all"}:
        return path
    return path.with_name(f"{path.stem}-{split}{path.suffix}")


def attach_repro_header(
    report: dict[str, Any],
    *,
    dataset_paths: list[Path],
    config_snapshot: dict[str, Any],
    prompt_rules_version: str,
    rule_source_paths: list[Path],
    eval_name: str,
    app_version: str | None = None,
) -> dict[str, Any]:
    """
    在报告顶部写入可复现绑定信息。

    不修改调用方传入的原始 report 引用语义：返回带 header 的新结构。
    """
    return {
        "evalName": eval_name,
        "createdAt": utc_now_iso(),
        "repro": build_repro_metadata(
            dataset_paths=dataset_paths,
            config_snapshot=config_snapshot,
            prompt_rules_version=prompt_rules_version,
            rule_source_paths=rule_source_paths,
            app_version=app_version,
        ),
        **report,
    }


def classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """
    计算多分类指标。

    返回 overall accuracy、macro P/R/F1、每类 P/R/F1、support、混淆矩阵。
    分母为 0 时对应指标记为 0.0，避免 NaN 污染报告。
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true 与 y_pred 长度必须一致")
    if not labels:
        labels = sorted(set(y_true) | set(y_pred))

    total = len(y_true)
    correct = sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred)
    matrix = {truth: {pred: 0 for pred in labels} for truth in labels}
    for truth, pred in zip(y_true, y_pred):
        if truth not in matrix:
            matrix[truth] = {candidate: 0 for candidate in labels}
        if pred not in matrix[truth]:
            matrix[truth][pred] = 0
        matrix[truth][pred] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for label in labels:
        tp = matrix.get(label, {}).get(label, 0)
        fp = sum(matrix.get(other, {}).get(label, 0) for other in labels if other != label)
        fn = sum(count for pred, count in matrix.get(label, {}).items() if pred != label)
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    n_labels = max(1, len(labels))
    return {
        "total": total,
        "accuracy": round(correct / total, 6) if total else 0.0,
        "macroPrecision": round(sum(precisions) / n_labels, 6),
        "macroRecall": round(sum(recalls) / n_labels, 6),
        "macroF1": round(sum(f1s) / n_labels, 6),
        "perClass": per_class,
        "confusionMatrix": matrix,
    }


def bucket_failures(cases: list[dict[str, Any]], *, key: str = "failureBucket") -> dict[str, int]:
    """按失败桶统计计数。"""
    counts: dict[str, int] = defaultdict(int)
    for case in cases:
        if case.get("correct"):
            continue
        counts[str(case.get(key) or "unclassified")] += 1
    return dict(sorted(counts.items()))
