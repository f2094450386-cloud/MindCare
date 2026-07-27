"""
应用版本与可复现评测元数据。

版本来源优先级：
1. 环境变量 APP_VERSION（部署覆盖）
2. 仓库根目录 VERSION 文件（正式版本号）
3. 回退常量 DEFAULT_APP_VERSION

评测报告还会额外绑定 git commit / dirty 状态，用于复现实验，
不等于把语义化版本号写进每一次本地改动。
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from functools import lru_cache
from pathlib import Path


# 与历史 FastAPI(version=...) 保持一致；若 VERSION 文件缺失则使用该值。
DEFAULT_APP_VERSION = "0.1.0"


def project_root() -> Path:
    """返回仓库根目录（app/core/version.py 的上两级）。"""
    return Path(__file__).resolve().parents[2]


@lru_cache
def get_app_version(
    explicit_version: str | None = None,
    root: Path | None = None,
) -> str:
    """
    读取应用语义化版本号。

    注意：git 提交说明里的「6.22version」是历史快照标签，
    不是本模块维护的结构化版本；正式版本以 VERSION / APP_VERSION 为准。
    """
    env_version = (explicit_version or os.environ.get("APP_VERSION") or "").strip()
    if env_version:
        return env_version

    version_file = (root or project_root()) / "VERSION"
    if version_file.is_file():
        text = version_file.read_text(encoding="utf-8").strip()
        if text:
            return text.splitlines()[0].strip()
    return DEFAULT_APP_VERSION


def get_git_fingerprint(cwd: Path | None = None) -> dict:
    """
    采集当前工作树的 git 指纹，供评测报告绑定。

    返回字段：
    - commit: 完整 SHA；非 git 仓库时为 null
    - shortCommit: 短 SHA
    - branch: 当前分支名
    - dirty: 工作树是否有未提交改动
    - describe: git describe --always --dirty（失败则为 null）
    """
    root = cwd or project_root()
    commit = _git(["rev-parse", "HEAD"], root)
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], root)
    describe = _git(["describe", "--always", "--dirty", "--tags"], root)
    status = _git(["status", "--porcelain"], root, allow_empty=True)
    dirty = bool(status) if status is not None else None
    tracked_diff = _git_bytes(["diff", "--binary", "--no-ext-diff", "HEAD", "--", "."], root)
    untracked = _untracked_fingerprints(root)
    untracked_manifest = "\n".join(
        f"{item['path']}\0{item['bytes']}\0{item['sha256']}" for item in (untracked or [])
    )
    return {
        "schema": "git-fingerprint-v1",
        "commit": commit,
        "shortCommit": commit[:12] if commit else None,
        "branch": branch,
        "dirty": dirty,
        "describe": describe,
        "trackedDiffSha256": (
            hashlib.sha256(tracked_diff).hexdigest() if tracked_diff is not None else None
        ),
        "trackedDiffBytes": len(tracked_diff) if tracked_diff is not None else None,
        "untrackedFiles": untracked,
        "untrackedContentSha256": (
            sha256_text(untracked_manifest) if untracked is not None else None
        ),
    }


def sha256_file(path: Path) -> str:
    """计算文件内容的 SHA-256，用于绑定评测数据集版本。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """计算文本的 SHA-256（UTF-8）。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_repro_metadata(
    *,
    dataset_paths: list[Path] | None = None,
    config_snapshot: dict | None = None,
    prompt_rules_version: str = "v0",
    rule_source_paths: list[Path] | None = None,
    app_version: str | None = None,
) -> dict:
    """
    构造可复现评测报告头部元数据。

    将代码版本、git 指纹、数据集 hash、关键配置快照绑在一起，
    避免简历数字与实际运行条件脱节。
    """
    datasets = []
    for path in dataset_paths or []:
        resolved = path.resolve()
        datasets.append(
            {
                "path": str(path).replace("\\", "/"),
                "exists": resolved.is_file(),
                "sha256": sha256_file(resolved) if resolved.is_file() else None,
                "bytes": resolved.stat().st_size if resolved.is_file() else None,
            }
        )
    rule_sources = []
    for path in rule_source_paths or []:
        resolved = path.resolve()
        if resolved.is_file():
            rule_sources.append(
                {
                    "path": _display_path(resolved),
                    "sha256": sha256_file(resolved),
                    "bytes": resolved.stat().st_size,
                }
            )
    rule_manifest = "\n".join(
        f"{item['path']}\0{item['bytes']}\0{item['sha256']}" for item in rule_sources
    )
    return {
        "appVersion": get_app_version(app_version),
        "git": get_git_fingerprint(),
        "promptRules": {
            "label": prompt_rules_version,
            "sha256": sha256_text(rule_manifest),
            "sources": rule_sources,
        },
        "datasets": datasets,
        "config": config_snapshot or {},
    }


def _git(args: list[str], cwd: Path, *, allow_empty: bool = False) -> str | None:
    """执行 git 子命令；失败时返回 None，不影响评测主流程。"""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    output = completed.stdout.strip()
    return output if output or allow_empty else None


def _git_bytes(args: list[str], cwd: Path) -> bytes | None:
    """Run git and preserve exact bytes for stable diff hashing."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=False,
            capture_output=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _untracked_fingerprints(root: Path) -> list[dict] | None:
    raw = _git_bytes(["ls-files", "--others", "--exclude-standard", "-z"], root)
    if raw is None:
        return None
    paths = sorted(
        item.decode("utf-8", errors="surrogateescape")
        for item in raw.split(b"\0")
        if item
    )
    fingerprints = []
    for relative in paths:
        path = root / relative
        if not path.is_file():
            continue
        fingerprints.append(
            {
                "path": relative.replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return fingerprints


def _display_path(path: Path) -> str:
    """Prefer project-relative report paths without assuming a git checkout."""
    try:
        return str(path.relative_to(project_root())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")
