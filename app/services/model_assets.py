"""
MindBridge 微调模型资产检查模块

检查本地微调 GGUF 模型的文件是否就绪：
- GGUF 权重文件是否存在
- Modelfile 是否存在
- GGUF 文件大小

用于 /api/agent/status 接口返回模型资产状态。
"""
from __future__ import annotations

from pathlib import Path

from app.core.config import Settings


def finetuned_model_status(settings: Settings) -> dict:
    """
    检查微调模型资产状态。

    返回：
    - name: 模型名称
    - directory: 模型目录相对路径
    - ggufFile: GGUF 文件名
    - ggufExists: GGUF 文件是否存在
    - ggufSizeBytes: GGUF 文件大小（字节）
    - modelfileExists: Modelfile 是否存在
    - ollamaCreateCommand: Ollama 创建模型的命令
    """
    root = settings.project_root
    model_dir = resolve_model_dir(settings)
    gguf_path = model_dir / settings.finetuned_model_file
    modelfile_path = model_dir / "Modelfile"
    return {
        "name": settings.finetuned_model_name,
        "directory": str(model_dir.relative_to(root)) if model_dir.is_relative_to(root) else str(model_dir),
        "ggufFile": settings.finetuned_model_file,
        "ggufExists": gguf_path.exists(),
        "ggufSizeBytes": gguf_path.stat().st_size if gguf_path.exists() else 0,
        "modelfileExists": modelfile_path.exists(),
        "ollamaCreateCommand": f"scripts/create-finetuned-model.sh",
    }


def resolve_model_dir(settings: Settings) -> Path:
    """解析模型目录为绝对路径。"""
    path = Path(settings.finetuned_model_dir)
    return path if path.is_absolute() else settings.project_root / path
