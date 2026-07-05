"""
MindBridge Chroma 向量存储模块

实现基于 ChromaDB 的向量检索，是 RAG 主检索路径的核心组件。

架构：
- 使用 OpenAI text-embedding-3-small 模型生成文本向量
- ChromaDB 本地持久化存储（cosine 距离）
- 支持向量库快照备份和自动清理

降级策略（三级）：
1. Chroma + OpenAI embedding 正常工作 → 主检索路径
2. 缺少 OPENAI_API_KEY 或 chromadb 未安装 → 回退到 BM25
3. knowledge_vector_required=true 时，不可用直接抛异常

向量检索流程：
1. embed_texts(): 调用 OpenAI API 将文本转为向量
2. upsert_chunks(): 将知识切块和向量写入 Chroma
3. query(): 用查询向量在 Chroma 中做相似度搜索
4. snapshot(): 每次写入后自动备份向量库
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx

from app.core.config import Settings
from app.models.entities import KnowledgeChunk


# 检索路径标签（用于日志和状态展示）
PRIMARY_RETRIEVAL_LABEL = "Chroma vector + BM25 hybrid + local reranker"
FALLBACK_RETRIEVAL_LABEL = "local BM25 + hybrid_score reranker"


class VectorStoreUnavailable(RuntimeError):
    """向量存储不可用异常。"""
    pass


@dataclass
class VectorSearchHit:
    """向量搜索结果条目。"""
    chunk_id: int | None
    source: str
    source_index: int
    content: str
    score: float


class ChromaKnowledgeStore:
    """
    Chroma 向量知识库。

    职责：
    - 管理 ChromaDB 集合（创建、查询、删除）
    - 调用 OpenAI embedding API 生成文本向量
    - 向量库快照备份和自动清理

    初始化时检查：
    1. knowledge_vector_enabled 配置
    2. OPENAI_API_KEY 是否存在
    3. chromadb 是否已安装
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.can_embed = False  # 是否可用作向量检索
        self.error = ""         # 不可用原因

        # 检查 1: 向量库是否启用
        if not settings.knowledge_vector_enabled:
            self.error = "Chroma 向量库未启用"
            return

        # 检查 2: OpenAI API Key
        if not settings.openai_api_key:
            if settings.knowledge_vector_required:
                raise VectorStoreUnavailable("缺少 OPENAI_API_KEY，无法启用 Chroma + text-embedding-3-small 主检索方案")
            self.error = f"缺少 OPENAI_API_KEY，Chroma + text-embedding-3-small 不可用，已回退到{FALLBACK_RETRIEVAL_LABEL}"
            return

        # 检查 3: chromadb 依赖
        try:
            import chromadb
        except ImportError as exc:
            if settings.knowledge_vector_required:
                raise VectorStoreUnavailable("缺少 chromadb 依赖，无法启用 Chroma + text-embedding-3-small 主检索方案") from exc
            self.error = f"缺少 chromadb 依赖，Chroma + text-embedding-3-small 不可用，已回退到{FALLBACK_RETRIEVAL_LABEL}"
            return

        # 初始化 Chroma 持久化客户端
        persist_dir = self._resolve_path(settings.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_name,
            embedding_function=None,  # 手动管理 embedding，不使用 Chroma 内置
            metadata={"hnsw:space": "cosine", "embedding_model": settings.openai_embedding_model},
        )
        self.can_embed = settings.knowledge_vector_enabled

    def upsert_chunks(self, chunks: list[KnowledgeChunk], embeddings: list[list[float]]) -> int:
        """
        将知识切块和对应的向量写入 Chroma。

        使用 upsert 语义：已存在则更新，不存在则插入。
        写入后自动触发快照备份。
        """
        rows = [chunk for chunk in chunks if chunk.id is not None and chunk.content.strip()]
        if not rows:
            return 0
        ids = [self._id(chunk.id) for chunk in rows]
        documents = [chunk.content for chunk in rows]
        metadatas = [
            {"db_id": int(chunk.id), "source": chunk.source, "source_index": int(chunk.source_index)}
            for chunk in rows
        ]
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        self.snapshot()
        return len(rows)

    def sync_chunks(self, chunks: list[KnowledgeChunk], embeddings: list[list[float]]) -> int:
        """
        全量同步知识切块到 Chroma。

        与 upsert_chunks 的区别：会删除 Chroma 中存在但数据库中已不存在的旧切块。
        用于 rebuild_vector_index 场景。
        """
        valid_ids = {self._id(int(chunk.id)) for chunk in chunks if chunk.id is not None}
        current_ids = set(self.collection.get().get("ids", []))
        stale_ids = sorted(current_ids - valid_ids)
        if stale_ids:
            self.collection.delete(ids=stale_ids)
        return self.upsert_chunks(chunks, embeddings)

    def has_exact_chunk_ids(self, chunks: list[KnowledgeChunk]) -> bool:
        """检查 Chroma 中的向量 ID 是否与数据库切块 ID 完全一致。"""
        valid_ids = {self._id(int(chunk.id)) for chunk in chunks if chunk.id is not None}
        current_ids = set(self.collection.get().get("ids", []))
        return current_ids == valid_ids

    def delete_source(self, source: str) -> None:
        """删除指定来源的所有向量。"""
        if not self.can_embed:
            return
        self.collection.delete(where={"source": source})

    def query(self, query_embedding: list[float], top_k: int) -> list[VectorSearchHit]:
        """
        用查询向量在 Chroma 中做相似度搜索。

        返回 top_k 个最相似的结果。
        分数计算：1.0 / (1.0 + distance)，将 cosine 距离转换为相似度分数。
        """
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        hits = []
        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = float(distances[index]) if index < len(distances) else 1.0
            hits.append(
                VectorSearchHit(
                    chunk_id=int(metadata["db_id"]) if metadata.get("db_id") is not None else None,
                    source=str(metadata.get("source", "")),
                    source_index=int(metadata.get("source_index", 0)),
                    content=document or "",
                    score=1.0 / (1.0 + max(0.0, distance)),
                )
            )
        return hits

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """调用 OpenAI embedding API 将文本列表转为向量列表。"""
        if not self.can_embed:
            raise VectorStoreUnavailable(self.error or "Chroma + text-embedding-3-small 主检索方案不可用")
        return self._embed(texts)

    def snapshot(self) -> str | None:
        """
        对 Chroma 持久化目录创建快照备份。

        快照目录格式：data/chroma-snapshots/YYYYMMDD-HHMMSS-microseconds/
        自动保留最近 N 个快照，清理旧的。
        """
        if not self.can_embed:
            return None
        if not self.persist_dir.exists():
            return None
        snapshot_root = self._resolve_path(self.settings.chroma_snapshot_dir)
        snapshot_root.mkdir(parents=True, exist_ok=True)
        destination = snapshot_root / datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
        shutil.copytree(self.persist_dir, destination)
        self._prune_snapshots(snapshot_root)
        return str(destination)

    def count(self) -> int:
        """返回 Chroma 中的向量总数。"""
        if not self.can_embed:
            return 0
        return int(self.collection.count())

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """
        调用 OpenAI embedding API 生成向量。

        空文本会被替换为空格（API 不接受空字符串）。
        返回的向量按 index 排序，确保与输入文本顺序一致。
        """
        payload = {
            "model": self.settings.openai_embedding_model,
            "input": [text if text.strip() else " " for text in texts],
        }
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        response = httpx.post(
            f"{self.settings.openai_base_url}/embeddings",
            headers=headers,
            json=payload,
            timeout=self.settings.embedding_timeout_seconds,
        )
        response.raise_for_status()
        rows = sorted(response.json().get("data", []), key=lambda item: item.get("index", 0))
        embeddings = [row.get("embedding") for row in rows]
        if len(embeddings) != len(texts) or any(not embedding for embedding in embeddings):
            raise VectorStoreUnavailable("OpenAI embeddings 接口返回向量数量不匹配")
        return [[float(value) for value in embedding] for embedding in embeddings]

    def _resolve_path(self, value: str) -> Path:
        """将相对路径解析为基于项目根目录的绝对路径。"""
        path = Path(value)
        return path if path.is_absolute() else self.settings.project_root / path

    def _prune_snapshots(self, snapshot_root: Path) -> None:
        """清理旧快照，只保留最近 N 个。"""
        keep = max(1, self.settings.chroma_snapshot_keep)
        snapshots = sorted([path for path in snapshot_root.iterdir() if path.is_dir()], reverse=True)
        for stale in snapshots[keep:]:
            shutil.rmtree(stale, ignore_errors=True)

    def _id(self, chunk_id: int) -> str:
        """将数据库 chunk_id 转换为 Chroma 文档 ID。"""
        return f"knowledge-chunk-{chunk_id}"


# 别名兼容
ChromaKnowledgeVectorStore = ChromaKnowledgeStore
