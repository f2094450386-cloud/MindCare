"""
MindBridge RAG 知识库服务模块

实现完整的检索增强生成（RAG）流程，是 CONSULT/RISK 意图的知识支撑核心。

检索策略（混合检索 + 融合 + rerank）：
1. 向量语义召回：Chroma + OpenAI embedding → 候选集
2. BM25 关键词召回：本地 BM25 算法 → 候选集
3. 分数融合：加权合并两路候选（默认向量 0.65 + BM25 0.35）
4. 本地 reranker：基于 token 覆盖率、短语匹配等信号重新打分
5. 上下文扩展：最佳结果自动扩展前后相邻切块

降级策略：
- Chroma/OpenAI 不可用 → 纯 BM25 + hybrid_score reranker
- PDF 解析：支持 pypdf 提取文本

数据模型：
- KnowledgeChunk: 数据库存储的切块（含 embedding_json 缓存）
- SearchResult: 检索结果（chunk_id, source, content, score）
- RetrievalCandidate: 融合中间结果（vector_score + bm25_score）
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Hashable

from pypdf import PdfReader
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models.entities import KnowledgeChunk
from app.services.vector_store import FALLBACK_RETRIEVAL_LABEL, PRIMARY_RETRIEVAL_LABEL, ChromaKnowledgeStore


logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """检索结果条目。"""
    chunk_id: int | None
    source: str
    content: str
    score: float


@dataclass
class RetrievalCandidate:
    """融合中间结果，同时保留向量分数和 BM25 分数。"""
    result: SearchResult
    vector_score: float = 0.0
    bm25_score: float = 0.0


class KnowledgeService:
    """
    RAG 知识库服务。

    职责：
    - 知识入库：切块 → 写入 MySQL → 生成 embedding → 写入 Chroma
    - 知识检索：向量召回 + BM25 召回 → 融合 → rerank → 上下文扩展
    - 知识管理：状态查询、重建索引、快照备份
    """

    def __init__(self, db: Session, settings: Settings):
        self.db = db
        self.settings = settings
        self.vector_store = ChromaKnowledgeStore(settings)

    def count(self) -> int:
        """返回数据库中的切块总数。"""
        return self.db.query(KnowledgeChunk).count()

    def ensure_source(self, source: str, content: str) -> int:
        """
        确保指定来源的知识已入库（幂等操作）。

        如果现有切块与新切块完全一致，跳过入库。
        用于 bootstrap 启动时同步内置知识库。
        """
        chunks = chunk_text(content, self.settings.knowledge_chunk_size, self.settings.knowledge_chunk_overlap)
        existing = [
            chunk.content
            for chunk in self.db.query(KnowledgeChunk)
            .filter(KnowledgeChunk.source == source)
            .order_by(KnowledgeChunk.source_index.asc())
            .all()
        ]
        if existing == chunks:
            return len(existing)
        return self.ingest(source, content)

    def status(self) -> dict:
        """
        返回知识库状态信息（管理员 API 用）。

        包含：检索路径、切块数量、向量库状态、配置参数等。
        """
        vector_chunks = None
        vector_error = getattr(self.vector_store, "error", "")
        if self.vector_store.can_embed:
            try:
                vector_chunks = self.vector_store.count()
            except Exception as exc:
                vector_error = f"{type(exc).__name__}: {exc}"
        return {
            "retrievalOrder": [
                PRIMARY_RETRIEVAL_LABEL,
                f"{FALLBACK_RETRIEVAL_LABEL} when OPENAI_API_KEY/chromadb/vector call is unavailable",
            ],
            "primaryRetrieval": PRIMARY_RETRIEVAL_LABEL,
            "fallbackRetrieval": FALLBACK_RETRIEVAL_LABEL,
            "databaseChunks": self.count(),
            "vectorEnabled": self.settings.knowledge_vector_enabled,
            "vectorAvailable": self.vector_store.can_embed,
            "vectorRequired": self.settings.knowledge_vector_required,
            "embeddingModel": self.settings.openai_embedding_model,
            "vectorChunks": vector_chunks,
            "chromaPersistDir": self.settings.chroma_persist_dir,
            "chromaCollectionName": self.settings.chroma_collection_name,
            "chromaSnapshotDir": self.settings.chroma_snapshot_dir,
            "candidateK": self.settings.knowledge_candidate_k,
            "hybridVectorWeight": self.settings.knowledge_hybrid_vector_weight,
            "hybridBm25Weight": self.settings.knowledge_hybrid_bm25_weight,
            "rerankEnabled": self.settings.knowledge_rerank_enabled,
            "vectorError": vector_error,
        }

    def rebuild_vector_index(self) -> int:
        """
        重建整个向量索引。

        全量读取数据库切块 → 生成 embedding → 全量同步到 Chroma。
        """
        if not self.vector_store.can_embed:
            raise RuntimeError(getattr(self.vector_store, "error", "") or "Chroma 向量库不可用")
        rows = self.db.query(KnowledgeChunk).order_by(KnowledgeChunk.source.asc(), KnowledgeChunk.source_index.asc()).all()
        self._sync_vector_chunks(rows)
        self.db.commit()
        return len(rows)

    def backup_vector_index(self) -> str:
        """创建向量库快照备份。"""
        if not self.vector_store.can_embed:
            raise RuntimeError(getattr(self.vector_store, "error", "") or "Chroma 向量库不可用")
        snapshot = self.vector_store.snapshot()
        if snapshot is None:
            raise RuntimeError("Chroma 持久化目录不存在，无法生成快照")
        return snapshot

    def ingest(self, source: str, content: str) -> int:
        """
        将文本内容入库。

        流程：
        1. 切块：按 chunk_size 和 overlap 切分文本
        2. 清除旧数据：删除该来源的旧向量和旧切块
        3. 写入 MySQL：逐块写入 KnowledgeChunk 表
        4. 生成 embedding 并写入 Chroma
        """
        chunks = chunk_text(content, self.settings.knowledge_chunk_size, self.settings.knowledge_chunk_overlap)
        self._delete_vector_source(source)
        self.db.query(KnowledgeChunk).filter(KnowledgeChunk.source == source).delete()
        rows = []
        for index, chunk in enumerate(chunks):
            row = KnowledgeChunk(source=source, source_index=index, content=chunk)
            self.db.add(row)
            rows.append(row)
        self.db.flush()
        self._index_vector_chunks(rows)
        self.db.commit()
        return len(chunks)

    def ingest_file(self, filename: str, data: bytes) -> int:
        """
        从文件内容入库。

        支持 PDF（通过 pypdf 提取文本）和纯文本文件。
        """
        lower = filename.lower()
        if lower.endswith(".pdf"):
            text = extract_pdf(data)
        else:
            text = data.decode("utf-8", errors="ignore")
        return self.ingest(filename, text)

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """
        执行混合检索。

        流程：
        1. 向量语义召回（Chroma）
        2. BM25 关键词召回
        3. 分数融合 + 本地 rerank
        4. 上下文扩展（最佳结果扩展前后相邻切块）
        """
        top_k = top_k or self.settings.knowledge_top_k
        candidate_k = self._candidate_k(top_k)
        chunks = self.db.query(KnowledgeChunk).all()

        # 两路召回
        vector_results = self._retrieve_vector(query, candidate_k)
        bm25_results = self._retrieve_bm25(query, candidate_k, chunks)

        # 融合 + rerank
        ranked = self._fuse_and_rerank(query, vector_results, bm25_results, top_k)
        if ranked:
            return self._expand_best(ranked, top_k)
        return []

    def _retrieve_bm25(self, query: str, top_k: int, chunks: list[KnowledgeChunk] | None = None) -> list[SearchResult]:
        """BM25 关键词召回。"""
        chunks = chunks if chunks is not None else self.db.query(KnowledgeChunk).all()
        scores = bm25_scores(query, chunks)
        ranked = [
            SearchResult(chunk.id, chunk.source, chunk.content, scores.get(chunk.id, 0.0))
            for chunk in chunks
            if chunk.id is not None and scores.get(chunk.id, 0.0) > 0
        ]
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:top_k]

    def _fuse_and_rerank(
        self,
        query: str,
        vector_results: list[SearchResult],
        bm25_results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """
        分数融合 + 本地 rerank。

        融合公式：
        final_score = (vector_score * vector_weight + bm25_score * bm25_weight) / total_weight

        两路分数先做 min-max 归一化到 [0, 1]，再加权融合。
        """
        candidates: dict[Hashable, RetrievalCandidate] = {}
        vector_scores = {result_key(item): item.score for item in vector_results if item.score > 0}
        bm25_scores_by_key = {result_key(item): item.score for item in bm25_results if item.score > 0}
        normalized_vector = normalize_scores(vector_scores)
        normalized_bm25 = normalize_scores(bm25_scores_by_key)

        # 合并两路候选，同一 chunk 取各路最高分
        for item in [*vector_results, *bm25_results]:
            key = result_key(item)
            candidate = candidates.get(key)
            if candidate is None:
                candidate = RetrievalCandidate(result=item)
                candidates[key] = candidate
            candidate.vector_score = max(candidate.vector_score, normalized_vector.get(key, 0.0))
            candidate.bm25_score = max(candidate.bm25_score, normalized_bm25.get(key, 0.0))

        if not candidates:
            return []

        # 加权融合
        vector_weight = max(0.0, self.settings.knowledge_hybrid_vector_weight) if vector_results else 0.0
        bm25_weight = max(0.0, self.settings.knowledge_hybrid_bm25_weight)
        if vector_weight == 0.0 and bm25_weight == 0.0:
            bm25_weight = 1.0
        total_weight = vector_weight + bm25_weight

        fused = []
        for candidate in candidates.values():
            score = (
                candidate.vector_score * vector_weight
                + candidate.bm25_score * bm25_weight
            ) / total_weight
            fused.append(replace_score(candidate.result, score))

        fused.sort(key=lambda item: item.score, reverse=True)
        fused = fused[:self._candidate_k(top_k)]
        return self._rerank(query, fused, top_k)

    def _rerank(self, query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:
        """
        本地 reranker。

        rerank_score 公式：
        base_score * 0.55 + lexical * 0.25 + coverage * 0.15 + phrase * 0.05

        - base_score: 融合后的原始分数
        - lexical: token cosine + keyword 混合分数
        - coverage: 查询 token 在内容中的覆盖率
        - phrase: 短语完全匹配加分
        """
        if not self.settings.knowledge_rerank_enabled:
            return candidates[:top_k]
        reranked = [
            replace_score(item, rerank_score(query, item.content, item.score))
            for item in candidates
        ]
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_k]

    def _candidate_k(self, top_k: int) -> int:
        """返回候选数量（至少为 top_k，至少为 candidate_k）。"""
        return max(top_k, self.settings.knowledge_candidate_k)

    def _retrieve_vector(self, query: str, top_k: int) -> list[SearchResult]:
        """
        向量语义召回。

        流程：
        1. 确保向量索引已同步（_ensure_vector_index）
        2. 将查询文本转为向量
        3. 在 Chroma 中做相似度搜索
        """
        if not self.vector_store.can_embed:
            return []
        try:
            self._ensure_vector_index()
            query_embedding = self.vector_store.embed_texts([query])[0]
            hits = self.vector_store.query(query_embedding, top_k)
        except Exception as exc:
            self._handle_vector_error("retrieve", exc)
            return []
        results = []
        for hit in hits:
            chunk = self.db.get(KnowledgeChunk, hit.chunk_id) if hit.chunk_id is not None else None
            results.append(
                SearchResult(
                    chunk.id if chunk is not None else hit.chunk_id,
                    chunk.source if chunk is not None else hit.source,
                    chunk.content if chunk is not None else hit.content,
                    hit.score,
                )
            )
        return results

    def _ensure_vector_index(self) -> None:
        """
        确保向量索引与数据库同步。

        检查条件：
        1. Chroma 向量数 == 数据库切块数
        2. 所有切块都有 embedding_json 缓存
        3. Chroma 中的 ID 集合与数据库完全一致
        任一条件不满足则全量同步。
        """
        rows = self.db.query(KnowledgeChunk).order_by(KnowledgeChunk.source.asc(), KnowledgeChunk.source_index.asc()).all()
        if not rows:
            return
        if (
            self.vector_store.count() == len(rows)
            and all(row.embedding_json for row in rows)
            and self.vector_store.has_exact_chunk_ids(rows)
        ):
            return
        self._sync_vector_chunks(rows)
        self.db.commit()

    def _delete_vector_source(self, source: str) -> None:
        """删除指定来源的向量（静默失败）。"""
        if not self.vector_store.can_embed:
            return
        try:
            self.vector_store.delete_source(source)
        except Exception as exc:
            self._handle_vector_error("delete_source", exc)

    def _index_vector_chunks(self, chunks: list[KnowledgeChunk]) -> None:
        """为切块生成 embedding 并写入 Chroma。"""
        if not chunks or not self.vector_store.can_embed:
            return
        try:
            embeddings = self._embeddings_for_chunks(chunks)
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding_json = json.dumps(embedding, separators=(",", ":"))
            self.vector_store.upsert_chunks(chunks, embeddings)
        except Exception as exc:
            self._handle_vector_error("index", exc)

    def _sync_vector_chunks(self, chunks: list[KnowledgeChunk]) -> None:
        """全量同步切块到 Chroma（删除旧的 + 写入新的）。"""
        if not chunks or not self.vector_store.can_embed:
            return
        try:
            embeddings = self._embeddings_for_chunks(chunks)
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding_json = json.dumps(embedding, separators=(",", ":"))
            self.vector_store.sync_chunks(chunks, embeddings)
        except Exception as exc:
            self._handle_vector_error("sync", exc)

    def _embeddings_for_chunks(self, chunks: list[KnowledgeChunk]) -> list[list[float]]:
        """
        获取切块的 embedding 向量。

        优先使用 embedding_json 缓存，缓存缺失的调用 OpenAI API 补建。
        这样可以避免重复调用 embedding API（节省成本和时间）。
        """
        embeddings: list[list[float] | None] = []
        missing_indexes = []
        missing_texts = []
        for index, chunk in enumerate(chunks):
            embedding = parse_embedding(chunk.embedding_json)
            embeddings.append(embedding)
            if embedding is None:
                missing_indexes.append(index)
                missing_texts.append(chunk.content)
        if missing_texts:
            new_embeddings = self.vector_store.embed_texts(missing_texts)
            for index, embedding in zip(missing_indexes, new_embeddings):
                embeddings[index] = embedding
        resolved = [embedding for embedding in embeddings if embedding is not None]
        if len(resolved) != len(chunks):
            raise ValueError("Embedding response count did not match knowledge chunks.")
        return resolved

    def _handle_vector_error(self, action: str, exc: Exception) -> None:
        """
        处理向量操作错误。

        knowledge_vector_required=true 时直接抛出异常。
        否则记录警告日志，降级到 BM25。
        """
        if self.settings.knowledge_vector_required:
            raise exc
        logger.warning(
            "%s %s failed; falling back to %s: %s",
            PRIMARY_RETRIEVAL_LABEL,
            action,
            FALLBACK_RETRIEVAL_LABEL,
            exc,
        )

    def _expand_best(self, ranked: list[SearchResult], top_k: int) -> list[SearchResult]:
        """
        上下文扩展：最佳结果自动扩展前后相邻切块。

        如果检索到的切块在原文中间，扩展前后各 1 个切块，
        拼接后返回更完整的上下文。
        """
        if not ranked:
            return []
        best = ranked[0]
        expanded = self._expand(best)
        results = [expanded]
        for item in ranked[1:]:
            if item.chunk_id != expanded.chunk_id and len(results) < top_k:
                results.append(item)
        return results

    def _expand(self, result: SearchResult) -> SearchResult:
        """扩展单个结果：获取同来源的前后相邻切块并拼接。"""
        if result.chunk_id is None:
            return result
        chunk = self.db.get(KnowledgeChunk, result.chunk_id)
        if chunk is None:
            return result
        neighbors = (
            self.db.query(KnowledgeChunk)
            .filter(KnowledgeChunk.source == chunk.source)
            .filter(KnowledgeChunk.source_index >= max(0, chunk.source_index - 1))
            .filter(KnowledgeChunk.source_index <= chunk.source_index + 1)
            .order_by(KnowledgeChunk.source_index.asc())
            .all()
        )
        return SearchResult(chunk.id, chunk.source, "\n\n".join(item.content for item in neighbors), result.score)


# ── 文本切块 ──────────────────────────────────────────────────────

def chunk_text(content: str, size: int, overlap: int) -> list[str]:
    """
    将文本按固定大小切块，支持重叠。

    切块策略：
    - 先将所有空白字符压缩为单个空格
    - 按 step = size - overlap 滑动窗口切分
    - 重叠防止语义在切块边界断裂
    """
    text = re.sub(r"\s+", " ", content or "").strip()
    if not text:
        return []
    chunks = []
    start = 0
    step = max(1, size - overlap)
    while start < len(text):
        chunks.append(text[start:start + size])
        start += step
    return chunks


# ── 混合打分函数 ──────────────────────────────────────────────────

def hybrid_score(query: str, content: str) -> float:
    """
    混合打分：token cosine * 0.75 + keyword * 0.25。

    用于 reranker 中的 lexical 信号。
    """
    return token_cosine(query, content) * 0.75 + keyword_score(query, content) * 0.25


def bm25_scores(query: str, chunks: list[KnowledgeChunk]) -> dict[int, float]:
    """
    BM25 关键词召回算法。

    标准 BM25 公式：
    - k1=1.5, b=0.75（经典参数）
    - IDF = log(1 + (N - df + 0.5) / (df + 0.5))
    - TF 归一化：tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl/avgdl))
    - 查询词频加权：1 + log(query_tf)
    """
    query_terms = counts(tokenize(query))
    if not query_terms or not chunks:
        return {}

    documents = []
    doc_freqs: dict[str, int] = {}
    for chunk in chunks:
        if chunk.id is None:
            continue
        token_counts = counts(tokenize(chunk.content))
        documents.append((chunk.id, token_counts, sum(token_counts.values())))
        for term in token_counts:
            doc_freqs[term] = doc_freqs.get(term, 0) + 1

    total_docs = len(documents)
    if total_docs == 0:
        return {}
    average_length = sum(length for _, _, length in documents) / total_docs or 1.0
    k1 = 1.5
    b = 0.75
    scores: dict[int, float] = {}

    for chunk_id, token_counts, doc_length in documents:
        score = 0.0
        length_norm = k1 * (1.0 - b + b * doc_length / average_length)
        for term, query_frequency in query_terms.items():
            term_frequency = token_counts.get(term, 0)
            if term_frequency == 0:
                continue
            doc_frequency = doc_freqs.get(term, 0)
            idf = math.log(1.0 + (total_docs - doc_frequency + 0.5) / (doc_frequency + 0.5))
            query_boost = 1.0 + math.log(query_frequency)
            score += idf * query_boost * (term_frequency * (k1 + 1.0)) / (term_frequency + length_norm)
        if score > 0:
            scores[chunk_id] = score
    return scores


def rerank_score(query: str, content: str, base_score: float) -> float:
    """
    本地 reranker 打分。

    综合四个信号：
    - base_score (0.55): 融合后的原始分数
    - lexical (0.25): token cosine + keyword 混合
    - coverage (0.15): 查询 token 覆盖率
    - phrase (0.05): 短语完全匹配
    """
    lexical = hybrid_score(query, content)
    coverage = query_token_coverage(query, content)
    phrase = phrase_score(query, content)
    return base_score * 0.55 + lexical * 0.25 + coverage * 0.15 + phrase * 0.05


def query_token_coverage(query: str, content: str) -> float:
    """计算查询 token 在内容中的覆盖率。"""
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    content_tokens = set(tokenize(content))
    return len(query_tokens & content_tokens) / len(query_tokens)


def phrase_score(query: str, content: str) -> float:
    """短语匹配打分：查询文本在内容中完全出现时返回 1.0。"""
    normalized_query = compact_text(query)
    if not normalized_query:
        return 0.0
    normalized_content = compact_text(content)
    if normalized_query in normalized_content:
        return 1.0
    return keyword_score(query, content)


def compact_text(text: str) -> str:
    """去除所有空白并转小写。"""
    return re.sub(r"\s+", "", text.lower())


def normalize_scores(scores: dict[Hashable, float]) -> dict[Hashable, float]:
    """
    Min-max 归一化分数到 [0, 1]。

    所有正分数线性映射到 [0, 1]，0 分保持为 0。
    所有分数相同时，正分数映射为 1.0。
    """
    positives = [score for score in scores.values() if score > 0]
    if not positives:
        return {key: 0.0 for key in scores}
    lowest = min(positives)
    highest = max(positives)
    if math.isclose(lowest, highest):
        return {key: 1.0 if score > 0 else 0.0 for key, score in scores.items()}
    return {
        key: (score - lowest) / (highest - lowest) if score > 0 else 0.0
        for key, score in scores.items()
    }


def result_key(result: SearchResult) -> Hashable:
    """生成结果的唯一标识（用于去重和融合）。"""
    return result.chunk_id if result.chunk_id is not None else (result.source, result.content)


def replace_score(result: SearchResult, score: float) -> SearchResult:
    """创建新的 SearchResult，替换分数。"""
    return SearchResult(result.chunk_id, result.source, result.content, score)


def parse_embedding(raw: str | None) -> list[float] | None:
    """解析缓存的 embedding JSON 字符串。"""
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list) or not data:
        return None
    if not all(isinstance(item, (int, float)) for item in data):
        return None
    return [float(item) for item in data]


# ── 中文分词 ──────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """
    中文分词。

    策略：
    1. 英文和数字：按单词分割
    2. 中文字符：逐字分割 + 二元组（bigram）
    3. 合并所有 token 并过滤空串

    bigram 可以捕获中文词组的局部语义（如"焦虑"→ ["焦", "虑", "焦虑"]）。
    """
    words = re.findall(r"[a-zA-Z0-9_]+|[一-鿿]", text.lower())
    grams = words[:]
    compact = "".join(ch for ch in text.lower() if "一" <= ch <= "鿿")
    grams.extend(compact[i:i + 2] for i in range(max(0, len(compact) - 1)))
    return [item for item in grams if item.strip()]


def token_cosine(left: str, right: str) -> float:
    """基于 token 的余弦相似度。"""
    left_counts = counts(tokenize(left))
    right_counts = counts(tokenize(right))
    if not left_counts or not right_counts:
        return 0.0
    dot = sum(value * right_counts.get(key, 0) for key, value in left_counts.items())
    left_norm = math.sqrt(sum(value * value for value in left_counts.values()))
    right_norm = math.sqrt(sum(value * value for value in right_counts.values()))
    return 0.0 if left_norm == 0 or right_norm == 0 else dot / (left_norm * right_norm)


def keyword_score(query: str, content: str) -> float:
    """关键词匹配分数：查询词在内容中出现的比例。"""
    terms = [term for term in re.split(r"[\s，。！？、；：,.!?;:]+", query.lower()) if len(term) >= 2]
    if not terms:
        return 0.0
    lower = content.lower()
    matched = sum(1 for term in terms if term in lower)
    return min(1.0, matched / len(terms))


def counts(values: list[str]) -> dict[str, int]:
    """统计列表中每个值的出现次数。"""
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


def extract_pdf(data: bytes) -> str:
    """从 PDF 二进制数据中提取文本。"""
    from io import BytesIO

    reader = PdfReader(BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)
