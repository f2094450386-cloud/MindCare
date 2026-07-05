"""
MindBridge RAG 评测模块

对 RAG 知识库检索质量进行量化评估。

评测指标：
- Recall@K: 在 top-K 结果中是否命中相关文档（0/1）
- Precision@K: top-K 结果中相关文档的比例
- MRR (Mean Reciprocal Rank): 第一个相关结果的排名倒数
- NDCG@K (Normalized Discounted Cumulative Gain): 排序质量
- HitRate: 所有测试用例中命中相关文档的比例

评测数据集：app/rag_eval/mindbridge-rag-eval.json
评测报告：target/rag-eval-report.json

使用方式：
  AI_PROVIDER=mock python -m app.rag_eval.runner
"""
import json
import math
from datetime import datetime
from pathlib import Path

from app.core.bootstrap import create_schema, seed_data
from app.core.config import get_settings
from app.core.database import SessionLocal
from app.services.knowledge import KnowledgeService


def evaluate() -> dict:
    """
    执行完整的 RAG 评测。

    流程：
    1. 初始化数据库和知识库
    2. 加载评测数据集
    3. 对每个测试用例执行检索
    4. 计算各项指标
    5. 输出评测报告
    """
    settings = get_settings()
    create_schema()
    db = SessionLocal()
    try:
        seed_data(db)
        service = KnowledgeService(db, settings)
        dataset_path = Path(settings.rag_eval_dataset)
        cases = json.loads(dataset_path.read_text(encoding="utf-8"))
        results = [evaluate_case(service, case, settings.knowledge_top_k) for case in cases]
        total = max(1, len(results))
        hits = [item for item in results if item["hit"]]
        report = {
            "createdAt": datetime.utcnow().isoformat(),
            "dataset": settings.rag_eval_dataset,
            "topK": settings.knowledge_top_k,
            "totalCases": len(results),
            "recallAtK": sum(item["recallAtK"] for item in results) / total,
            "precisionAtK": sum(item["precisionAtK"] for item in results) / total,
            "mrr": sum(item["reciprocalRank"] for item in results) / total,
            "ndcgAtK": sum(item["ndcgAtK"] for item in results) / total,
            "hitRate": len(hits) / total,
            "averageFirstRelevantRank": sum(item["firstRelevantRank"] for item in hits) / max(1, len(hits)),
            "results": results,
        }
        output = Path(settings.rag_eval_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
    finally:
        db.close()


def evaluate_case(service: KnowledgeService, case: dict, top_k: int) -> dict:
    """
    评测单个测试用例。

    每个用例包含：
    - id: 用例 ID
    - question: 查询问题
    - expectedSources: 期望命中的来源文件列表
    - expectedTerms: 期望在结果中出现的关键词列表

    相关性判断：
    - 来源文件名在 expectedSources 中 → 相关
    - 内容中包含 expectedTerms 中的任意关键词 → 相关
    """
    retrieved = service.retrieve(case["question"], top_k)
    expected_sources = {source.lower() for source in case.get("expectedSources", [])}
    expected_terms = [term.lower() for term in case.get("expectedTerms", [])]
    items = []
    first_rank = 0
    relevant_count = 0
    for index, item in enumerate(retrieved, start=1):
        relevant = is_relevant(item.source, item.content, expected_sources, expected_terms)
        if relevant:
            relevant_count += 1
            if first_rank == 0:
                first_rank = index
        items.append({
            "rank": index,
            "chunkId": item.chunk_id,
            "source": item.source,
            "score": item.score,
            "relevant": relevant,
            "preview": " ".join(item.content.split())[:160],
        })
    hit = first_rank > 0
    return {
        "id": case["id"],
        "question": case["question"],
        "expectedSources": case.get("expectedSources", []),
        "expectedTerms": case.get("expectedTerms", []),
        "retrieved": items,
        "hit": hit,
        "firstRelevantRank": first_rank,
        "recallAtK": 1.0 if hit else 0.0,
        "precisionAtK": relevant_count / top_k if top_k > 0 else 0.0,
        "reciprocalRank": 1.0 / first_rank if hit else 0.0,
        "ndcgAtK": ndcg(items),
    }


def is_relevant(source: str, content: str, expected_sources: set[str], expected_terms: list[str]) -> bool:
    """判断检索结果是否相关。"""
    if source.lower() in expected_sources:
        return True
    lower = content.lower()
    return any(len(term) >= 2 and term in lower for term in expected_terms)


def ndcg(items: list[dict]) -> float:
    """
    计算 NDCG@K (Normalized Discounted Cumulative Gain)。

    DCG = Σ(rel_i / log2(i + 1))
    NDCG = DCG / IDCG（理想排序的 DCG）
    """
    dcg = 0.0
    relevant = 0
    for index, item in enumerate(items):
        if item["relevant"]:
            relevant += 1
            dcg += 1.0 / math.log(index + 2.0)
    if relevant == 0:
        return 0.0
    ideal = sum(1.0 / math.log(index + 2.0) for index in range(relevant))
    return dcg / ideal


if __name__ == "__main__":
    report = evaluate()
    print("RAG evaluation completed.")
    for key in ["totalCases", "topK", "recallAtK", "precisionAtK", "mrr", "ndcgAtK", "hitRate", "averageFirstRelevantRank"]:
        print(f"{key}={report[key]}")
