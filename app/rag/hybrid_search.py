"""Hybrid Retrieval：Dense + Keyword 加权分数融合。

为什么混合（ADR-003）：Dense 检索擅长语义相似（同义改写），
关键词检索擅长精确匹配（套餐名/数字/专有名词），两者互补。

融合方式（第一版 Weighted Score Fusion）：
    1. 两个列表分别 min-max 归一化（分数尺度不同，直接相加无意义）；
    2. final = vector_weight * norm_vector + keyword_weight * norm_keyword；
    3. 权重可配置（VECTOR_WEIGHT / KEYWORD_WEIGHT），
       默认 0.6/0.4 只是起点，最终值应由 evaluation 数据决定（不声称最优）。
"""

import time

from app.core.config import settings
from app.observability import metrics
from app.rag.keyword_search import keyword_search
from app.rag.vector_search import get_vector_store


def _minmax(scores: list[tuple[str, float]]) -> dict[str, float]:
    if not scores:
        return {}
    values = [s for _, s in scores]
    lo, hi = min(values), max(values)
    if hi == lo:
        return {doc_id: 1.0 for doc_id, _ in scores}
    return {doc_id: (s - lo) / (hi - lo) for doc_id, s in scores}


def hybrid_search(
    query: str,
    query_vector: list[float],
    *,
    top_k: int | None = None,
    category: str | None = None,
) -> list[dict]:
    """Dense + Keyword 融合检索。"""
    top_k = top_k or settings.retrieval_top_k

    dense = get_vector_store().search(query_vector, top_k=top_k, category=category)
    sparse = keyword_search(query, top_k=top_k, category=category)

    start = time.perf_counter()
    norm_dense = _minmax([(d["doc_id"], d["score"]) for d in dense])
    norm_sparse = _minmax([(d["doc_id"], d["score"]) for d in sparse])

    merged: dict[str, dict] = {}
    for doc in dense:
        merged[doc["doc_id"]] = doc
    for doc in sparse:
        merged.setdefault(doc["doc_id"], doc)

    w_v, w_k = settings.vector_weight, settings.keyword_weight
    fused = []
    for doc_id, doc in merged.items():
        fused_score = w_v * norm_dense.get(doc_id, 0.0) + w_k * norm_sparse.get(doc_id, 0.0)
        fused.append(
            {
                **doc,
                "vector_score": doc.get("score", 0.0),
                "keyword_score": next(
                    (d["score"] for d in sparse if d["doc_id"] == doc_id), 0.0
                ),
                "score": round(fused_score, 4),
            }
        )
    fused.sort(key=lambda d: d["score"], reverse=True)
    metrics.rag_retrieval_latency.labels("hybrid").observe(time.perf_counter() - start)
    metrics.rag_retrieval_count.labels("hybrid").inc()
    return fused[:top_k]
