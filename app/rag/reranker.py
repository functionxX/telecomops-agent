"""BGE Reranker：对 Top-K 候选做精排。

为什么需要 Reranker（ADR-004）：召回（向量/关键词）追求高召回、
用粗粒度分数；重排用交叉编码器逐对判断 query-document 相关性，
语义交互更精细，能把真正相关的文档提到最前，且为后续
Context 截断提供高质量排序。模型与 Top-K 全配置化。
"""

import time
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.observability import metrics


class Reranker:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self._model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, documents: list[dict], top_k: int) -> list[dict]:
        """返回按 rerank 分数降序的 top_k 文档（附 rerank_score）。"""
        if not documents:
            return []
        start = time.perf_counter()
        pairs = [(query, d["content"]) for d in documents]
        scores = self._model.predict(pairs)
        metrics.rag_retrieval_latency.labels("rerank").observe(time.perf_counter() - start)
        metrics.rag_retrieval_count.labels("rerank").inc()
        ranked = [
            {**doc, "rerank_score": float(s)}
            for doc, s in sorted(
                zip(documents, scores, strict=False), key=lambda x: x[1], reverse=True
            )
        ]
        return ranked[:top_k]


@lru_cache
def get_reranker() -> Reranker:
    return Reranker(settings.reranker_model, device=settings.embedding_device)
