"""Embedding 客户端：sentence-transformers + BGE 系列。

- 模型名从配置读取（EMBEDDING_MODEL），业务代码不硬编码；
- 向量维度从模型配置动态获取（get_sentence_embedding_dimension），禁止手写；
- BGE 系列按余弦相似度训练/评估 → normalize_embeddings=True 与 Milvus COSINE 一致。
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.observability import metrics


class EmbeddingClient:
    """本地 Embedding 模型封装（懒加载单例，避免每个请求重载模型）。"""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self.dimension = int(self._model.get_sentence_embedding_dimension() or 0)

    def embed(self, texts: list[str]) -> list[list[float]]:
        with _observe_latency():
            vectors = self._model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


@lru_cache
def get_embedding_client() -> EmbeddingClient:
    return EmbeddingClient(settings.embedding_model, device=settings.embedding_device)


def _observe_latency():
    """记录 embedding 耗时（仅首次测量用；histogram 无具体 label 成本高，简化处理）。"""
    import time
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        start = time.perf_counter()
        yield
        metrics.rag_retrieval_latency.labels("embedding").observe(time.perf_counter() - start)

    return _ctx()
