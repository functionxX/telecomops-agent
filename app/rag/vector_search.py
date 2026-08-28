"""Milvus 向量检索（Dense Retrieval）。

Index 选择说明（ADR-002）：
- IVF_FLAT：10 万级文档规模下召回率高、内存友好；构建快，适合第一版；
  nlist 取 sqrt(N) 量级（104 条文档配置 128 已足够，留余量）；nprobe 越大
  召回越高、越慢，16 为经验起点。全部参数可配置。
- COSINE：BGE 系列按余弦相似度训练/评估，Embedding 侧已 normalize，
  与余弦度量方向一致。
"""

import time
from functools import lru_cache
from typing import Any

import os  # noqa: E402

# pymilvus 在 import 时会 load_dotenv() 并解析全局 MILVUS_URI，
# 且只接受 http 形式（本地文件路径只能传给 MilvusClient 实例）。
# 在 import pymilvus 之前兜底一个合法默认值：load_dotenv 不覆盖
# 已存在的环境变量，因此 Lite 路径（.env 中配置）不会被全局解析炸掉。
os.environ.setdefault("MILVUS_URI", "http://localhost:19530")

from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient  # noqa: E402
from pymilvus.milvus_client.index import IndexParams  # noqa: E402

from app.core.config import settings
from app.core.logging import get_logger
from app.observability import metrics

logger = get_logger(__name__)


def _schema(dim: int) -> CollectionSchema:
    return CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="version", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="effective_date", dtype=DataType.VARCHAR, max_length=16),
        ]
    )


class VectorStore:
    """Milvus 封装：collection 初始化 / 插入 / 检索 / 元数据过滤。"""

    def __init__(self, uri: str, collection: str, dimension: int) -> None:
        self._client = MilvusClient(uri=uri)
        self.collection = collection
        self.dimension = dimension

    def ensure_collection(self) -> None:
        if not self._client.has_collection(self.collection):
            self._client.create_collection(
                collection_name=self.collection,
                schema=_schema(self.dimension),
                index_params=self._index_params(),
            )
            logger.info(
                "milvus_collection_created",
                extra={"collection": self.collection, "dim": self.dimension},
            )
        # 防御：历史版本可能留下无索引的 collection，补齐索引
        if not self._client.list_indexes(self.collection):
            self._client.create_index(
                collection_name=self.collection,
                index_params=self._index_params(),
            )
        # load 是进程内状态：每次进程启动都要确保已加载（幂等）
        self._client.load_collection(self.collection)

    def _index_params(self) -> IndexParams:
        params = IndexParams()
        params.add_index(
            field_name="vector",
            index_type=settings.milvus_index_type,  # IVF_FLAT
            metric_type=settings.milvus_metric_type,  # COSINE
            params={"nlist": settings.milvus_index_nlist},
        )
        return params

    def upsert(self, records: list[dict[str, Any]]) -> None:
        """按 doc_id 幂等 upsert（id=doc_id 作为主键）。"""
        if not records:
            return
        self.ensure_collection()
        rows = []
        for r in records:
            rows.append(
                {
                    "id": r["doc_id"],
                    "vector": r["vector"],
                    "doc_id": r["doc_id"],
                    "content": r["content"][:2048],
                    "category": r["category"],
                    "source": r["source"],
                    "version": r["version"],
                    "effective_date": r["effective_date"],
                }
            )
        self._client.upsert(collection_name=self.collection, data=rows)

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """向量检索，返回 [{doc_id, content, category, source, score, ...}]。"""
        self.ensure_collection()
        filter_expr = f'category == "{category}"' if category else None
        start = time.perf_counter()
        hits = self._client.search(
            collection_name=self.collection,
            data=[query_vector],
            limit=top_k,
            filter=filter_expr,
            search_params={
                "metric_type": settings.milvus_metric_type,
                "params": {"nprobe": settings.milvus_nprobe},
            },
            output_fields=["doc_id", "content", "category", "source", "version", "effective_date"],
        )
        metrics.rag_retrieval_latency.labels("dense").observe(time.perf_counter() - start)
        metrics.rag_retrieval_count.labels("dense").inc()
        results = []
        for hit in hits[0]:
            entity = hit.get("entity", hit)
            results.append(
                {
                    "doc_id": entity["doc_id"],
                    "content": entity["content"],
                    "category": entity["category"],
                    "source": entity["source"],
                    "version": entity.get("version", ""),
                    "effective_date": entity.get("effective_date", ""),
                    "score": hit.get("distance", 0.0),  # COSINE 下即相似度，越大越相关
                }
            )
        return results


@lru_cache
def get_vector_store() -> VectorStore:
    from app.rag.embedding import get_embedding_client

    embedding = get_embedding_client()
    store = VectorStore(settings.milvus_uri, settings.milvus_collection, embedding.dimension)
    store.ensure_collection()
    return store
