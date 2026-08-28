"""知识库导入：data/knowledge/*.json → PostgreSQL（含 tsvector）+ Milvus（向量）。

幂等策略：
- PostgreSQL：按 doc_id upsert（content_tsv 同步重算）；
- Milvus：主键 = doc_id，upsert 覆盖。
"""

import json
from pathlib import Path

from sqlalchemy import text

from app.core.logging import get_logger
from app.db.session import engine
from app.rag.embedding import get_embedding_client
from app.rag.keyword_search import to_tsvector
from app.rag.vector_search import get_vector_store

logger = get_logger(__name__)

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "knowledge"


def load_knowledge_documents(data_dir: Path | None = None) -> list[dict]:
    """加载知识库 JSON 文件（faq/package/roaming/billing/promotion/policy）。"""
    data_dir = data_dir or KNOWLEDGE_DIR
    docs: list[dict] = []
    for path in sorted(data_dir.glob("**/*.json")):
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        for record in records:
            required = ("id", "content", "category", "source", "version", "effective_date")
            missing = [k for k in required if k not in record]
            if missing:
                raise ValueError(f"{path} 中记录缺少字段: {missing}")
            docs.append(
                {
                    "doc_id": record["id"],
                    "content": record["content"],
                    "category": record["category"],
                    "source": record["source"],
                    "version": record["version"],
                    "effective_date": record["effective_date"],
                }
            )
    return docs


def ingest(data_dir: Path | None = None) -> dict:
    """全量导入。返回统计信息。"""
    docs = load_knowledge_documents(data_dir)
    if not docs:
        logger.warning("no_documents_found", extra={"dir": str(data_dir or KNOWLEDGE_DIR)})
        return {"total": 0}

    # 1) PostgreSQL：upsert + tsvector
    with engine.begin() as conn:
        for d in docs:
            conn.execute(
                text(
                    "INSERT INTO knowledge_documents"
                    " (doc_id, content, category, source, version, effective_date, content_tsv)"
                    " VALUES (:doc_id, :content, :category, :source, :version, :effective_date, "
                    + to_tsvector(d["content"])
                    + ") ON CONFLICT (doc_id) DO UPDATE SET content = EXCLUDED.content,"
                    " category = EXCLUDED.category, source = EXCLUDED.source,"
                    " version = EXCLUDED.version, effective_date = EXCLUDED.effective_date,"
                    " content_tsv = EXCLUDED.content_tsv"
                ),
                {
                    "doc_id": d["doc_id"],
                    "content": d["content"],
                    "category": d["category"],
                    "source": d["source"],
                    "version": d["version"],
                    "effective_date": d["effective_date"],
                },
            )

    # 2) Milvus：批量 embedding + upsert
    embedding = get_embedding_client()
    batch_size = 32
    store = get_vector_store()
    total = 0
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        vectors = embedding.embed([d["content"] for d in batch])
        store.upsert([{**d, "vector": v} for d, v in zip(batch, vectors, strict=False)])
        total += len(batch)

    logger.info("ingest_done", extra={"total": total, "dimension": embedding.dimension})
    return {"total": total, "dimension": embedding.dimension}
