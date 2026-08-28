"""检索演示：Dense / Keyword / Hybrid 三路对比。

用法：uv run python scripts/demo_retrieval.py "国际漫游怎么开通"
"""

import sys

from app.core.config import settings
from app.core.logging import setup_logging
from app.rag.embedding import get_embedding_client
from app.rag.hybrid_search import hybrid_search
from app.rag.keyword_search import keyword_search
from app.rag.vector_search import get_vector_store

setup_logging(settings.log_level)


def main() -> int:
    query = sys.argv[1] if len(sys.argv) > 1 else "5G套餐有哪些"
    qv = get_embedding_client().embed_query(query)
    print(f"查询: {query}\n")

    print("--- Dense（Milvus 向量检索）---")
    for d in get_vector_store().search(qv, top_k=3):
        print(f"  {d['doc_id']:16s} [{d['score']:.4f}] {d['content'][:44]}")

    print("--- Keyword（PostgreSQL FTS）---")
    for d in keyword_search(query, top_k=3):
        print(f"  {d['doc_id']:16s} [{d['score']:.4f}] {d['content'][:44]}")

    print("--- Hybrid（加权融合）---")
    for d in hybrid_search(query, qv, top_k=5):
        print(
            f"  {d['doc_id']:16s} fused={d['score']:.4f} "
            f"(vec={d['vector_score']:.4f}, kw={d['keyword_score']:.4f}) {d['content'][:36]}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
