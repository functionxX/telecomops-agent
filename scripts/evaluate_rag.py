"""RAG 评测：Vector vs Hybrid vs Hybrid+Rerank 三组对比。

真实运行数据集，不伪造数字：
    uv run python scripts/evaluate_rag.py [--limit N]
结果写入 evaluation/results/rag_results.json。
"""

import argparse
import json
import sys
import time
from pathlib import Path

from app.core.config import settings
from app.core.logging import setup_logging
from app.rag.embedding import get_embedding_client
from app.rag.hybrid_search import hybrid_search
from app.rag.reranker import get_reranker
from app.rag.vector_search import get_vector_store

setup_logging(settings.log_level)

DATASET = Path(__file__).resolve().parent.parent / "evaluation" / "datasets" / "rag_queries.json"
RESULTS = Path(__file__).resolve().parent.parent / "evaluation" / "results" / "rag_results.json"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.rag_eval.metrics import mrr, recall_at_k  # noqa: E402


def run_variant(queries: list[dict], variant: str) -> tuple[list[list[str]], float]:
    """执行一个检索变体，返回 (rankings, 总耗时 ms)。"""
    embedding = get_embedding_client()
    store = get_vector_store()
    reranker = get_reranker() if variant == "hybrid_rerank" else None

    rankings: list[list[str]] = []
    start = time.perf_counter()
    for item in queries:
        q = item["query"]
        qv = embedding.embed_query(q)
        if variant == "vector":
            docs = store.search(qv, top_k=settings.retrieval_top_k)
            rankings.append([d["doc_id"] for d in docs])
        elif variant == "hybrid":
            docs = hybrid_search(q, qv, top_k=settings.retrieval_top_k)
            rankings.append([d["doc_id"] for d in docs])
        else:  # hybrid_rerank
            candidates = hybrid_search(q, qv, top_k=settings.retrieval_top_k)
            assert reranker is not None
            docs = reranker.rerank(q, candidates, top_k=settings.rerank_top_k)
            rankings.append([d["doc_id"] for d in docs])
    elapsed_ms = (time.perf_counter() - start) * 1000
    return rankings, round(elapsed_ms, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条查询（调试用）")
    args = parser.parse_args()

    with open(DATASET, encoding="utf-8") as f:
        queries = json.load(f)
    if args.limit:
        queries = queries[: args.limit]

    expected = [q["expected_document_ids"] for q in queries]

    variants = ["vector", "hybrid", "hybrid_rerank"]
    report: dict = {"dataset": str(DATASET), "dataset_size": len(queries), "variants": {}}
    for variant in variants:
        print(f"运行变体: {variant}（{len(queries)} 条查询）...")
        rankings, elapsed = run_variant(queries, variant)
        result = {
            "recall@5": recall_at_k(rankings, expected, 5),
            "recall@10": recall_at_k(rankings, expected, 10),
            "mrr": mrr(rankings, expected),
            "total_ms": elapsed,
        }
        report["variants"][variant] = result
        print(f"  Recall@5={result['recall@5']}  Recall@10={result['recall@10']}  "
              f"MRR={result['mrr']}  耗时 {elapsed}ms")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
