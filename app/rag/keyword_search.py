"""PostgreSQL 全文检索（Keyword Retrieval）。

为什么用 PG FTS 而不是 OpenSearch（ADR-003）：
- 知识库规模（数百~数千条）下，PG FTS 完全够用，避免多引入一个
  独立检索引擎的运维成本；
- 中文分词：PG 内置没有中文 parser，第一版用 bigram（二元组）分词
  写入 tsvector（'simple' 配置）——零扩展依赖、任何 PG 都能跑，
  对短查询和短文档效果稳定。

tsvector 在 ingestion 时计算（content_tsv 列），查询时对 query 做
同样的 bigram 化。
"""

import re
import time

from sqlalchemy import text

from app.db.session import engine
from app.observability import metrics

_BIGRAM_STRIP = re.compile(r"[^一-鿿A-Za-z0-9]")


def bigram_tokenize(text: str) -> str:
    """中文/英文数字混合的 bigram 分词：空格分隔的二元组字符串。"""
    cleaned = _BIGRAM_STRIP.sub("", text)
    if not cleaned:
        return ""
    tokens = [cleaned[i : i + 2] for i in range(len(cleaned) - 1)]
    if not tokens:  # 单字符输入
        tokens = [cleaned]
    return " ".join(tokens)


def to_tsvector(text: str) -> str:
    """生成写入 content_tsv 的 SQL 表达式片段（ingestion 用）。"""
    tokens = bigram_tokenize(text)
    if not tokens:
        return "to_tsvector('simple', '')"
    # 参数化注入安全：单引号转义
    safe = tokens.replace("'", "''")
    return f"to_tsvector('simple', '{safe}')"


def keyword_search(query: str, *, top_k: int, category: str | None = None) -> list[dict]:
    """关键词检索：返回 [{doc_id, content, category, source, score, ...}]。"""
    tokens = bigram_tokenize(query)
    if not tokens:
        return []
    # OR 语义：命中任一 bigram 即候选，命中越多 ts_rank 越高。
    # （AND 语义等价于精确短语匹配，对短文档过严；OR 更适合模糊关键词检索。）
    tsquery = " | ".join(tokens.split())
    start = time.perf_counter()
    with engine.connect() as conn:
        params: dict = {"q": tsquery, "top_k": top_k}
        where = ""
        if category:
            where = "AND category = :cat"
            params["cat"] = category
        rows = conn.execute(
            text(
                "SELECT doc_id, content, category, source, version, effective_date,"
                " ts_rank(content_tsv, to_tsquery('simple', :q)) AS score"
                " FROM knowledge_documents"
                " WHERE content_tsv @@ to_tsquery('simple', :q) "
                + where
                + " ORDER BY score DESC LIMIT :top_k"
            ),
            params,
        ).fetchall()
    metrics.rag_retrieval_latency.labels("keyword").observe(time.perf_counter() - start)
    metrics.rag_retrieval_count.labels("keyword").inc()
    return [
        {
            "doc_id": r.doc_id,
            "content": r.content,
            "category": r.category,
            "source": r.source,
            "version": r.version,
            "effective_date": r.effective_date,
            "score": float(r.score),
        }
        for r in rows
    ]
