"""Context Builder：把检索结果组织成 LLM 上下文 + 引用列表。

- 截断保护：MAX_CONTEXT_CHARS 内拼接，超出部分丢弃并标记；
- 引用编号 [1][2]… 与文档 source 一一对应，答案必须带来源。
"""

from app.core.config import settings


def build_context(
    documents: list[dict], *, max_chars: int | None = None
) -> tuple[str, list[dict]]:
    """返回 (context 文本, citations)。

    citations: [{index, doc_id, source, content, score}]
    """
    max_chars = max_chars or settings.max_context_chars
    blocks: list[str] = []
    citations: list[dict] = []
    truncated_docs = 0
    used = 0
    for i, doc in enumerate(documents, start=1):
        block = f"[{i}] {doc['content']}"
        if used + len(block) > max_chars:
            truncated_docs += 1
            continue
        blocks.append(block)
        used += len(block)
        citations.append(
            {
                "index": i,
                "doc_id": doc["doc_id"],
                "source": doc["source"],
                "content": doc["content"],
                "score": doc.get("rerank_score", doc.get("score")),
            }
        )
    context = "\n\n".join(blocks)
    if truncated_docs:
        context += f"\n\n（另有 {truncated_docs} 条文档因长度限制被截断）"
    return context, citations
