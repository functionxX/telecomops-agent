"""RAG Pipeline：Query → Rewrite → Hybrid Retrieval → Rerank → Context → Answer。

核心逻辑自己实现（Embedding / 检索 / 融合 / 重排 / 上下文），
不包装成 LangChain Retriever/Chain——每个环节内部发生什么都可解释。
"""

from app.core.config import settings
from app.core.logging import get_logger
from app.llm.client import get_llm_client
from app.llm.schemas import ChatMessage
from app.observability.tracing import start_span
from app.rag.context_builder import build_context
from app.rag.embedding import get_embedding_client
from app.rag.hybrid_search import hybrid_search
from app.rag.query_rewrite import rewrite_query
from app.rag.reranker import get_reranker

logger = get_logger(__name__)

NO_KNOWLEDGE_ANSWER = "知识库中没有找到足够相关的信息，无法回答这个问题。建议联系人工客服获取帮助。"

ANSWER_SYSTEM_PROMPT = """你是电信 CRM 智能助手。只根据提供的知识文档回答用户问题。
规则：
1. 只使用文档中的信息，不得编造；
2. 引用文档时在句末标注编号，如 [1][2]；
3. 如果文档信息不足以回答，明确说"知识库中没有找到足够相关的信息"；
4. 回答简洁、面向用户、中文。"""


def answer_with_rag(query: str, conversation_id: str = "") -> dict:
    """完整 RAG 回答。返回 {answer, citations, documents, rewritten_query}。"""
    # 1. Query Rewrite（失败自动 fallback 原始 query）
    with start_span("rag.query_rewrite"):
        rewritten = rewrite_query(query)

    # 2. Hybrid Retrieval（Dense + Keyword 融合，Top 20）
    with start_span("rag.retrieval", {"top_k": settings.retrieval_top_k}) as span:
        embedding = get_embedding_client()
        query_vector = embedding.embed_query(rewritten)
        candidates = hybrid_search(rewritten, query_vector, top_k=settings.retrieval_top_k)
        span.set_attribute("candidates", len(candidates))

    # 3. Rerank（Top 5）
    with start_span("rag.rerank", {"top_k": settings.rerank_top_k}):
        documents = get_reranker().rerank(rewritten, candidates, top_k=settings.rerank_top_k)

    # 4. 最低相关性阈值：低于阈值 → 明确回答"无相关信息"，不让 LLM 编造
    best_similarity = max((d.get("vector_score", 0.0) for d in documents), default=0.0)
    if not documents or best_similarity < settings.min_relevance_score:
        logger.info(
            "rag_no_relevant_docs",
            extra={"query": query, "best_similarity": best_similarity},
        )
        return {
            "answer": NO_KNOWLEDGE_ANSWER,
            "citations": [],
            "documents": [],
            "rewritten_query": rewritten,
        }

    # 5. Context 构建（截断保护 + 引用编号）
    context, citations = build_context(documents)

    # 6. LLM 生成答案（Mock 模式用模板）
    if settings.mock_llm:
        sources = "、".join(f"[{c['index']}] {c['source']}" for c in citations)
        answer = (
            f"（Mock 模式 RAG 回答）根据知识库检索结果：\n"
            f"{documents[0]['content']}\n\n"
            f"参考来源：{sources}"
        )
    else:
        try:
            resp = get_llm_client().chat(
                [
                    ChatMessage(role="system", content=ANSWER_SYSTEM_PROMPT),
                    ChatMessage(
                        role="user",
                        content=f"用户问题：{query}\n\n知识文档：\n{context}",
                    ),
                ],
                temperature=0.2,
                purpose="rag_answer",
            )
            answer = resp.content
        except Exception as exc:  # noqa: BLE001
            logger.exception("rag_answer_failed", extra={"error": str(exc)[:200]})
            answer = "回答生成失败，请稍后重试。"

    return {
        "answer": answer,
        "citations": citations,
        "documents": documents,
        "rewritten_query": rewritten,
    }
