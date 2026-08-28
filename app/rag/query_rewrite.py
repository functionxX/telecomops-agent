"""Query Rewrite：检索前把口语化查询改写成检索友好的表述。

解决什么：用户口语（"那个套餐流量咋没了"）与知识库书面语
（"查询套餐剩余流量"）存在词汇鸿沟，改写能显著提升召回。
改写只服务于检索，不改变用户原始语义；失败时 fallback 原始 query。
"""

from pydantic import BaseModel, Field

from app.llm.client import get_llm_client
from app.llm.schemas import ChatMessage

REWRITE_SYSTEM_PROMPT = """你是电信领域查询改写器。把用户口语化的问题改写为检索友好的一句话查询。
规则：
1. 保留用户原始语义，不得增删业务事实（如套餐档位、数量、国家）；
2. 补全指代（"那个套餐"→"用户当前套餐"），使用规范术语（"流量没了"→"剩余流量"）；
3. 去掉语气词；只输出改写后的查询文本。
"""


class QueryRewriteResult(BaseModel):
    rewritten_query: str = Field(description="改写后的检索查询")


def rewrite_query(query: str) -> str:
    """改写查询；任何失败都返回原始 query（fallback，不影响主流程）。"""
    try:
        result = get_llm_client().structured_output(
            [
                ChatMessage(role="system", content=REWRITE_SYSTEM_PROMPT),
                ChatMessage(role="user", content=query),
            ],
            QueryRewriteResult,
            purpose="query_rewrite",
        )
        rewritten = result.rewritten_query.strip()
        return rewritten if rewritten else query
    except Exception:  # noqa: BLE001 — 改写失败必须 fallback 原始 query
        return query
