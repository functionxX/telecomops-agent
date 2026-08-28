"""Intent Router：只负责判断用户请求类型，不执行任何工具。

- LLM 结构化输出（temperature=0，JSON mode + Pydantic 校验）
- 非法输出兜底 / LLM 失败 → UNKNOWN（安全方向：不做总好过错做）
- 记录 latency、model、token usage、最终 intent（metadata）
"""

import time
from typing import Literal

from pydantic import BaseModel, Field

from app.llm.client import get_llm_client
from app.llm.schemas import ChatMessage

ROUTER_SYSTEM_PROMPT = """你是电信 CRM 智能助手的意图路由器。只判断用户请求属于哪一类，不回答问题、不调用工具。

分类定义：
- FAQ：知识类问题——产品/套餐介绍、资费、政策、流程、优惠活动等，可以通过知识库回答。
  例：“5G套餐有哪些”“国际漫游怎么开通”“违约金怎么算”
- QUERY：单一数据查询——需要查询一个具体业务数据，一次工具调用即可。
  例：“我的积分还有多少”“我的套餐还剩多少流量”“查一下订单”
- TASK：任务类——多步操作、办理/变更业务、或带条件分支的请求，需要生成计划执行。
  例：“帮我办理30GB流量包”“帮我开通国际漫游”
  “帮我查一下套餐，如果剩余流量低于10GB就推荐一个流量包”
- UNKNOWN：闲聊、无法判断、超出业务范围。

规则：
1. 含“如果…就/则…”等条件分支的 → TASK
2. 含办理/开通/关闭/取消/订购类动作 → TASK
3. 一个请求里包含两个以上不同数据的查询（如“查询我的套餐和剩余流量”）→ TASK
4. 问“有哪些/怎么/资费/政策/流程”等知识 → FAQ
5. 单一数据查询 → QUERY
6. 无法判断 → UNKNOWN
"""


class RouterDecision(BaseModel):
    intent: Literal["FAQ", "QUERY", "TASK", "UNKNOWN"]
    reason: str = Field(description="分类理由，一句话")
    confidence: float | None = Field(default=None, description="置信度 0-1，可空")


def route(user_message: str) -> tuple[RouterDecision, dict]:
    """执行意图路由。返回 (决策, 统计信息)。

    任何异常都兜底为 UNKNOWN——路由失败不能让 Workflow 瘫痪。
    """
    client = get_llm_client()
    start = time.perf_counter()
    stats: dict = {"model": client.model, "purpose": "router"}
    try:
        decision = client.structured_output(
            [
                ChatMessage(role="system", content=ROUTER_SYSTEM_PROMPT),
                ChatMessage(role="user", content=user_message),
            ],
            RouterDecision,
            purpose="router",
        )
        stats["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 — 兜底，任何失败进 UNKNOWN
        decision = RouterDecision(intent="UNKNOWN", reason=f"路由失败兜底: {exc}")
        stats["status"] = "fallback"
        stats["error"] = str(exc)[:200]
    stats["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
    stats["intent"] = decision.intent
    return decision, stats
