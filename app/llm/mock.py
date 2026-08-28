"""MockLLMClient：确定性规则实现（CI / 离线开发 / Eval 复现）。

重要：Mock 不是随机假数据，而是一套明确的规则引擎，
与真实 LLM 共享同一 LLMClient 接口。README 会明确标注：
Mock 模式下的 Eval 结果不是正式 Benchmark。
"""

import json
import re
from collections.abc import Iterator
from typing import Any

from app.llm.client import LLMClient
from app.llm.schemas import ChatMessage, ChatResponse, LLMUsage

# ---------- 规则（确定性，按 schema 名称分派） ----------

_TASK_MARKERS = ["如果", "然后", "再", "之后", "先查", "办理", "开通", "关闭", "取消", "订购", "下单", "购买"]
_FAQ_MARKERS = ["哪些", "有什么", "介绍", "怎么", "如何", "什么是", "资费", "政策", "流程", "优惠", "活动", "协议", "规则", "条件", "方式", "可以", "适合"]
_QUERY_MARKERS = ["查询", "查一下", "还剩", "还有多少", "是多少", "状态", "我的", "积分", "资料", "档案", "搜索"]
_UNKNOWN_MARKERS = ["天气", "你好", "谢谢", "再见", "你是谁"]

# 条件性多步任务：如果…就 / 不够…的话 等
_CONDITIONAL_RE = re.compile(r"(如果.{0,12}(就|则))|((不够|低于|少于|不足).{0,10}(的话|就|则))")


def _mock_intent(query: str) -> tuple[str, str]:
    """Router 的 Mock 规则：条件性多步任务 → TASK；知识问题 → FAQ；查询 → QUERY。"""
    if any(m in query for m in _UNKNOWN_MARKERS):
        return "UNKNOWN", "mock: 闲聊/业务外内容"
    if _CONDITIONAL_RE.search(query):
        return "TASK", "mock: 条件性多步任务"
    # 套餐+流量 双实体查询 → 多步任务
    if ("和" in query or "并且" in query) and "套餐" in query and "流量" in query:
        return "TASK", "mock: 多实体查询任务"
    # "我的/当前 X 是什么" → 个人数据查询（区别于"什么是 X"的知识问题）
    if "是什么" in query and re.search(r"(我的|现在用|当前)", query):
        return "QUERY", "mock: 个人数据查询"
    # 知识型标记优先于动作动词："怎么开通"是 FAQ，"帮我开通"是 TASK
    if any(m in query for m in _FAQ_MARKERS):
        return "FAQ", "mock: 知识类问题"
    if any(m in query for m in _TASK_MARKERS):
        return "TASK", "mock: 包含办理/变更类动作词"
    if any(m in query for m in _QUERY_MARKERS):
        return "QUERY", "mock: 查询类问题"
    return "UNKNOWN", "mock: 无法判断"


def _mock_plan(query: str, user_id: str) -> list[dict[str, Any]]:
    """Planner 的 Mock 模板：覆盖演示场景 3/4 的结构。"""
    if "取消" in query and "订单" in query:
        m = re.search(r"ord_[a-z0-9_]+", query)
        return [
            {
                "step_id": "step_1",
                "tool": "cancel_order",
                "arguments": {"order_id": m.group(0) if m else ""},
                "description": "取消指定订单",
                "status": "PENDING",
            }
        ]
    if "套餐" in query and "流量" in query and "推荐" not in query and "如果" not in query:
        # 套餐 + 流量双查询（无条件分支）
        return [
            {
                "step_id": "step_1",
                "tool": "get_current_package",
                "arguments": {"user_id": user_id},
                "description": "查询当前套餐",
                "status": "PENDING",
            },
            {
                "step_id": "step_2",
                "tool": "get_remaining_data",
                "arguments": {"user_id": user_id},
                "description": "查询剩余流量",
                "status": "PENDING",
            },
        ]
    if "流量" in query and ("推荐" in query or "如果" in query):
        # 场景3：查套餐 → 查流量 → 条件判断 → 推荐
        return [
            {
                "step_id": "step_1",
                "tool": "get_current_package",
                "arguments": {"user_id": user_id},
                "description": "查询当前主套餐",
                "status": "PENDING",
            },
            {
                "step_id": "step_2",
                "tool": "get_remaining_data",
                "arguments": {"user_id": user_id},
                "description": "查询剩余流量",
                "status": "PENDING",
            },
            {
                "step_id": "step_3",
                "tool": None,
                "arguments": {
                    "left": "$step_2.total_remaining_gb",
                    "op": "<",
                    "right": 10,
                    "then_step": "step_4",
                    "else_step": "END",
                },
                "description": "判断剩余流量是否低于 10GB",
                "status": "PENDING",
            },
            {
                "step_id": "step_4",
                "tool": "recommend_package",
                "arguments": {"user_id": user_id, "min_data_gb": 10},
                "description": "推荐流量加餐包",
                "status": "PENDING",
            },
        ]

    m = re.search(r"(\d+)\s*GB", query)
    if m and ("办" in query or "购买" in query or "订" in query):
        # 场景4：办理流量包 → 下单
        return [
            {
                "step_id": "step_1",
                "tool": "create_order",
                "arguments": {"user_id": user_id, "package_id": f"addon_{m.group(1)}g"},
                "description": f"办理 {m.group(1)}GB 流量包",
                "status": "PENDING",
            }
        ]
    if "漫游" in query:
        if "开通" in query:
            return [
                {
                    "step_id": "step_1",
                    "tool": "enable_roaming",
                    "arguments": {"user_id": user_id},
                    "description": "开通国际漫游",
                    "status": "PENDING",
                }
            ]
        if "关闭" in query:
            return [
                {
                    "step_id": "step_1",
                    "tool": "disable_roaming",
                    "arguments": {"user_id": user_id},
                    "description": "关闭国际漫游",
                    "status": "PENDING",
                }
            ]
    if "流量" in query or "剩余" in query:
        return [
            {
                "step_id": "step_1",
                "tool": "get_remaining_data",
                "arguments": {"user_id": user_id},
                "description": "查询剩余流量",
                "status": "PENDING",
            }
        ]
    # 兜底：查询当前套餐
    return [
        {
            "step_id": "step_1",
            "tool": "get_current_package",
            "arguments": {"user_id": user_id},
            "description": "查询当前套餐",
            "status": "PENDING",
        }
    ]


_TOOL_KEYWORDS: list[tuple[str, str]] = [
    ("加餐包|搜索|在售", "search_packages"),
    ("语音|分钟", "get_current_package"),
    ("剩余|还剩|还有多少流量", "get_remaining_data"),
    ("套餐.*多少|现在用什么套餐|现在用的|当前套餐|套餐情况|套餐详情", "get_current_package"),
    ("漫游", "query_roaming_status"),
    ("积分|星级|客户资料|档案", "get_customer_profile"),
    ("等级", "get_customer_level"),
    ("订单", "query_order"),
]


def _mock_tool_selection(query: str, tools: list[dict[str, Any]]) -> dict[str, Any] | None:
    """QUERY 路径 bind_tools 的 Mock 选择：按关键词匹配工具清单。"""
    available = {t["function"]["name"] for t in tools}
    for pattern, tool in _TOOL_KEYWORDS:
        if tool in available and re.search(pattern, query):
            args: dict[str, Any] = {}
            if tool == "query_order":
                m = re.search(r"ord_[a-z0-9_]+", query)
                args = {"order_id": m.group(0) if m else "ord_20260601_001"}
            return {
                "id": "call_mock_1",
                "type": "function",
                "function": {"name": tool, "arguments": json.dumps(args, ensure_ascii=False)},
            }
    fallback = "get_customer_profile" if "get_customer_profile" in available else next(iter(available), None)
    if fallback is None:
        return None
    return {
        "id": "call_mock_1",
        "type": "function",
        "function": {"name": fallback, "arguments": json.dumps({}, ensure_ascii=False)},
    }


def _mock_usage(tokens: int = 10) -> LLMUsage:
    return LLMUsage(prompt_tokens=tokens, completion_tokens=tokens, total_tokens=tokens * 2)


class MockLLMClient(LLMClient):
    """确定性规则 Mock。schema 按类名分派（避免与 agent 模块循环依赖）。"""

    def __init__(self, model: str = "mock") -> None:
        self.model = model

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        purpose: str = "chat",
    ) -> ChatResponse:
        user_msg = next((m.content for m in reversed(messages) if m.role == "user"), "")
        content = ""
        tool_calls = None
        if tools is not None:
            tool_calls = [tc for tc in [_mock_tool_selection(user_msg, tools)] if tc]
        else:
            content = "（Mock 回答：无真实 LLM 调用）"
        return ChatResponse(
            content=content,
            usage=_mock_usage(),
            model=self.model,
            finish_reason="tool_calls" if tool_calls else "stop",
            tool_calls=tool_calls,
        )

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        purpose: str = "chat",
    ) -> Iterator[str]:
        yield from ["（", "Mock", "回答", "）"]

    def structured_output(
        self,
        messages: list[ChatMessage],
        schema: type,
        *,
        purpose: str = "structured",
        temperature: float = 0.0,
    ) -> Any:
        user_msg = next((m.content for m in reversed(messages) if m.role == "user"), "")
        system_msg = next((m.content for m in messages if m.role == "system"), "")
        user_id = re.search(r"user_\d{3}", system_msg + user_msg) or "user_001"
        user_id = user_id.group(0) if hasattr(user_id, "group") else user_id

        name = schema.__name__
        if name == "RouterDecision":
            intent, reason = _mock_intent(user_msg)
            return schema(intent=intent, reason=reason)
        if name == "Plan":
            return schema(steps=_mock_plan(user_msg, user_id))
        if name == "QueryRewriteResult":
            return schema(rewritten_query=user_msg)
        if name == "ValidatorVerdict":
            return schema(satisfied=True, reason="mock 规则校验通过")
        raise NotImplementedError(f"Mock 未实现 schema 分派: {name}")
