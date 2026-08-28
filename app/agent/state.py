"""AgentState：Workflow 的核心数据载体。

节点之间通过 State 显式传递数据，不存在隐式全局变量。
列表字段用 operator.add 归并（循环里追加），标量默认覆盖。
"""

import operator
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):
    # ---- 会话标识 ----
    conversation_id: str
    user_id: str
    trace_id: str

    # ---- 输入与历史 ----
    query: str  # 用户本轮原始消息
    messages: Annotated[list[dict[str, Any]], operator.add]  # 会话历史（可跨轮追加）
    intent: str  # FAQ / QUERY / TASK / UNKNOWN

    # ---- RAG ----
    rewritten_query: str
    retrieved_documents: Annotated[list[dict[str, Any]], operator.add]
    citations: list[dict[str, Any]]

    # ---- 计划与执行 ----
    plan: list[dict[str, Any]]  # Planner 产出的结构化计划
    current_step: str  # 当前 step_id
    tool_calls: Annotated[list[dict[str, Any]], operator.add]  # 调用审计
    tool_results: dict[str, dict[str, Any]]  # step_id -> result
    total_tool_executions: int  # 全局执行计数（失控循环硬兜底）

    # ---- 校验 / 重试 / 重规划 ----
    execution_status: str  # 内部路由标记：success / plan_complete / error / approval_required
    validation_result: dict[str, Any]
    retry_count: int
    replan_count: int
    failure_reason: str
    failed_tool: str

    # ---- Human-in-the-loop ----
    requires_human_approval: bool
    pending_approval: dict[str, Any]  # {approval_id, tool_name, arguments, summary, step_id}
    human_decision: str  # "" / "approved" / "rejected"

    # ---- 输出 ----
    final_answer: str
    error: str

    # ---- 可观测元数据 ----
    metadata: dict[str, Any]  # router/planner 的 latency、model、token usage 等
