"""LangGraph Workflow 装配。

图结构（含真实 Conditional Edges 与 Loop）：

    START → router ──FAQ──> rag ──────────────> answer → END
                 ├─QUERY─> executor ──┐
                 ├─TASK──> planner ───┤
                 └UNKNOWN─────────────┴──> answer

    executor ──success──> executor（推进下一步，Loop）
        ├──plan_complete──> answer
        ├──error──> validator ──retry(未超限)──> retry ──> executor
        │                     ├──replan(未超限)─> replan ──> executor
        │                     └──超限──> fail ──> answer
        └──approval_required──> human_approval ──approved──> executor
                                               └─rejected──> answer
"""

from functools import lru_cache
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.agent import nodes
from app.agent.state import AgentState
from app.core.config import settings
from app.db.session import session_scope
from app.tools.executor import ToolExecutor

# ---------- 条件边 ----------


def route_after_router(state: AgentState) -> str:
    """Router 输出 → 分支节点。"""
    return {
        "FAQ": "rag",
        "QUERY": "executor",
        "TASK": "planner",
        "UNKNOWN": "answer",
    }.get(state.get("intent", "UNKNOWN"), "answer")


def route_after_executor(state: AgentState) -> str:
    """执行结果 → 继续 / 校验 / 审批 / 回答。"""
    return {
        "success": "executor",  # Loop：推进到下一步
        "plan_complete": "answer",
        "error": "validator",
        "approval_required": "human_approval",
    }.get(state.get("execution_status", ""), "answer")


def route_after_validator(state: AgentState) -> str:
    """校验结论 → Retry / Replan / 快速失败（上限写死，杜绝失控循环）。"""
    verdict = state.get("validation_result", {})
    kind = verdict.get("kind")
    if kind == "retry" and state.get("retry_count", 0) < settings.max_retries:
        return "retry"
    if kind == "replan" and state.get("replan_count", 0) < settings.max_replans:
        return "replan"
    return "fail"


def route_after_approval(state: AgentState) -> str:
    """人工决策 → 批准执行 / 拒绝终止。"""
    return "executor" if state.get("human_decision") == "approved" else "answer"


def fail_node(state: AgentState) -> dict[str, Any]:
    """失败收口：组装失败说明（重试/重规划已到上限）。"""
    reason = (
        state.get("validation_result", {}).get("reason")
        or state.get("failure_reason")
        or "未知错误"
    )
    return {
        "error": f"{reason}（已重试 {state.get('retry_count', 0)} 次、重规划 {state.get('replan_count', 0)} 次）",
        "execution_status": "plan_complete",
    }


# ---------- 图装配 ----------


def build_graph(
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    executor: ToolExecutor | None = None,
):
    """构建 Agent Workflow 图。executor 可注入（测试用）。"""
    executor = executor or ToolExecutor(session_factory=session_scope)

    graph = StateGraph(AgentState)

    graph.add_node("router", nodes.router_node)
    graph.add_node("planner", nodes.planner_node)
    graph.add_node("replan", nodes.replan_node)
    graph.add_node("executor", lambda state: nodes.executor_node(state, executor))
    graph.add_node("validator", nodes.validator_node)
    graph.add_node("retry", nodes.retry_node)
    graph.add_node("fail", fail_node)
    graph.add_node("human_approval", nodes.human_approval_node)
    graph.add_node("rag", nodes.rag_node)
    graph.add_node("answer", nodes.answer_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_after_router)
    graph.add_edge("planner", "executor")
    graph.add_edge("replan", "executor")
    graph.add_conditional_edges("executor", route_after_executor)
    graph.add_conditional_edges("validator", route_after_validator)
    graph.add_edge("retry", "executor")
    graph.add_edge("fail", "answer")
    graph.add_conditional_edges("human_approval", route_after_approval)
    graph.add_edge("rag", "answer")
    graph.add_edge("answer", END)

    if checkpointer is not None:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()


def build_workflow(checkpointer: BaseCheckpointSaver | None = None):
    """默认工作流（生产路径：外部传入 PostgresCheckpointer）。"""
    return build_graph(checkpointer=checkpointer)


@lru_cache
def get_workflow():
    """全局单例工作流：PostgresCheckpointer 持久化，支持 interrupt/resume。

    lru_cache 保证 API 进程内共享同一实例（checkpointer 状态一致），
    线程安全由 LangGraph 的并发控制保证。
    """
    from app.memory.checkpoint import PostgresCheckpointer

    return build_graph(checkpointer=PostgresCheckpointer())
