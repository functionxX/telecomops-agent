"""Agent 侧执行引擎：计划步骤的确定性执行。

- 状态引用解析：参数里的 "$step_2.xxx" 在执行前被解析为前序结果值
  （tool_results[step_id] 即该步结果 dict）
- condition 步骤：确定性比较（left op right），然后/否则跳转到指定步骤，
  其余分支步骤标记 SKIPPED —— 运行时控制流不经过 LLM
- 工具步骤：交给 ToolExecutor（校验/权限/风险/超时/截断），
  结果按 step_id 归档到 tool_results
"""

from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.tools.executor import ToolExecution, ToolExecutionStatus, ToolExecutor

logger = get_logger(__name__)

_OPS = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
}

REF_PREFIX = "$"


def resolve_references(value: Any, tool_results: dict[str, dict[str, Any]]) -> Any:
    """把状态引用（"$step_N.字段"）解析为实际值；非引用原样返回。

    容错：LLM 偶尔写成 "$step_N.result.字段"，若直连解析失败，
    尝试跳过 "result" 段再解析。
    """
    if isinstance(value, str) and value.startswith(REF_PREFIX):
        path = value[1:].split(".")
        node: Any = tool_results
        failed_at = None
        for i, part in enumerate(path):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                failed_at = i
                break
        if failed_at is not None and path[failed_at] == "result":
            # 容错路径：跳过 "result" 段
            node = tool_results
            for part in path[:failed_at] + path[failed_at + 1 :]:
                if isinstance(node, dict) and part in node:
                    node = node[part]
                else:
                    raise ValueError(f"无法解析状态引用: {value}")
        elif failed_at is not None:
            raise ValueError(f"无法解析状态引用: {value}")
        return node
    if isinstance(value, dict):
        return {k: resolve_references(v, tool_results) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_references(v, tool_results) for v in value]
    return value


def evaluate_condition(cond_args: dict[str, Any], tool_results: dict[str, dict[str, Any]]) -> str:
    """计算条件步骤。返回 "then" 或 "else"（不满足/非法一律 else，安全方向）。"""
    try:
        left = resolve_references(cond_args["left"], tool_results)
        right = resolve_references(cond_args.get("right"), tool_results)
        op = _OPS.get(cond_args.get("op", "=="))
        if op is None:
            return "else"
        return "then" if op(left, right) else "else"
    except (KeyError, ValueError, TypeError):
        return "else"


def find_step(plan: list[dict[str, Any]], step_id: str) -> dict[str, Any] | None:
    for step in plan:
        if step["step_id"] == step_id:
            return step
    return None


def _step_index(plan: list[dict[str, Any]], step_id: str) -> int:
    for i, step in enumerate(plan):
        if step["step_id"] == step_id:
            return i
    return -1


def apply_condition_jump(
    plan: list[dict[str, Any]],
    current_step_id: str,
    cond_args: dict[str, Any],
    tool_results: dict[str, dict[str, Any]],
) -> str:
    """执行条件跳转：标记分支步骤 SKIPPED，返回下一个 step_id（或 END）。

    约定：then_step 在计划中位于 else_step 之前；
    选中 then → else_step 及之后全部 SKIPPED；
    选中 else → then_step 到 else_step 之间的步骤 SKIPPED。
    """
    chosen = evaluate_condition(cond_args, tool_results)
    cur_idx = _step_index(plan, current_step_id)
    then_step = cond_args.get("then_step")
    else_step = cond_args.get("else_step")

    def mark_skipped(from_idx: int | None, to_idx: int | None) -> None:
        if from_idx is None:
            return
        end = to_idx if to_idx is not None else len(plan)
        for step in plan[from_idx:end]:
            if step["status"] != "SUCCESS":
                step["status"] = "SKIPPED"

    if chosen == "then":
        # 选中 then 分支：else_step 及其后的步骤全部 SKIPPED
        if else_step and else_step != "END":
            mark_skipped(_step_index(plan, else_step), None)
        return then_step or "END"
    # 选中 else 分支：cond 之后到 else_step 之间的步骤 SKIPPED
    else_idx = _step_index(plan, else_step) if else_step and else_step != "END" else len(plan)
    mark_skipped(cur_idx + 1, else_idx)
    return else_step or "END"


def run_step(
    state: dict[str, Any],
    executor: ToolExecutor,
    approval_granted: bool = False,
) -> dict[str, Any]:
    """执行 current_step 一步。返回状态更新片段（不修改原 state）。"""
    plan: list[dict[str, Any]] = state["plan"]
    current_id: str = state.get("current_step", "")
    updates: dict[str, Any] = {}
    current = find_step(plan, current_id)
    if current is None:
        updates["execution_status"] = "plan_complete"
        return updates

    # ---------- condition 步骤：确定性跳转 ----------
    if current.get("tool") is None:
        current["status"] = "SUCCESS"
        next_id = apply_condition_jump(plan, current_id, current.get("arguments", {}), state.get("tool_results", {}))
        updates["plan"] = plan
        updates["current_step"] = next_id
        updates["execution_status"] = "plan_complete" if next_id == "END" else "success"
        return updates

    # ---------- 工具步骤 ----------
    current["status"] = "RUNNING"
    updates["plan"] = plan

    try:
        args = resolve_references(current.get("arguments", {}), state.get("tool_results", {}))
    except ValueError as exc:
        current["status"] = "FAILED"
        updates["execution_status"] = "error"
        updates["failed_tool"] = current["tool"]
        updates["failure_reason"] = str(exc)
        updates["validation_result"] = {"status": "pending"}
        updates["tool_calls"] = [
            {
                "step_id": current_id,
                "tool_name": current["tool"],
                "arguments": current.get("arguments", {}),
                "status": "error",
                "error": str(exc),
            }
        ]
        return updates

    try:
        from langgraph.config import get_stream_writer

        get_stream_writer()(
            {"type": "tool_started", "step_id": current_id, "tool_name": current["tool"], "arguments": args}
        )
    except Exception:  # noqa: BLE001
        pass

    result: ToolExecution = executor.execute(
        current["tool"],
        args,
        actor_user_id=state["user_id"],
        approval_granted=approval_granted,
        run_context={"conversation_id": state.get("conversation_id", ""), "step_id": current_id},
    )

    try:
        from langgraph.config import get_stream_writer

        event: dict[str, Any] = {
            "type": "approval_required" if result.status == ToolExecutionStatus.APPROVAL_REQUIRED else "tool_finished",
            "step_id": current_id,
            "tool_name": current["tool"],
            "status": result.status.value,
        }
        if result.status == ToolExecutionStatus.APPROVAL_REQUIRED and result.approval:
            event.update(result.approval.model_dump())
        get_stream_writer()(event)
    except Exception:  # noqa: BLE001
        pass

    record: dict[str, Any] = {
        "step_id": current_id,
        "tool_name": current["tool"],
        "arguments": args,
        "status": result.status.value,
    }

    if result.status == ToolExecutionStatus.APPROVAL_REQUIRED:
        updates["requires_human_approval"] = True
        approval_payload = result.approval.model_dump() if result.approval else {}
        updates["pending_approval"] = {**approval_payload, "step_id": current_id}
        updates["execution_status"] = "approval_required"
        record["approval_id"] = result.approval.approval_id if result.approval else None
        updates["tool_calls"] = [record]
        return updates

    if result.status == ToolExecutionStatus.ERROR:
        current["status"] = "FAILED"
        updates["plan"] = plan
        updates["execution_status"] = "error"
        updates["failed_tool"] = current["tool"]
        updates["failure_reason"] = result.failure.message if result.failure else "未知错误"
        updates["validation_result"] = {"status": "pending"}
        record.update(
            {
                "error": result.failure.message if result.failure else "",
                "failure_kind": result.failure.kind.value if result.failure else "",
            }
        )
        updates["tool_calls"] = [record]
        return updates

    # SUCCESS
    current["status"] = "SUCCESS"
    updates["plan"] = plan
    updates["tool_results"] = {**state.get("tool_results", {}), current_id: result.result}
    updates["total_tool_executions"] = state.get("total_tool_executions", 0) + 1
    if approval_granted:
        # 一次审批只放行一个动作：执行成功后立即作废，后续高风险步骤需重新审批
        updates["human_decision"] = ""
        updates["requires_human_approval"] = False
        updates["pending_approval"] = {}
    record.update({"result": result.result, "duration_ms": result.duration_ms, "truncated": result.truncated})
    updates["tool_calls"] = [record]
    updates["execution_status"] = "success"

    # 推进到下一步
    cur_idx = _step_index(plan, current_id)
    if cur_idx + 1 < len(plan):
        updates["current_step"] = plan[cur_idx + 1]["step_id"]
    else:
        updates["current_step"] = "END"
        updates["execution_status"] = "plan_complete"
    return updates


def check_global_cap(state: dict[str, Any]) -> bool:
    """全局工具执行次数硬兜底：超过上限强制终止计划。"""
    return state.get("total_tool_executions", 0) >= settings.max_tool_executions
