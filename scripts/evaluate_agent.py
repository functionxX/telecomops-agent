"""Agent 评测：意图 / 工具选择 / 参数 / 成功率 / Retry / Replan / 审批。

默认 Mock LLM（确定性、零成本、可复现）；--real-llm 切换真实 DeepSeek。
真实运行工作流（含真实 Retry/Replan 循环与 interrupt/resume），不伪造数字：
    uv run python scripts/evaluate_agent.py [--limit N] [--real-llm]
结果写入 evaluation/results/agent_results.json。
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# --real-llm 必须在导入 app 前生效（settings 在导入时读取）
if "--real-llm" in sys.argv:
    os.environ["MOCK_LLM"] = "false"

from app.agent.graph import build_graph  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.core.exceptions import FailureKind  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.db.session import session_scope  # noqa: E402
from app.guardrails.input import check_input  # noqa: E402
from app.guardrails.tool import USER_SCOPED_TOOLS  # noqa: E402
from app.tools.executor import (  # noqa: E402
    ToolExecution,
    ToolExecutionStatus,
    ToolExecutor,
    ToolFailure,
)
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

setup_logging(settings.log_level)

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "evaluation" / "datasets" / "agent_tasks.json"
RESULTS = ROOT / "evaluation" / "results" / "agent_results.json"

sys.path.insert(0, str(ROOT))
from evaluation.agent_eval.metrics import aggregate  # noqa: E402


class FlakyExecutor(ToolExecutor):
    """对指定工具注入一次失败：真实走 Retry / Replan 循环（其余工具正常）。"""

    def __init__(self, *args, fail_kind: str = "transient", fail_tool: str | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._fail_kind = fail_kind
        self._fail_tool = fail_tool
        self._failed = False

    def execute(self, tool_name: str, arguments: dict, **kwargs) -> ToolExecution:
        if self._fail_tool == tool_name and not self._failed:
            self._failed = True
            if self._fail_kind == "transient":
                failure = ToolFailure(
                    code="execution_error", message="评测注入：瞬时故障（第 1 次调用）",
                    kind=FailureKind.TRANSIENT,
                )
            else:
                failure = ToolFailure(
                    code="invalid_arguments", message="评测注入：计划性故障（第 1 次调用）",
                    kind=FailureKind.PLAN_ERROR,
                )
            return ToolExecution(
                status=ToolExecutionStatus.ERROR, tool_name=tool_name, failure=failure
            )
        return super().execute(tool_name, arguments, **kwargs)


def run_task(task: dict, index: int) -> dict:
    """运行单个任务（含 Retry/Replan/审批），返回逐任务指标。"""
    thread_id = f"eval_agent_{index}_{task['task_id']}"

    # guardrail 任务：只验证 Input Guardrail 拦截，不进工作流
    if task["mode"] == "guardrail":
        guard = check_input(task["user_message"])
        return {
            "task_id": task["task_id"],
            "intent_match": True,
            "tool_selection_match": True,
            "argument_check_pass": True,
            "success": not guard.ok,
            "steps": 0,
            "retry_count": 0,
            "replan_count": 0,
            "approval_occurred": False,
            "latency_ms": 0.0,
            "error": "" if not guard.ok else guard.reason,
        }

    executor: ToolExecutor = ToolExecutor(session_factory=session_scope)
    if task["mode"] == "flaky_retry":
        executor = FlakyExecutor(
            session_factory=session_scope,
            fail_kind="transient",
            fail_tool=task["expected_tools"][0] if task["expected_tools"] else None,
        )
    elif task["mode"] == "flaky_replan":
        executor = FlakyExecutor(
            session_factory=session_scope,
            fail_kind="plan_error",
            fail_tool=task["expected_tools"][0] if task["expected_tools"] else None,
        )

    graph = build_graph(checkpointer=MemorySaver(), executor=executor)
    config = {"configurable": {"thread_id": thread_id}}
    start = time.perf_counter()

    state = graph.invoke(
        {
            "query": task["user_message"],
            "user_id": task["user_id"],
            "conversation_id": thread_id,
            "trace_id": f"eval_{task['task_id']}",
        },
        config=config,
    )
    approval_occurred = False
    if state.get("__interrupt__"):
        approval_occurred = True
        interrupt = state["__interrupt__"][0]
        state = graph.invoke(
            Command(resume={"approval_id": interrupt.value["approval_id"], "decision": "approved"}),
            config=config,
        )
    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    tool_calls = state.get("tool_calls", [])
    executed_tools = [tc["tool_name"] for tc in tool_calls if tc.get("status") == "success"]
    intent_match = state.get("intent") == task["expected_intent"]
    tool_selection_match = all(t in executed_tools for t in task["expected_tools"])
    if not task["expected_tools"] and state.get("intent") in ("UNKNOWN",):
        tool_selection_match = True

    # 参数校验：只检查「包含该 key 的记录」（无关工具的参数里没有该 key 属正常）
    argument_ok = True
    for key, expected_value in (task.get("arg_checks") or {}).items():
        for tc in tool_calls:
            args = tc.get("arguments", {})
            if key in args and args[key] != expected_value:
                argument_ok = False
    for tc in tool_calls:
        if tc["tool_name"] in USER_SCOPED_TOOLS:
            # user_id 缺失 = 由执行器注入（安全）；显式给出但不同于任务用户 = 越权
            provided = tc.get("arguments", {}).get("user_id")
            if provided not in (None, task["user_id"]):
                argument_ok = False

    no_error = not state.get("error")
    approval_ok = (not task["expect_approval"]) or approval_occurred
    if task["expect_success"]:
        success = no_error and intent_match and tool_selection_match and argument_ok and approval_ok
    else:
        # UNKNOWN/越权防护类：意图正确 + 无跨用户数据访问（工具被拦截或按
        # actor 注入后执行都算正确处理——两种都是安全结果）
        success = intent_match and argument_ok

    return {
        "task_id": task["task_id"],
        "intent_match": intent_match,
        "tool_selection_match": tool_selection_match,
        "argument_check_pass": argument_ok,
        "success": success,
        "steps": len(tool_calls),
        "retry_count": state.get("retry_count", 0),
        "replan_count": state.get("replan_count", 0),
        "approval_occurred": approval_occurred,
        "latency_ms": latency_ms,
        "error": state.get("error", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 个任务（调试用）")
    parser.add_argument("--real-llm", action="store_true", help="使用真实 DeepSeek（默认 Mock）")
    args = parser.parse_args()

    with open(DATASET, encoding="utf-8") as f:
        tasks = json.load(f)
    if args.limit:
        tasks = tasks[: args.limit]

    print(f"Agent 评测：{len(tasks)} 个任务，LLM 模式={'真实 DeepSeek' if settings.deepseek_api_key and not settings.mock_llm else 'Mock（确定性规则）'}")
    results = []
    for i, task in enumerate(tasks):
        r = run_task(task, i)
        results.append(r)
        mark = "✅" if r["success"] else "❌"
        print(f"  [{mark}] {task['task_id']} intent={task['expected_intent']} "
              f"tools={','.join(task['expected_tools']) or '-'} "
              f"retry={r['retry_count']} replan={r['replan_count']} approval={r['approval_occurred']} "
              f"{r['latency_ms']}ms")

    report = {
        "dataset": str(DATASET),
        "llm_mode": "mock" if settings.mock_llm else "real",
        "metrics": aggregate(results),
        "tasks": results,
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n汇总指标: {json.dumps(report['metrics'], ensure_ascii=False, indent=2)}")
    print(f"结果已写入 {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
