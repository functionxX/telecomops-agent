"""Agent 评测指标汇总。"""

from typing import Any


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """把逐任务结果聚合为整体指标。"""
    n = len(results)
    if n == 0:
        return {}

    def ratio(pred) -> float:
        return round(sum(1 for r in results if pred(r)) / n, 4)

    success_count = sum(1 for r in results if r["success"])
    steps = [r["steps"] for r in results if r.get("steps") is not None]
    latencies = [r["latency_ms"] for r in results if r.get("latency_ms") is not None]
    retry_tasks = [r for r in results if r.get("retry_count", 0) > 0]
    replan_tasks = [r for r in results if r.get("replan_count", 0) > 0]

    return {
        "dataset_size": n,
        "intent_accuracy": ratio(lambda r: r["intent_match"]),
        "tool_selection_accuracy": ratio(lambda r: r["tool_selection_match"]),
        "tool_argument_accuracy": ratio(lambda r: r["argument_check_pass"]),
        "task_success_rate": round(success_count / n, 4),
        "failure_rate": round(1 - success_count / n, 4),
        "retry_rate": round(len(retry_tasks) / n, 4),
        "replan_rate": round(len(replan_tasks) / n, 4),
        "average_steps": round(sum(steps) / len(steps), 2) if steps else None,
        "average_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
    }
