"""Planner：把复杂任务拆解为结构化执行计划。只生成计划，不执行工具。

支持 Replan：携带「已有结果 + 失败原因」重新生成，禁止重复原计划。
计划内支持 condition 步骤（tool=None）：
    arguments = {left, op, right, then_step, else_step}
    left 可为状态引用 "$step_2.result.total_remaining_gb"，
    由执行器在运行前解析——运行时控制流是确定性的，不经过 LLM。
"""

import json
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.llm.client import get_llm_client
from app.llm.schemas import ChatMessage
from app.tools.registry import registry

PLANNER_SYSTEM_PROMPT = """你是电信 CRM 业务办理的计划器。把用户任务拆解为可执行步骤，只生成计划，不执行。

工具清单（JSON）：
{tools}

可用工具：
{names}

计划格式（严格 JSON）：
{{"steps": [
  {{"step_id": "step_1", "tool": "get_current_package",
    "arguments": {{"user_id": "{user_id}"}},
    "description": "查询当前套餐", "status": "PENDING"}},
  ...
]}}

规则：
1. step_id 按 step_1、step_2 递增；tool 必须是工具清单中的名字；arguments 必须匹配该工具参数 schema。
   user_id 参数由系统自动填充，**不要填写 user_id 字段**。
2. 条件分支用 condition 步骤表达：tool 为 null，arguments 为
   {{"left": "$step_2.total_remaining_gb", "op": "<", "right": 10,
     "then_step": "step_4", "else_step": "END"}}
   引用格式是 $step_N.字段 直连（N 为已有步骤编号，字段为该步返回 dict 的键，
   例如 "$step_2.total_remaining_gb"），不要写成 $step_N.result.字段。
3. 需要补充流量时调用 recommend_package（参数 min_data_gb 为期望补充的流量，按 10 取值即可）。
4. 每个步骤 status 一律 PENDING。
5. 只输出 JSON，不要任何解释文字。
"""

REPLAN_SYSTEM_PROMPT = """你是电信 CRM 业务办理的计划器。上一次计划执行失败了，请基于失败信息重新规划。

工具清单（JSON）：
{tools}

失败信息：
- 失败工具：{failed_tool}
- 失败原因：{failure_reason}
- 已成功执行步骤及结果：{previous_results}

重新规划要求：
1. 不要重复调用已失败的工具（除非换了有实质区别的参数，如换一个套餐/换一种查询方式）。
2. 利用已有成功结果，只规划缺失的步骤；若已有结果已能完成任务，输出空 steps 列表。
3. 其余格式规则与首次规划一致，只输出 JSON。
"""


class PlanStep(BaseModel):
    step_id: str
    tool: str | None = None  # None → condition 步骤
    arguments: dict[str, Any] = Field(default_factory=dict)
    description: str = ""
    status: Literal["PENDING", "RUNNING", "SUCCESS", "FAILED", "SKIPPED"] = "PENDING"


class Plan(BaseModel):
    steps: list[PlanStep]


def plan_task(user_message: str, *, user_id: str) -> tuple[Plan, dict]:
    """首次规划。返回 (计划, 统计)。"""
    client = get_llm_client()
    tools_json = json.dumps(registry.describe_all(), ensure_ascii=False)
    start = time.perf_counter()
    stats: dict = {"model": client.model, "purpose": "planner", "phase": "plan"}
    try:
        plan = client.structured_output(
            [
                ChatMessage(
                    role="system",
                    content=PLANNER_SYSTEM_PROMPT.format(
                        tools=tools_json, names=", ".join(registry.names()), user_id=user_id
                    ),
                ),
                ChatMessage(role="user", content=user_message),
            ],
            Plan,
            purpose="planner",
        )
        stats["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        plan = _fallback_plan(user_id)
        stats["status"] = "fallback"
        stats["error"] = str(exc)[:200]
    stats["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
    stats["steps"] = len(plan.steps)
    return plan, stats


def replan(
    user_message: str,
    *,
    user_id: str,
    failed_tool: str,
    failure_reason: str,
    previous_results: dict[str, Any],
) -> tuple[Plan, dict]:
    """失败后重规划：基于失败原因生成新计划，不重复原计划。"""
    client = get_llm_client()
    tools_json = json.dumps(registry.describe_all(), ensure_ascii=False)
    start = time.perf_counter()
    stats: dict = {"model": client.model, "purpose": "planner", "phase": "replan"}
    try:
        plan = client.structured_output(
            [
                ChatMessage(
                    role="system",
                    content=REPLAN_SYSTEM_PROMPT.format(
                        tools=tools_json,
                        failed_tool=failed_tool,
                        failure_reason=failure_reason,
                        previous_results=json.dumps(previous_results, ensure_ascii=False, default=str),
                    ),
                ),
                ChatMessage(role="user", content=user_message),
            ],
            Plan,
            purpose="planner",
        )
        stats["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        plan = Plan(steps=[])
        stats["status"] = "fallback"
        stats["error"] = str(exc)[:200]
    stats["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
    stats["steps"] = len(plan.steps)
    return plan, stats


def _fallback_plan(user_id: str) -> Plan:
    """规划失败兜底：保守地只做一步安全查询。"""
    return Plan(
        steps=[
            PlanStep(
                step_id="step_1",
                tool="get_current_package",
                arguments={"user_id": user_id},
                description="兜底：查询当前套餐",
            )
        ]
    )
