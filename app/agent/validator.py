"""Validator：判定「这一步的结果是否可用」。

两层校验（共识 Q5）：
1. 确定性结构校验（永远执行）：失败分类（FailureKind）来自执行器异常模型；
2. LLM 语义校验（可选，VALIDATOR_USE_LLM，仅 TASK 步骤）：
   「结果是否满足步骤目标」——抓“工具成功了但结果不对”的场景。

输出 validation_result = {status, kind, reason}，kind ∈
  retry | replan | fail | satisfied
这是条件边的唯一依据，全部写死，不用 LLM 临场判断控制流。
"""

import json
from typing import Any, Literal

from pydantic import BaseModel

from app.core.config import settings
from app.core.exceptions import FailureKind
from app.llm.client import get_llm_client
from app.llm.schemas import ChatMessage
from app.tools.executor import ToolExecutionStatus

VALIDATOR_SYSTEM_PROMPT = """你是执行结果校验器。判断工具执行结果是否真正满足步骤目标。

步骤目标：{step_description}
工具名：{tool_name}
工具结果（JSON）：{tool_result}

判断标准：结果内容与步骤目标一致（字段齐全、数值合理、能回答目标问题）。
输出 JSON：{{"satisfied": true/false, "reason": "一句话理由"}}。
"""


class ValidatorVerdict(BaseModel):
    satisfied: bool
    reason: str = ""


def validate_step(
    *,
    execution_status: str,
    failure_kind: FailureKind | None,
    failure_message: str | None,
    step_description: str,
    tool_name: str,
    tool_result: dict[str, Any] | None,
    use_semantic_check: bool,
    purpose_hint: Literal["query", "task"] = "task",
) -> dict[str, Any]:
    """判定一步执行结果。返回 validation_result dict。"""

    # 1. 确定性：执行失败 → 按 FailureKind 分流（retry / replan / fail）
    if execution_status != ToolExecutionStatus.SUCCESS.value:
        kind = {
            FailureKind.TRANSIENT: "retry",
            FailureKind.PLAN_ERROR: "replan",
            FailureKind.FAST_FAIL: "fail",
        }.get(failure_kind or FailureKind.FAST_FAIL, "fail")
        return {
            "status": "failed",
            "kind": kind,
            "reason": failure_message or "工具执行失败",
        }

    # 2. 确定性：结果形态
    if not tool_result or not isinstance(tool_result, dict):
        return {
            "status": "failed",
            "kind": "replan",
            "reason": "工具返回空结果或非 dict，计划可能有误",
        }

    # 3. 可选：LLM 语义校验（仅 TASK；QUERY 单步省一次调用）
    if use_semantic_check and purpose_hint == "task":
        client = get_llm_client()
        try:
            verdict = client.structured_output(
                [
                    ChatMessage(
                        role="system",
                        content=VALIDATOR_SYSTEM_PROMPT.format(
                            step_description=step_description,
                            tool_name=tool_name,
                            tool_result=json.dumps(tool_result, ensure_ascii=False, default=str),
                        ),
                    )
                ],
                ValidatorVerdict,
                purpose="validator",
            )
            if not verdict.satisfied:
                return {
                    "status": "failed",
                    "kind": "replan",
                    "reason": f"语义校验不通过: {verdict.reason}",
                }
        except Exception as exc:  # noqa: BLE001 — 校验失败降级为结构校验通过
            return {
                "status": "satisfied",
                "kind": "satisfied",
                "reason": f"语义校验降级（{exc}），按结构校验通过",
            }

    return {"status": "satisfied", "kind": "satisfied", "reason": "结构校验通过"}


def semantic_check_enabled() -> bool:
    """语义校验开关（配置化；Mock 模式默认关闭走确定性路径）。"""
    return settings.validator_use_llm and not settings.mock_llm
