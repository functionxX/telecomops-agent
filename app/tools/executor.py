"""Tool Executor：独立的工具执行模块。

调用流程：
    Tool Exists? → user_id 注入（Guardrail）→ Pydantic 校验 → 权限检查 → 风险检查
    → （高风险未批准 → APPROVAL_REQUIRED）
    → 超时控制 → 执行 → 结果校验 → 截断 → Tool Result

Executor 不吞异常也不把异常抛给 Workflow：所有失败都归一为
ToolExecution（status + failure.kind），由 Workflow 按 FailureKind
决定 Retry / Replan / 快速失败。
"""

import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any

from pydantic import BaseModel, TypeAdapter
from pydantic import ValidationError as PydanticValidationError

from app.core.config import settings
from app.core.exceptions import (
    ExecutionError,
    FailureKind,
    InvalidArgumentsError,
    PermissionDeniedError,
    ToolNotFoundError,
    ToolTimeoutError,
)
from app.guardrails.tool import check_actor_scope
from app.observability import metrics
from app.observability.tracing import start_span
from app.tools.policies import get_policy
from app.tools.registry import registry

# 执行线程池：超时不会中断线程，但能把控制权交还 Workflow
_executor_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool")


class ToolExecutionStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    APPROVAL_REQUIRED = "approval_required"


class ApprovalRequest(BaseModel):
    """高风险工具的审批请求。"""

    approval_id: str
    tool_name: str
    arguments: dict[str, Any]
    summary: str


class ToolFailure(BaseModel):
    """归一化的失败信息（code/message + 失败分类）。"""

    code: str
    message: str
    kind: FailureKind


class ToolExecution(BaseModel):
    """一次工具调用的完整结果。"""

    status: ToolExecutionStatus
    tool_name: str
    result: dict[str, Any] | None = None
    failure: ToolFailure | None = None
    approval: ApprovalRequest | None = None
    duration_ms: float | None = None
    truncated: bool = False


class ToolExecutor:
    """工具执行器（Registry + 校验 + 权限 + 风险 + 超时 + 结果规范化）。"""

    def __init__(
        self,
        session_factory: Callable[..., Any],
        actor_role: str = "customer_service",
    ) -> None:
        self._session_factory = session_factory
        self._actor_role = actor_role

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        actor_user_id: str,
        approval_granted: bool = False,
        run_context: dict[str, Any] | None = None,
    ) -> ToolExecution:
        """执行一次工具调用（带 OTel span）。所有异常归一为 ToolExecution。"""
        with start_span("tool.execute", {"tool.name": tool_name}) as span:
            result = self._execute_inner(
                tool_name,
                arguments,
                actor_user_id=actor_user_id,
                approval_granted=approval_granted,
                run_context=run_context,
            )
            span.set_attribute("tool.status", result.status.value)
            if result.failure is not None:
                span.set_attribute("tool.error", result.failure.code)
            return result

    def _execute_inner(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        actor_user_id: str,
        approval_granted: bool = False,
        run_context: dict[str, Any] | None = None,
    ) -> ToolExecution:
        start = time.perf_counter()

        # 1. 工具存在性
        spec = registry.get(tool_name)
        if spec is None:
            return self._fail(tool_name, start, ToolNotFoundError(f"工具 {tool_name} 不存在"))

        # 2. Guardrail：注入/校验 actor 的 user_id（先于 Schema 校验——
        #    user_id 是会话上下文，应由执行器注入，不信任也不依赖 LLM 提供）
        try:
            arguments = check_actor_scope(tool_name, dict(arguments), actor_user_id)
        except PermissionDeniedError as exc:
            return self._fail(tool_name, start, exc)

        # 3. Pydantic 参数校验（计划参数错误 → Replan 的信号）
        try:
            validated = TypeAdapter(spec.args_schema).validate_python(arguments)
        except PydanticValidationError as exc:
            return self._fail(tool_name, start, InvalidArgumentsError(str(exc.errors())))
        validated_args: dict[str, Any] = dict(validated)

        # 4. 权限检查（角色）
        policy = get_policy(tool_name)
        if policy is not None and self._actor_role != policy.role:
            return self._fail(
                tool_name,
                start,
                PermissionDeniedError(
                    f"角色 {self._actor_role} 无权限调用 {tool_name}（要求 {policy.role}）"
                ),
            )

        # 5. 风险检查：高风险且未经人工批准 → 请求审批（不是失败）
        if (
            policy is not None
            and policy.require_confirmation
            and not approval_granted
        ):
            metrics.tool_call_count.labels(tool_name, "approval_required").inc()
            return ToolExecution(
                status=ToolExecutionStatus.APPROVAL_REQUIRED,
                tool_name=tool_name,
                approval=ApprovalRequest(
                    approval_id=f"apv_{uuid.uuid4().hex[:12]}",
                    tool_name=tool_name,
                    arguments=validated_args,
                    summary=self._approval_summary(tool_name, validated_args),
                ),
            )

        # create_order 幂等键兜底：从业务语义派生 (会话, 套餐) ——
        # 与计划步骤编号无关（LLM 每次生成的 step_id 不稳定），
        # 同一会话内重复办理同一套餐 → 返回已有订单，杜绝重复下单。
        if tool_name == "create_order" and not validated_args.get("idempotency_key"):
            run_context = run_context or {}
            cid = run_context.get("conversation_id", "unknown")
            pid = validated_args.get("package_id", "unknown")
            validated_args["idempotency_key"] = f"ik_{cid}_{pid}"

        # 6. 超时控制 + 执行
        try:
            future = _executor_pool.submit(self._run, spec.handler, validated_args)
            result = future.result(timeout=settings.tool_timeout)
        except TimeoutError:
            return self._fail(tool_name, start, ToolTimeoutError(f"工具 {tool_name} 执行超时（>{settings.tool_timeout}s）"))
        except Exception as exc:  # 工具内部异常（含 BusinessError/ExecutionError）
            return self._fail(tool_name, start, exc)

        # 7. 结果校验
        if not isinstance(result, dict):
            return self._fail(
                tool_name, start, ExecutionError(f"工具 {tool_name} 返回了非 dict 结果")
            )

        # 8. 截断（保护 LLM 上下文）
        result, truncated = truncate_result(
            result, max_rows=settings.max_rows, max_cell_length=settings.max_cell_length
        )

        duration = (time.perf_counter() - start) * 1000
        metrics.tool_call_count.labels(tool_name, "success").inc()
        return ToolExecution(
            status=ToolExecutionStatus.SUCCESS,
            tool_name=tool_name,
            result=result,
            duration_ms=round(duration, 1),
            truncated=truncated,
        )

    # ---------- 内部 ----------

    def _run(self, handler: Callable[..., Any], arguments: dict[str, Any]) -> Any:
        with self._session_factory() as db:
            return handler(db, **arguments)

    def _fail(self, tool_name: str, start: float, exc: Exception) -> ToolExecution:
        from app.core.exceptions import AppError, classify_failure

        if isinstance(exc, AppError):
            code, message, kind = exc.code, exc.message, classify_failure(exc)
        else:  # 未预期异常：归一为 ExecutionError（瞬时可重试）
            code, message, kind = "execution_error", f"{type(exc).__name__}: {exc}", FailureKind.TRANSIENT
        duration = (time.perf_counter() - start) * 1000
        metrics.tool_call_count.labels(tool_name, "error").inc()
        metrics.tool_error_count.labels(tool_name, code).inc()
        return ToolExecution(
            status=ToolExecutionStatus.ERROR,
            tool_name=tool_name,
            failure=ToolFailure(code=code, message=message, kind=kind),
            duration_ms=round(duration, 1),
        )

    def _approval_summary(self, tool_name: str, arguments: dict[str, Any]) -> str:
        if tool_name == "create_order":
            return f"办理套餐 {arguments.get('package_id')}（用户 {arguments.get('user_id')}）"
        if tool_name == "cancel_order":
            return f"取消订单 {arguments.get('order_id')}"
        if tool_name == "enable_roaming":
            return f"为 {arguments.get('user_id')} 开通国际漫游"
        if tool_name == "disable_roaming":
            return f"为 {arguments.get('user_id')} 关闭国际漫游"
        return f"{tool_name}({arguments})"


def truncate_result(
    result: dict[str, Any], *, max_rows: int, max_cell_length: int
) -> tuple[dict[str, Any], bool]:
    """递归截断结果：行数 / 单元格长度受限，返回是否发生截断。"""

    truncated = False

    def _cell(value: Any) -> Any:
        nonlocal truncated
        if isinstance(value, str) and len(value) > max_cell_length:
            truncated = True
            return value[:max_cell_length] + f"...(截断，原长 {len(value)})"
        return value

    def _rows(items: list[Any]) -> list[Any]:
        nonlocal truncated
        if len(items) > max_rows:
            truncated = True
            items = items[:max_rows]
        return items

    def _walk(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_walk(item) for item in _rows(value)]
        return _cell(value)

    return _walk(result), truncated
