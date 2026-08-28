"""统一异常模型。

API 层把 AppError 映射为统一错误 Schema（code / message / trace_id），
绝不把 Python 堆栈直接返回给客户端。

工具域异常自带错误分类（FailureKind），这是 Retry / Replan / 快速失败
分流规则的唯一事实来源——分类写死在代码里，不靠 LLM 临场判断。
"""

from enum import Enum
from typing import Any


class FailureKind(str, Enum):
    """失败分类：决定 Workflow 对失败的处理路径。"""

    TRANSIENT = "transient"  # 瞬时错误 → Retry（指数退避，per-step ≤ max_retries）
    PLAN_ERROR = "plan_error"  # 计划性错误 → Replan（≤ max_replans）
    FAST_FAIL = "fast_fail"  # 权限/业务性失败 → 不进循环，直接回答失败原因


class AppError(Exception):
    """应用异常基类。"""

    code: str = "app_error"
    http_status: int = 500

    def __init__(self, message: str, *, detail: Any = None):
        self.message = message
        self.detail = detail
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "detail": self.detail}


# ---------- 通用错误 ----------


class ValidationError(AppError):
    code = "validation_error"
    http_status = 422


class AuthenticationError(AppError):
    code = "authentication_error"
    http_status = 401


class AuthorizationError(AppError):
    code = "authorization_error"
    http_status = 403


class LLMError(AppError):
    """LLM 调用失败。retryable=True 表示瞬时错误（5xx/429/网络抖动）。"""

    code = "llm_error"
    http_status = 502

    def __init__(self, message: str, *, retryable: bool = False, detail: Any = None):
        self.retryable = retryable
        super().__init__(message, detail=detail)


class RetrievalError(AppError):
    code = "retrieval_error"
    http_status = 502


class WorkflowError(AppError):
    code = "workflow_error"
    http_status = 500


class OperationTimeoutError(AppError):
    code = "timeout_error"
    http_status = 504


# ---------- 工具域异常（统一模型 + 失败分类） ----------


class ToolError(AppError):
    """工具调用错误基类。"""

    code = "tool_error"
    http_status = 400
    kind: FailureKind = FailureKind.FAST_FAIL


class ToolNotFoundError(ToolError):
    """工具不存在 → 计划引用了错误的工具 → Replan。"""

    code = "tool_not_found"
    kind = FailureKind.PLAN_ERROR


class InvalidArgumentsError(ToolError):
    """参数不符合工具 Schema → 计划参数有误 → Replan。"""

    code = "invalid_arguments"
    kind = FailureKind.PLAN_ERROR


class PermissionDeniedError(ToolError):
    """权限拒绝 → 快速失败，不重试。"""

    code = "permission_denied"
    http_status = 403
    kind = FailureKind.FAST_FAIL


class ToolTimeoutError(ToolError):
    """工具执行超时 → 瞬时 → Retry。"""

    code = "tool_timeout"
    http_status = 504
    kind = FailureKind.TRANSIENT


class BusinessError(ToolError):
    """业务性失败（如套餐不存在）。默认快速失败；计划前提错误时由
    Validator 显式标记为 PLAN_ERROR。"""

    code = "business_error"
    kind = FailureKind.FAST_FAIL


class ExecutionError(ToolError):
    """执行环境错误（DB 连接抖动等），默认瞬时可重试。"""

    code = "execution_error"
    kind = FailureKind.TRANSIENT


def classify_failure(exc: Exception) -> FailureKind:
    """把异常映射为失败分类（Retry / Replan / 快速失败的分流依据）。"""
    if isinstance(exc, ToolError):
        return exc.kind
    if isinstance(exc, LLMError):
        return FailureKind.TRANSIENT if exc.retryable else FailureKind.FAST_FAIL
    if isinstance(exc, (OperationTimeoutError, RetrievalError)):
        return FailureKind.TRANSIENT
    return FailureKind.FAST_FAIL
