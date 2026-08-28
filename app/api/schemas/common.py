"""通用 API Schema（统一错误结构等）。"""

from typing import Any

from pydantic import BaseModel


class ErrorBody(BaseModel):
    """统一 API 错误结构（规格书第二十七节）。"""

    code: str
    message: str
    trace_id: str | None = None
    detail: Any = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class DependencyStatus(BaseModel):
    name: str
    status: str  # ok / error
    detail: str | None = None


class ReadyResponse(BaseModel):
    status: str  # ready / degraded
    dependencies: list[DependencyStatus]
