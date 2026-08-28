"""FastAPI 应用入口。

只负责：配置加载、日志、中间件、路由挂载、统一异常处理。
业务逻辑一律在各模块中，禁止堆在 main.py。
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from app.api.routes import approvals, chat, health
from app.api.schemas.common import ErrorBody, ErrorResponse
from app.core.config import settings
from app.core.exceptions import AppError
from app.core.logging import get_logger, setup_logging
from app.observability import metrics
from app.observability.tracing import generate_trace_id, get_trace_id, set_trace_id

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.log_level)
    from app.observability.tracing import init_otel, instrument_fastapi

    init_otel()
    instrument_fastapi(app)
    logger.info("service_started", extra={"env": settings.app_env})
    yield
    logger.info("service_stopped")


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)


@app.middleware("http")
async def trace_and_metrics(request: Request, call_next):
    """为每个请求注入 trace_id 并记录 request 指标。"""
    set_trace_id(generate_trace_id())
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    route = request.url.path
    metrics.request_count.labels(request.method, route, response.status_code).inc()
    metrics.request_latency.labels(request.method, route).observe(elapsed)
    response.headers["X-Trace-Id"] = get_trace_id() or ""
    return response


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """统一错误 Schema：code / message / trace_id，绝不泄漏堆栈。"""
    body = ErrorResponse(
        error=ErrorBody(code=exc.code, message=exc.message, trace_id=get_trace_id(), detail=exc.detail)
    )
    return JSONResponse(status_code=exc.http_status, content=body.model_dump())


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorBody(
            code="validation_error",
            message="请求参数校验失败",
            trace_id=get_trace_id(),
            detail=exc.errors(),
        )
    )
    return JSONResponse(status_code=422, content=body.model_dump())


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_error", extra={"path": request.url.path})
    body = ErrorResponse(
        error=ErrorBody(code="internal_error", message="服务器内部错误", trace_id=get_trace_id())
    )
    return JSONResponse(status_code=500, content=body.model_dump())


# ---------- 路由 ----------
app.include_router(health.router)
app.include_router(approvals.router)
app.include_router(chat.router)

# Prometheus /metrics（spec 第二十六节要求）
app.mount("/metrics", make_asgi_app())
