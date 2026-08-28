"""OpenTelemetry：trace_id 贯穿 + OTLP 导出 + 埋点辅助。

分工（见 monitoring/otel-collector.yaml 注释）：
- 业务自定义指标 → prometheus_client 直接暴露 /metrics
- OTel 自动埋点指标（HTTP server 等）→ OTLP → collector → Prometheus
- Traces → OTLP → collector（第一版 exporter 为 debug/console）

安全：span 只记录 latency/status/model/tokens/tool name 等，
绝不记录 API Key、密码、完整敏感用户信息。
"""

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from app.core.config import settings

_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)

tracer = trace.get_tracer(settings.otel_service_name)


def generate_trace_id() -> str:
    return uuid.uuid4().hex


def set_trace_id(trace_id: str) -> None:
    _trace_id.set(trace_id)


def get_trace_id() -> str | None:
    return _trace_id.get()


def init_otel() -> None:
    """初始化 OTel SDK（进程启动时调用一次；OTEL_ENABLED=false 则跳过）。"""
    if not settings.otel_enabled:
        return
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(
        resource=Resource.create({"service.name": settings.otel_service_name})
    )
    exporter = OTLPSpanExporter(
        endpoint=f"{settings.otel_exporter_otlp_endpoint.rstrip('/')}/v1/traces"
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)


def instrument_fastapi(app: Any) -> None:
    """FastAPI 自动埋点（HTTP server span）。"""
    if not settings.otel_enabled:
        return
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app)


@contextmanager
def start_span(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Iterator[Span]:
    """业务埋点上下文：统一记录 trace_id 关联 + 状态 + 异常。

    用法：
        with start_span("llm.chat", {"model": ..., "purpose": ...}) as span:
            ...
            span.set_attribute("tokens", n)
    """
    attrs = {"trace_id": get_trace_id() or ""}
    if attributes:
        attrs.update(attributes)
    with tracer.start_as_current_span(name, attributes=attrs) as span:
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)[:200]))
            raise
