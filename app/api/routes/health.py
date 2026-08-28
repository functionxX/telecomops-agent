"""健康检查端点。

- GET /health：进程存活（liveness），不依赖任何外部服务。
- GET /ready：依赖就绪（readiness），逐个探测 PostgreSQL / Milvus，
  任一依赖不可用返回 503，供负载均衡/编排器摘流。
"""

from fastapi import APIRouter, Response, status

from app.api.schemas.common import DependencyStatus, HealthResponse, ReadyResponse
from app.core.config import settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", service=settings.app_name, version=settings.app_version)


@router.get("/ready", response_model=ReadyResponse)
async def ready(response: Response) -> ReadyResponse:
    deps: list[DependencyStatus] = []

    # PostgreSQL：SELECT 1，2 秒连接超时
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(settings.postgres_url, connect_args={"connect_timeout": 2})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        deps.append(DependencyStatus(name="postgresql", status="ok"))
        engine.dispose()
    except Exception as exc:  # noqa: BLE001
        deps.append(DependencyStatus(name="postgresql", status="error", detail=str(exc)[:200]))

    # Milvus：list_collections 探测（standalone / Lite 同一 API）
    try:
        from pymilvus import MilvusClient

        client = MilvusClient(uri=settings.milvus_uri, timeout=2)
        client.list_collections()
        deps.append(DependencyStatus(name="milvus", status="ok"))
    except Exception as exc:  # noqa: BLE001
        deps.append(DependencyStatus(name="milvus", status="error", detail=str(exc)[:200]))

    all_ok = all(d.status == "ok" for d in deps)
    if not all_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadyResponse(status="ready" if all_ok else "degraded", dependencies=deps)
