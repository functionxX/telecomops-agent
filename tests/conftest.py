"""pytest 全局配置。

- 默认 MOCK_LLM=true（LLM 相关测试确定性、零成本）；
  真实 LLM 测试用 --real-llm 标记单独跑（见 pyproject markers）。
- 集成测试需要 compose 的 PostgreSQL/Milvus，不可达时自动 skip。
"""

import os

# 必须在导入任何 app 模块之前生效
os.environ.setdefault("MOCK_LLM", "true")

import pytest  # noqa: E402


def _pg_available() -> bool:
    try:
        from app.core.config import settings
        from sqlalchemy import create_engine, text

        engine = create_engine(settings.postgres_url, connect_args={"connect_timeout": 2})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception:  # noqa: BLE001
        return False


def _milvus_available() -> bool:
    try:
        from app.core.config import settings
        from pymilvus import MilvusClient

        client = MilvusClient(uri=settings.milvus_uri, timeout=2)
        client.list_collections()
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(scope="session")
def require_postgres():
    if not _pg_available():
        pytest.skip("PostgreSQL 不可达（请先 docker compose up -d postgres）")
    return True


@pytest.fixture(scope="session")
def require_milvus():
    if not _milvus_available():
        pytest.skip("Milvus 不可达（请先 docker compose up -d milvus）")
    return True


@pytest.fixture(scope="session")
def require_services(require_postgres, require_milvus):
    return True
