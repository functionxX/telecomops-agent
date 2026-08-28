"""初始化业务数据库：建表 + 导入种子数据（幂等）。

用法：uv run python scripts/init_db.py
"""

import json
import sys
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.db.models import (
    Base,
    CustomerProfile,
    Order,
    Package,
    Service,
    User,
    UserPackage,
)
from app.db.session import engine
from sqlalchemy import select, text

logger = get_logger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "seed"

# (模型, 种子文件名)
SEED_TABLES: list[tuple[type[Base], str]] = [
    (User, "users.json"),
    (CustomerProfile, "customer_profiles.json"),
    (Package, "packages.json"),
    (UserPackage, "user_packages.json"),
    (Service, "services.json"),
    (Order, "orders.json"),
]


def create_tables() -> None:
    Base.metadata.create_all(engine)
    logger.info("tables_created", extra={"tables": list(Base.metadata.tables)})


def seed_table(model: type[Base], filename: str) -> tuple[int, int]:
    """导入单个种子文件。已存在数据则跳过（幂等）。"""
    path = DATA_DIR / filename
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    with engine.begin() as conn:
        count = conn.execute(select(model)).first()
        if count is not None:
            return 0, len(records)  # 已初始化，跳过

    from app.db.session import SessionLocal

    with SessionLocal() as session:
        session.add_all([model(**record) for record in records])
        session.commit()
    return len(records), len(records)


def main() -> int:
    setup_logging(settings.log_level)
    create_tables()
    total_inserted = 0
    for model, filename in SEED_TABLES:
        inserted, total = seed_table(model, filename)
        status = "seeded" if inserted else "skipped (exists)"
        total_inserted += inserted
        logger.info(
            "seed_table",
            extra={"table": model.__tablename__, "status": status, "records": total},
        )
    logger.info("init_db_done", extra={"inserted": total_inserted})

    # 校验关键表行数
    with engine.connect() as conn:
        for table in ("users", "packages", "user_packages", "services", "orders"):
            n = conn.execute(text(f"SELECT count(*) FROM {table}")).scalar_one()
            logger.info("table_count", extra={"table": table, "count": n})
    return 0


if __name__ == "__main__":
    sys.exit(main())
