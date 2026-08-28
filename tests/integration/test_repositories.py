"""集成测试：真实 PostgreSQL 上的 repositories（幂等/upsert/查询口径）。"""

import pytest
from app.db.models import Order
from app.db.repositories import customer_repo, order_repo, package_repo, service_repo
from app.db.session import SessionLocal, session_scope
from sqlalchemy import func, select

pytestmark = pytest.mark.integration


@pytest.fixture()
def db():
    with session_scope() as session:
        yield session


def test_requires_postgres(require_postgres):
    pass


def test_customer_profile_seeded(db):
    profile = customer_repo.get_customer_profile(db, "user_001")
    assert profile is not None
    assert profile["customer_level"] == "金卡"
    assert profile["customer_name"] == "张伟"


def test_current_package_and_remaining(db):
    pkg = package_repo.get_current_package(db, "user_001")
    assert pkg is not None
    assert pkg["package_id"] == "pk_5g_129"
    assert pkg["remaining_data_gb"] == 8.0  # 30 - 22

    remaining = package_repo.get_remaining_data(db, "user_001")
    assert remaining["total_remaining_gb"] == 8.0


def test_create_order_idempotency(db):
    """相同 idempotency_key 重复下单必须返回同一订单，不产生第二行。"""
    key = "test_idem_key_001"
    order1, created1 = order_repo.create_order(
        db, user_id="user_001", item_name="测试套餐", amount=10.0, idempotency_key=key,
        package_id="addon_5g",
    )
    assert created1 is True

    order2, created2 = order_repo.create_order(
        db, user_id="user_001", item_name="测试套餐", amount=10.0, idempotency_key=key,
        package_id="addon_5g",
    )
    assert created2 is False
    assert order2["order_id"] == order1["order_id"]

    with SessionLocal() as s:
        count = s.execute(
            select(func.count()).select_from(Order).where(Order.idempotency_key == key)
        ).scalar_one()
    assert count == 1

    # 清理
    with SessionLocal() as s:
        row = s.execute(select(Order).where(Order.idempotency_key == key)).scalar_one()
        s.delete(row)
        s.commit()


def test_service_upsert(db):
    before = service_repo.query_service_status(db, "user_008", "roaming")
    assert before is None or before["status"] == "disabled"

    enabled = service_repo.set_service_status(db, "user_008", "roaming", enabled=True)
    assert enabled["status"] == "enabled"

    again = service_repo.query_service_status(db, "user_008", "roaming")
    assert again["status"] == "enabled"

    # 还原
    service_repo.set_service_status(db, "user_008", "roaming", enabled=False)


def test_order_cancel_business_rule(db):
    """已完成的订单不可取消：返回原状态（由 Tool 层抛 BusinessError）。"""
    result = order_repo.cancel_order(db, "ord_20260601_001")
    assert result is not None
    assert result["status"] == "completed"  # 未变更
