"""订单数据访问。

create_order 具备幂等性：idempotency_key 上有数据库唯一约束，
Agent 重试同一业务请求时不会产生第二张订单，而是返回已有订单。
"""

import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import Order


def _to_dict(o: Order) -> dict:
    return {
        "order_id": o.order_id,
        "user_id": o.user_id,
        "package_id": o.package_id,
        "item_name": o.item_name,
        "amount": o.amount,
        "status": o.status,
        "idempotency_key": o.idempotency_key,
        "created_at": o.created_at.isoformat(),
    }


def create_order(
    db: Session,
    *,
    user_id: str,
    item_name: str,
    amount: float,
    idempotency_key: str,
    package_id: str | None = None,
) -> tuple[dict, bool]:
    """创建订单。返回 (订单, 是否新建)。

    相同 idempotency_key 重复调用时返回已有订单（created=False），
    这是防止 Agent Retry / 网络重试导致重复下单的关键保证。
    """
    existing = db.execute(
        select(Order).where(Order.idempotency_key == idempotency_key)
    ).scalar_one_or_none()
    if existing is not None:
        return _to_dict(existing), False

    order = Order(
        order_id=f"ord_{uuid.uuid4().hex[:12]}",
        user_id=user_id,
        package_id=package_id,
        item_name=item_name,
        amount=amount,
        status="pending",
        idempotency_key=idempotency_key,
    )
    db.add(order)
    try:
        db.commit()
    except IntegrityError:
        # 并发下唯一约束兜底：另一个请求已插入同 key 的订单
        db.rollback()
        existing = db.execute(
            select(Order).where(Order.idempotency_key == idempotency_key)
        ).scalar_one()
        return _to_dict(existing), False
    return _to_dict(order), True


def query_order(db: Session, order_id: str) -> dict | None:
    """按订单号查询。"""
    order = db.execute(select(Order).where(Order.order_id == order_id)).scalar_one_or_none()
    return _to_dict(order) if order else None


def cancel_order(db: Session, order_id: str) -> dict | None:
    """取消订单（仅 pending 状态可取消）。"""
    order = db.execute(select(Order).where(Order.order_id == order_id)).scalar_one_or_none()
    if order is None:
        return None
    if order.status != "pending":
        return _to_dict(order)  # 业务上不可取消，由 Tool 层抛 BusinessError
    order.status = "cancelled"
    db.commit()
    return _to_dict(order)
