"""订单类工具：创建/查询/取消。

create_order 为高风险资金操作：需要人工确认 + 数据库幂等约束
（idempotency_key 唯一），Agent 重试不会产生重复订单。
"""

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.exceptions import BusinessError
from app.db.repositories import order_repo
from app.tools.registry import register


class CreateOrderArgs(BaseModel):
    user_id: str = Field(description="用户ID")
    package_id: str = Field(description="要办理的套餐ID，如 addon_30g")
    # 可选：由 ToolExecutor 按 (会话, 步骤) 派生兜底，LLM 无法伪造跨请求的稳定键
    idempotency_key: str | None = Field(
        default=None,
        description="幂等键：同一业务请求重试时保持不变；留空由系统自动生成",
    )


@register(
    "create_order",
    description="为用户办理套餐下单（高风险资金操作，需人工确认；幂等：相同 idempotency_key 不会重复下单）。",
    args_schema=CreateOrderArgs,
)
def create_order(db: Session, user_id: str, package_id: str, idempotency_key: str) -> dict:
    from sqlalchemy import select

    from app.db.models import Package

    pkg = db.execute(select(Package).where(Package.package_id == package_id)).scalar_one_or_none()
    if pkg is None:
        raise BusinessError(f"套餐 {package_id} 不存在或已下架")
    order, created = order_repo.create_order(
        db,
        user_id=user_id,
        package_id=package_id,
        item_name=pkg.name,
        amount=pkg.monthly_fee,
        idempotency_key=idempotency_key,
    )
    return {**order, "created": created, "note": "已存在相同请求的订单" if not created else None}


class QueryOrderArgs(BaseModel):
    order_id: str = Field(description="订单号，如 ord_xxx")


@register(
    "query_order",
    description="按订单号查询订单详情。只读。",
    args_schema=QueryOrderArgs,
)
def query_order(db: Session, order_id: str) -> dict:
    order = order_repo.query_order(db, order_id)
    if order is None:
        raise BusinessError(f"订单 {order_id} 不存在")
    return order


class CancelOrderArgs(BaseModel):
    order_id: str = Field(description="订单号，如 ord_xxx")


@register(
    "cancel_order",
    description="取消订单（仅未支付订单可取消；高风险变更操作，需人工确认）。",
    args_schema=CancelOrderArgs,
)
def cancel_order(db: Session, order_id: str) -> dict:
    order = order_repo.cancel_order(db, order_id)
    if order is None:
        raise BusinessError(f"订单 {order_id} 不存在")
    if order["status"] != "cancelled":
        raise BusinessError(f"订单 {order_id} 当前状态为 {order['status']}，不可取消")
    return {**order, "message": "订单已取消"}
