"""客户信息数据访问。"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import CustomerProfile, User


def get_customer_profile(db: Session, user_id: str) -> dict | None:
    """获取客户档案（用户 + 客户画像）。"""
    stmt = (
        select(
            User.user_id,
            User.username,
            User.phone,
            User.status,
            User.created_at,
            CustomerProfile.customer_name,
            CustomerProfile.id_card,
            CustomerProfile.customer_level,
            CustomerProfile.credit_score,
            CustomerProfile.joined_date,
        )
        .join(CustomerProfile, CustomerProfile.user_id == User.user_id)
        .where(User.user_id == user_id)
    )
    row = db.execute(stmt).first()
    if row is None:
        return None
    return {
        "user_id": row.user_id,
        "username": row.username,
        "phone": row.phone,
        "status": row.status,
        "customer_name": row.customer_name,
        "id_card": row.id_card,
        "customer_level": row.customer_level,
        "credit_score": row.credit_score,
        "joined_date": row.joined_date.isoformat() if row.joined_date else None,
    }


def get_customer_level(db: Session, user_id: str) -> str | None:
    """获取客户等级（钻石/金卡/银卡/普通）。"""
    stmt = select(CustomerProfile.customer_level).where(CustomerProfile.user_id == user_id)
    return db.execute(stmt).scalar_one_or_none()
