"""套餐数据访问。

注意：剩余流量的计算口径 = 主套餐 data_gb - data_used_gb。
"""
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from app.db.models import Package, UserPackage


def get_current_package(db: Session, user_id: str) -> dict | None:
    """用户当前生效的主套餐。"""
    stmt = (
        select(Package, UserPackage)
        .join(UserPackage, UserPackage.package_id == Package.package_id)
        .where(
            and_(
                UserPackage.user_id == user_id,
                UserPackage.status == "active",
                Package.category == "main",
            )
        )
        .order_by(UserPackage.subscribed_at.desc())
    )
    row = db.execute(stmt).first()
    if row is None:
        return None
    pkg, sub = row
    return {
        "package_id": pkg.package_id,
        "package_name": pkg.name,
        "monthly_fee": pkg.monthly_fee,
        "data_gb": pkg.data_gb,
        "voice_minutes": pkg.voice_minutes,
        "data_used_gb": sub.data_used_gb,
        "remaining_data_gb": round(pkg.data_gb - sub.data_used_gb, 2),
        "subscribed_at": sub.subscribed_at.isoformat(),
    }


def get_remaining_data(db: Session, user_id: str) -> dict:
    """用户剩余流量（含叠加包，逐包明细）。"""
    stmt = (
        select(Package.name, Package.data_gb, UserPackage.data_used_gb, UserPackage.expires_at)
        .join(UserPackage, UserPackage.package_id == Package.package_id)
        .where(and_(UserPackage.user_id == user_id, UserPackage.status == "active"))
    )
    details = []
    total_remaining = 0.0
    for row in db.execute(stmt):
        remaining = round(row.data_gb - row.data_used_gb, 2)
        total_remaining += remaining
        details.append(
            {
                "package_name": row.name,
                "total_gb": row.data_gb,
                "used_gb": row.data_used_gb,
                "remaining_gb": remaining,
                "expires_at": row.expires_at.isoformat() if row.expires_at else None,
            }
        )
    return {"user_id": user_id, "total_remaining_gb": round(total_remaining, 2), "details": details}


def search_packages(
    db: Session,
    category: str | None = None,
    min_data_gb: float | None = None,
    max_monthly_fee: float | None = None,
) -> list[dict]:
    """按条件搜索在售套餐。"""
    stmt = select(Package).where(Package.status == "active")
    if category:
        stmt = stmt.where(Package.category == category)
    if min_data_gb is not None:
        stmt = stmt.where(Package.data_gb >= min_data_gb)
    if max_monthly_fee is not None:
        stmt = stmt.where(Package.monthly_fee <= max_monthly_fee)
    stmt = stmt.order_by(Package.monthly_fee)
    return [
        {
            "package_id": p.package_id,
            "name": p.name,
            "category": p.category,
            "monthly_fee": p.monthly_fee,
            "data_gb": p.data_gb,
            "voice_minutes": p.voice_minutes,
            "description": p.description,
        }
        for p in db.execute(stmt).scalars()
    ]


def recommend_addon(db: Session, user_id: str, min_data_gb: float) -> dict | None:
    """推荐满足至少 min_data_gb 的最便宜流量加餐包。"""
    stmt = (
        select(Package)
        .where(
            and_(
                Package.category == "data_addon",
                Package.status == "active",
                Package.data_gb >= min_data_gb,
            )
        )
        .order_by(Package.monthly_fee, Package.data_gb)
    )
    best = db.execute(stmt).scalars().first()
    if best is None:
        return None
    return {
        "user_id": user_id,
        "min_data_gb": min_data_gb,
        "recommended_package_id": best.package_id,
        # 别名：下游计划引用 $step_N.package_id 时可解析
        "package_id": best.package_id,
        "name": best.name,
        "monthly_fee": best.monthly_fee,
        "data_gb": best.data_gb,
        "reason": f"为满足至少 {min_data_gb}GB 需求，在售加餐包中性价比最优（价格最低档）",
    }
