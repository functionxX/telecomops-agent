"""增值服务（漫游等）数据访问。"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Service


def _to_dict(s: Service) -> dict:
    return {
        "service_id": s.id,
        "user_id": s.user_id,
        "service_type": s.service_type,
        "status": s.status,
        "updated_at": s.updated_at.isoformat(),
    }


def query_service_status(db: Session, user_id: str, service_type: str) -> dict | None:
    """查询某类服务的开通状态。"""
    stmt = select(Service).where(
        Service.user_id == user_id, Service.service_type == service_type
    )
    svc = db.execute(stmt).scalar_one_or_none()
    return _to_dict(svc) if svc else None


def set_service_status(db: Session, user_id: str, service_type: str, enabled: bool) -> dict:
    """开通/关闭服务（upsert）。"""
    stmt = select(Service).where(
        Service.user_id == user_id, Service.service_type == service_type
    )
    svc = db.execute(stmt).scalar_one_or_none()
    status = "enabled" if enabled else "disabled"
    if svc is None:
        svc = Service(
            id=f"svc_{uuid.uuid4().hex[:10]}",
            user_id=user_id,
            service_type=service_type,
            status=status,
            updated_at=datetime.now(UTC),
        )
        db.add(svc)
    else:
        svc.status = status
        svc.updated_at = datetime.now(UTC)
    db.commit()
    return _to_dict(svc)
