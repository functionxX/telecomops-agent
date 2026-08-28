"""审批数据访问：审批请求的持久化（可查询、可审计）。"""

from datetime import UTC

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Approval


def create_or_get(db: Session, approval_id: str, *, conversation_id: str, agent_run_id: str, tool_name: str, arguments: dict) -> Approval:
    """幂等创建审批记录（中断恢复时节点会重跑）。"""
    existing = db.execute(select(Approval).where(Approval.id == approval_id)).scalar_one_or_none()
    if existing is not None:
        return existing
    approval = Approval(
        id=approval_id,
        conversation_id=conversation_id,
        agent_run_id=agent_run_id,
        tool_name=tool_name,
        arguments=arguments,
        status="pending",
    )
    db.add(approval)
    db.commit()
    return approval


def get(db: Session, approval_id: str) -> Approval | None:
    return db.execute(select(Approval).where(Approval.id == approval_id)).scalar_one_or_none()


def decide(db: Session, approval_id: str, decision: str, decided_by: str) -> Approval | None:
    """记录审批决策（approved / rejected）。"""
    approval = get(db, approval_id)
    if approval is None:
        return None
    from datetime import datetime

    approval.status = "approved" if decision == "approved" else "rejected"
    approval.decided_by = decided_by
    approval.decided_at = datetime.now(UTC)
    db.commit()
    return approval
