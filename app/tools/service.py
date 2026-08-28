"""服务类工具：漫游查询/开通/关闭。"""

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.repositories import service_repo
from app.tools.registry import register


class QueryRoamingStatusArgs(BaseModel):
    user_id: str = Field(description="用户ID")


@register(
    "query_roaming_status",
    description="查询用户国际漫游服务的开通状态。只读。",
    args_schema=QueryRoamingStatusArgs,
)
def query_roaming_status(db: Session, user_id: str) -> dict:
    status = service_repo.query_service_status(db, user_id, "roaming")
    if status is None:
        return {"user_id": user_id, "service_type": "roaming", "status": "not_opened"}
    return status


class ToggleRoamingArgs(BaseModel):
    user_id: str = Field(description="用户ID")


@register(
    "enable_roaming",
    description="为用户开通国际漫游服务（高风险变更操作，需人工确认）。",
    args_schema=ToggleRoamingArgs,
)
def enable_roaming(db: Session, user_id: str) -> dict:
    result = service_repo.set_service_status(db, user_id, "roaming", enabled=True)
    return {**result, "message": "国际漫游已开通"}


@register(
    "disable_roaming",
    description="为用户关闭国际漫游服务（高风险变更操作，需人工确认）。",
    args_schema=ToggleRoamingArgs,
)
def disable_roaming(db: Session, user_id: str) -> dict:
    result = service_repo.set_service_status(db, user_id, "roaming", enabled=False)
    return {**result, "message": "国际漫游已关闭"}
