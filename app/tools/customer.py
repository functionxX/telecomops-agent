"""客户类工具：档案、等级。"""

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.exceptions import BusinessError
from app.db.repositories import customer_repo
from app.tools.registry import register


class GetCustomerProfileArgs(BaseModel):
    user_id: str = Field(description="用户ID")


@register(
    "get_customer_profile",
    description="获取客户档案：姓名、等级、信用分、入网时间等。只读。",
    args_schema=GetCustomerProfileArgs,
)
def get_customer_profile(db: Session, user_id: str) -> dict:
    profile = customer_repo.get_customer_profile(db, user_id)
    if profile is None:
        raise BusinessError(f"用户 {user_id} 不存在或未建立客户档案")
    return profile


class GetCustomerLevelArgs(BaseModel):
    user_id: str = Field(description="用户ID")


@register(
    "get_customer_level",
    description="获取客户等级（钻石/金卡/银卡/普通）。只读。",
    args_schema=GetCustomerLevelArgs,
)
def get_customer_level(db: Session, user_id: str) -> dict:
    level = customer_repo.get_customer_level(db, user_id)
    if level is None:
        raise BusinessError(f"用户 {user_id} 不存在或未建立客户档案")
    return {"user_id": user_id, "customer_level": level}
