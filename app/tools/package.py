"""套餐类工具：当前套餐、剩余流量、搜索、推荐。"""


from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.exceptions import BusinessError
from app.db.repositories import package_repo
from app.tools.registry import register


class GetCurrentPackageArgs(BaseModel):
    user_id: str = Field(description="用户ID")


@register(
    "get_current_package",
    description="查询用户当前生效的主套餐（名称、月费、流量总量、已用、剩余）。只读。",
    args_schema=GetCurrentPackageArgs,
)
def get_current_package(db: Session, user_id: str) -> dict:
    result = package_repo.get_current_package(db, user_id)
    if result is None:
        raise BusinessError(f"用户 {user_id} 当前没有生效中的主套餐")
    return result


class GetRemainingDataArgs(BaseModel):
    user_id: str = Field(description="用户ID")


@register(
    "get_remaining_data",
    description="查询用户剩余流量（含叠加包，逐包明细与合计）。只读。",
    args_schema=GetRemainingDataArgs,
)
def get_remaining_data(db: Session, user_id: str) -> dict:
    return package_repo.get_remaining_data(db, user_id)


class SearchPackagesArgs(BaseModel):
    category: str | None = Field(
        default=None, description="套餐类别：main / data_addon，可空"
    )
    min_data_gb: float | None = Field(default=None, description="最小流量 GB，可空")
    max_monthly_fee: float | None = Field(default=None, description="最高月费，可空")


@register(
    "search_packages",
    description="按类别/流量/价格条件搜索在售套餐。只读。",
    args_schema=SearchPackagesArgs,
)
def search_packages(
    db: Session,
    category: str | None = None,
    min_data_gb: float | None = None,
    max_monthly_fee: float | None = None,
) -> dict:
    packages = package_repo.search_packages(db, category, min_data_gb, max_monthly_fee)
    return {"count": len(packages), "packages": packages}


class RecommendPackageArgs(BaseModel):
    user_id: str = Field(description="用户ID")
    min_data_gb: float = Field(description="需要补充的最小流量（GB）")


@register(
    "recommend_package",
    description="根据需要的流量缺口推荐最便宜的流量加餐包。只读。",
    args_schema=RecommendPackageArgs,
)
def recommend_package(db: Session, user_id: str, min_data_gb: float) -> dict:
    result = package_repo.recommend_addon(db, user_id, min_data_gb)
    if result is None:
        raise BusinessError(f"没有满足 {min_data_gb}GB 需求的在售加餐包")
    return result
