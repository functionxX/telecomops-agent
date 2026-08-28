"""Tool 安全策略。

LLM 生成 Tool Name 不代表拥有调用权限：每个工具都有程序级策略
（角色、风险等级、是否必须人工确认），由 ToolExecutor 强制检查。
Prompt 只是提示，Policy 才是防线。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolPolicy:
    name: str
    role: str = "customer_service"  # 允许调用的角色
    risk: str = "LOW"  # LOW / HIGH
    require_confirmation: bool = False  # 高风险工具必须人工确认


TOOL_POLICIES: dict[str, ToolPolicy] = {
    # 客户类（只读，低风险）
    "get_customer_profile": ToolPolicy("get_customer_profile", risk="LOW"),
    "get_customer_level": ToolPolicy("get_customer_level", risk="LOW"),
    # 套餐类（只读，低风险）
    "get_current_package": ToolPolicy("get_current_package", risk="LOW"),
    "get_remaining_data": ToolPolicy("get_remaining_data", risk="LOW"),
    "search_packages": ToolPolicy("search_packages", risk="LOW"),
    "recommend_package": ToolPolicy("recommend_package", risk="LOW"),
    # 服务类（查询低风险；开通/关闭 = 变更操作，高风险）
    "query_roaming_status": ToolPolicy("query_roaming_status", risk="LOW"),
    "enable_roaming": ToolPolicy(
        "enable_roaming", risk="HIGH", require_confirmation=True
    ),
    "disable_roaming": ToolPolicy(
        "disable_roaming", risk="HIGH", require_confirmation=True
    ),
    # 订单类（查询低风险；下单/取消 = 资金操作，高风险）
    "query_order": ToolPolicy("query_order", risk="LOW"),
    "create_order": ToolPolicy("create_order", risk="HIGH", require_confirmation=True),
    "cancel_order": ToolPolicy("cancel_order", risk="HIGH", require_confirmation=True),
}


def get_policy(tool_name: str) -> ToolPolicy | None:
    """获取工具策略；未知工具返回 None（由 Registry 层面报 ToolNotFound）。"""
    return TOOL_POLICIES.get(tool_name)
