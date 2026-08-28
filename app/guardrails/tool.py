"""Tool Guardrail：user_id 越权检查。

核心规则：工具参数中的 user_id 必须与当前会话的 actor 一致。
LLM 生成 user_id=其他用户 的调用会被拦截——权限不是靠 prompt 保证的。
"""

from app.core.exceptions import PermissionDeniedError

USER_SCOPED_TOOLS = {
    "get_customer_profile",
    "get_customer_level",
    "get_current_package",
    "get_remaining_data",
    "recommend_package",
    "query_roaming_status",
    "enable_roaming",
    "disable_roaming",
    "create_order",
}


def check_actor_scope(tool_name: str, arguments: dict, actor_user_id: str) -> dict:
    """校验并注入 actor 的 user_id。

    - 工具按 user_id 作用域：arguments 中的 user_id 必须等于 actor_user_id，
      且由程序强制写回可信的 actor_user_id（不信任 LLM 提供的值）。
    - 若 LLM 提供了不同的 user_id，说明存在越权尝试 → PermissionDenied。
    """
    if tool_name not in USER_SCOPED_TOOLS:
        return arguments

    provided = arguments.get("user_id")
    if provided is not None and provided != actor_user_id:
        raise PermissionDeniedError(
            f"工具 {tool_name} 的 user_id={provided} 与当前会话用户 {actor_user_id} 不一致，已拦截"
        )
    arguments = dict(arguments)
    arguments["user_id"] = actor_user_id
    return arguments
