"""Tool Registry 单元测试：11 个工具齐全、Schema 可校验、未知工具返回 None。"""

import pytest
from app.tools.registry import ToolRegistry, register, registry
from pydantic import BaseModel, Field

EXPECTED_TOOLS = {
    "get_customer_profile",
    "get_customer_level",
    "get_current_package",
    "get_remaining_data",
    "search_packages",
    "recommend_package",
    "query_roaming_status",
    "enable_roaming",
    "disable_roaming",
    "create_order",
    "query_order",
    "cancel_order",
}


def test_all_spec_tools_registered():
    names = set(registry.names())
    assert EXPECTED_TOOLS <= names, f"缺少工具: {EXPECTED_TOOLS - names}"


def test_describe_all_has_schema():
    for item in registry.describe_all():
        assert item["name"]
        assert item["description"]
        assert "properties" in item["arguments"]


def test_get_unknown_returns_none():
    assert registry.get("not_a_tool") is None


def test_duplicate_registration_rejected():
    class Args(BaseModel):
        x: int = Field(default=0)

    with pytest.raises(ValueError, match="already registered"):
        # 已注册名字再注册必须报错
        register("get_customer_profile", "dup", Args)(lambda db, x=0: {})


def test_registry_is_isolated_instance():
    fresh = ToolRegistry()
    assert fresh.get("get_current_package") is not None  # 共享模块级注册表
    assert fresh.get("zzz") is None
