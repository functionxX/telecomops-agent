"""统一工具注册中心。

Tool = 能力，不是 Workflow。每个工具声明：name / description /
typed arguments（Pydantic Schema）/ handler。
Planner 通过 Registry 拿到工具清单生成计划，执行由 ToolExecutor 完成。
"""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict

_registry: dict[str, "ToolSpec"] = {}


class ToolSpec(BaseModel):
    """工具规格：名称、描述、参数 Schema、处理器。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    args_schema: type[BaseModel]
    handler: Callable[..., Any]


def register(name: str, description: str, args_schema: type[BaseModel]):
    """装饰器：注册一个工具。"""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if name in _registry:
            raise ValueError(f"tool already registered: {name}")
        _registry[name] = ToolSpec(
            name=name, description=description, args_schema=args_schema, handler=func
        )
        return func

    return decorator


class ToolRegistry:
    """工具注册表（Registry 模式）。"""

    def get(self, name: str) -> ToolSpec | None:
        return _registry.get(name)

    def all(self) -> dict[str, ToolSpec]:
        return dict(_registry)

    def names(self) -> list[str]:
        return sorted(_registry)

    def describe_all(self) -> list[dict[str, Any]]:
        """给 Planner / LLM 的工具清单（name + description + 参数 schema）。"""
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "arguments": spec.args_schema.model_json_schema(),
            }
            for spec in _registry.values()
        ]


registry = ToolRegistry()
