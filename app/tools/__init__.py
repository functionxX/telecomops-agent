"""工具包：导入即注册（@register 装饰器在模块导入时填充 Registry）。"""

from app.tools import customer, order, package, service  # noqa: F401
from app.tools.registry import ToolRegistry, registry  # noqa: F401
