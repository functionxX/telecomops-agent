"""ToolExecutor 单元测试（注入 Fake 工具，零外部依赖）。

覆盖：工具不存在 / 参数校验 / 权限拒绝 / 越权拦截 / 风险审批 /
超时 / 业务错误 / 成功 + 截断。
"""


import pytest
from app.core.exceptions import BusinessError, ExecutionError
from app.guardrails.tool import USER_SCOPED_TOOLS
from app.tools import registry as global_registry  # noqa: F401
from app.tools.executor import ToolExecutionStatus, ToolExecutor
from app.tools.registry import _registry
from pydantic import BaseModel, Field


class FakeArgs(BaseModel):
    user_id: str = Field(description="用户ID")
    value: int = Field(default=1, description="值")


class SlowArgs(BaseModel):
    value: int = Field(default=1)


class _Result:
    """假查询结果：所有取数方法返回空/None（模拟"查无记录"）。"""

    def scalar_one_or_none(self):
        return None

    def scalar_one(self):
        return None

    def first(self):
        return None

    def fetchall(self):
        return []

    def scalars(self):
        return []


class _FakeDB:
    """极简假 DB：只实现 repo 用到的 execute（总是查无记录）。"""

    def execute(self, *args, **kwargs):
        return _Result()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_session_factory():
    def factory():
        return _FakeDB()

    return factory


@pytest.fixture()
def executor_with_fakes():
    """注入 Fake 工具（测试后清理，不污染全局注册表）。"""
    saved = dict(_registry)
    saved_scoped = set(USER_SCOPED_TOOLS)

    def _register(name, description, schema, handler):
        from app.tools.registry import ToolSpec

        _registry[name] = ToolSpec(
            name=name, description=description, args_schema=schema, handler=handler
        )

    _register("fake_ok", "ok 工具", FakeArgs, lambda db, user_id, value=1: {"user_id": user_id, "value": value})
    _register(
        "fake_business_error", "业务错误工具", FakeArgs,
        lambda db, user_id, value=1: (_ for _ in ()).throw(BusinessError("业务不允许")),
    )
    _register(
        "fake_exec_error", "执行错误工具", FakeArgs,
        lambda db, user_id, value=1: (_ for _ in ()).throw(ExecutionError("DB 抖动")),
    )
    _register("fake_slow", "慢工具", SlowArgs, lambda db, value=1: __import__("time").sleep(10))
    _register("fake_unscoped", "无用户作用域工具", SlowArgs, lambda db, value=1: {"v": value})
    # 让 fake 工具走 user_id 注入路径（测试越权拦截/注入行为）
    USER_SCOPED_TOOLS.update({"fake_ok", "fake_business_error", "fake_exec_error", "fake_slow"})

    executor = ToolExecutor(session_factory=_fake_session_factory())
    yield executor
    _registry.clear()
    _registry.update(saved)
    USER_SCOPED_TOOLS.clear()
    USER_SCOPED_TOOLS.update(saved_scoped)


def test_tool_not_found(executor_with_fakes):
    r = executor_with_fakes.execute("nope", {}, actor_user_id="user_001")
    assert r.status == ToolExecutionStatus.ERROR
    assert r.failure.code == "tool_not_found"
    assert r.failure.kind.value == "plan_error"


def test_invalid_arguments(executor_with_fakes):
    r = executor_with_fakes.execute("fake_ok", {"value": "not-int"}, actor_user_id="user_001")
    assert r.status == ToolExecutionStatus.ERROR
    assert r.failure.code == "invalid_arguments"


def test_cross_user_blocked(executor_with_fakes):
    r = executor_with_fakes.execute(
        "fake_ok", {"user_id": "user_999"}, actor_user_id="user_001"
    )
    assert r.status == ToolExecutionStatus.ERROR
    assert r.failure.code == "permission_denied"


def test_business_error_fast_fail(executor_with_fakes):
    r = executor_with_fakes.execute(
        "fake_business_error", {}, actor_user_id="user_001"
    )
    assert r.status == ToolExecutionStatus.ERROR
    assert r.failure.code == "business_error"
    assert r.failure.kind.value == "fast_fail"


def test_execution_error_transient(executor_with_fakes):
    r = executor_with_fakes.execute("fake_exec_error", {}, actor_user_id="user_001")
    assert r.failure.code == "execution_error"
    assert r.failure.kind.value == "transient"


def test_success_with_actor_injection(executor_with_fakes):
    r = executor_with_fakes.execute("fake_ok", {}, actor_user_id="user_007")
    assert r.status == ToolExecutionStatus.SUCCESS
    assert r.result["user_id"] == "user_007"  # 执行器注入，不信任调用方


def test_timeout(executor_with_fakes):
    from app.core.config import settings

    original = settings.tool_timeout
    settings.tool_timeout = 1  # 注入 1 秒超时
    try:
        r = executor_with_fakes.execute("fake_slow", {}, actor_user_id="user_001")
    finally:
        settings.tool_timeout = original
    assert r.status == ToolExecutionStatus.ERROR
    assert r.failure.code == "tool_timeout"
    assert r.failure.kind.value == "transient"


def test_truncation(executor_with_fakes):
    long_text = "很长的内容" * 200
    _registry["fake_ok"].handler = lambda db, user_id, value=1: {"user_id": user_id, "text": long_text}
    r = executor_with_fakes.execute("fake_ok", {}, actor_user_id="user_001")
    assert r.status == ToolExecutionStatus.SUCCESS
    assert r.truncated is True
    assert "截断" in r.result["text"]


def test_high_risk_requires_approval():
    # create_order 是真实注册的高风险工具：未批准必须返回 APPROVAL_REQUIRED
    executor = ToolExecutor(session_factory=_fake_session_factory())
    r = executor.execute(
        "create_order", {"package_id": "addon_30g"}, actor_user_id="user_001"
    )
    assert r.status == ToolExecutionStatus.APPROVAL_REQUIRED
    assert r.approval is not None
    assert r.approval.tool_name == "create_order"


def test_high_risk_approved_executes():
    executor = ToolExecutor(session_factory=_fake_session_factory())
    r = executor.execute(
        "create_order",
        {"package_id": "addon_30g"},
        actor_user_id="user_001",
        approval_granted=True,
    )
    # 假 DB 上没有套餐 → BusinessError（说明审批通过后确实进入了执行阶段）
    assert r.status == ToolExecutionStatus.ERROR
    assert r.failure.code == "business_error"
