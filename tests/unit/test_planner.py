"""Planner 单元测试（MockLLM 确定性模板 + Schema 校验）。"""

from app.agent.planner import Plan, PlanStep, _fallback_plan, plan_task, replan


def test_plan_schema_validation():
    plan = Plan(
        steps=[
            PlanStep(
                step_id="step_1",
                tool="get_current_package",
                arguments={"user_id": "u1"},
                description="查询",
            )
        ]
    )
    assert plan.steps[0].status == "PENDING"


def test_plan_task_condition_structure():
    """条件场景必须产出含 condition 步骤的计划。"""
    plan, stats = plan_task("帮我查一下当前套餐，如果剩余流量低于10GB就推荐一个流量包。", user_id="user_001")
    assert stats["status"] == "ok"
    tools = [s.tool for s in plan.steps]
    assert "get_current_package" in tools
    assert "get_remaining_data" in tools
    assert "recommend_package" in tools
    cond = [s for s in plan.steps if s.tool is None]
    assert cond, "必须包含 condition 步骤"
    args = cond[0].arguments
    assert args["left"].startswith("$step_")  # 状态引用
    assert "then_step" in args and "else_step" in args


def test_plan_task_order_creation():
    plan, _ = plan_task("帮我办理30GB流量包。", user_id="user_001")
    assert plan.steps[0].tool == "create_order"
    assert plan.steps[0].arguments["package_id"] == "addon_30g"


def test_replan_returns_new_plan():
    plan, stats = replan(
        "帮我办理30GB流量包。",
        user_id="user_001",
        failed_tool="create_order",
        failure_reason="套餐不存在",
        previous_results={},
    )
    assert stats["phase"] == "replan"
    assert isinstance(plan, Plan)


def test_fallback_plan_is_safe():
    plan = _fallback_plan("user_001")
    assert plan.steps[0].tool == "get_current_package"  # 兜底只做安全查询
