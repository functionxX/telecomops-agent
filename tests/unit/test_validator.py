"""Validator 单元测试：失败分类分流（Retry / Replan / Fail / satisfied）。"""

from app.agent.validator import validate_step
from app.core.exceptions import FailureKind
from app.tools.executor import ToolExecutionStatus


def test_transient_goes_retry():
    verdict = validate_step(
        execution_status=ToolExecutionStatus.ERROR.value,
        failure_kind=FailureKind.TRANSIENT,
        failure_message="timeout",
        step_description="查询流量",
        tool_name="get_remaining_data",
        tool_result=None,
        use_semantic_check=False,
    )
    assert verdict["kind"] == "retry"


def test_plan_error_goes_replan():
    verdict = validate_step(
        execution_status=ToolExecutionStatus.ERROR.value,
        failure_kind=FailureKind.PLAN_ERROR,
        failure_message="tool not found",
        step_description="x",
        tool_name="no_such_tool",
        tool_result=None,
        use_semantic_check=False,
    )
    assert verdict["kind"] == "replan"


def test_fast_fail_goes_fail():
    verdict = validate_step(
        execution_status=ToolExecutionStatus.ERROR.value,
        failure_kind=FailureKind.FAST_FAIL,
        failure_message="permission denied",
        step_description="x",
        tool_name="create_order",
        tool_result=None,
        use_semantic_check=False,
    )
    assert verdict["kind"] == "fail"


def test_success_with_empty_result_goes_replan():
    verdict = validate_step(
        execution_status=ToolExecutionStatus.SUCCESS.value,
        failure_kind=None,
        failure_message=None,
        step_description="x",
        tool_name="t",
        tool_result={},
        use_semantic_check=False,
    )
    assert verdict["status"] == "failed"
    assert verdict["kind"] == "replan"


def test_success_passes():
    verdict = validate_step(
        execution_status=ToolExecutionStatus.SUCCESS.value,
        failure_kind=None,
        failure_message=None,
        step_description="x",
        tool_name="t",
        tool_result={"ok": 1},
        use_semantic_check=False,
    )
    assert verdict["status"] == "satisfied"
