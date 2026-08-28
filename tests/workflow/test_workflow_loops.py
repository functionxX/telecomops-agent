"""Workflow 核心测试：真实 Retry / Replan / 条件跳转 / Human Approval 循环。

这是项目最重要的测试：不是 mock 流程图，而是让图真实跑完
失败→分流→循环→成功的完整路径。
"""

from app.agent.graph import build_graph
from app.core.exceptions import FailureKind
from app.db.session import session_scope
from app.tools.executor import (
    ToolExecution,
    ToolExecutionStatus,
    ToolExecutor,
    ToolFailure,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command


class FailOnceExecutor(ToolExecutor):
    """指定工具第一次调用注入指定类型失败，之后正常。"""

    def __init__(self, *args, fail_tool: str, fail_kind: FailureKind, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._fail_tool = fail_tool
        self._fail_kind = fail_kind
        self._failed = False

    def execute(self, tool_name: str, arguments: dict, **kwargs) -> ToolExecution:
        if tool_name == self._fail_tool and not self._failed:
            self._failed = True
            return ToolExecution(
                status=ToolExecutionStatus.ERROR,
                tool_name=tool_name,
                failure=ToolFailure(
                    code="execution_error" if self._fail_kind == FailureKind.TRANSIENT else "invalid_arguments",
                    message=f"测试注入故障（{self._fail_kind.value}）",
                    kind=self._fail_kind,
                ),
            )
        return super().execute(tool_name, arguments, **kwargs)


class AlwaysFailExecutor(ToolExecutor):
    """永远瞬时失败：验证循环上限（不会无限循环）。"""

    def execute(self, tool_name: str, arguments: dict, **kwargs) -> ToolExecution:
        return ToolExecution(
            status=ToolExecutionStatus.ERROR,
            tool_name=tool_name,
            failure=ToolFailure(code="execution_error", message="永远失败", kind=FailureKind.TRANSIENT),
        )


def _invoke(graph, query: str, user_id: str, thread_id: str) -> dict:
    return graph.invoke(
        {
            "query": query,
            "user_id": user_id,
            "conversation_id": thread_id,
            "trace_id": f"test_{thread_id}",
        },
        config={"configurable": {"thread_id": thread_id}},
    )


class TestRetryLoop:
    def test_transient_failure_retries_then_succeeds(self, require_postgres):
        """工具第一次瞬时失败 → Validator → Retry → 第二次成功。"""
        executor = FailOnceExecutor(
            session_factory=session_scope,
            fail_tool="get_remaining_data",
            fail_kind=FailureKind.TRANSIENT,
        )
        graph = build_graph(checkpointer=MemorySaver(), executor=executor)
        state = _invoke(graph, "我的套餐还剩多少流量？", "user_001", "test_retry_1")

        assert state.get("intent") == "QUERY"
        assert state.get("retry_count", 0) == 1  # 真实重试过一次
        assert not state.get("error")
        assert "step_query" in state.get("tool_results", {})

    def test_retry_exhausted_terminates_not_infinite(self, require_postgres):
        """永久瞬时失败：重试 2 次到上限后收口失败，绝不无限循环。"""
        graph = build_graph(checkpointer=MemorySaver(), executor=AlwaysFailExecutor(session_factory=session_scope))
        state = _invoke(graph, "我的套餐还剩多少流量？", "user_001", "test_retry_cap")

        assert state.get("retry_count", 0) == 2  # 到达 max_retries 上限
        assert state.get("error")  # 收口为失败回答
        assert "已重试 2 次" in state.get("final_answer", "")


class TestReplanLoop:
    def test_plan_error_replans_then_succeeds(self, require_postgres):
        """工具计划性失败 → Validator → Replan → Planner → 新执行成功。"""
        executor = FailOnceExecutor(
            session_factory=session_scope,
            fail_tool="get_current_package",
            fail_kind=FailureKind.PLAN_ERROR,
        )
        graph = build_graph(checkpointer=MemorySaver(), executor=executor)
        state = _invoke(graph, "我现在用的是什么套餐？", "user_001", "test_replan_1")

        assert state.get("intent") == "QUERY"
        assert state.get("replan_count", 0) == 1  # 真实重规划过一次
        assert not state.get("error")
        assert "step_query" in state.get("tool_results", {})


class TestHumanApprovalLoop:
    def test_approval_interrupt_resume_executes(self, require_postgres):
        """高风险工具 → interrupt → 批准 resume → 执行成功（订单幂等）。"""
        graph = build_graph(checkpointer=MemorySaver(), executor=ToolExecutor(session_factory=session_scope))
        thread = "test_approval_flow"
        state = _invoke(graph, "帮我办理30GB流量包。", "user_001", thread)

        interrupts = state.get("__interrupt__", [])
        assert interrupts, "高风险操作必须中断等待审批"
        payload = interrupts[0].value
        assert payload["tool_name"] == "create_order"
        assert state["requires_human_approval"] is True

        # 批准：从 checkpoint 恢复继续执行
        final_state = graph.invoke(
            Command(resume={"approval_id": payload["approval_id"], "decision": "approved"}),
            config={"configurable": {"thread_id": thread}},
        )
        assert not final_state.get("error")
        assert "step_1" in final_state.get("tool_results", {})
        order = final_state["tool_results"]["step_1"]
        assert order["package_id"] == "addon_30g"

    def test_approval_reject_terminates(self, require_postgres):
        """拒绝审批：不执行工具，Workflow 正常收口。"""
        graph = build_graph(checkpointer=MemorySaver(), executor=ToolExecutor(session_factory=session_scope))
        thread = "test_approval_reject"
        state = _invoke(graph, "帮我开通国际漫游", "user_001", thread)

        interrupts = state.get("__interrupt__", [])
        assert interrupts
        payload = interrupts[0].value

        final_state = graph.invoke(
            Command(resume={"approval_id": payload["approval_id"], "decision": "rejected"}),
            config={"configurable": {"thread_id": thread}},
        )
        assert final_state.get("human_decision") == "rejected"
        assert "取消" in final_state.get("final_answer", "")

    def test_second_high_risk_action_requires_new_approval(self, require_postgres):
        """一次审批只放行一个动作：批准执行后状态重置，下次高风险需重新审批。"""
        graph = build_graph(checkpointer=MemorySaver(), executor=ToolExecutor(session_factory=session_scope))
        thread = "test_approval_reset"
        state = _invoke(graph, "帮我开通国际漫游", "user_001", thread)
        payload = state["__interrupt__"][0].value
        graph.invoke(
            Command(resume={"approval_id": payload["approval_id"], "decision": "approved"}),
            config={"configurable": {"thread_id": thread}},
        )
        # 同一会话再次发起高风险操作：必须再次中断（不能沿用上次批准）
        state2 = _invoke(graph, "帮我关闭国际漫游", "user_001", thread)
        assert state2.get("__interrupt__"), "第二次高风险操作必须重新审批"
