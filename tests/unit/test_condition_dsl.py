"""condition 步骤 mini-DSL 单元测试（纯函数，无外部依赖）。

覆盖：状态引用解析、条件比较、分支跳转与 SKIPPED 标记、异常兜底。
"""

import pytest
from app.agent.executor import (
    apply_condition_jump,
    evaluate_condition,
    find_step,
    resolve_references,
)


class TestResolveReferences:
    def test_simple_reference(self):
        results = {"step_2": {"total_remaining_gb": 8.0}}
        assert resolve_references("$step_2.total_remaining_gb", results) == 8.0

    def test_nested_reference(self):
        results = {"step_1": {"detail": {"a": {"b": 42}}}}
        assert resolve_references("$step_1.detail.a.b", results) == 42

    def test_dict_and_list_recursion(self):
        results = {"s1": {"v": 1}}
        assert resolve_references({"x": "$s1.v", "y": ["$s1.v"]}, results) == {"x": 1, "y": [1]}

    def test_non_reference_passthrough(self):
        assert resolve_references("plain text", {}) == "plain text"

    def test_broken_reference_raises(self):
        with pytest.raises(ValueError):
            resolve_references("$step_9.missing", {})


class TestEvaluateCondition:
    def test_lt_true(self):
        args = {"left": "$s2.total", "op": "<", "right": 10}
        assert evaluate_condition(args, {"s2": {"total": 8}}) == "then"

    def test_ge_false(self):
        args = {"left": "$s2.total", "op": ">=", "right": 10}
        assert evaluate_condition(args, {"s2": {"total": 8}}) == "else"

    def test_eq(self):
        args = {"left": "$s1.name", "op": "==", "right": "x"}
        assert evaluate_condition(args, {"s1": {"name": "x"}}) == "then"

    def test_invalid_safely_else(self):
        # 引用缺失 / 运算符非法 → else（安全方向，不抛异常）
        assert evaluate_condition({"left": "$missing.x", "op": "<", "right": 1}, {}) == "else"
        assert evaluate_condition({"left": 1, "op": "?&", "right": 1}, {}) == "else"


class TestApplyConditionJump:
    def _plan(self):
        return [
            {"step_id": "s1", "tool": "t1", "arguments": {}, "description": "", "status": "SUCCESS"},
            {"step_id": "s2", "tool": None, "arguments": {}, "description": "", "status": "PENDING"},
            {"step_id": "s3", "tool": "t3", "arguments": {}, "description": "", "status": "PENDING"},
            {"step_id": "s4", "tool": "t4", "arguments": {}, "description": "", "status": "PENDING"},
        ]

    def test_then_branch_skips_else(self):
        plan = self._plan()
        cond = {"left": 5, "op": "<", "right": 10, "then_step": "s3", "else_step": "END"}
        next_id = apply_condition_jump(plan, "s2", cond, {})
        assert next_id == "s3"
        assert find_step(plan, "s3")["status"] == "PENDING"

    def test_else_branch_skips_then(self):
        plan = self._plan()
        cond = {"left": 15, "op": "<", "right": 10, "then_step": "s3", "else_step": "s4"}
        next_id = apply_condition_jump(plan, "s2", cond, {})
        assert next_id == "s4"
        assert find_step(plan, "s3")["status"] == "SKIPPED"

    def test_else_end_marks_rest_skipped(self):
        plan = self._plan()
        cond = {"left": 15, "op": "<", "right": 10, "then_step": "s3", "else_step": "END"}
        next_id = apply_condition_jump(plan, "s2", cond, {})
        assert next_id == "END"
        assert find_step(plan, "s3")["status"] == "SKIPPED"
