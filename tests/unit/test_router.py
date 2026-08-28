"""Intent Router 单元测试（MockLLM 确定性规则 + 兜底路径）。"""

from unittest.mock import patch

from app.agent.router import route
from app.core.exceptions import LLMError


def test_route_faq():
    decision, stats = route("5G套餐有哪些？")
    assert decision.intent == "FAQ"
    assert stats["status"] == "ok"
    assert "latency_ms" in stats and "model" in stats


def test_route_query():
    decision, _ = route("我的积分还有多少？")
    assert decision.intent == "QUERY"


def test_route_task_conditional():
    decision, _ = route("帮我查一下套餐，如果流量低于10GB就推荐流量包。")
    assert decision.intent == "TASK"


def test_route_task_action():
    decision, _ = route("帮我办理30GB流量包。")
    assert decision.intent == "TASK"


def test_route_unknown():
    decision, _ = route("今天天气怎么样？")
    assert decision.intent == "UNKNOWN"


def test_route_howto_is_faq_not_task():
    # "怎么开通" 是知识问题，不能因为含"开通"被误判为 TASK
    decision, _ = route("国际漫游怎么开通？")
    assert decision.intent == "FAQ"


def test_route_llm_failure_falls_back_to_unknown():
    with patch("app.agent.router.get_llm_client") as mock_client:
        mock_client.return_value.structured_output.side_effect = LLMError("boom")
        decision, stats = route("任意内容")
    assert decision.intent == "UNKNOWN"
    assert stats["status"] == "fallback"
