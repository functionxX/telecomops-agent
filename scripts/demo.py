"""端到端演示脚本：Mock 或真实 LLM 模式跑 7 个业务场景。

用法：
  uv run python scripts/demo.py            # 全部场景
  uv run python scripts/demo.py 3          # 只跑场景 3
"""

import json
import sys

from app.agent.graph import build_graph
from langgraph.checkpoint.memory import MemorySaver

SCENARIOS = [
    ("场景1: QUERY 单步查询", "我的套餐还剩多少流量？"),
    ("场景2: FAQ 知识检索", "5G套餐有哪些？"),
    ("场景3: TASK 条件分支", "帮我查一下当前套餐，如果剩余流量低于10GB就推荐一个流量包。"),
    ("场景4: TASK 人工审批", "帮我办理30GB流量包。"),
    ("场景5: QUERY 漫游查询", "查一下我的国际漫游状态。"),
    ("场景6: QUERY 订单查询", "查一下订单 ord_20260601_001。"),
    ("场景7: UNKNOWN 兜底", "今天天气怎么样？"),
]


def run_scenario(index: int, graph) -> None:
    title, query = SCENARIOS[index - 1]
    thread_id = f"demo_conv_{index}"
    print(f"\n{'=' * 60}\n{title}\n用户: {query}\n{'=' * 60}")
    result = graph.invoke(
        {
            "query": query,
            "user_id": "user_001",
            "conversation_id": thread_id,
            "trace_id": f"demo_trace_{index}",
        },
        config={"configurable": {"thread_id": thread_id}},
    )
    print(f"intent: {result.get('intent')}")
    if result.get("tool_calls"):
        print("tool_calls:")
        for tc in result["tool_calls"]:
            print(f"  - {tc.get('step_id')} {tc.get('tool_name')} [{tc.get('status')}]")
    if result.get("pending_approval"):
        print(f"⚠️ 需要审批: {json.dumps(result['pending_approval'], ensure_ascii=False)}")
    print(f"回答: {result.get('final_answer')}")
    if result.get("error"):
        print(f"ERROR: {result['error']}")


def main() -> int:
    indices = [int(sys.argv[1])] if len(sys.argv) > 1 else list(range(1, len(SCENARIOS) + 1))
    graph = build_graph(checkpointer=MemorySaver())
    for i in indices:
        run_scenario(i, graph)
    return 0


if __name__ == "__main__":
    sys.exit(main())
