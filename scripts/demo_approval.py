"""Human-in-the-loop 演示：中断 → checkpoint 落库 → 批准恢复 → 执行 + 幂等验证。

LangGraph 1.x 语义：interrupt() 不抛异常，invoke 正常返回，
状态中的 __interrupt__ 字段携带中断信息；恢复用 Command(resume=...)。

用法：uv run python scripts/demo_approval.py
"""

import json
import sys
import uuid

from app.agent.graph import get_workflow
from app.core.config import settings
from app.core.logging import setup_logging
from app.db.models import Order
from app.db.session import SessionLocal, engine
from langgraph.types import Command
from sqlalchemy import select, text

setup_logging(settings.log_level)

# 每次运行使用全新 thread（旧 checkpoint 中可能残留 human_decision 状态）
THREAD_ID = f"demo_approval_conv_{uuid.uuid4().hex[:8]}"


def _order_count() -> int:
    with SessionLocal() as db:
        return len(db.execute(select(Order)).scalars().all())


def _latest_order() -> Order:
    with SessionLocal() as db:
        order = db.execute(select(Order).order_by(Order.created_at.desc()).limit(1)).scalars().first()
    if order is None:
        raise RuntimeError("订单表为空")
    return order


def main() -> int:
    workflow = get_workflow()
    config = {"configurable": {"thread_id": THREAD_ID}}
    before_count = _order_count()

    print("=" * 60)
    print("步骤 1：发起高风险任务（办理 30GB 流量包）")
    print("=" * 60)
    state = workflow.invoke(
        {
            "query": "帮我办理30GB流量包。",
            "user_id": "user_001",
            "conversation_id": THREAD_ID,
            "trace_id": "demo_approval_trace",
        },
        config=config,
    )
    interrupts = state.get("__interrupt__", [])
    assert interrupts, "应当产生中断"
    payload = interrupts[0].value
    approval_id = payload.get("approval_id", "")
    print("⏸  Workflow 已中断（INTERRUPTED），等待人工审批")
    print(f"   {json.dumps(payload, ensure_ascii=False, indent=2)}")

    # checkpoint 落库验证
    with engine.connect() as conn:
        n = conn.execute(
            text("SELECT count(*) FROM workflow_checkpoints WHERE thread_id = :t"),
            {"t": THREAD_ID},
        ).scalar_one()
    print(f"\n步骤 2：checkpoint 已持久化到 PostgreSQL（workflow_checkpoints 行数={n}）")
    assert n > 0, "checkpoint 未落库"
    assert _order_count() == before_count, "中断时不应产生订单"

    print("\n" + "=" * 60)
    print("步骤 3：用户批准（Command(resume=approved) 从 checkpoint 恢复）")
    print("=" * 60)
    final_state = workflow.invoke(
        Command(resume={"approval_id": approval_id, "decision": "approved"}),
        config=config,
    )
    print(f"✅ RESUME 完成，最终回答:\n{final_state.get('final_answer')}")

    order = _latest_order()
    print(
        f"\n步骤 4：数据库订单验证：order_id={order.order_id} item={order.item_name} "
        f"amount={order.amount} idempotency_key={order.idempotency_key}"
    )
    assert _order_count() == before_count + 1, "批准后应恰好产生 1 张订单"

    print("\n" + "=" * 60)
    print("步骤 5：幂等验证——同一对话重新发起同语义请求，不产生重复订单")
    print("=" * 60)
    state2 = workflow.invoke(
        {
            "query": "帮我办理30GB流量包。",
            "user_id": "user_001",
            "conversation_id": THREAD_ID,
            "trace_id": "demo_approval_trace_2",
        },
        config=config,
    )
    interrupts2 = state2.get("__interrupt__", [])
    if interrupts2:
        final_state2 = workflow.invoke(
            Command(resume={"approval_id": interrupts2[0].value["approval_id"], "decision": "approved"}),
            config=config,
        )
        print(f"回答: {final_state2.get('final_answer')}")
    count_after = _order_count()
    print(f"订单数：重试前 {before_count + 1} → 重试后 {count_after}")
    assert count_after == before_count + 1, "幂等失效：产生了重复订单！"
    print("\n✅ Human-in-the-loop + checkpoint 持久化 + 幂等 全链路验证通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
