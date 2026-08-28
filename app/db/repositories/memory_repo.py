"""Agent 记忆数据访问：会话、消息、运行记录、工具调用审计。

三层记忆的区分（spec 第十一节）：
- Conversation History → conversations / messages 表（多轮对话上下文）
- Workflow State     → workflow_checkpoints 表 + LangGraph State（中断可恢复）
- Long-term User Info→ customer_profiles 等业务表（不是聊天记录）
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import AgentRun, Conversation, Message, ToolCallRecord


def get_or_create_conversation(db: Session, conversation_id: str | None, user_id: str) -> Conversation:
    """获取或创建会话。"""
    if conversation_id:
        conv = db.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        ).scalar_one_or_none()
        if conv is not None:
            return conv
    conv = Conversation(id=conversation_id or f"conv_{uuid.uuid4().hex[:12]}", user_id=user_id)
    db.add(conv)
    db.commit()
    return conv


def add_message(db: Session, conversation_id: str, role: str, content: str) -> Message:
    message = Message(conversation_id=conversation_id, role=role, content=content)
    db.add(message)
    db.commit()
    return message


def get_recent_messages(db: Session, conversation_id: str, limit: int = 20) -> list[dict]:
    """按时间正序返回最近 N 条消息（注入 LLM 上下文用）。"""
    rows = db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    ).scalars().all()
    return [
        {"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()}
        for m in reversed(rows)
    ]


def start_agent_run(
    db: Session, *, conversation_id: str, user_id: str, trace_id: str | None
) -> AgentRun:
    run = AgentRun(
        id=f"run_{uuid.uuid4().hex[:12]}",
        conversation_id=conversation_id,
        user_id=user_id,
        status="running",
        trace_id=trace_id,
    )
    db.add(run)
    db.commit()
    return run


def finish_agent_run(
    db: Session, run_id: str, *, status: str, intent: str | None = None, error: str | None = None
) -> None:
    run = db.execute(select(AgentRun).where(AgentRun.id == run_id)).scalar_one_or_none()
    if run is None:
        return
    run.status = status
    run.intent = intent
    run.error = error
    run.finished_at = datetime.now(UTC)
    db.commit()


def record_tool_call(
    db: Session,
    *,
    agent_run_id: str,
    tool_name: str,
    arguments: dict,
    result: dict | None,
    status: str,
    error: str | None = None,
    duration_ms: float | None = None,
) -> ToolCallRecord:
    record = ToolCallRecord(
        agent_run_id=agent_run_id,
        tool_name=tool_name,
        arguments=arguments,
        result=result,
        status=status,
        error=error,
        duration_ms=duration_ms,
    )
    db.add(record)
    db.commit()
    return record
