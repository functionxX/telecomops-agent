"""会话历史注入：把多轮对话历史组织为 LLM 上下文。

只注入「对话历史」，不注入 workflow 内部状态——两者职责分离
（见 app/db/repositories/memory_repo.py 的说明）。
"""

from app.db.repositories import memory_repo
from app.db.session import session_scope

MAX_HISTORY_MESSAGES = 20


def load_history(conversation_id: str) -> list[dict]:
    """加载会话最近消息（不含当前轮）。"""
    with session_scope() as db:
        return memory_repo.get_recent_messages(db, conversation_id, limit=MAX_HISTORY_MESSAGES)


def history_as_text(history: list[dict]) -> str:
    """历史转文本（注入 Planner/Router 的上下文）。"""
    if not history:
        return "（无历史对话）"
    lines = [f"{m['role']}: {m['content']}" for m in history]
    return "\n".join(lines)
