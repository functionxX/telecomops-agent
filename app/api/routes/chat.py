"""Chat API：普通 JSON 响应 + SSE 流式。

- POST /api/v1/chat        普通请求（含 interrupt 状态返回 approval 信息）
- POST /api/v1/chat/stream SSE：workflow 生命周期事件 + 答案 token 流
- POST /api/v1/feedback    用户反馈（写入 evaluation_results）
- GET  /api/v1/conversations/{id}  会话详情（消息 + 运行记录）
"""

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from app.agent.graph import get_workflow
from app.api.schemas.chat import (
    ApprovalInfo,
    ChatRequest,
    ChatResponse,
    ConversationDetail,
    FeedbackRequest,
    FeedbackResponse,
)
from app.core.exceptions import AppError
from app.core.logging import get_logger
from app.db.models import AgentRun, EvaluationResult
from app.db.repositories import memory_repo
from app.db.session import session_scope
from app.guardrails.input import check_input
from app.memory.conversation import load_history
from app.observability.tracing import get_trace_id

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])


def _build_initial_state(req: ChatRequest, trace_id: str | None) -> dict[str, Any]:
    history = load_history(req.conversation_id) if req.conversation_id else []
    return {
        "query": req.message,
        "user_id": req.user_id,
        "conversation_id": req.conversation_id or "",
        "trace_id": trace_id or "",
        "messages": history,  # 会话历史（Memory：Conversation History 层）
    }


def _persist_run(
    req: ChatRequest,
    conversation_id: str,
    final_state: dict[str, Any],
    *,
    status: str,
    run_id: str,
    intent: str | None,
    error: str | None = None,
) -> None:
    """持久化会话消息 + 工具调用审计 + agent_run 状态。"""
    with session_scope() as db:
        memory_repo.add_message(db, conversation_id, "user", req.message)
        answer = final_state.get("final_answer")
        if answer:
            memory_repo.add_message(db, conversation_id, "assistant", answer)
        for tc in final_state.get("tool_calls", []):
            memory_repo.record_tool_call(
                db,
                agent_run_id=run_id,
                tool_name=tc.get("tool_name", ""),
                arguments=tc.get("arguments", {}),
                result=tc.get("result"),
                status=tc.get("status", ""),
                error=tc.get("error"),
                duration_ms=tc.get("duration_ms"),
            )
        memory_repo.finish_agent_run(
            db, run_id, status=status, intent=intent, error=error
        )


def _to_response(
    req: ChatRequest,
    conversation_id: str,
    state: dict[str, Any],
) -> ChatResponse:
    approval = None
    if state.get("requires_human_approval") and not state.get("human_decision"):
        pending = state.get("pending_approval", {})
        approval = ApprovalInfo(
            approval_id=pending.get("approval_id", ""),
            tool_name=pending.get("tool_name", ""),
            summary=pending.get("summary", ""),
            status="pending",
        )
    return ChatResponse(
        conversation_id=conversation_id,
        answer=state.get("final_answer", ""),
        intent=state.get("intent", "UNKNOWN"),
        citations=state.get("citations", []),
        tool_calls=state.get("tool_calls", []),
        approval=approval,
        trace_id=state.get("trace_id"),
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    # Input Guardrail：注入/超长检查（第一道防线）
    guard = check_input(req.message)
    if not guard.ok:
        raise HTTPException(status_code=422, detail=guard.reason)

    trace_id = get_trace_id()
    with session_scope() as db:
        conversation = memory_repo.get_or_create_conversation(db, req.conversation_id, req.user_id)
        conversation_id = conversation.id
        run = memory_repo.start_agent_run(
            db, conversation_id=conversation_id, user_id=req.user_id, trace_id=trace_id
        )
        run_id = run.id

    state = _build_initial_state(req, trace_id)
    state["conversation_id"] = conversation_id
    state["metadata"] = {"agent_run_id": run_id}

    workflow = get_workflow()
    try:
        final_state = workflow.invoke(
            state, config={"configurable": {"thread_id": conversation_id}}
        )
    except AppError as exc:
        _persist_run(
            req, conversation_id, {}, status="failed", run_id=run_id,
            intent=None, error=exc.message,
        )
        raise HTTPException(status_code=exc.http_status, detail=exc.to_dict()) from exc

    if final_state.get("__interrupt__"):
        # Human-in-the-loop：Workflow 已暂停，审批请求已由审批节点落库
        _persist_run(
            req, conversation_id, final_state, status="interrupted",
            run_id=run_id, intent=final_state.get("intent"),
        )
    else:
        _persist_run(
            req, conversation_id, final_state, status="succeeded",
            run_id=run_id, intent=final_state.get("intent"),
            error=final_state.get("error"),
        )
    return _to_response(req, conversation_id, final_state)


def _sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    guard = check_input(req.message)
    if not guard.ok:
        raise HTTPException(status_code=422, detail=guard.reason)

    trace_id = get_trace_id()
    with session_scope() as db:
        conversation = memory_repo.get_or_create_conversation(db, req.conversation_id, req.user_id)
        conversation_id = conversation.id
        run = memory_repo.start_agent_run(
            db, conversation_id=conversation_id, user_id=req.user_id, trace_id=trace_id
        )
        run_id = run.id

    async def event_source() -> AsyncIterator[str]:
        state = _build_initial_state(req, trace_id)
        state["conversation_id"] = conversation_id
        state["metadata"] = {"agent_run_id": run_id, "stream": True}
        workflow = get_workflow()
        final_state: dict[str, Any] = {}
        try:
            async for mode, chunk in workflow.astream(
                state,
                config={"configurable": {"thread_id": conversation_id}},
                stream_mode=["custom", "values"],
            ):
                if mode == "custom":
                    yield _sse(chunk)
                elif mode == "values":
                    final_state = chunk
            yield _sse({"type": "workflow_completed", "conversation_id": conversation_id})
        except AppError as exc:
            yield _sse(
                {"type": "error", "code": exc.code, "message": exc.message}
            )
        finally:
            status = "succeeded"
            if final_state.get("__interrupt__"):
                status = "interrupted"
            _persist_run(
                req, conversation_id, final_state, status=status,
                run_id=run_id, intent=final_state.get("intent"),
                error=final_state.get("error"),
            )

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(req: FeedbackRequest) -> FeedbackResponse:
    """用户反馈：作为在线评估信号落库（evaluation_results）。"""
    with session_scope() as db:
        db.add(
            EvaluationResult(
                eval_type="feedback",
                dataset_name=req.conversation_id,
                metrics={
                    "rating": req.rating,
                    "comment": req.comment,
                    "feedback_type": req.feedback_type,
                },
            )
        )
    return FeedbackResponse()


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(conversation_id: str) -> ConversationDetail:
    with session_scope() as db:
        conv = db.execute(
            select(AgentRun).where(AgentRun.conversation_id == conversation_id)
        ).first()
        if conv is None:
            messages = memory_repo.get_recent_messages(db, conversation_id, limit=100)
            if not messages:
                raise HTTPException(status_code=404, detail=f"会话 {conversation_id} 不存在")
            return ConversationDetail(
                conversation_id=conversation_id, user_id="", messages=messages, runs=[]
            )
        runs = db.execute(
            select(AgentRun).where(AgentRun.conversation_id == conversation_id)
        ).scalars().all()
        return ConversationDetail(
            conversation_id=conversation_id,
            user_id=runs[0].user_id if runs else "",
            messages=memory_repo.get_recent_messages(db, conversation_id, limit=100),
            runs=[
                {
                    "run_id": r.id,
                    "status": r.status,
                    "intent": r.intent,
                    "error": r.error,
                    "created_at": r.created_at.isoformat(),
                }
                for r in runs
            ],
        )
