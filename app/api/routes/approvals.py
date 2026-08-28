"""人工审批 API：查看待审批请求 + 批准/拒绝（触发 Workflow resume）。"""

from typing import Literal

from fastapi import APIRouter, HTTPException
from langgraph.types import Command
from pydantic import BaseModel

from app.agent.graph import get_workflow
from app.core.exceptions import AppError
from app.core.logging import get_logger
from app.db.repositories import approval_repo
from app.db.session import session_scope

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/approvals", tags=["approvals"])


class ApprovalDecisionRequest(BaseModel):
    decision: Literal["approve", "reject"]
    user_id: str | None = None


class ApprovalInfoResponse(BaseModel):
    approval_id: str
    conversation_id: str
    tool_name: str
    arguments: dict
    status: str
    summary: str | None = None


class ApprovalDecisionResponse(BaseModel):
    approval_id: str
    status: str
    conversation_id: str
    answer: str | None = None


@router.get("/{approval_id}", response_model=ApprovalInfoResponse)
async def get_approval(approval_id: str) -> ApprovalInfoResponse:
    with session_scope() as db:
        approval = approval_repo.get(db, approval_id)
    if approval is None:
        raise HTTPException(status_code=404, detail=f"审批 {approval_id} 不存在")
    return ApprovalInfoResponse(
        approval_id=approval.id,
        conversation_id=approval.conversation_id,
        tool_name=approval.tool_name,
        arguments=approval.arguments,
        status=approval.status,
    )


@router.post("/{approval_id}", response_model=ApprovalDecisionResponse)
async def decide_approval(
    approval_id: str, req: ApprovalDecisionRequest
) -> ApprovalDecisionResponse:
    """批准/拒绝，并用 Command(resume=...) 从 checkpoint 恢复 Workflow。"""
    with session_scope() as db:
        approval = approval_repo.get(db, approval_id)
    if approval is None:
        raise HTTPException(status_code=404, detail=f"审批 {approval_id} 不存在")
    if approval.status != "pending":
        raise HTTPException(status_code=409, detail=f"审批 {approval_id} 已处理（{approval.status}）")

    decision = "approved" if req.decision == "approve" else "rejected"
    resume_value = {"approval_id": approval_id, "decision": decision}
    workflow = get_workflow()
    try:
        final_state = workflow.invoke(
            Command(resume=resume_value),
            config={"configurable": {"thread_id": approval.conversation_id}},
        )
    except AppError as exc:
        raise HTTPException(status_code=exc.http_status, detail=exc.to_dict()) from exc
    if final_state.get("__interrupt__"):
        raise HTTPException(status_code=409, detail="恢复过程中产生了新的审批中断")

    return ApprovalDecisionResponse(
        approval_id=approval_id,
        status="approved" if decision == "approved" else "rejected",
        conversation_id=approval.conversation_id,
        answer=final_state.get("final_answer"),
    )
