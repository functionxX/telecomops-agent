"""Chat API Schema。"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str = Field(description="用户ID，如 user_001")
    conversation_id: str | None = Field(
        default=None, description="会话ID；为空则新建会话"
    )
    message: str = Field(description="用户消息")


class ApprovalInfo(BaseModel):
    approval_id: str
    tool_name: str
    summary: str
    status: str


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    intent: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    approval: ApprovalInfo | None = None  # 高风险操作待审批时非空
    trace_id: str | None = None


class FeedbackRequest(BaseModel):
    conversation_id: str
    rating: int = Field(ge=1, le=5, description="1-5 分")
    comment: str | None = None
    feedback_type: Literal["answer", "rag", "tool"] = "answer"


class FeedbackResponse(BaseModel):
    status: str = "ok"


class ConversationDetail(BaseModel):
    conversation_id: str
    user_id: str
    messages: list[dict[str, Any]]
    runs: list[dict[str, Any]]
