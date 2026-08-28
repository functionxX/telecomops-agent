"""LLM 协议层数据结构（与具体供应商无关）。"""

from typing import Any, Literal

from pydantic import BaseModel


class LLMUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None


class ChatResponse(BaseModel):
    """LLMClient.chat 的统一返回。"""

    content: str
    usage: LLMUsage
    model: str
    finish_reason: str | None = None
    # bind_tools 场景下的工具调用（OpenAI 格式）
    tool_calls: list[dict[str, Any]] | None = None
