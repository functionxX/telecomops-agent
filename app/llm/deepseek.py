"""CustomDeepSeekChatModel：DeepSeek ↔ LangChain 接口适配层。

解决的问题：LangGraph / LangChain 生态需要 BaseChatModel 接口，
而我们的核心调用路径在 LLMClient（httpx + 重试 + 用量统计）。
本适配层很薄：把 LangChain 消息/工具格式翻译给 LLMClient，
把结果翻译回 AIMessage / tool_calls。
"""

from collections.abc import Iterator
from typing import Any, cast

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

from app.llm.client import LLMClient, get_llm_client
from app.llm.schemas import ChatMessage

_RoleMap = {"system": "system", "human": "user", "assistant": "assistant", "tool": "tool"}


def _to_openai_messages(messages: list[BaseMessage]) -> list[ChatMessage]:
    out: list[ChatMessage] = []
    for m in messages:
        if m.type == "ai" and getattr(m, "tool_calls", None):
            # tool call 消息在 OpenAI 协议里 content 可为空
            out.append(ChatMessage(role="assistant", content=_content_of(m)))
            continue
        role = cast(Any, _RoleMap.get(m.type, "user"))
        out.append(ChatMessage(role=role, content=_content_of(m)))
    return out


def _content_of(m: BaseMessage) -> str:
    c = m.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return str(c)


class CustomDeepSeekChatModel(BaseChatModel):
    """包装 LLMClient 的 LangChain ChatModel（含 bind_tools 支持）。"""

    client: LLMClient
    temperature: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "custom-deepseek"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.client.model}

    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """工具调用绑定：把工具 Schema 传给 LLMClient，返回可直接 invoke 的 Runnable。

        langchain-core 1.x 中 BaseChatModel.bind_tools 默认抛 NotImplementedError，
        支持工具调用的模型必须自行实现。
        """
        from langchain_core.runnables import RunnableLambda

        openai_tools = [convert_to_openai_tool(t) for t in tools]
        model = self

        def _invoke(messages: Any, config: Any = None, **kw: Any) -> AIMessage:
            if isinstance(messages, BaseMessage):
                msgs = [messages]
            else:
                msgs = list(messages)
            result = model._generate(
                msgs,
                tools=openai_tools,
                tool_choice=tool_choice,
                **{**kwargs, **kw},
            )
            return cast(AIMessage, result.generations[0].message)

        return RunnableLambda(_invoke)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")
        openai_tools = None
        if tools:
            openai_tools = [convert_to_openai_tool(t) for t in tools]

        resp = self.client.chat(
            _to_openai_messages(messages),
            temperature=self.temperature,
            max_tokens=kwargs.get("max_tokens"),
            tools=openai_tools,
            tool_choice=tool_choice,
            purpose="langchain",
        )

        tool_calls: list[dict[str, Any]] = []
        if resp.tool_calls:
            for tc in resp.tool_calls:
                fn = tc.get("function", {})
                import json

                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(
                    {"name": fn.get("name", ""), "args": args, "id": tc.get("id", "")}
                )
        # langchain-core 1.x 不允许 tool_calls=None，空则省略
        message_kwargs: dict[str, Any] = {"content": resp.content}
        if tool_calls:
            message_kwargs["tool_calls"] = tool_calls
        message = AIMessage(**message_kwargs)
        return ChatResult(
            generations=[ChatGeneration(message=message)],
            llm_output={
                "token_usage": resp.usage.model_dump(),
                "model_name": resp.model,
            },
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        # 同步模型流式：整体生成后单块产出。
        # 注意：SSE 的 token 级流式走 LLMClient.stream（直连），不走本适配层。
        result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        text = result.generations[0].message.content
        if text:
            yield ChatGenerationChunk(message=AIMessageChunk(content=text))


def get_chat_model() -> CustomDeepSeekChatModel:
    """ChatModel 工厂（QUERY 路径 bind_tools 用）。"""
    return CustomDeepSeekChatModel(client=get_llm_client(), temperature=0.0)
