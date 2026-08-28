"""LLMClient：统一的 LLM 访问层（核心，业务代码只依赖它）。

职责：chat / structured_output / stream / token_usage / timeout /
指数退避（仅 transient 错误）/ 统一异常（LLMError）。
不散落在各模块；CustomDeepSeekChatModel 只是它上面的 LangChain 适配层。
"""

import json
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from functools import lru_cache
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from app.core.config import settings
from app.core.exceptions import LLMError
from app.llm.schemas import ChatMessage, ChatResponse, LLMUsage
from app.observability import metrics
from app.observability.tracing import start_span

T = TypeVar("T", bound=BaseModel)


class LLMClient(ABC):
    """LLM 访问抽象。Mock 与真实实现共用同一接口。"""

    model: str

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        purpose: str = "chat",
    ) -> ChatResponse: ...

    @abstractmethod
    def stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        purpose: str = "chat",
    ) -> Iterator[str]: ...

    def structured_output(
        self,
        messages: list[ChatMessage],
        schema: type[T],
        *,
        purpose: str = "structured",
        temperature: float = 0.0,
    ) -> T:
        """JSON mode + Pydantic 校验；一次修复重试；仍失败抛 LLMError。"""
        # DeepSeek/OpenAI 要求 json_object 模式时 prompt 中出现 "json" 字样；
        # 同时把目标 JSON Schema 注入 prompt（显著降低缺字段/类型错误）
        messages = _ensure_json_hint(messages, schema)
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                resp = self.chat(
                    messages, temperature=temperature, json_mode=True, purpose=purpose
                )
                data = json.loads(_extract_json(resp.content))
                return schema.model_validate(data)
            except (json.JSONDecodeError, PydanticValidationError, LLMError) as exc:
                last_error = exc
                if attempt == 0:
                    messages = messages + [
                        ChatMessage(
                            role="assistant",
                            content=f"（上次输出无法解析：{exc}，请严格输出合法 JSON）",
                        )
                    ]
        raise LLMError(f"结构化输出校验失败（schema={schema.__name__}）：{last_error}")


class DeepSeekClient(LLMClient):
    """DeepSeek API 直连实现（httpx + 超时 + 指数退避）。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float,
        max_retries: int,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self.model = model
        self._timeout = timeout
        self._max_retries = max_retries

    # ---------- 协议实现 ----------

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        purpose: str = "chat",
    ) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.model_dump(exclude_none=True) for m in messages],
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if tools is not None:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice

        start = time.perf_counter()
        try:
            with start_span(
                "llm.chat",
                {"model": self.model, "purpose": purpose, "json_mode": json_mode},
            ) as span:
                data = self._post("/chat/completions", payload)
                usage = data.get("usage", {})
                span.set_attribute("tokens.prompt", usage.get("prompt_tokens", 0))
                span.set_attribute("tokens.completion", usage.get("completion_tokens", 0))
        except Exception as exc:
            metrics.llm_error_count.labels(self.model, purpose).inc()
            raise exc
        metrics.llm_latency.labels(self.model, purpose).observe(time.perf_counter() - start)

        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        return ChatResponse(
            content=msg.get("content") or "",
            usage=LLMUsage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
            model=data.get("model", self.model),
            finish_reason=choice.get("finish_reason"),
            tool_calls=msg.get("tool_calls"),
        )

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        purpose: str = "chat",
    ) -> Iterator[str]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.model_dump(exclude_none=True) for m in messages],
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        start = time.perf_counter()
        try:
            with httpx.Client(timeout=httpx.Timeout(self._timeout, read=self._timeout * 3)) as c:
                with c.stream(
                    "POST",
                    f"{self._base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                ) as resp:
                    if resp.status_code >= 400:
                        raise LLMError(
                            f"DeepSeek stream 请求失败 HTTP {resp.status_code}: {resp.text[:200]}",
                            retryable=resp.status_code in self._TRANSIENT_STATUS,
                        )
                    for line in resp.iter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        chunk = line[len("data:") :].strip()
                        if chunk == "[DONE]":
                            break
                        delta = json.loads(chunk).get("choices", [{}])[0].get("delta", {})
                        if delta.get("content"):
                            yield delta["content"]
        except LLMError:
            metrics.llm_error_count.labels(self.model, purpose).inc()
            raise
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            metrics.llm_error_count.labels(self.model, purpose).inc()
            raise LLMError(f"DeepSeek stream 失败: {exc}", retryable=True) from exc
        finally:
            metrics.llm_latency.labels(self.model, purpose).observe(time.perf_counter() - start)

    # ---------- 内部 ----------

    _TRANSIENT_STATUS = {429, 500, 502, 503, 504}

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST + 指数退避（仅 transient：429/5xx/网络错误）。"""
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                with httpx.Client(timeout=httpx.Timeout(self._timeout)) as client:
                    resp = client.post(
                        f"{self._base_url}{path}", headers=self._headers(), json=payload
                    )
                if resp.status_code < 400:
                    return resp.json()
                if resp.status_code in self._TRANSIENT_STATUS:
                    last_exc = LLMError(
                        f"DeepSeek HTTP {resp.status_code}: {resp.text[:200]}", retryable=True
                    )
                else:
                    raise LLMError(
                        f"DeepSeek HTTP {resp.status_code}: {resp.text[:200]}",
                        retryable=False,
                    )
            except (httpx.HTTPError, httpx.TimeoutException) as exc:
                last_exc = LLMError(f"DeepSeek 网络错误: {exc}", retryable=True)
            if attempt < self._max_retries and isinstance(last_exc, LLMError) and last_exc.retryable:
                time.sleep(0.5 * (2**attempt) + random.uniform(0, 0.1))
        raise last_exc  # type: ignore[misc]

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }


def _ensure_json_hint(messages: list[ChatMessage], schema: type[BaseModel]) -> list[ChatMessage]:
    """在 system 消息中注入 JSON 输出要求与目标 Schema（json_object 模式的前置条件）。"""
    hint = (
        "请以合法 JSON 格式输出（不要输出任何解释文字或 markdown 围栏）。"
        f"输出必须符合以下 JSON Schema：{json.dumps(schema.model_json_schema(), ensure_ascii=False)}"
    )
    if messages and messages[0].role == "system":
        if "JSON" not in messages[0].content:
            messages[0] = ChatMessage(role="system", content=messages[0].content + "\n" + hint)
        return messages
    return [ChatMessage(role="system", content=hint)] + messages


def _extract_json(text: str) -> str:
    """从可能带 markdown 围栏的输出中提取 JSON。"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return text


@lru_cache
def get_llm_client() -> LLMClient:
    """LLMClient 工厂：Mock（确定性，CI/离线）或真实 DeepSeek。"""
    from app.llm.mock import MockLLMClient

    if settings.mock_llm:
        return MockLLMClient(model=f"mock-{settings.deepseek_model}")
    return DeepSeekClient(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_model,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
    )
