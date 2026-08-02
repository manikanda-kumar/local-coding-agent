from __future__ import annotations

from typing import Any, Self

import httpx

from agent_runtime.errors import IncompleteModelOutputError, InvalidModelOutputError
from agent_runtime.models import ChatRequest, ChatResponse, ToolCall, Usage


class OpenAICompatibleProvider:
    """Client for OpenAI-compatible APIs such as OpenRouter, vLLM, and SGLang."""

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 60.0,
        headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        default_headers = dict(headers or {})
        if api_key:
            default_headers["Authorization"] = f"Bearer {api_key}"
        self._name = name
        self._client = httpx.Client(
            base_url=base_url.rstrip("/") + "/",
            headers=default_headers,
            timeout=timeout,
            transport=transport,
        )

    @property
    def name(self) -> str:
        return self._name

    def chat(self, model: str, request: ChatRequest) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [message.to_dict() for message in request.messages],
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = [tool.to_dict() for tool in request.tools]

        response = self._client.post("chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        return self._parse_response(data, fallback_model=model)

    def list_models(self) -> tuple[str, ...]:
        response = self._client.get("models")
        response.raise_for_status()
        return tuple(model["id"] for model in response.json().get("data", ()))

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _parse_response(self, data: dict[str, Any], *, fallback_model: str) -> ChatResponse:
        choices = data.get("choices") or []
        if not choices:
            raise InvalidModelOutputError(f"{self.name} returned no completion choices")

        choice = choices[0]
        if not isinstance(choice, dict):
            raise InvalidModelOutputError(f"{self.name} returned an invalid completion choice")
        message = choice.get("message") or {}
        if not isinstance(message, dict):
            raise InvalidModelOutputError(f"{self.name} returned an invalid completion message")
        tool_calls = self._parse_tool_calls(message.get("tool_calls") or ())
        content = message.get("content") or ""
        finish_reason = choice.get("finish_reason")
        if finish_reason == "length" and not content and not tool_calls:
            raise IncompleteModelOutputError(
                f"{self.name} exhausted its output budget before producing a response"
            )
        usage = data.get("usage") or {}
        return ChatResponse(
            content=content,
            model=data.get("model") or fallback_model,
            provider=self.name,
            finish_reason=finish_reason,
            reasoning=message.get("reasoning") or message.get("reasoning_content"),
            reasoning_details=tuple(message.get("reasoning_details") or ()),
            usage=Usage(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            ),
            tool_calls=tool_calls,
            raw=data,
        )

    def _parse_tool_calls(self, values: object) -> tuple[ToolCall, ...]:
        if not isinstance(values, (list, tuple)):
            raise InvalidModelOutputError(f"{self.name} returned invalid tool_calls")

        tool_calls = []
        for value in values:
            if not isinstance(value, dict) or not isinstance(value.get("function"), dict):
                raise InvalidModelOutputError(f"{self.name} returned an invalid tool call")
            function = value["function"]
            identifier = value.get("id")
            name = function.get("name")
            arguments = function.get("arguments")
            if not all(isinstance(item, str) and item for item in (identifier, name)):
                raise InvalidModelOutputError(f"{self.name} returned an incomplete tool call")
            if not isinstance(arguments, str):
                raise InvalidModelOutputError(f"{self.name} returned non-string tool arguments")
            tool_calls.append(ToolCall(id=identifier, name=name, arguments=arguments))
        return tuple(tool_calls)
