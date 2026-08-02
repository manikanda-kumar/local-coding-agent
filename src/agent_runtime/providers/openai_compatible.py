from __future__ import annotations

from typing import Any, Self

import httpx

from agent_runtime.models import ChatRequest, ChatResponse, Usage


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
            payload["tools"] = list(request.tools)

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
            raise ValueError(f"{self.name} returned no completion choices")

        choice = choices[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        return ChatResponse(
            content=message.get("content") or "",
            model=data.get("model") or fallback_model,
            provider=self.name,
            finish_reason=choice.get("finish_reason"),
            reasoning=message.get("reasoning") or message.get("reasoning_content"),
            reasoning_details=tuple(message.get("reasoning_details") or ()),
            usage=Usage(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            ),
            tool_calls=tuple(message.get("tool_calls") or ()),
            raw=data,
        )
