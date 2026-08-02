from __future__ import annotations

import json
import re
from typing import Any, Self

import httpx

from agent_runtime.errors import IncompleteModelOutputError, InvalidModelOutputError
from agent_runtime.models import ChatRequest, ChatResponse, ToolCall, Usage
from agent_runtime.request_validation import validate_json, validate_request_options


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
        extensions = validate_request_options(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            top_k=request.top_k,
            seed=request.seed,
            stop=request.stop,
            timeout=request.timeout,
            reasoning_mode=request.reasoning_mode,
            reasoning_field=request.reasoning_field,
            extensions=request.extensions,
        )
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                message.to_dict(
                    reasoning_field=request.reasoning_field, reasoning_mode=request.reasoning_mode
                )
                for message in request.messages
            ],
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.top_k is not None:
            payload["top_k"] = request.top_k
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.stop:
            payload["stop"] = list(request.stop)
        if request.tools:
            payload["tools"] = [tool.to_dict() for tool in request.tools]
        payload.update(extensions)

        kwargs: dict[str, Any] = {"json": payload}
        if request.timeout is not None:
            kwargs["timeout"] = request.timeout
        response = self._client.post("chat/completions", **kwargs)
        response.raise_for_status()
        data = response.json()
        return self._parse_response(
            data, fallback_model=model, reasoning_mode=request.reasoning_mode
        )

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

    def _parse_response(
        self, data: dict[str, Any], *, fallback_model: str, reasoning_mode: str = "preserve"
    ) -> ChatResponse:
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
        content_value = message.get("content")
        if content_value is not None and not isinstance(content_value, str):
            raise InvalidModelOutputError(f"{self.name} returned non-string content")
        content = content_value or ""
        finish_reason = choice.get("finish_reason")
        aliases = [key for key in ("reasoning", "reasoning_content") if key in message]
        if len(aliases) > 1:
            raise InvalidModelOutputError(f"{self.name} returned conflicting reasoning aliases")
        reasoning_field = aliases[0] if aliases else None
        reasoning = message.get(reasoning_field) if reasoning_field else None
        if reasoning is not None and not isinstance(reasoning, str):
            raise InvalidModelOutputError(f"{self.name} returned non-string reasoning")
        wire_content = None
        if reasoning_mode in {"preserve", "tagged_content"} and content.lstrip().startswith(
            "<think>"
        ):
            wire_content = content
            tagged_reasoning, content = self._separate_thinking(content, bool(tool_calls))
            if reasoning is not None and tagged_reasoning is not None:
                raise InvalidModelOutputError(
                    f"{self.name} returned conflicting reasoning protocols"
                )
            reasoning = tagged_reasoning
        elif reasoning_mode == "tagged_content" and reasoning is not None:
            raise InvalidModelOutputError(
                f"{self.name} returned structured reasoning in tagged-content mode"
            )
        if (
            not content
            and not tool_calls
            and (reasoning is not None or message.get("reasoning_details"))
        ):
            if finish_reason == "length":
                raise IncompleteModelOutputError(
                    f"{self.name} exhausted its output budget before producing a response"
                )
            raise InvalidModelOutputError(
                f"{self.name} returned reasoning without a final answer or tool call"
            )
        if finish_reason == "length" and not content and not tool_calls:
            raise IncompleteModelOutputError(
                f"{self.name} exhausted its output budget before producing a response"
            )
        usage = data.get("usage") or {}
        try:
            return ChatResponse(
                content=content,
                model=data.get("model") or fallback_model,
                provider=self.name,
                finish_reason=finish_reason,
                reasoning=reasoning,
                reasoning_details=self._parse_reasoning_details(message.get("reasoning_details")),
                usage=Usage(
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                ),
                tool_calls=tool_calls,
                raw=data,
                reasoning_field=reasoning_field,
                wire_content=wire_content,
            )
        except (TypeError, ValueError) as error:
            raise InvalidModelOutputError(
                f"{self.name} returned invalid reasoning: {error}"
            ) from error

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
            if isinstance(arguments, dict):
                try:
                    arguments = json.dumps(
                        arguments, sort_keys=True, separators=(",", ":"), allow_nan=False
                    )
                except ValueError as error:
                    raise InvalidModelOutputError(
                        f"{self.name} returned invalid tool arguments"
                    ) from error
            elif not isinstance(arguments, str):
                raise InvalidModelOutputError(
                    f"{self.name} returned tool arguments that are not an object or string"
                )
            tool_calls.append(ToolCall(id=identifier, name=name, arguments=arguments))
        return tuple(tool_calls)

    def _parse_reasoning_details(self, value: object) -> tuple[dict[str, Any], ...]:
        if value is None:
            return ()
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise InvalidModelOutputError(f"{self.name} returned invalid reasoning_details")
        try:
            return tuple(
                validate_json(
                    value,
                    label="reasoning_details",
                    max_bytes=1024 * 1024,
                    max_depth=12,
                    max_items=4096,
                    max_string=1024 * 1024,
                )
            )
        except ValueError as error:
            raise InvalidModelOutputError(
                f"{self.name} returned invalid reasoning_details: {error}"
            ) from error

    def _separate_thinking(self, content: str, has_tool_calls: bool) -> tuple[str | None, str]:
        if not content.lstrip().startswith("<think>"):
            return None, content
        match = re.match(r"\s*<think>(.*?)</think>(.*)\Z", content, re.DOTALL)
        if match is None:
            raise IncompleteModelOutputError(f"{self.name} returned an unclosed thinking block")
        reasoning, final = match.group(1).strip(), match.group(2).strip()
        if not final and not has_tool_calls:
            raise InvalidModelOutputError(
                f"{self.name} returned thinking without a final answer or tool call"
            )
        return reasoning, final
