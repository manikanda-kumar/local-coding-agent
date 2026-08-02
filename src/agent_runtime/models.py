from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

from agent_runtime.errors import ToolArgumentsError
from agent_runtime.request_validation import validate_json

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str

    def parse_arguments(self) -> dict[str, Any]:
        try:
            parsed = json.loads(self.arguments)
        except json.JSONDecodeError as error:
            raise ToolArgumentsError(self.id, f"invalid JSON arguments for {self.name}") from error
        if not isinstance(parsed, dict):
            raise ToolArgumentsError(self.id, f"arguments for {self.name} must be a JSON object")
        return parsed

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass(frozen=True, slots=True)
class ToolResult:
    tool_call_id: str
    content: str
    name: str | None = None

    def to_message(self) -> ChatMessage:
        return ChatMessage(
            role="tool",
            content=self.content,
            name=self.name,
            tool_call_id=self.tool_call_id,
        )


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Role
    content: str | None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    reasoning: str | None = None
    reasoning_details: tuple[dict[str, Any], ...] = ()
    reasoning_field: Literal["reasoning", "reasoning_content"] | None = None
    wire_content: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id")
        if self.role != "tool" and self.tool_call_id is not None:
            raise ValueError("tool_call_id is only valid for tool messages")
        if self.role != "assistant" and self.tool_calls:
            raise ValueError("tool_calls are only valid for assistant messages")
        if self.role != "assistant" and (
            self.reasoning is not None
            or self.reasoning_details
            or self.reasoning_field is not None
            or self.wire_content is not None
        ):
            raise ValueError("reasoning is only valid for assistant messages")
        if self.reasoning is not None and not isinstance(self.reasoning, str):
            raise ValueError("reasoning must be a string")
        if not isinstance(self.reasoning_details, tuple) or not all(
            isinstance(detail, dict) for detail in self.reasoning_details
        ):
            raise ValueError("reasoning_details must contain objects")
        if self.reasoning_field not in {None, "reasoning", "reasoning_content"}:
            raise ValueError("unsupported reasoning field")
        if self.wire_content is not None and not isinstance(self.wire_content, str):
            raise ValueError("wire content must be a string")
        bounded = validate_json(
            {"reasoning": self.reasoning, "details": self.reasoning_details},
            label="assistant reasoning",
            max_bytes=1024 * 1024,
            max_depth=12,
            max_items=4096,
            max_string=1024 * 1024,
        )
        object.__setattr__(self, "reasoning_details", tuple(bounded["details"]))
        if self.wire_content is not None:
            validate_json(
                self.wire_content,
                label="assistant wire content",
                max_bytes=1024 * 1024,
                max_string=1024 * 1024,
            )
        if self.content is None and not (self.role == "assistant" and self.tool_calls):
            raise ValueError("content can only be null for assistant tool calls")

    def to_dict(
        self, *, reasoning_field: str = "reasoning", reasoning_mode: str = "preserve"
    ) -> dict[str, Any]:
        if reasoning_field not in {"reasoning", "reasoning_content"}:
            raise ValueError("unsupported reasoning field")
        content = (
            self.wire_content
            if self.wire_content is not None and reasoning_mode != "structured"
            else self.content
        )
        message = {"role": self.role, "content": content}
        if self.name is not None:
            message["name"] = self.name
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            message["tool_calls"] = [tool_call.to_dict() for tool_call in self.tool_calls]
        selected_field = (
            reasoning_field
            if reasoning_mode == "structured"
            else self.reasoning_field or reasoning_field
        )
        emit_structured = reasoning_mode == "structured" or (
            reasoning_mode == "preserve" and self.wire_content is None
        )
        if self.reasoning is not None and emit_structured:
            message[selected_field] = self.reasoning
        if self.reasoning_details and emit_structured:
            message["reasoning_details"] = list(self.reasoning_details)
        return message


@dataclass(frozen=True, slots=True)
class ChatRequest:
    messages: tuple[ChatMessage, ...]
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    stop: tuple[str, ...] = ()
    timeout: float | None = None
    reasoning_field: Literal["reasoning", "reasoning_content"] = "reasoning"
    reasoning_mode: Literal["preserve", "structured", "tagged_content"] = "preserve"
    extensions: dict[str, Any] = field(default_factory=dict)
    tools: tuple[ToolDefinition, ...] = ()


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class ChatResponse:
    content: str
    model: str
    provider: str
    finish_reason: str | None = None
    reasoning: str | None = None
    reasoning_details: tuple[dict[str, Any], ...] = ()
    usage: Usage = Usage()
    tool_calls: tuple[ToolCall, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    reasoning_field: Literal["reasoning", "reasoning_content"] | None = None
    wire_content: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError("response content must be a string")
        if self.reasoning is not None and not isinstance(self.reasoning, str):
            raise ValueError("response reasoning must be a string")
        bounded = validate_json(
            {"reasoning": self.reasoning, "details": self.reasoning_details},
            label="response reasoning",
            max_bytes=1024 * 1024,
            max_depth=12,
            max_items=4096,
            max_string=1024 * 1024,
        )
        object.__setattr__(self, "reasoning_details", tuple(bounded["details"]))
        if self.wire_content is not None:
            validate_json(
                self.wire_content,
                label="response wire content",
                max_bytes=1024 * 1024,
                max_string=1024 * 1024,
            )
        object.__setattr__(self, "raw", deepcopy(self.raw))

    def to_assistant_message(self) -> ChatMessage:
        content = self.content or None if self.tool_calls else self.content
        return ChatMessage(
            role="assistant",
            content=content,
            tool_calls=self.tool_calls,
            reasoning=self.reasoning,
            reasoning_details=self.reasoning_details,
            reasoning_field=self.reasoning_field,
            wire_content=self.wire_content,
        )


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    context_window: int
    supports_tools: bool = False
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
