from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from agent_runtime.errors import ToolArgumentsError

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

    def __post_init__(self) -> None:
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id")
        if self.role != "tool" and self.tool_call_id is not None:
            raise ValueError("tool_call_id is only valid for tool messages")
        if self.role != "assistant" and self.tool_calls:
            raise ValueError("tool_calls are only valid for assistant messages")
        if self.content is None and not (self.role == "assistant" and self.tool_calls):
            raise ValueError("content can only be null for assistant tool calls")

    def to_dict(self) -> dict[str, Any]:
        message = {"role": self.role, "content": self.content}
        if self.name is not None:
            message["name"] = self.name
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            message["tool_calls"] = [tool_call.to_dict() for tool_call in self.tool_calls]
        return message


@dataclass(frozen=True, slots=True)
class ChatRequest:
    messages: tuple[ChatMessage, ...]
    temperature: float | None = None
    max_tokens: int | None = None
    tools: tuple[ToolDefinition, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


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

    def to_assistant_message(self) -> ChatMessage:
        content = self.content or None if self.tool_calls else self.content
        return ChatMessage(role="assistant", content=content, tool_calls=self.tool_calls)


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    context_window: int
    supports_tools: bool = False
    supports_streaming: bool = False
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
