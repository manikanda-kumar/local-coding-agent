from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, str]:
        message = {"role": self.role, "content": self.content}
        if self.name is not None:
            message["name"] = self.name
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        return message


@dataclass(frozen=True, slots=True)
class ChatRequest:
    messages: tuple[ChatMessage, ...]
    temperature: float | None = None
    max_tokens: int | None = None
    tools: tuple[dict[str, Any], ...] = ()
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
    tool_calls: tuple[dict[str, Any], ...] = ()
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    context_window: int
    supports_tools: bool = False
    supports_streaming: bool = True
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
