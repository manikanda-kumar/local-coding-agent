from __future__ import annotations

from typing import Protocol

from agent_runtime.models import ChatRequest, ChatResponse


class Provider(Protocol):
    @property
    def name(self) -> str: ...

    def chat(self, model: str, request: ChatRequest) -> ChatResponse: ...

    def close(self) -> None: ...
