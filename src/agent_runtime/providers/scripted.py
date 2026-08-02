from __future__ import annotations

from collections import deque

from agent_runtime.models import ChatRequest, ChatResponse


class ScriptedProvider:
    """Deterministic provider for agent-loop and state-machine tests."""

    def __init__(self, responses: tuple[ChatResponse, ...], *, name: str = "scripted") -> None:
        self._name = name
        self._responses = deque(responses)
        self.requests: list[tuple[str, ChatRequest]] = []

    @property
    def name(self) -> str:
        return self._name

    def chat(self, model: str, request: ChatRequest) -> ChatResponse:
        self.requests.append((model, request))
        if not self._responses:
            raise RuntimeError("scripted provider has no responses remaining")
        return self._responses.popleft()

    def close(self) -> None:
        pass
