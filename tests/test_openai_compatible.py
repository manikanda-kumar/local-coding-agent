import json

import httpx
import pytest

from agent_runtime import (
    ChatMessage,
    ChatRequest,
    IncompleteModelOutputError,
    InvalidModelOutputError,
    ToolDefinition,
)
from agent_runtime.providers import OpenAICompatibleProvider


def test_chat_uses_openai_contract_and_normalizes_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://models.example/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer secret"
        assert request.read() == (
            b'{"model":"gemma4-31b","messages":[{"role":"user","content":"Hello"}],'
            b'"temperature":0.2,"max_tokens":64}'
        )
        return httpx.Response(
            200,
            json={
                "model": "gemma4-31b",
                "choices": [
                    {
                        "message": {
                            "content": "Hi",
                            "reasoning": "A short greeting is appropriate.",
                            "reasoning_details": [{"type": "reasoning.text", "text": "..."}],
                        },
                        "finish_reason": "stop",
                    },
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
        )

    provider = OpenAICompatibleProvider(
        name="local-vllm",
        base_url="https://models.example/v1",
        api_key="secret",
        transport=httpx.MockTransport(handler),
    )
    response = provider.chat(
        "gemma4-31b",
        ChatRequest(
            messages=(ChatMessage(role="user", content="Hello"),),
            temperature=0.2,
            max_tokens=64,
        ),
    )

    assert response.content == "Hi"
    assert response.provider == "local-vllm"
    assert response.reasoning == "A short greeting is appropriate."
    assert response.reasoning_details[0]["type"] == "reasoning.text"
    assert response.usage.total_tokens == 4


def test_chat_rejects_response_without_choices() -> None:
    provider = OpenAICompatibleProvider(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"choices": []})),
    )

    with pytest.raises(InvalidModelOutputError, match="no completion choices"):
        provider.chat("some-model", ChatRequest(messages=()))


def test_tool_call_round_trips_through_conversation_history() -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.read()))
        if len(requests) == 1:
            return httpx.Response(
                200,
                json={
                    "model": "tool-model",
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {
                                            "name": "capability_search",
                                            "arguments": '{"query":"payments"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                },
            )
        return httpx.Response(
            200,
            json={
                "model": "tool-model",
                "choices": [
                    {"message": {"content": "Found it."}, "finish_reason": "stop"},
                ],
            },
        )

    provider = OpenAICompatibleProvider(
        name="test",
        base_url="https://models.example/v1",
        transport=httpx.MockTransport(handler),
    )
    tool = ToolDefinition(
        name="capability_search",
        description="Search approved capabilities",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    )
    first = provider.chat(
        "tool-model",
        ChatRequest(
            messages=(ChatMessage(role="user", content="Find code search"),), tools=(tool,)
        ),
    )
    assert first.tool_calls[0].parse_arguments() == {"query": "payments"}

    provider.chat(
        "tool-model",
        ChatRequest(
            messages=(
                ChatMessage(role="user", content="Find code search"),
                first.to_assistant_message(),
                ChatMessage(role="tool", content='{"matches":[]}', tool_call_id="call-1"),
            ),
            tools=(tool,),
        ),
    )

    assert requests[1]["messages"][1] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "capability_search",
                    "arguments": '{"query":"payments"}',
                },
            }
        ],
    }
    assert requests[1]["messages"][2]["tool_call_id"] == "call-1"
    assert requests[0]["tools"] == [tool.to_dict()]


def test_incomplete_empty_output_has_structured_retryable_error() -> None:
    provider = OpenAICompatibleProvider(
        name="reasoning-model",
        base_url="https://models.example/v1",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": "", "reasoning": "Still reasoning"},
                            "finish_reason": "length",
                        }
                    ]
                },
            )
        ),
    )

    with pytest.raises(IncompleteModelOutputError) as error:
        provider.chat("minimax", ChatRequest(messages=()))

    assert error.value.code == "incomplete_output"
    assert error.value.retryable is True
