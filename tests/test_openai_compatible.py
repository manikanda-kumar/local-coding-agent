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


def test_reasoning_round_trips_to_second_turn_with_selected_field() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.read()))
        choices = (
            [{"message": {"content": "answer"}, "finish_reason": "stop"}]
            if len(requests) == 2
            else [
                {
                    "message": {
                        "content": "first",
                        "reasoning_content": "private chain",
                        "reasoning_details": [{"opaque": {"server": 42}}],
                    }
                }
            ]
        )
        return httpx.Response(
            200,
            json={"choices": choices},
        )

    provider = OpenAICompatibleProvider(
        name="local",
        base_url="https://models.example/v1",
        transport=httpx.MockTransport(handler),
    )
    first = provider.chat("glm", ChatRequest(messages=(ChatMessage("user", "one"),)))
    provider.chat(
        "glm",
        ChatRequest(
            messages=(first.to_assistant_message(), ChatMessage("user", "two")),
            reasoning_field="reasoning_content",
            reasoning_mode="structured",
        ),
    )

    assistant = requests[1]["messages"][0]
    assert assistant["reasoning_content"] == "private chain"
    assert "reasoning" not in assistant
    assert assistant["reasoning_details"] == [{"opaque": {"server": 42}}]


@pytest.mark.parametrize("arguments", [[1], 3, None])
def test_tool_call_rejects_non_object_structured_arguments(arguments) -> None:
    provider = OpenAICompatibleProvider(name="local", base_url="https://example.test/v1")
    data = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"id": "c", "function": {"name": "tool", "arguments": arguments}}
                    ]
                }
            }
        ]
    }
    with pytest.raises(InvalidModelOutputError, match="not an object or string"):
        provider._parse_response(data, fallback_model="model")


def test_tool_call_canonicalizes_object_arguments() -> None:
    provider = OpenAICompatibleProvider(name="local", base_url="https://example.test/v1")
    data = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"id": "c", "function": {"name": "tool", "arguments": {"b": 2, "a": 1}}}
                    ]
                }
            }
        ]
    }
    assert (
        provider._parse_response(data, fallback_model="m").tool_calls[0].arguments
        == '{"a":1,"b":2}'
    )


def test_leading_think_block_is_separated() -> None:
    provider = OpenAICompatibleProvider(name="local", base_url="https://example.test/v1")
    response = provider._parse_response(
        {"choices": [{"message": {"content": "<think>work</think>Final"}}]},
        fallback_model="m",
        reasoning_mode="tagged_content",
    )
    assert (response.reasoning, response.content) == ("work", "Final")


@pytest.mark.parametrize(
    "content,error",
    [("<think>work", IncompleteModelOutputError), ("<think>work</think>", InvalidModelOutputError)],
)
def test_thinking_without_complete_final_output_is_rejected(content, error) -> None:
    provider = OpenAICompatibleProvider(name="local", base_url="https://example.test/v1")
    with pytest.raises(error):
        provider._parse_response(
            {"choices": [{"message": {"content": content}}]},
            fallback_model="m",
            reasoning_mode="tagged_content",
        )


def test_no_profile_preserves_tagged_wire_content_without_translation() -> None:
    provider = OpenAICompatibleProvider(name="local", base_url="https://example.test/v1")
    response = provider._parse_response(
        {"choices": [{"message": {"content": "<think>work</think>Final"}}]},
        fallback_model="m",
    )
    assert response.content == "Final"
    assert response.reasoning == "work"
    assert response.to_assistant_message().to_dict()["content"] == "<think>work</think>Final"
    assert "reasoning" not in response.to_assistant_message().to_dict()


def test_tagged_tool_call_replays_exact_original_content_on_second_turn() -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.read()))
        if len(requests) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": "  <think> exact work </think>  ",
                                "tool_calls": [
                                    {
                                        "id": "c",
                                        "type": "function",
                                        "function": {"name": "tool", "arguments": "{}"},
                                    }
                                ],
                            }
                        }
                    ]
                },
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": "done"}}]})

    provider = OpenAICompatibleProvider(
        name="local", base_url="https://example.test/v1", transport=httpx.MockTransport(handler)
    )
    mode = {"reasoning_mode": "tagged_content"}
    first = provider.chat("m", ChatRequest((ChatMessage("user", "one"),), **mode))
    assert first.content == ""
    provider.chat(
        "m",
        ChatRequest((first.to_assistant_message(), ChatMessage("user", "two")), **mode),
    )
    assert requests[1]["messages"][0]["content"] == "  <think> exact work </think>  "
    assert "reasoning" not in requests[1]["messages"][0]


@pytest.mark.parametrize(
    "extensions",
    [
        {"stream": True},
        {"max_completion_tokens": 1},
        {"temperature": 0},
        {"chat_template_kwargs": {"api_token": "bad"}},
        {"chat_template_kwargs": {"value": float("nan")}},
    ],
)
def test_direct_request_extensions_cannot_bypass_validation(extensions) -> None:
    provider = OpenAICompatibleProvider(name="local", base_url="https://example.test/v1")
    with pytest.raises((TypeError, ValueError)):
        provider.chat("m", ChatRequest((), extensions=extensions))


@pytest.mark.parametrize("field,value", [("temperature", float("nan")), ("timeout", float("inf"))])
def test_direct_request_rejects_non_finite_options(field, value) -> None:
    provider = OpenAICompatibleProvider(name="local", base_url="https://example.test/v1")
    with pytest.raises(ValueError):
        provider.chat("m", ChatRequest((), **{field: value}))


@pytest.mark.parametrize("field", ["max_tokens", "top_k", "seed"])
def test_direct_request_rejects_fractional_integer_options(field) -> None:
    provider = OpenAICompatibleProvider(name="local", base_url="https://example.test/v1")
    with pytest.raises(ValueError):
        provider.chat("m", ChatRequest((), **{field: 1.5}))


def test_request_timeout_omission_preserves_client_timeout_and_override_wins() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"]["read"])
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    provider = OpenAICompatibleProvider(
        name="local",
        base_url="https://example.test/v1",
        timeout=17,
        transport=httpx.MockTransport(handler),
    )
    provider.chat("m", ChatRequest(()))
    provider.chat("m", ChatRequest((), timeout=3))
    assert seen == [17, 3]


def test_conflicting_reasoning_aliases_are_rejected() -> None:
    provider = OpenAICompatibleProvider(name="local", base_url="https://example.test/v1")
    with pytest.raises(InvalidModelOutputError, match="conflicting reasoning aliases"):
        provider._parse_response(
            {
                "choices": [
                    {"message": {"content": "ok", "reasoning": "a", "reasoning_content": "b"}}
                ]
            },
            fallback_model="m",
        )


def test_reasoning_details_are_bounded() -> None:
    provider = OpenAICompatibleProvider(name="local", base_url="https://example.test/v1")
    with pytest.raises(InvalidModelOutputError, match="reasoning_details"):
        provider._parse_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": "ok",
                            "reasoning_details": [{"x": "a" * (1024 * 1024 + 1)}],
                        }
                    }
                ]
            },
            fallback_model="m",
        )
