import httpx
import pytest

from agent_runtime import ChatMessage, ChatRequest
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

    with pytest.raises(ValueError, match="no completion choices"):
        provider.chat("some-model", ChatRequest(messages=()))
