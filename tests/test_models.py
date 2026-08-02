import pytest

from agent_runtime import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ToolArgumentsError,
    ToolCall,
    ToolResult,
)
from agent_runtime.providers import ScriptedProvider


def test_tool_call_preserves_malformed_arguments_for_gateway_validation() -> None:
    tool_call = ToolCall(id="call-1", name="capability_invoke", arguments="{not-json")

    with pytest.raises(ToolArgumentsError) as error:
        tool_call.parse_arguments()

    assert tool_call.arguments == "{not-json"
    assert error.value.tool_call_id == "call-1"
    assert error.value.code == "invalid_tool_arguments"


def test_tool_messages_require_correlation_id() -> None:
    with pytest.raises(ValueError, match="require tool_call_id"):
        ChatMessage(role="tool", content="result")


def test_tool_result_creates_correlated_message() -> None:
    message = ToolResult(tool_call_id="call-1", content='{"ok":true}').to_message()

    assert message.to_dict() == {
        "role": "tool",
        "content": '{"ok":true}',
        "tool_call_id": "call-1",
    }


def test_scripted_provider_records_requests_and_is_deterministic() -> None:
    expected = ChatResponse(content="done", model="fixture", provider="scripted")
    provider = ScriptedProvider((expected,))
    request = ChatRequest(messages=(ChatMessage(role="user", content="work"),))

    assert provider.chat("fixture", request) is expected
    assert provider.requests == [("fixture", request)]

    with pytest.raises(RuntimeError, match="no responses remaining"):
        provider.chat("fixture", request)
