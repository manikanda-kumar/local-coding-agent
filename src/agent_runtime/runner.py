from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from uuid import uuid4

from agent_runtime.gateway import CapabilityGateway, GatewayError, to_jsonable
from agent_runtime.metrics import MetricEvent, MetricsSink, emit_metric
from agent_runtime.models import ChatMessage, ChatRequest, ToolCall, ToolDefinition
from agent_runtime.profiles import ModelRequestProfile
from agent_runtime.providers.base import Provider


def _tool(name: str, properties: dict, required: list[str]) -> ToolDefinition:
    return ToolDefinition(
        name,
        name.replace("_", " "),
        {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    )


GATEWAY_TOOLS = (
    _tool("capability_search", {"query": {"type": "string"}}, ["query"]),
    _tool("capability_describe", {"capability_id": {"type": "string"}}, ["capability_id"]),
    _tool(
        "capability_invoke",
        {"capability_id": {"type": "string"}, "arguments": {"type": "object"}},
        ["capability_id", "arguments"],
    ),
    _tool("execution_status", {"execution_id": {"type": "string"}}, ["execution_id"]),
    _tool("execution_cancel", {"execution_id": {"type": "string"}}, ["execution_id"]),
)


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """Deterministically retain a stable prefix and newest complete tool rounds."""

    max_context_tokens: int = 65_536
    reserve_output_tokens: int = 8_192
    chars_per_token: int = 1
    retain_groups: int = 2
    request_overhead_tokens: int = 64
    message_overhead_tokens: int = 16

    def __post_init__(self) -> None:
        if (
            not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in (
                    self.max_context_tokens,
                    self.reserve_output_tokens,
                    self.chars_per_token,
                    self.retain_groups,
                    self.request_overhead_tokens,
                    self.message_overhead_tokens,
                )
            )
            or self.max_context_tokens <= self.reserve_output_tokens
            or self.reserve_output_tokens < 0
            or self.chars_per_token < 1
            or self.retain_groups < 1
            or self.request_overhead_tokens < 0
            or self.message_overhead_tokens < 0
        ):
            raise ValueError("context budget is invalid")

    def estimate_json(self, value: object) -> int:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        return (len(encoded) + self.chars_per_token - 1) // self.chars_per_token

    def _tokens(
        self,
        messages: tuple[ChatMessage, ...],
        *,
        reasoning_field: str = "reasoning",
        reasoning_mode: str = "preserve",
    ) -> int:
        wire = [
            message.to_dict(reasoning_field=reasoning_field, reasoning_mode=reasoning_mode)
            for message in messages
        ]
        return (
            self.estimate_json(wire)
            + self.request_overhead_tokens
            + len(messages) * self.message_overhead_tokens
        )

    def compose(
        self,
        messages: tuple[ChatMessage, ...],
        *,
        fixed_tokens: int = 0,
        reasoning_field: str = "reasoning",
        reasoning_mode: str = "preserve",
    ) -> tuple[ChatMessage, ...]:
        if not messages or fixed_tokens < 0:
            raise ValueError("context messages and fixed token estimate are required")
        available = self.max_context_tokens - self.reserve_output_tokens - fixed_tokens
        if messages[0].role == "system":
            if len(messages) < 2 or messages[1].role != "user":
                raise ValueError("context must start with system then user")
            prefix_length = 2
        elif messages[0].role == "user":
            prefix_length = 1
        else:
            raise ValueError("context must start with a user message")
        prefix = messages[:prefix_length]
        groups: list[tuple[ChatMessage, ...]] = []
        current: list[ChatMessage] = []
        for message in messages[prefix_length:]:
            if message.role == "assistant" and current:
                groups.append(tuple(current))
                current = []
            current.append(message)
        if current:
            groups.append(tuple(current))
        for group in groups:
            if group[0].role != "assistant":
                raise ValueError("context rounds must start with an assistant message")
            expected = [call.id for call in group[0].tool_calls]
            actual = [message.tool_call_id for message in group[1:]]
            if (
                not expected
                or len(set(expected)) != len(expected)
                or actual != expected
                or any(message.role != "tool" for message in group[1:])
            ):
                raise ValueError("context tool calls and results must form a complete round")
        token_options = {
            "reasoning_field": reasoning_field,
            "reasoning_mode": reasoning_mode,
        }
        if self._tokens(messages, **token_options) <= available:
            return messages
        mandatory = groups[-self.retain_groups :]
        selected = list(mandatory)
        if (
            self._tokens(
                prefix + tuple(item for group in selected for item in group), **token_options
            )
            > available
        ):
            raise RuntimeError("newest complete tool rounds exceed the context budget")
        for group in reversed(groups[: -self.retain_groups]):
            candidate = [group, *selected]
            flattened = prefix + tuple(item for item_group in candidate for item in item_group)
            if self._tokens(flattened, **token_options) > available:
                break
            selected = candidate
        return prefix + tuple(item for group in selected for item in group)


class AgentRunner:
    def __init__(
        self,
        provider: Provider,
        model: str,
        gateway: CapabilityGateway,
        *,
        max_turns: int = 8,
        max_invocations: int = 4,
        metrics: MetricsSink | None = None,
        profile: ModelRequestProfile | None = None,
        context_budget: ContextBudget | None = None,
    ) -> None:
        if max_turns < 1 or max_invocations < 0:
            raise ValueError("limits must be non-negative and max_turns positive")
        self.provider, self.model, self.gateway = provider, model, gateway
        self.max_turns, self.max_invocations = max_turns, max_invocations
        self.metrics = metrics
        self.profile = profile
        if (
            profile is not None
            and context_budget is not None
            and (
                context_budget.reserve_output_tokens < profile.max_output_tokens
                or context_budget.max_context_tokens > profile.context_window_tokens
            )
        ):
            raise ValueError("custom context budget exceeds the model profile")
        self.context_budget = context_budget or ContextBudget(
            max_context_tokens=(profile.context_window_tokens if profile is not None else 65_536),
            reserve_output_tokens=(profile.max_output_tokens if profile is not None else 8_192),
        )
        self._fixed_tool_tokens = self.context_budget.estimate_json(
            [tool.to_dict() for tool in GATEWAY_TOOLS]
        )

    def run(self, prompt: str, *, session_id: str | None = None) -> str:
        if session_id is None:
            session_id = str(uuid4())
        elif (
            not isinstance(session_id, str)
            or not session_id
            or len(session_id.encode("utf-8")) > 256
        ):
            raise ValueError("agent session ID is invalid or unbounded")
        messages = [ChatMessage(role="user", content=prompt)]
        invocations = 0
        for turn in range(1, self.max_turns + 1):
            started = time.monotonic()
            try:
                options = {} if self.profile is None else self.profile.chat_request_options()
                request_messages = self.context_budget.compose(
                    tuple(messages),
                    fixed_tokens=self._fixed_tool_tokens,
                    reasoning_field=(self.profile.reasoning_field if self.profile else "reasoning"),
                    reasoning_mode=(self.profile.reasoning_mode if self.profile else "preserve"),
                )
                response = self.provider.chat(
                    self.model, ChatRequest(request_messages, tools=GATEWAY_TOOLS, **options)
                )
            except Exception:
                emit_metric(
                    self.metrics,
                    MetricEvent(
                        "model",
                        "chat",
                        "failure",
                        run_id=self.gateway.context.run_id,
                        duration_ms=(time.monotonic() - started) * 1000,
                    ),
                )
                raise
            assistant_message = response.to_assistant_message()
            emit_metric(
                self.metrics,
                MetricEvent(
                    "model",
                    "chat",
                    "success",
                    run_id=self.gateway.context.run_id,
                    duration_ms=(time.monotonic() - started) * 1000,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                ),
            )
            messages.append(assistant_message)
            if not response.tool_calls:
                return response.content
            for tool_index, call in enumerate(response.tool_calls, start=1):
                if call.name == "capability_invoke":
                    if invocations >= self.max_invocations:
                        raise RuntimeError("agent invocation limit reached")
                    invocations += 1
                result = self._dispatch(
                    call,
                    invocation_key=hashlib.sha256(
                        f"agent:{session_id}:turn:{turn}:tool:{tool_index}".encode()
                    ).hexdigest(),
                )
                messages.append(
                    ChatMessage(role="tool", content=json.dumps(result), tool_call_id=call.id)
                )
        raise RuntimeError("agent turn limit reached")

    def _dispatch(self, call: ToolCall, *, invocation_key: str | None = None) -> dict:
        try:
            args = call.parse_arguments()
            known = {tool.name: tool for tool in GATEWAY_TOOLS}
            if call.name not in known:
                raise GatewayError("invalid_request", "unsupported tool")
            from agent_runtime.gateway import _validate

            _validate(args, known[call.name].parameters)
        except Exception as error:  # noqa: BLE001 - tool errors are returned to the model
            self.gateway.audit_invalid(call.name, str(error))
            return {
                "ok": False,
                "error": {"code": getattr(error, "code", "invalid_request"), "message": str(error)},
            }

        operations = {
            "capability_search": lambda: self.gateway.search(args["query"]),
            "capability_describe": lambda: self.gateway.describe(args["capability_id"]),
            "capability_invoke": lambda: self.gateway.invoke(
                args["capability_id"], args["arguments"], invocation_key=invocation_key
            ),
            "execution_status": lambda: self.gateway.status(args["execution_id"]),
            "execution_cancel": lambda: self.gateway.cancel(args["execution_id"]),
        }
        try:
            return {"ok": True, "data": to_jsonable(operations[call.name]())}
        except Exception as error:  # noqa: BLE001 - gateway errors are returned to the model
            return {
                "ok": False,
                "error": {"code": getattr(error, "code", "invalid_request"), "message": str(error)},
            }
