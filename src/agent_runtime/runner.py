from __future__ import annotations

import json
import time

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
    ) -> None:
        if max_turns < 1 or max_invocations < 0:
            raise ValueError("limits must be non-negative and max_turns positive")
        self.provider, self.model, self.gateway = provider, model, gateway
        self.max_turns, self.max_invocations = max_turns, max_invocations
        self.metrics = metrics
        self.profile = profile

    def run(self, prompt: str) -> str:
        messages = [ChatMessage(role="user", content=prompt)]
        invocations = 0
        for _ in range(self.max_turns):
            started = time.monotonic()
            try:
                options = {} if self.profile is None else self.profile.chat_request_options()
                response = self.provider.chat(
                    self.model, ChatRequest(tuple(messages), tools=GATEWAY_TOOLS, **options)
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
            for call in response.tool_calls:
                if call.name == "capability_invoke":
                    if invocations >= self.max_invocations:
                        raise RuntimeError("agent invocation limit reached")
                    invocations += 1
                result = self._dispatch(call)
                messages.append(
                    ChatMessage(role="tool", content=json.dumps(result), tool_call_id=call.id)
                )
        raise RuntimeError("agent turn limit reached")

    def _dispatch(self, call: ToolCall) -> dict:
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
                args["capability_id"], args["arguments"]
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
