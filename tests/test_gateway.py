import json

import pytest

from agent_runtime import (
    GATEWAY_TOOLS,
    AgentRunner,
    Capability,
    CapabilityCard,
    CapabilityDescriptor,
    CapabilityGateway,
    ChatResponse,
    ExecutionStatus,
    GatewayError,
    InMemoryAuditSink,
    InMemoryCapabilityCatalog,
    InvocationContext,
    PolicyDecision,
    StaticPolicyEngine,
    ToolCall,
    fixture_read_capability,
)
from agent_runtime.providers import ScriptedProvider


def make_gateway(*, allowed=frozenset({"fixture.read"}), extra=()):
    audit = InMemoryAuditSink()
    gateway = CapabilityGateway(
        InMemoryCapabilityCatalog((fixture_read_capability(), *extra)),
        StaticPolicyEngine(allowed),
        audit,
        InvocationContext(
            principal_id="principal",
            run_id="run",
            stage="ANALYZE",
            story_id="story",
            repository_id="repository",
            workspace_id="workspace",
        ),
    )
    return gateway, audit


def test_deny_by_default_hides_and_rechecks_guessed_capabilities() -> None:
    gateway, audit = make_gateway(allowed=frozenset())

    assert gateway.search("fixture") == ()
    with pytest.raises(GatewayError, match="denied"):
        gateway.describe("fixture.read")
    with pytest.raises(GatewayError, match="denied"):
        gateway.invoke("fixture.read", {"key": "x"})

    assert [event.outcome for event in audit.events] == ["success", "denial", "denial"]
    assert (
        StaticPolicyEngine(require_approval=frozenset({"fixture.read"})).decide(
            "fixture.read", gateway.context
        )
        == PolicyDecision.REQUIRE_APPROVAL
    )


def test_validation_rejects_unknown_and_runtime_owned_fields_and_audits() -> None:
    gateway, audit = make_gateway()
    for field in ("principal_id", "stage", "workspace_id", "policy", "idempotency_key"):
        with pytest.raises(GatewayError, match="unknown field"):
            gateway.invoke("fixture.read", {"key": "x", field: "attacker"})
    assert all(event.outcome == "invalid_request" for event in audit.events)


def test_success_failure_status_and_invalid_operations_are_audited() -> None:
    failure = Capability(
        CapabilityDescriptor(
            CapabilityCard("fixture.failure", "Failure", "fail harmlessly"),
            {"type": "object", "properties": {}, "additionalProperties": False},
        ),
        lambda _args, _context: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    gateway, audit = make_gateway(
        allowed=frozenset({"fixture.read", "fixture.failure"}), extra=(failure,)
    )
    record = gateway.invoke("fixture.read", {"key": "hello"})
    assert record.status == ExecutionStatus.SUCCEEDED
    assert record.result["value"] == "fixture-value"
    assert gateway.status(record.execution_id) is record
    assert gateway.invoke("fixture.failure", {}).status == ExecutionStatus.FAILED
    with pytest.raises(GatewayError, match="already terminal"):
        gateway.cancel(record.execution_id)
    assert record.status == ExecutionStatus.SUCCEEDED
    with pytest.raises(GatewayError, match="unknown execution"):
        gateway.status("guessed")
    assert {event.outcome for event in audit.events} >= {
        "success",
        "failure",
        "invalid_request",
    }


def response(call_id, name, arguments):
    return ChatResponse(
        "",
        "fixture-model",
        "scripted",
        tool_calls=(ToolCall(call_id, name, json.dumps(arguments)),),
    )


def test_scripted_search_describe_invoke_final_loop_and_five_safe_tools() -> None:
    provider = ScriptedProvider(
        (
            response("1", "capability_search", {"query": "fixture"}),
            response("2", "capability_describe", {"capability_id": "fixture.read"}),
            response(
                "3",
                "capability_invoke",
                {"capability_id": "fixture.read", "arguments": {"key": "x"}},
            ),
            ChatResponse("The fixture value is fixture-value.", "fixture-model", "scripted"),
        )
    )
    gateway, _ = make_gateway()
    answer = AgentRunner(provider, "fixture-model", gateway).run("Read the fixture")

    assert answer == "The fixture value is fixture-value."
    assert {tool.name for tool in GATEWAY_TOOLS} == {
        "capability_search",
        "capability_describe",
        "capability_invoke",
        "execution_status",
        "execution_cancel",
    }
    assert all(request.tools == GATEWAY_TOOLS for _, request in provider.requests)
    invoke_schema = next(tool for tool in GATEWAY_TOOLS if tool.name == "capability_invoke")
    forbidden = {"principal", "run_id", "stage", "workspace", "policy", "idempotency_key"}
    assert forbidden.isdisjoint(invoke_schema.parameters["properties"])


def test_runner_stops_at_turn_and_invocation_limits() -> None:
    gateway, _ = make_gateway()
    looping = tuple(response(str(i), "capability_search", {"query": "x"}) for i in range(3))
    with pytest.raises(RuntimeError, match="turn limit"):
        AgentRunner(ScriptedProvider(looping), "model", gateway, max_turns=2).run("loop")

    invokes = tuple(
        response(
            str(i),
            "capability_invoke",
            {"capability_id": "fixture.read", "arguments": {"key": "x"}},
        )
        for i in range(2)
    )
    with pytest.raises(RuntimeError, match="invocation limit"):
        AgentRunner(ScriptedProvider(invokes), "model", gateway, max_invocations=1).run("loop")


def test_runner_audits_malformed_and_unsupported_gateway_calls() -> None:
    gateway, audit = make_gateway()
    provider = ScriptedProvider(
        (
            ChatResponse(
                "",
                "fixture-model",
                "scripted",
                tool_calls=(ToolCall("1", "capability_search", "{invalid"),),
            ),
            response("2", "untrusted_tool", {}),
            ChatResponse("done", "fixture-model", "scripted"),
        )
    )

    assert AgentRunner(provider, "fixture-model", gateway).run("test") == "done"
    assert [(event.operation, event.outcome) for event in audit.events] == [
        ("capability_search", "invalid_request"),
        ("untrusted_tool", "invalid_request"),
    ]
