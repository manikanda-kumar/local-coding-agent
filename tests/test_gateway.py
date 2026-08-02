import json

import pytest

from agent_runtime import (
    GATEWAY_TOOLS,
    AgentRunner,
    ArtifactStore,
    Capability,
    CapabilityCard,
    CapabilityDescriptor,
    CapabilityGateway,
    ChatMessage,
    ChatResponse,
    ContextBudget,
    Effect,
    ExecutionStatus,
    GatewayError,
    GatewayOutputPolicy,
    InMemoryAuditSink,
    InMemoryCapabilityCatalog,
    InvocationContext,
    PolicyDecision,
    StaticPolicyEngine,
    ToolCall,
    fixture_read_capability,
)
from agent_runtime.metrics import RedactionPolicy, redact_sensitive_text
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

    assert [event.outcome for event in audit.events] == ["denial", "success", "denial", "denial"]
    assert (
        StaticPolicyEngine(require_approval=frozenset({"fixture.read"})).decide(
            "fixture.read", gateway.context
        )
        == PolicyDecision.REQUIRE_APPROVAL
    )


def test_catalog_rejects_unknown_and_mislabeled_workspace_effects() -> None:
    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    for capability_id, effect in (
        ("workspace.write", Effect.TRUSTED_WORKSPACE_WRITE),
        ("workspace.patch.apply", Effect.TRUSTED_WORKSPACE_READ),
        ("ordinary.read", Effect.TRUSTED_WORKSPACE_WRITE),
        ("ordinary.unknown", "invented"),
    ):
        capability = Capability(
            CapabilityDescriptor(CapabilityCard(capability_id, "x", "x"), schema, effect),
            lambda _a, _c: None,
        )
        with pytest.raises(ValueError):
            InMemoryCapabilityCatalog((capability,))


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


def test_gateway_redacts_all_results_and_capability_errors() -> None:
    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    result_capability = Capability(
        CapabilityDescriptor(CapabilityCard("fixture.secrets", "Secrets", "return data"), schema),
        lambda _args, _context: {
            "access_token": "must-not-escape",
            "Proxy-Authorization": "proxy-secret",
            "passwd": "passwd-secret",
            "Authorization: Basic key-secret": "failed request",
            "nested": {
                "databasePassword": "also-secret",
                "safe": '{"Authorization":"Basic value-secret"}',
            },
        },
    )
    failure_capability = Capability(
        CapabilityDescriptor(CapabilityCard("fixture.error", "Error", "fail"), schema),
        lambda _args, _context: (_ for _ in ()).throw(
            RuntimeError("request failed Authorization: Bearer-sensitive api_key=key-sensitive")
        ),
    )
    gateway, audit = make_gateway(
        allowed=frozenset({"fixture.secrets", "fixture.error"}),
        extra=(result_capability, failure_capability),
    )

    result = gateway.invoke("fixture.secrets", {}).result
    assert result == {
        "access_token": "[REDACTED]",
        "Proxy-Authorization": "[REDACTED]",
        "passwd": "[REDACTED]",
        "Authorization=[REDACTED]": "[REDACTED]",
        "nested": {
            "databasePassword": "[REDACTED]",
            "safe": "{Authorization=[REDACTED]",
        },
    }
    failure = gateway.invoke("fixture.error", {})
    assert failure.error == "capability execution failed"
    assert audit.events[-1].detail == failure.error


@pytest.mark.parametrize(
    ("value", "secret"),
    [
        ("Authorization: Basic dXNlcjpwYXNz", "dXNlcjpwYXNz"),
        ("Proxy-Authorization: Digest auth-secret", "auth-secret"),
        ("Cookie: session=one; csrf=two", "csrf=two"),
        ("Set-Cookie: session=one; Secure", "session=one"),
        ("access_token=oauth-secret", "oauth-secret"),
        ("refresh_token=refresh-secret", "refresh-secret"),
        ("client_secret=client-secret", "client-secret"),
        ('api_key="secret with spaces"', "secret with spaces"),
        ('{"Authorization":"Basic json-secret"}', "json-secret"),
        ('{"Cookie":"session=one; csrf=two"}', "csrf=two"),
        ('{"api_key":"json secret with spaces"}', "json secret with spaces"),
        ("{'client_secret': 'python-secret'}", "python-secret"),
        ("stripe=sk_live_123456789abcdef", "sk_live_123456789abcdef"),
    ],
)
def test_sensitive_text_redacts_complete_standard_credentials(value, secret) -> None:
    assert secret not in redact_sensitive_text(value)


def test_gateway_preserves_small_shapes_and_marks_large_strings(tmp_path) -> None:
    policy = GatewayOutputPolicy(inline_bytes=1024, artifact_bytes=32 * 1024)
    values = list(range(129))
    assert policy.normalize(values) == values

    artifacts = ArtifactStore(tmp_path / "artifacts")
    result = policy.normalize({"text": "é" * 10_000}, artifacts)
    assert result["truncated"] and "artifact_sha256" in result
    assert len(policy._encode(result)) <= policy.inline_bytes
    assert artifacts.get(result["artifact_sha256"]).decode().count("é") == 10_000


def test_gateway_output_policy_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="bounds"):
        GatewayOutputPolicy(inline_bytes=100)
    with pytest.raises(ValueError, match="bounds"):
        GatewayOutputPolicy(inline_bytes=1024, artifact_bytes=512)
    with pytest.raises(ValueError, match="finite"):
        GatewayOutputPolicy(redaction=RedactionPolicy(max_string=1024)).normalize(
            {"value": float("inf")}
        )


def test_validation_execution_is_discoverable_and_callable_only_in_validate() -> None:
    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    called = []
    capability = Capability(
        CapabilityDescriptor(
            CapabilityCard("workspace.test.run", "Run trusted test", "validation test profile"),
            schema,
            Effect.TRUSTED_PROCESS_EXECUTION,
        ),
        lambda _arguments, _context: called.append(True) or {"passed": True},
    )
    catalog = InMemoryCapabilityCatalog((capability,))
    policy = StaticPolicyEngine(frozenset({"workspace.test.run"}))
    for stage in ("IMPLEMENT", "REPORT"):
        gateway = CapabilityGateway(
            catalog, policy, InMemoryAuditSink(), InvocationContext("p", "r", stage)
        )
        assert gateway.search("validation") == ()
        with pytest.raises(GatewayError, match="VALIDATE"):
            gateway.describe("workspace.test.run")
        with pytest.raises(GatewayError, match="VALIDATE"):
            gateway.invoke("workspace.test.run", {})
    gateway = CapabilityGateway(
        catalog, policy, InMemoryAuditSink(), InvocationContext("p", "r", "VALIDATE")
    )
    assert gateway.search("validation")[0].capability_id == "workspace.test.run"
    assert gateway.describe("workspace.test.run").effect == Effect.TRUSTED_PROCESS_EXECUTION
    assert gateway.invoke("workspace.test.run", {}).result == {"passed": True}
    assert called == [True]


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


def test_context_budget_preserves_prefix_and_complete_newest_tool_rounds() -> None:
    prefix = ChatMessage("user", "stable task prefix")
    old_assistant = ChatMessage(
        "assistant", "old", tool_calls=(ToolCall("old", "capability_search", '{"query":"old"}'),)
    )
    old_tool = ChatMessage("tool", "old-result-" + "x" * 100, tool_call_id="old")
    new_assistant = ChatMessage(
        "assistant", "new", tool_calls=(ToolCall("new", "capability_search", '{"query":"new"}'),)
    )
    new_tool = ChatMessage("tool", "new-result", tool_call_id="new")
    newest = (prefix, new_assistant, new_tool)
    measuring = ContextBudget(10_000, 0, chars_per_token=1, retain_groups=1)
    budget = ContextBudget(
        measuring._tokens(newest) + 1,
        0,
        chars_per_token=1,
        retain_groups=1,
    )
    composed = budget.compose((prefix, old_assistant, old_tool, new_assistant, new_tool))
    assert composed == newest
    assert composed[0] is prefix
    assert {message.tool_call_id for message in composed if message.role == "tool"} == {"new"}


def test_context_budget_rejects_invalid_or_oversized_minimum_context() -> None:
    with pytest.raises(ValueError, match="context budget"):
        ContextBudget(10, 10)
    budget = ContextBudget(100, 0, chars_per_token=1, retain_groups=1)
    with pytest.raises(RuntimeError, match="newest complete tool rounds"):
        budget.compose(
            (
                ChatMessage("user", "task"),
                ChatMessage(
                    "assistant",
                    "x" * 200,
                    tool_calls=(ToolCall("call", "capability_search", '{"query":"x"}'),),
                ),
                ChatMessage("tool", "result", tool_call_id="call"),
            )
        )
    with pytest.raises(ValueError, match="complete round"):
        ContextBudget(1_000, 0).compose(
            (
                ChatMessage("user", "task"),
                ChatMessage(
                    "assistant",
                    "",
                    tool_calls=(ToolCall("call", "capability_search", '{"query":"x"}'),),
                ),
                ChatMessage("tool", "result", tool_call_id="wrong"),
            )
        )
    with pytest.raises(ValueError, match="complete round"):
        ContextBudget(1_000, 0).compose(
            (
                ChatMessage("user", "task"),
                ChatMessage(
                    "assistant",
                    "",
                    tool_calls=(
                        ToolCall("duplicate", "capability_search", '{"query":"a"}'),
                        ToolCall("duplicate", "capability_search", '{"query":"b"}'),
                    ),
                ),
                ChatMessage("tool", "a", tool_call_id="duplicate"),
                ChatMessage("tool", "b", tool_call_id="duplicate"),
            )
        )


def test_runner_counts_fixed_gateway_tools_with_the_configured_estimator() -> None:
    gateway, _ = make_gateway()
    measuring = ContextBudget(100_000, 0, chars_per_token=1, retain_groups=1)
    probe = AgentRunner(ScriptedProvider(()), "model", gateway, context_budget=measuring)
    prompt = (ChatMessage("user", "task"),)
    constrained = ContextBudget(
        measuring._tokens(prompt) + probe._fixed_tool_tokens - 1,
        0,
        chars_per_token=1,
        retain_groups=1,
    )
    with pytest.raises(RuntimeError, match="context budget"):
        AgentRunner(ScriptedProvider(()), "model", gateway, context_budget=constrained).run("task")


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
