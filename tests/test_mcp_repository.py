import json
import subprocess
import threading

import httpx
import pytest

from agent_runtime import (
    ArtifactStore,
    CapabilityGateway,
    ChatResponse,
    GatewayError,
    InMemoryAuditSink,
    InMemoryCapabilityCatalog,
    IntakePlanningService,
    InvocationContext,
    JiraAuth,
    JiraReadAdapter,
    MCPAllowlistAdapter,
    MCPFailure,
    RepositorySnapshotError,
    RepositorySnapshotProvider,
    ReviewedToolMapping,
    RunState,
    SQLiteRunStore,
    StaticPolicyEngine,
    StreamableHTTPTransport,
    canonical_schema_hash,
)
from agent_runtime.providers import ScriptedProvider

SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": False,
}


def transport_for(*, schema=SCHEMA, output=None):
    def handler(request):
        body = json.loads(request.content)
        if body["method"] == "notifications/initialized":
            return httpx.Response(202)
        request_id = body["id"]
        if body["method"] == "initialize":
            return httpx.Response(
                200,
                headers={"Mcp-Session-Id": "fixture-session"},
                json={"jsonrpc": "2.0", "id": request_id, "result": {"protocolVersion": "1"}},
            )
        if body["method"] == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "search",
                        "inputSchema": schema,
                        "annotations": {"readOnlyHint": False},
                    },
                    {"name": "unreviewed-shell", "inputSchema": {}},
                ]
            }
        else:
            result = output or {"content": [{"type": "text", "text": "src/app.py: Widget"}]}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": request_id, "result": result})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return StreamableHTTPTransport("https://mcp.example.test", client=client)


def mapping(expected=None):
    return ReviewedToolMapping(
        "code-intelligence",
        "search",
        "repository.code_search",
        "Code search",
        "Search symbols and code in the pinned repository",
        "1",
        SCHEMA,
        expected or canonical_schema_hash(SCHEMA),
    )


def test_only_reviewed_healthy_mapping_becomes_read_capability_and_drift_quarantines():
    adapter = MCPAllowlistAdapter("code-intelligence", transport_for(), (mapping(),))
    capabilities = adapter.capabilities()
    assert [item.descriptor.card.capability_id for item in capabilities] == [
        "repository.code_search"
    ]
    assert capabilities[0].descriptor.effect == "read"  # MCP annotation cannot raise/lower trust.

    drifted = MCPAllowlistAdapter(
        "code-intelligence", transport_for(schema={"type": "string"}), (mapping(),)
    )
    assert drifted.capabilities() == ()
    assert drifted.quarantined == {"repository.code_search": "schema_drift"}

    gateway = CapabilityGateway(
        InMemoryCapabilityCatalog(capabilities),
        StaticPolicyEngine(frozenset({"repository.code_search"})),
        InMemoryAuditSink(),
        InvocationContext("principal", "run", "ANALYZE"),
    )
    assert gateway.search("code")
    with pytest.raises(GatewayError, match="denied"):
        gateway.invoke("unreviewed-shell", {})


def test_recursive_redaction_spill_and_hard_maximum(tmp_path):
    output = {
        "content": [
            {
                "password": "bad",
                "Cookie": "session=structured-secret",
                "Set-Cookie": "csrf=structured-secret",
                "Authorization: Basic key-secret": "failed",
                "nested": {"access_token": "also-bad"},
                "text": '{"Authorization":"Basic value-secret"}' + "x" * 80,
            }
        ]
    }
    store = ArtifactStore(tmp_path / "artifacts")
    adapter = MCPAllowlistAdapter(
        "code-intelligence",
        transport_for(output=output),
        (mapping(),),
        artifact_store=store,
        inline_bytes=20,
    )
    result = adapter.capabilities()[0].handler({"query": "Widget"}, None)
    full = store.get(result["artifact_sha256"])
    for secret in (b"bad", b"value-secret", b"key-secret", b"structured-secret"):
        assert secret not in full
    assert full.count(b"[REDACTED]") == 7

    too_large = MCPAllowlistAdapter(
        "code-intelligence",
        transport_for(output=output),
        (mapping(),),
        artifact_store=store,
        hard_bytes=20,
    )
    with pytest.raises(MCPFailure) as caught:
        too_large.capabilities()[0].handler({"query": "Widget"}, None)
    assert caught.value.code == "output_too_large"


def test_timeout_malformed_and_cancellation_are_structured():
    def timeout(_request):
        raise httpx.ReadTimeout("sensitive upstream detail")

    transport = StreamableHTTPTransport(
        "https://mcp.example.test", client=httpx.Client(transport=httpx.MockTransport(timeout))
    )
    with pytest.raises(MCPFailure) as caught:
        transport.list_tools()
    assert caught.value.code == "timeout" and "sensitive" not in str(caught.value)

    event = threading.Event()
    event.set()
    with pytest.raises(MCPFailure, match="cancelled"):
        transport_for().call_tool("search", {}, cancelled=event)


def git(path, *args):
    return subprocess.run(
        ["git", *args], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_repository_snapshot_is_origin_validated_revision_pinned_and_read_only(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    git(repository, "init")
    git(repository, "config", "user.email", "fixture@example.test")
    git(repository, "config", "user.name", "Fixture")
    git(repository, "remote", "add", "origin", "fixture://trusted/repository")
    (repository / "app.py").write_text("class Widget:\n    pass\n")
    git(repository, "add", "app.py")
    git(repository, "commit", "-m", "base")
    base = git(repository, "rev-parse", "HEAD")
    snapshot = RepositorySnapshotProvider(repository, "fixture://trusted/repository").acquire(base)
    (repository / "app.py").write_text("drifted working tree")
    assert snapshot.read_file("app.py") == b"class Widget:\n    pass\n"
    with pytest.raises(RepositorySnapshotError, match="origin mismatch"):
        RepositorySnapshotProvider(repository, "fixture://attacker/repository").acquire(base)


def test_mcp_evidence_is_persisted_into_story_plan(tmp_path):
    def jira_handler(request):
        if request.url.path.endswith("/comment"):
            return httpx.Response(200, json={"comments": [], "total": 0})
        return httpx.Response(
            200,
            json={
                "key": "ABC-1",
                "fields": {
                    "summary": "Update Widget",
                    "description": "Change Widget safely",
                    "status": {"name": "Open"},
                    "issuetype": {"name": "Story"},
                    "updated": "2026-01-01",
                },
            },
        )

    store = SQLiteRunStore(tmp_path / "runs.db")
    store.create_run(
        "run",
        story_hash="pending",
        repository="org/repo",
        base_revision="abc",
        provider="scripted",
        model="model",
        prompt_version="v1",
        policy_version="v1",
        profile_id="gemma-4-31b-it-vllm",
    )
    jira = JiraReadAdapter(
        "https://jira.example.test",
        JiraAuth("bearer", "not-real"),
        client=httpx.Client(transport=httpx.MockTransport(jira_handler)),
    )
    capability = MCPAllowlistAdapter(
        "code-intelligence", transport_for(), (mapping(),)
    ).capabilities()[0]
    gateway = CapabilityGateway(
        InMemoryCapabilityCatalog((capability,)),
        StaticPolicyEngine(frozenset({"repository.code_search"})),
        InMemoryAuditSink(),
        InvocationContext("principal", "run", "ANALYZE"),
    )
    evidence = gateway.invoke("repository.code_search", {"query": "Widget"}).result
    provider = ScriptedProvider(
        (ChatResponse("Update `src/app.py` at `Widget`.", "model", "scripted"),)
    )
    workflow = IntakePlanningService(store, jira, provider)

    workflow.intake("run", "ABC-1")
    plan = workflow.plan("run", (evidence,))

    assert store.get_run("run").state == RunState.PLAN_READY
    assert "src/app.py" in plan
    assert "src/app.py" in provider.requests[0][1].messages[1].content
    assert store.analysis_evidence("run", 1)[0] == evidence
