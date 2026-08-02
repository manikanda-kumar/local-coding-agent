import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from agent_runtime import (
    AgentRunner,
    CapabilityGateway,
    ChatResponse,
    InMemoryAuditSink,
    InMemoryCapabilityCatalog,
    InvocationContext,
    StaticPolicyEngine,
    Usage,
    fixture_read_capability,
)
from agent_runtime.durable import RunState, SQLiteRunStore
from agent_runtime.evaluation import GoldenTask, evaluate
from agent_runtime.metrics import (
    MetricEvent,
    RedactionPolicy,
    RetentionPolicy,
    SQLiteMetricsCollector,
)
from agent_runtime.providers import ScriptedProvider


def _run(store, name):
    store.create_run(
        name,
        story_hash="s",
        repository="r",
        base_revision="b",
        provider="p",
        model="m",
        prompt_version="v",
        policy_version="v",
    )


def test_checked_in_golden_contract_and_regression_details():
    task = GoldenTask.load("tests/fixtures/golden/add_greeting.json")
    # Deterministic execution consumes both fixture inputs; no provider or network is involved.
    assert task.jira["key"] == task.task_id
    workspace = dict(task.repository)
    assert workspace["greeting.txt"] == "hello\n"
    workspace["greeting.txt"] = "hello world\n"
    changed = {
        name: value for name, value in workspace.items() if task.repository.get(name) != value
    }
    result = evaluate(task, changed, {"unit": True, "lint": True})
    assert result.success and result.quality_score == 100
    regression = evaluate(
        task, {"greeting.txt": "wrong\n", "extra": "x"}, {"unit": True, "lint": False}
    )
    assert regression.quality_score == 25
    assert regression.regressions == ("file:greeting.txt", "unexpected-files", "validation:lint")


def test_restart_matrix_every_nonterminal_state_and_retry(tmp_path):
    flow = list(RunState)[:10]
    for index, expected in enumerate(flow):
        path = tmp_path / f"{index}.db"
        store = SQLiteRunStore(path)
        _run(store, "run")
        for state in flow[1 : index + 1]:
            store.transition("run", state, {"resume": state.value})
        store.close()
        resumed = SQLiteRunStore(path)
        assert resumed.get_run("run").state == expected
        assert json.loads(resumed.checkpoint("run")["payload_json"]) == (
            {} if index == 0 else {"resume": expected.value}
        )
    path = tmp_path / "retry.db"
    store = SQLiteRunStore(path)
    _run(store, "run")
    for state in flow[1:6]:
        store.transition("run", state)
    store.transition("run", RunState.IMPLEMENT, {"retry": 1})
    store.close()
    resumed = SQLiteRunStore(path)
    assert resumed.get_run("run").state == RunState.IMPLEMENT
    assert json.loads(resumed.checkpoint("run")["payload_json"]) == {"retry": 1}


def test_metrics_are_durable_redacted_bounded_exportable_and_retained(tmp_path):
    path = tmp_path / "metrics.db"
    collector = SQLiteMetricsCollector(path, redaction=RedactionPolicy(max_string=8, max_items=2))
    collector.record(
        MetricEvent(
            "model",
            "chat",
            "success",
            run_id="r",
            duration_ms=3,
            input_tokens=10,
            output_tokens=2,
            cost_usd=0.01,
            attributes={
                "access_token": "unsafe",
                "nested": {
                    "dbPassword": "bad",
                    "X-API-Key": "bad",
                    "authorization_header": "bad",
                    "session_cookie": "bad",
                    "text": "long-value",
                },
                "ignored": 1,
            },
        )
    )
    collector.connection.close()
    collector = SQLiteMetricsCollector(path)
    event = collector.events()[0]
    assert event.attributes == {
        "access_t": "[REDACTED]",
        "nested": {"X-API-Ke": "[REDACTED]", "dbPasswo": "[REDACTED]"},
    }
    assert RedactionPolicy().redact(
        {
            "X-API-Key": "bad",
            "authorization_header": "bad",
            "session_cookie": "bad",
        }
    ) == {
        "X-API-Key": "[REDACTED]",
        "authorization_header": "[REDACTED]",
        "session_cookie": "[REDACTED]",
    }
    assert collector.events(run_id="other") == []

    class Exporter:
        def export(self, events):
            self.events = events

    exporter = Exporter()
    assert collector.export(exporter) == 1
    assert len(exporter.events) == 1
    assert collector.export(exporter) == 0
    collector.connection.execute(
        "UPDATE metric_events SET timestamp=?",
        ((datetime.now(UTC) - timedelta(days=31)).isoformat(),),
    )
    collector.connection.commit()
    assert collector.enforce_retention(RetentionPolicy(metric_days=30)) == 1
    with pytest.raises(ValueError):
        RetentionPolicy(metric_days=0)


def test_metrics_reject_unbounded_labels_and_invalid_numbers(tmp_path):
    collector = SQLiteMetricsCollector(tmp_path / "metrics.db")
    with pytest.raises(ValueError, match="low-cardinality"):
        collector.record(MetricEvent("model", "secret value in a label", "success"))
    with pytest.raises(ValueError, match="finite"):
        collector.record(MetricEvent("model", "chat", "success", value=math.inf))
    with pytest.raises(ValueError, match="token"):
        collector.record(MetricEvent("model", "chat", "success", input_tokens=-1))


def test_failed_export_remains_pending_for_retry(tmp_path):
    collector = SQLiteMetricsCollector(tmp_path / "metrics.db")
    collector.record(MetricEvent("policy", "authorize", "denied"))

    class BrokenExporter:
        def export(self, events):
            raise RuntimeError("unavailable")

    with pytest.raises(RuntimeError, match="unavailable"):
        collector.export(BrokenExporter())
    assert len(collector.events(pending_only=True)) == 1


def test_metrics_batch_snapshot_ids_nonfinite_attributes_and_pending_retention(tmp_path):
    collector = SQLiteMetricsCollector(tmp_path / "metrics.db", batch_size=1)
    first = collector.record(MetricEvent("policy", "authorize", "allowed"))
    second = collector.record(MetricEvent("policy", "authorize", "denied"))

    class Exporter:
        def export(self, events):
            self.events = events

    exporter = Exporter()
    assert collector.export(exporter) == 1
    assert [event.event_id for event in exporter.events] == [first]
    assert [event.event_id for event in collector.events(pending_only=True)] == [second]
    collector.connection.execute(
        "UPDATE metric_events SET timestamp=? WHERE id=?",
        ((datetime.now(UTC) - timedelta(days=31)).isoformat(), second),
    )
    collector.connection.commit()
    assert collector.enforce_retention(RetentionPolicy()) == 0
    with pytest.raises(ValueError, match="finite"):
        collector.record(MetricEvent("model", "chat", "success", attributes={"nested": [math.nan]}))
    with pytest.raises(ValueError, match="JSON primitives"):
        collector.record(MetricEvent("model", "chat", "success", attributes={"data": b"secret"}))
    with pytest.raises(TypeError, match="keys must be strings"):
        collector.record(MetricEvent("model", "chat", "success", attributes={1: "value"}))


def test_golden_task_rejects_duplicate_validation_names(tmp_path):
    fixture = json.loads(Path("tests/fixtures/golden/add_greeting.json").read_text())
    fixture["required_validations"] = ["unit", "unit"]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(fixture))
    with pytest.raises(ValueError, match="unique"):
        GoldenTask.load(path)


def test_runner_and_gateway_emit_model_policy_and_capability_metrics(tmp_path):
    metrics = SQLiteMetricsCollector(tmp_path / "metrics.db")
    capability = fixture_read_capability()
    context = InvocationContext("principal", "run", "ANALYZE", "story", "repo", "workspace")
    gateway = CapabilityGateway(
        InMemoryCapabilityCatalog((capability,)),
        StaticPolicyEngine(frozenset({"fixture.read"})),
        InMemoryAuditSink(),
        context,
        metrics=metrics,
    )
    provider = ScriptedProvider((ChatResponse("done", "model", "scripted", usage=Usage(7, 3)),))

    assert AgentRunner(provider, "model", gateway, metrics=metrics).run("task") == "done"
    assert gateway.invoke("fixture.read", {"key": "x"}).result["value"] == "fixture-value"

    events = metrics.events(run_id="run")
    assert {(event.category, event.name, event.outcome) for event in events} == {
        ("model", "chat", "success"),
        ("policy", "capability_decision", "allow"),
        ("capability", "invoke", "success"),
    }
    model = next(event for event in events if event.category == "model")
    assert (model.input_tokens, model.output_tokens) == (7, 3)


def test_observability_failure_does_not_change_runtime_semantics():
    class BrokenMetrics:
        def record(self, event):
            raise RuntimeError("metrics unavailable")

    capability = fixture_read_capability()
    gateway = CapabilityGateway(
        InMemoryCapabilityCatalog((capability,)),
        StaticPolicyEngine(frozenset({"fixture.read"})),
        InMemoryAuditSink(),
        InvocationContext("principal", "run", "ANALYZE", "story", "repo", "workspace"),
        metrics=BrokenMetrics(),
    )
    assert gateway.invoke("fixture.read", {"key": "x"}).status.value == "SUCCEEDED"
