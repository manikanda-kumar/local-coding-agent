import pytest

from agent_runtime import (
    CapabilityGateway,
    ContinuityDecision,
    ContinuityService,
    DurableAuditSink,
    ExecutionStatus,
    GatewayError,
    InMemoryCapabilityCatalog,
    InvocationContext,
    RunState,
    SQLiteRunStore,
    StaticPolicyEngine,
    continuity_memory_capability,
    fixture_read_capability,
)
from agent_runtime.continuity import MAX_ACTIVITY


def make_store(path):
    store = SQLiteRunStore(path)
    store.create_run(
        "run",
        story_hash="story",
        repository="org/repo",
        base_revision="abc",
        provider="provider",
        model="model",
        prompt_version="v1",
        policy_version="v1",
    )
    return store


def gateway(store, stage="IMPLEMENT", allowed=True):
    capability = continuity_memory_capability(ContinuityService(store))
    return CapabilityGateway(
        InMemoryCapabilityCatalog((capability,)),
        StaticPolicyEngine(frozenset({"continuity.memory.update"}) if allowed else frozenset()),
        DurableAuditSink(store),
        InvocationContext("runtime", "run", stage),
        store,
    )


def test_exact_resume_and_runtime_owned_immutable_decisions(tmp_path):
    path = tmp_path / "runs.db"
    store = make_store(path)
    memory = ContinuityService(store)
    memory.initialize("run", "Fix the issue", ("Do not publish",))
    memory.add_decision("run", "Use SQLite", source="runtime", provenance="story:ABC-1")
    expected = memory.update("run", {"completed": ("Read code",), "next": ("Test",)}, revision=0)
    store.close()

    resumed = ContinuityService(SQLiteRunStore(path)).get("run")
    assert resumed == expected
    assert resumed.decisions == (ContinuityDecision("Use SQLite", "runtime", "story:ABC-1"),)
    with pytest.raises(AttributeError):
        resumed.decisions[0].source = "model"


def test_bounds_revision_atomicity_and_bounded_history(tmp_path):
    store = make_store(tmp_path / "runs.db")
    memory = ContinuityService(store)
    memory.initialize("run", "goal")
    with pytest.raises(ValueError, match="stale"):
        memory.update("run", {"completed": ("x",)}, revision=1)
    with pytest.raises(ValueError, match="at most"):
        memory.update("run", {"completed": ("x" * 2001,)}, revision=0)
    assert memory.get("run").revision == 0

    store.connection.execute(
        "CREATE TRIGGER fail_memory_history BEFORE INSERT ON continuity_updates "
        "BEGIN SELECT RAISE(ABORT, 'history failure'); END"
    )
    with pytest.raises(Exception, match="history failure"):
        memory.update("run", {"completed": ("x",)}, revision=0)
    assert memory.get("run").revision == 0
    store.connection.execute("DROP TRIGGER fail_memory_history")
    fields = ("completed", "next", "working_set", "learnings")
    for revision in range(33):
        memory.update(
            "run", {fields[revision % len(fields)]: (f"item-{revision}",)}, revision=revision
        )
    assert store.connection.execute("SELECT COUNT(*) FROM continuity_updates").fetchone()[0] == 32
    with pytest.raises(ValueError, match="append at least one"):
        memory.update("run", {"next": ()}, revision=33)


def test_capability_cannot_mutate_authority_and_obeys_schema_policy_and_stage(tmp_path):
    store = make_store(tmp_path / "runs.db")
    ContinuityService(store).initialize("run", "goal")
    active = gateway(store)
    for field in ("goal", "constraints", "decisions", "policy", "authority", "principal_id"):
        with pytest.raises(GatewayError, match="unknown field"):
            active.invoke("continuity.memory.update", {"revision": 0, field: "attacker"})
    with pytest.raises(GatewayError, match="too long"):
        active.invoke("continuity.memory.update", {"revision": 0, "learnings": ["x" * 2001]})
    assert (
        active.invoke(
            "continuity.memory.update", {"revision": 0, "working_set": ["src/a.py"]}
        ).status
        == ExecutionStatus.SUCCEEDED
    )
    assert ContinuityService(store).get("run").working_set == ("src/a.py",)

    with pytest.raises(GatewayError, match="denied"):
        gateway(store, allowed=False).describe("continuity.memory.update")
    for stage in ("PUBLISH", "REPORT", "SUCCEEDED", "FAILED", "CANCELLED"):
        denied = gateway(store, stage=stage)
        assert denied.search("continuity") == ()
        with pytest.raises(GatewayError, match="external or terminal"):
            denied.invoke("continuity.memory.update", {"revision": 1, "next": ["x"]})
    with pytest.raises(GatewayError, match="external or terminal"):
        gateway(store, stage="invented").invoke(
            "continuity.memory.update", {"revision": 1, "next": ["x"]}
        )


def test_durable_state_recheck_denies_stale_gateway_context(tmp_path):
    store = make_store(tmp_path / "runs.db")
    ContinuityService(store).initialize("run", "goal")
    stale = gateway(store, stage="IMPLEMENT")
    for state in (
        RunState.INTAKE,
        RunState.ANALYZE,
        RunState.PLAN_READY,
        RunState.IMPLEMENT,
        RunState.VALIDATE,
        RunState.AWAITING_PUBLISH_APPROVAL,
        RunState.PUBLISH,
    ):
        store.transition("run", state)
    record = stale.invoke(
        "continuity.memory.update", {"revision": 0, "completed": ["must not commit"]}
    )
    assert record.status == ExecutionStatus.FAILED
    assert ContinuityService(store).get("run").revision == 0
    assert store.connection.execute("SELECT COUNT(*) FROM continuity_updates").fetchone()[0] == 0


def test_interrupted_memory_invocation_replays_exact_update_once(tmp_path):
    path = tmp_path / "runs.db"
    store = make_store(path)
    memory = ContinuityService(store)
    memory.initialize("run", "goal")
    arguments = {"revision": 0, "completed": ["once"]}
    invocation = store.begin_invocation("run", "IMPLEMENT", "continuity.memory.update", arguments)
    memory.update("run", {"completed": ("once",)}, revision=0)
    store.close()

    resumed = SQLiteRunStore(path)
    assert resumed.recover_running_invocations("run") == 1
    record = gateway(resumed).invoke("continuity.memory.update", arguments)
    assert record.status == ExecutionStatus.SUCCEEDED
    assert record.invocation_id == invocation.invocation_id
    ledger = ContinuityService(resumed).get("run")
    assert ledger.completed == ("once",) and ledger.revision == 1
    assert resumed.connection.execute("SELECT COUNT(*) FROM continuity_updates").fetchone()[0] == 1


def test_activity_is_derived_from_persisted_transitions_and_capability_events(tmp_path):
    store = make_store(tmp_path / "runs.db")
    ContinuityService(store).initialize("run", "model prose must not become activity")
    store.transition("run", RunState.INTAKE)
    record = gateway(store, stage="INTAKE").invoke(
        "continuity.memory.update", {"revision": 0, "completed": ["model supplied note"]}
    )
    activity = ContinuityService(store).activity("run")
    assert {item["kind"] for item in activity} == {"transition", "capability"}
    assert any(item.get("to") == "INTAKE" for item in activity)
    assert any(item.get("status") == record.status for item in activity)
    assert "model supplied note" not in repr(activity)

    fixture = fixture_read_capability()
    for index in range(MAX_ACTIVITY + 5):
        invocation = store.begin_invocation(
            "run", "INTAKE", fixture.descriptor.card.capability_id, {"key": str(index)}
        )
        store.finish_invocation(
            invocation.invocation_id, result={"index": index}, normalized_version=1
        )
    store.close()
    bounded = ContinuityService(SQLiteRunStore(tmp_path / "runs.db")).activity("run")
    assert len(bounded) == MAX_ACTIVITY
    assert [item["occurred_at"] for item in bounded] == sorted(
        item["occurred_at"] for item in bounded
    )


def test_decision_rows_reject_update_and_delete(tmp_path):
    store = make_store(tmp_path / "runs.db")
    memory = ContinuityService(store)
    memory.initialize("run", "goal")
    memory.add_decision("run", "decision", source="runtime", provenance="event:1")
    with pytest.raises(Exception, match="immutable"):
        store.connection.execute("UPDATE continuity_decisions SET source='model'")
    with pytest.raises(Exception, match="immutable"):
        store.connection.execute("DELETE FROM continuity_decisions")
    assert memory.get("run").decisions == (ContinuityDecision("decision", "runtime", "event:1"),)
