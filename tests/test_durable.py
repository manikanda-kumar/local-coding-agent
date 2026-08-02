import hashlib
import json

import pytest

from agent_runtime import (
    ArtifactStore,
    CapabilityGateway,
    DurableAuditSink,
    ExecutionStatus,
    InMemoryCapabilityCatalog,
    InvalidTransition,
    InvocationContext,
    RunCancelled,
    RunState,
    SQLiteRunStore,
    StaticPolicyEngine,
    fixture_read_capability,
)


def create_run(store: SQLiteRunStore, run_id: str = "run") -> None:
    store.create_run(
        run_id,
        story_hash="story-sha",
        repository="org/repo",
        base_revision="abc123",
        provider="provider",
        model="fixed-model",
        prompt_version="prompt-v1",
        policy_version="policy-v2",
        usage={"input_tokens": 2},
        limits={"turns": 8},
    )


def test_state_machine_checkpoint_resume_and_pins(tmp_path) -> None:
    path = tmp_path / "runs.db"
    store = SQLiteRunStore(path)
    create_run(store)
    with pytest.raises(InvalidTransition):
        store.transition("run", RunState.ANALYZE)
    assert store.get_run("run").state == RunState.NEW
    assert (
        store.connection.execute("SELECT committed FROM transitions ORDER BY id DESC").fetchone()[0]
        == 0
    )

    store.transition("run", RunState.INTAKE, {"snapshot": "artifact"})
    store.close()
    resumed = SQLiteRunStore(path)
    run = resumed.get_run("run")
    checkpoint = resumed.checkpoint("run")
    assert run.state == RunState.INTAKE
    assert (run.story_hash, run.base_revision, run.usage, run.limits) == (
        "story-sha",
        "abc123",
        {"input_tokens": 2},
        {"turns": 8},
    )
    assert (checkpoint["state"], checkpoint["model"], checkpoint["policy_version"]) == (
        "INTAKE",
        "fixed-model",
        "policy-v2",
    )
    assert json.loads(checkpoint["payload_json"]) == {"snapshot": "artifact"}


def test_transition_rolls_back_if_checkpoint_write_fails(tmp_path) -> None:
    store = SQLiteRunStore(tmp_path / "runs.db")
    create_run(store)
    store.connection.execute(
        "CREATE TRIGGER fail_intake_checkpoint BEFORE INSERT ON checkpoints "
        "WHEN NEW.state='INTAKE' BEGIN SELECT RAISE(ABORT, 'checkpoint failure'); END"
    )

    with pytest.raises(Exception, match="checkpoint failure"):
        store.transition("run", RunState.INTAKE)

    assert store.get_run("run").state == RunState.NEW
    assert store.connection.execute("SELECT COUNT(*) FROM transitions").fetchone()[0] == 0


def test_completed_replay_does_not_execute_handler_again(tmp_path) -> None:
    path = tmp_path / "runs.db"
    store = SQLiteRunStore(path)
    create_run(store)
    calls = 0
    capability = fixture_read_capability()
    original = capability.handler

    def counted(arguments, context):
        nonlocal calls
        calls += 1
        return original(arguments, context)

    capability = type(capability)(capability.descriptor, counted)

    def gateway(active_store):
        return CapabilityGateway(
            InMemoryCapabilityCatalog((capability,)),
            StaticPolicyEngine(frozenset({"fixture.read"})),
            DurableAuditSink(active_store),
            InvocationContext("principal", "run", "NEW", policy_version="policy-v2"),
            active_store,
        )

    first = gateway(store).invoke("fixture.read", {"key": "x"})
    store.close()
    restarted = SQLiteRunStore(path)
    replay = gateway(restarted).invoke("fixture.read", {"key": "x"})
    assert calls == 1
    assert replay.execution_id == first.execution_id
    assert replay.status == ExecutionStatus.SUCCEEDED
    assert replay.result == first.result


def test_crash_after_intent_is_terminal_failure_not_false_success(tmp_path) -> None:
    path = tmp_path / "runs.db"
    store = SQLiteRunStore(path)
    create_run(store)
    intent = store.begin_invocation("run", "ANALYZE", "fixture.read", {"key": "x"})
    assert intent.status == "RUNNING"  # durable boundary before external execution
    store.close()  # simulate process death before result persistence

    restarted = SQLiteRunStore(path)
    recovered = restarted.begin_invocation("run", "ANALYZE", "fixture.read", {"key": "x"})
    assert recovered.replayed
    assert recovered.status == "FAILED"
    assert "interrupted" in recovered.error
    with pytest.raises(RuntimeError, match="already terminal"):
        restarted.finish_invocation(recovered.invocation_id, result={"false": "success"})


def test_cancellation_blocks_new_work_and_is_terminal(tmp_path) -> None:
    store = SQLiteRunStore(tmp_path / "runs.db")
    create_run(store)
    store.transition("run", RunState.CANCELLED)
    with pytest.raises(RunCancelled):
        store.begin_invocation("run", "NEW", "fixture.read", {"key": "x"})
    with pytest.raises(InvalidTransition):
        store.transition("run", RunState.INTAKE)


def test_artifacts_are_sha256_addressed_verified_and_deduplicated(tmp_path) -> None:
    artifacts = ArtifactStore(tmp_path / "artifacts")
    content = b"bounded large result"
    digest = artifacts.put(content)
    assert digest == hashlib.sha256(content).hexdigest()
    assert artifacts.put(content) == digest
    assert artifacts.get(digest) == content
    assert len([path for path in (tmp_path / "artifacts").rglob("*") if path.is_file()]) == 1

    path = tmp_path / "artifacts" / digest[:2] / digest[2:]
    path.write_bytes(b"tampered")
    with pytest.raises(OSError, match="hash mismatch"):
        artifacts.get(digest)
