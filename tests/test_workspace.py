import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent_runtime import (
    AgentRunner,
    CapabilityGateway,
    ChatResponse,
    DurableAuditSink,
    InMemoryCapabilityCatalog,
    InvocationContext,
    RunState,
    SQLiteRunStore,
    StaticPolicyEngine,
    ToolCall,
    WorkspaceError,
    WorkspaceLimits,
    WorkspaceManager,
    workspace_capabilities,
)


def git(path, *args):
    return subprocess.run(
        ["git", *args], cwd=path, check=True, capture_output=True, text=True
    ).stdout


def setup_workspace(tmp_path, *, limits=None):
    repository = tmp_path / "repository"
    repository.mkdir()
    git(repository, "init", "-q")
    git(repository, "config", "user.email", "test@example.test")
    git(repository, "config", "user.name", "Test")
    (repository / "file.txt").write_text("one\ntwo\n")
    git(repository, "add", "file.txt")
    git(repository, "commit", "-qm", "base")
    revision = git(repository, "rev-parse", "HEAD").strip()
    store = SQLiteRunStore(tmp_path / "runs.db")
    store.create_run(
        "run",
        story_hash="x",
        repository="repo",
        base_revision=revision,
        provider="p",
        model="m",
        prompt_version="1",
        policy_version="1",
    )
    for state in (RunState.INTAKE, RunState.ANALYZE, RunState.PLAN_READY, RunState.IMPLEMENT):
        store.transition("run", state)
    manager = WorkspaceManager(tmp_path / "runtime", store, {"repo": repository}, limits)
    return repository, store, manager


def test_archive_ignores_repo_filters_and_generations_replay(tmp_path):
    repository, store, manager = setup_workspace(tmp_path)
    marker = tmp_path / "marker"
    git(repository, "config", "filter.evil.smudge", f"sh -c 'touch {marker}; cat'")
    (repository / ".git" / "info" / "attributes").write_text("* filter=evil\n")
    workspace = manager.acquire("run")
    assert not marker.exists()
    result = manager.create_file("run", "nested/new.txt", "safe\n")
    assert result["changed_files"] == ["nested/new.txt"]
    assert manager.create_file("run", "nested/new.txt", "safe\n") == result
    assert store.connection.execute("SELECT current_generation FROM workspaces").fetchone()[0] == 1
    assert (workspace.path / "gen-0" / "file.txt").read_text() == "one\ntwo\n"


def test_bounded_file_read_tracks_current_generation(tmp_path):
    _, _, manager = setup_workspace(tmp_path)
    first = manager.read_file("run", "file.txt", max_lines=1)
    assert first["generation"] == 0
    assert first["content"] == "one\n"
    assert first["next_start_line"] == 2 and not first["eof"]
    second = manager.read_file("run", "file.txt", start_line=2)
    assert second["content"] == "two\n" and second["eof"]

    manager.apply_patch(
        "run", "--- a/file.txt\n+++ b/file.txt\n@@ -1,2 +1,2 @@\n-one\n+ONE\n two\n"
    )
    current = manager.read_file("run", "file.txt")
    assert current["generation"] == 1
    assert current["content"] == "ONE\ntwo\n"
    assert current["content_sha256"] != first["content_sha256"]


def test_file_read_rejects_unsafe_binary_and_unbounded_lines(tmp_path):
    _, _, manager = setup_workspace(tmp_path)
    for path in ("../escape", "/absolute", ".git/config", "a/.GIT/x"):
        with pytest.raises(WorkspaceError):
            manager.read_file("run", path)
    workspace = manager.acquire("run")
    (workspace.path / "gen-0" / "binary.dat").write_bytes(b"\xff\x00")
    with pytest.raises(WorkspaceError, match="UTF-8"):
        manager.read_file("run", "binary.dat")
    (workspace.path / "gen-0" / "long.txt").write_text("x" * 32_001)
    with pytest.raises(WorkspaceError, match="line exceeds"):
        manager.read_file("run", "long.txt")


def test_file_read_rejects_parent_swapped_after_verification(tmp_path):
    _, _, manager = setup_workspace(tmp_path)
    manager.create_file("run", "nested/file.txt", "safe\n")
    workspace = manager.acquire("run")
    _, current = manager._generation(workspace)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "file.txt").write_text("secret\n")
    original_verify = manager._verify
    calls = 0

    def racing_verify(active_workspace):
        nonlocal calls
        calls += 1
        original_verify(active_workspace)
        if calls == 2:
            shutil.rmtree(current / "nested")
            os.symlink(outside, current / "nested")

    manager._verify = racing_verify
    with pytest.raises(WorkspaceError, match="opened safely"):
        manager.read_file("run", "nested/file.txt")


def test_durable_workspace_reads_use_runtime_invocation_keys_and_recover(tmp_path):
    _, store, manager = setup_workspace(tmp_path)
    workspace = manager.acquire("run")
    capabilities = workspace_capabilities(manager)
    gateway = CapabilityGateway(
        InMemoryCapabilityCatalog(capabilities),
        StaticPolicyEngine(frozenset({"workspace.file.read"})),
        DurableAuditSink(store),
        InvocationContext("principal", "run", "IMPLEMENT", workspace_id=workspace.workspace_id),
        store,
    )
    arguments = {"path": "file.txt"}
    first = gateway.invoke(
        "workspace.file.read", arguments, invocation_key="attempt:1:turn:1:tool:1"
    )
    manager.apply_patch(
        "run", "--- a/file.txt\n+++ b/file.txt\n@@ -1,2 +1,2 @@\n-one\n+ONE\n two\n"
    )
    second = gateway.invoke(
        "workspace.file.read", arguments, invocation_key="attempt:1:turn:2:tool:1"
    )
    assert first.result["generation"] == 0
    assert second.result["generation"] == 1

    intent = store.begin_invocation(
        "run",
        "IMPLEMENT",
        "workspace.file.read",
        arguments,
        invocation_key="attempt:1:turn:3:tool:1",
    )
    assert store.recover_running_invocations("run") == 1
    recovered = gateway.invoke(
        "workspace.file.read", arguments, invocation_key="attempt:1:turn:3:tool:1"
    )
    assert recovered.execution_id == intent.execution_id
    assert recovered.status == "SUCCEEDED" and recovered.result["generation"] == 1


def test_agent_runner_keys_identical_reads_by_session_turn(tmp_path):
    _, store, manager = setup_workspace(tmp_path)
    workspace = manager.acquire("run")
    gateway = CapabilityGateway(
        InMemoryCapabilityCatalog(workspace_capabilities(manager)),
        StaticPolicyEngine(frozenset({"workspace.file.read"})),
        DurableAuditSink(store),
        InvocationContext("principal", "run", "IMPLEMENT", workspace_id=workspace.workspace_id),
        store,
    )

    def read_response(call_id):
        return ChatResponse(
            "",
            "model",
            "scripted",
            tool_calls=(
                ToolCall(
                    call_id,
                    "capability_invoke",
                    json.dumps(
                        {
                            "capability_id": "workspace.file.read",
                            "arguments": {"path": "file.txt"},
                        }
                    ),
                ),
            ),
        )

    class ReadsAcrossMutation:
        name = "scripted"

        def __init__(self):
            self.turn = 0

        def chat(self, _model, request):
            self.turn += 1
            if self.turn == 1:
                return read_response("read-1")
            result = json.loads(request.messages[-1].content)["data"]["result"]
            if self.turn == 2:
                assert result["generation"] == 0
                manager.apply_patch(
                    "run",
                    "--- a/file.txt\n+++ b/file.txt\n@@ -1,2 +1,2 @@\n-one\n+ONE\n two\n",
                )
                return read_response("read-2")
            assert result["generation"] == 1
            return ChatResponse("done", "model", "scripted")

        def close(self):
            pass

    assert (
        AgentRunner(ReadsAcrossMutation(), "model", gateway).run(
            "read twice", session_id="attempt-1"
        )
        == "done"
    )
    assert (
        store.connection.execute(
            "SELECT COUNT(*) FROM invocations WHERE capability_id='workspace.file.read'"
        ).fetchone()[0]
        == 2
    )

    class RetryFirstTurn:
        name = "scripted"

        def __init__(self):
            self.turn = 0

        def chat(self, _model, request):
            self.turn += 1
            if self.turn == 1:
                return read_response("provider-regenerated-call-id")
            result = json.loads(request.messages[-1].content)["data"]["result"]
            assert result["generation"] == 0  # same durable turn replays the same execution
            return ChatResponse("recovered", "model", "scripted")

        def close(self):
            pass

    assert (
        AgentRunner(RetryFirstTurn(), "model", gateway).run("read twice", session_id="attempt-1")
        == "recovered"
    )
    assert (
        store.connection.execute(
            "SELECT COUNT(*) FROM invocations WHERE capability_id='workspace.file.read'"
        ).fetchone()[0]
        == 2
    )


def test_failed_candidate_and_interrupted_candidate_do_not_publish(tmp_path):
    _, store, manager = setup_workspace(tmp_path, limits=WorkspaceLimits(maximum_bytes=20))
    workspace = manager.acquire("run")
    with pytest.raises(WorkspaceError, match="limit"):
        manager.create_file("run", "large.txt", "x" * 100)
    assert manager.diff("run")["changed_files"] == []
    candidate = workspace.path / "candidate-interrupted"
    candidate.mkdir()
    (candidate / "bad").write_text("bad")
    manager.acquire("run")
    assert not candidate.exists()
    assert store.connection.execute("SELECT current_generation FROM workspaces").fetchone()[0] == 0


@pytest.mark.parametrize("path", ["../escape", "/absolute", ".git/config", "a/.GIT/x"])
def test_create_rejects_traversal_and_git_paths(tmp_path, path):
    _, _, manager = setup_workspace(tmp_path)
    with pytest.raises(WorkspaceError):
        manager.create_file("run", path, "x")
    assert manager.diff("run")["changed_files"] == []


def test_patch_rejects_modes_symlinks_and_context_mismatch(tmp_path):
    _, _, manager = setup_workspace(tmp_path)
    workspace = manager.acquire("run")
    mode_patch = "old mode 100644\nnew mode 120000\n--- a/file.txt\n+++ b/file.txt\n"
    with pytest.raises(WorkspaceError, match="mode"):
        manager.apply_patch("run", mode_patch)
    os.symlink("/tmp", workspace.path / "gen-0" / "escape")
    with pytest.raises(WorkspaceError, match="unsafe|non-regular"):
        manager.diff("run")


def test_patch_success_stage_recheck_and_bounded_diff(tmp_path):
    _, store, manager = setup_workspace(tmp_path, limits=WorkspaceLimits(maximum_diff_bytes=100))
    patch = "--- a/file.txt\n+++ b/file.txt\n@@ -1,2 +1,2 @@\n-one\n+ONE\n two\n"
    assert manager.apply_patch("run", patch)["changed_files"] == ["file.txt"]
    store.transition("run", RunState.VALIDATE)
    with pytest.raises(WorkspaceError, match="IMPLEMENT"):
        manager.create_file("run", "later.txt", "no")
    assert "later.txt" not in manager.diff("run")["changed_files"]


def test_snapshot_failure_rolls_back_generation(tmp_path):
    _, store, manager = setup_workspace(tmp_path, limits=WorkspaceLimits(maximum_diff_bytes=10))
    with pytest.raises(WorkspaceError, match="diff"):
        manager.create_file("run", "new.txt", "a long new value\n")
    assert store.connection.execute("SELECT current_generation FROM workspaces").fetchone()[0] == 0
    assert not list(manager.acquire("run").path.glob("candidate-*"))


def test_checkpoint_failure_does_not_publish_generation(tmp_path):
    _, store, manager = setup_workspace(tmp_path)
    manager.acquire("run")
    store.connection.execute(
        "CREATE TRIGGER fail_workspace_checkpoint BEFORE INSERT ON workspace_checkpoints "
        "BEGIN SELECT RAISE(ABORT, 'checkpoint failure'); END"
    )

    with pytest.raises(Exception, match="checkpoint failure"):
        manager.create_file("run", "new.txt", "value\n")

    assert store.connection.execute("SELECT current_generation FROM workspaces").fetchone()[0] == 0
    assert manager.diff("run")["changed_files"] == []


def test_two_managers_serialize_mutations(tmp_path):
    repository, store, manager = setup_workspace(tmp_path)
    database = tmp_path / "runs.db"
    runtime = tmp_path / "runtime"
    manager.acquire("run")

    def create(path: str) -> None:
        worker_store = SQLiteRunStore(database)
        try:
            WorkspaceManager(runtime, worker_store, {"repo": repository}).create_file(
                "run", path, path
            )
        finally:
            worker_store.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(create, ("one.txt", "two.txt")))

    assert manager.diff("run")["changed_files"] == ["one.txt", "two.txt"]
    assert store.connection.execute("SELECT current_generation FROM workspaces").fetchone()[0] == 2


def test_recovered_outer_invocation_reconciles_committed_generation(tmp_path):
    repository, store, manager = setup_workspace(tmp_path)
    database = tmp_path / "runs.db"
    workspace = manager.acquire("run")
    arguments = {"path": "recovered.txt", "content": "once\n"}
    intent = store.begin_invocation("run", "IMPLEMENT", "workspace.file.create", arguments)
    manager.create_file("run", **arguments)
    store.close()

    resumed = SQLiteRunStore(database)
    assert resumed.recover_running_invocations("run") == 1
    resumed_manager = WorkspaceManager(tmp_path / "runtime", resumed, {"repo": repository})
    capability = next(
        item
        for item in workspace_capabilities(resumed_manager)
        if item.descriptor.card.capability_id == "workspace.file.create"
    )
    gateway = CapabilityGateway(
        InMemoryCapabilityCatalog((capability,)),
        StaticPolicyEngine(frozenset({"workspace.file.create"})),
        DurableAuditSink(resumed),
        InvocationContext("principal", "run", "IMPLEMENT", workspace_id=workspace.workspace_id),
        resumed,
    )

    result = gateway.invoke("workspace.file.create", arguments)

    assert result.status == "SUCCEEDED"
    assert result.execution_id == intent.execution_id
    assert (
        resumed.connection.execute("SELECT current_generation FROM workspaces").fetchone()[0] == 1
    )
    assert resumed_manager.diff("run")["changed_files"] == ["recovered.txt"]
