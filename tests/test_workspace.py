import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent_runtime import (
    CapabilityGateway,
    DurableAuditSink,
    InMemoryCapabilityCatalog,
    InvocationContext,
    RunState,
    SQLiteRunStore,
    StaticPolicyEngine,
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
