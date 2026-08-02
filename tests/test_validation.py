import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from agent_runtime import (
    ArtifactStore,
    CoordinatorLimits,
    ImplementationValidationCoordinator,
    RunState,
    SQLiteRunStore,
    ValidationService,
    WorkspaceManager,
)
from agent_runtime.validation import (
    BubblewrapSandboxBackend,
    SandboxLimits,
    ValidationProfile,
    ValidationResult,
)

requires_bubblewrap = pytest.mark.skipif(
    sys.platform != "linux", reason="Bubblewrap backend is Linux-only"
)


def backend(tmp_path: Path, **limits: object) -> BubblewrapSandboxBackend:
    return BubblewrapSandboxBackend(
        tmp_path / "scratch",
        limits=SandboxLimits(**limits),
        artifacts=ArtifactStore(tmp_path / "artifacts"),
    )


def run_python(sandbox: BubblewrapSandboxBackend, workspace: Path, source: str):
    return sandbox.run(
        ValidationProfile("fixture", "test", ("/usr/bin/python3", "-c", source)), workspace
    )


@requires_bubblewrap
def test_copy_is_disposable_environment_and_host_are_hidden(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "generation"
    workspace.mkdir()
    (workspace / "value").write_text("before")
    sentinel = tmp_path / "host-secret"
    sentinel.write_text("secret")
    monkeypatch.setenv("UNAPPROVED_SECRET", "secret")
    result = run_python(
        backend(tmp_path),
        workspace,
        f"import os; assert 'UNAPPROVED_SECRET' not in os.environ; "
        f"assert not os.path.exists({str(sentinel)!r}); open('value','w').write('after')",
    )
    assert result.passed, result.stderr
    assert (workspace / "value").read_text() == "before"


@requires_bubblewrap
def test_network_namespace_has_no_interfaces_except_loopback(tmp_path: Path):
    workspace = tmp_path / "generation"
    workspace.mkdir()
    result = run_python(
        backend(tmp_path), workspace, "import socket; s=socket.socket(); s.connect(('1.1.1.1',80))"
    )
    assert not result.passed


@requires_bubblewrap
def test_timeout_and_cancel(tmp_path: Path):
    workspace = tmp_path / "generation"
    workspace.mkdir()
    result = run_python(backend(tmp_path, wall_seconds=0.1), workspace, "while True: pass")
    assert result.timed_out and not result.passed

    cancel = threading.Event()
    sandbox = backend(tmp_path / "cancel", wall_seconds=10)
    output = []
    thread = threading.Thread(
        target=lambda: output.append(
            sandbox.run(
                ValidationProfile(
                    "cancel", "test", ("/usr/bin/python3", "-c", "import time; time.sleep(20)")
                ),
                workspace,
                cancel=cancel,
            )
        )
    )
    thread.start()
    time.sleep(0.1)
    cancel.set()
    thread.join(3)
    assert not thread.is_alive() and output[0].cancelled


@requires_bubblewrap
def test_output_is_bounded_and_spilled(tmp_path: Path):
    workspace = tmp_path / "generation"
    workspace.mkdir()
    sandbox = backend(tmp_path, output_bytes=32, full_output_bytes=4096)
    result = run_python(sandbox, workspace, "print('x'*1000)")
    assert result.passed
    assert len(result.stdout.encode()) == 32
    assert result.stdout_truncated and result.output_artifact


@requires_bubblewrap
def test_profile_is_argv_not_shell(tmp_path: Path):
    workspace = tmp_path / "generation"
    workspace.mkdir()
    result = backend(tmp_path).run(
        ValidationProfile("argv", "lint", ("/usr/bin/printf", "%s", "$(id)")), workspace
    )
    assert result.passed and result.stdout == "$(id)"


def git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()


def coordinator_fixture(tmp_path: Path):
    repository = tmp_path / "repository"
    repository.mkdir()
    git(repository, "init", "-q")
    git(repository, "config", "user.email", "test@example.test")
    git(repository, "config", "user.name", "Test")
    (repository / "value.txt").write_text("old\n")
    git(repository, "add", "value.txt")
    git(repository, "commit", "-qm", "base")
    revision = git(repository, "rev-parse", "HEAD")
    store = SQLiteRunStore(tmp_path / "runs.db")
    store.create_run(
        "run",
        story_hash="story",
        repository="repo",
        base_revision=revision,
        provider="scripted",
        model="model",
        prompt_version="1",
        policy_version="1",
    )
    for state in (RunState.INTAKE, RunState.ANALYZE, RunState.PLAN_READY):
        store.transition("run", state)
    manager = WorkspaceManager(tmp_path / "workspaces", store, {"repo": repository})
    profile = ValidationProfile(
        "unit",
        "test",
        (
            "/usr/bin/python3",
            "-c",
            "import pathlib,sys; sys.exit(pathlib.Path('value.txt').read_text() != 'good\\n')",
        ),
    )

    class DeterministicBackend:
        def run(self, selected, generation, *, cancel=None):
            del cancel
            passed = (generation / "value.txt").read_text() == "good\n"
            return ValidationResult(
                selected.profile_id, passed, 0 if passed else 1, "", "", False, False
            )

    service = ValidationService(manager, store, DeterministicBackend(), (profile,))
    return store, manager, service


def patch(before: str, after: str) -> str:
    return f"--- a/value.txt\n+++ b/value.txt\n@@ -1,1 +1,1 @@\n-{before}\n+{after}\n"


def test_bounded_coordinator_corrects_failure_and_requires_passing_validation(tmp_path: Path):
    store, manager, service = coordinator_fixture(tmp_path)
    attempts = []

    def implement(attempt: int):
        attempts.append(attempt)
        manager.apply_patch(
            "run", patch("old" if attempt == 1 else "bad", "bad" if attempt == 1 else "good")
        )
        return 1, 10, 0.01

    coordinator = ImplementationValidationCoordinator(
        store, service, CoordinatorLimits(attempts=3, turns=3, tokens=100, cost=1)
    )

    assert coordinator.run("run", implement)
    assert attempts == [1, 2]
    assert store.get_run("run").state == RunState.AWAITING_PUBLISH_APPROVAL
    assert manager.diff("run")["changed_files"] == ["value.txt"]
    assert (
        store.connection.execute("SELECT COUNT(*) FROM validation_checkpoints").fetchone()[0] == 2
    )


def test_coordinator_resumes_validation_after_persisted_edit_without_reimplementation(
    tmp_path: Path,
):
    store, manager, service = coordinator_fixture(tmp_path)
    store.transition("run", RunState.IMPLEMENT)
    manager.apply_patch("run", patch("old", "good"))
    store.transition("run", RunState.VALIDATE)
    coordinator = ImplementationValidationCoordinator(store, service)

    assert coordinator.run(
        "run", lambda _attempt: (_ for _ in ()).throw(AssertionError("must not implement"))
    )
    assert store.get_run("run").state == RunState.AWAITING_PUBLISH_APPROVAL


def test_coordinator_resumes_passed_final_checkpoint_without_rerunning(tmp_path: Path):
    store, manager, service = coordinator_fixture(tmp_path)
    store.transition("run", RunState.IMPLEMENT)
    manager.apply_patch("run", patch("old", "good"))
    store.transition("run", RunState.VALIDATE)
    result = ValidationResult("unit", True, 0, "", "", False, False)
    generation, _ = manager._generation(manager.acquire("run"))
    store.begin_validation("run", 3, "unit", generation)
    store.finish_validation("run", 3, "unit", asdict(result))
    store.save_validation_checkpoint("run", 3, [{"profile_id": "unit", "passed": True}])

    class FailingBackend:
        def run(self, *_args, **_kwargs):
            raise AssertionError("validation must not rerun")

    resumed_service = ValidationService(
        manager, store, FailingBackend(), tuple(service.profiles.values())
    )
    coordinator = ImplementationValidationCoordinator(
        store, resumed_service, CoordinatorLimits(attempts=3)
    )
    assert coordinator.run(
        "run", lambda _attempt: (_ for _ in ()).throw(AssertionError("must not implement"))
    )
    assert store.get_run("run").state == RunState.AWAITING_PUBLISH_APPROVAL


def test_coordinator_stops_before_validation_when_budget_is_exceeded(tmp_path: Path):
    store, _manager, service = coordinator_fixture(tmp_path)
    coordinator = ImplementationValidationCoordinator(
        store, service, CoordinatorLimits(attempts=2, turns=1, tokens=10, cost=0.1)
    )

    assert not coordinator.run("run", lambda _attempt: (2, 1, 0.01))
    assert store.get_run("run").state == RunState.IMPLEMENT
    assert store.connection.execute("SELECT COUNT(*) FROM validation_attempts").fetchone()[0] == 0
