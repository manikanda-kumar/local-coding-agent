import hashlib
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_runtime import (
    ChatResponse,
    ImplementationValidationCoordinator,
    IntakePlanningService,
    JiraCodingAgentWorkflow,
    JiraReportingService,
    PublicationResult,
    PublicationService,
    RunState,
    SQLiteRunStore,
    StorySnapshot,
    ValidationProfile,
    ValidationResult,
    ValidationService,
    WorkspaceManager,
)
from agent_runtime.providers import ScriptedProvider
from agent_runtime.publication import GitHubError
from agent_runtime.reporting import JiraWriteError


def run_store(tmp_path, state=RunState.NEW):
    tmp_path.mkdir(parents=True, exist_ok=True)
    store = SQLiteRunStore(tmp_path / "runs.db")
    store.create_run(
        "run",
        story_hash="pending",
        repository="org/repo",
        base_revision="base",
        provider="scripted",
        model="model",
        prompt_version="v1",
        policy_version="v1",
    )
    flow = [
        RunState.INTAKE,
        RunState.ANALYZE,
        RunState.PLAN_READY,
        RunState.IMPLEMENT,
        RunState.VALIDATE,
        RunState.AWAITING_PUBLISH_APPROVAL,
        RunState.PUBLISH,
        RunState.REPORT,
        RunState.SUCCEEDED,
    ]
    for target in flow:
        if store.get_run("run").state == state:
            break
        store.transition("run", target)
    return store


def workflow(store, *, validation=True):
    calls = []

    class Planning:
        def intake(self, run_id, issue_key):
            calls.append(("intake", issue_key))
            store.transition(run_id, RunState.INTAKE)

        def plan(self, run_id, evidence):
            calls.append(("plan", evidence))
            store.transition(run_id, RunState.ANALYZE)
            store.transition(run_id, RunState.PLAN_READY)
            return "plan"

    class Coordinator:
        def run(self, run_id, implement):
            calls.append(("validate",))
            if validation:
                while store.get_run(run_id).state != RunState.AWAITING_PUBLISH_APPROVAL:
                    current = store.get_run(run_id).state
                    store.transition(
                        run_id,
                        {
                            RunState.PLAN_READY: RunState.IMPLEMENT,
                            RunState.IMPLEMENT: RunState.VALIDATE,
                            RunState.VALIDATE: RunState.AWAITING_PUBLISH_APPROVAL,
                        }[current],
                    )
            return validation

    class Publication:
        def publish(self, run_id, approval_id, *, title):
            calls.append(("publish", approval_id, title))
            store.transition(run_id, RunState.PUBLISH)
            store.transition(run_id, RunState.REPORT)
            return SimpleNamespace(pull_request_url="pr")

        def resume(self, run_id):
            calls.append(("resume_publish",))
            store.transition(run_id, RunState.REPORT)
            return SimpleNamespace(pull_request_url="pr")

    class Reporting:
        def resume(self, run_id, *, transition_approval_id=None):
            calls.append(("report", transition_approval_id))
            store.transition(run_id, RunState.SUCCEEDED)
            return SimpleNamespace(comment_id="1")

    return JiraCodingAgentWorkflow(
        store, Planning(), Coordinator(), Publication(), Reporting()
    ), calls


def test_waits_for_external_approval_then_completes_and_terminal_is_noop(tmp_path):
    store = run_store(tmp_path)
    service, calls = workflow(store)
    first = service.advance("run", "ABC-1", ({"path": "x"},), lambda _: (1, 1, 0.0))
    assert first.state == RunState.AWAITING_PUBLISH_APPROVAL
    assert first.awaiting_publication_approval
    assert not any(call[0] in {"publish", "report"} for call in calls)
    second = service.advance(
        "run",
        "ABC-1",
        (),
        lambda _: (1, 1, 0.0),
        publication_approval_id="human-approved",
        publication_title="Title",
    )
    assert second.state == RunState.SUCCEEDED
    before = list(calls)
    assert service.advance("run", "ABC-1", (), lambda _: (1, 1, 0.0)).state == RunState.SUCCEEDED
    assert calls == before


def test_validation_exhaustion_is_resumable_and_publish_resume_is_used(tmp_path):
    store = run_store(tmp_path, RunState.PLAN_READY)
    service, calls = workflow(store, validation=False)
    outcome = service.advance("run", "ABC-1", (), lambda _: (1, 1, 0.0))
    assert outcome.validation_exhausted and outcome.state == RunState.PLAN_READY
    assert not any(call[0] == "publish" for call in calls)

    resumed = run_store(tmp_path / "other", RunState.PUBLISH)
    service, calls = workflow(resumed)
    assert service.advance("run", "ABC-1", (), lambda _: (1, 1, 0.0)).state == RunState.SUCCEEDED
    assert calls == [("resume_publish",), ("report", None)]


def test_explicit_transition_policy_pauses_reporting_for_external_approval(tmp_path):
    store = run_store(tmp_path, RunState.REPORT)
    service, calls = workflow(store)
    waiting = service.advance(
        "run",
        "ABC-1",
        (),
        lambda _: (1, 1, 0.0),
        require_transition_approval=True,
    )
    assert waiting.state == RunState.REPORT
    assert waiting.awaiting_transition_approval
    assert calls == []

    completed = service.advance(
        "run",
        "ABC-1",
        (),
        lambda _: (1, 1, 0.0),
        transition_approval_id="human-transition-approval",
        require_transition_approval=True,
    )
    assert completed.state == RunState.SUCCEEDED
    assert calls == [("report", "human-transition-approval")]


def git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_real_services_execute_story_to_approved_pr_and_report_once(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    git(repository, "init", "-q")
    git(repository, "config", "user.email", "test@example.test")
    git(repository, "config", "user.name", "Test")
    (repository / "greeting.txt").write_text("hello\n")
    git(repository, "add", ".")
    git(repository, "commit", "-qm", "base")
    revision = git(repository, "rev-parse", "HEAD")
    story_hash = hashlib.sha256(b"story").hexdigest()
    snapshot = StorySnapshot(
        "JIRA-101",
        "Add deterministic greeting",
        "Change greeting.txt to hello world.",
        "Open",
        "Story",
        "2026-08-03T00:00:00Z",
        (),
        story_hash,
    )

    class ReadAdapter:
        calls = 0

        def snapshot(self, issue_key):
            assert issue_key == "JIRA-101"
            self.calls += 1
            return snapshot

    class Sandbox:
        def run(self, profile, generation, *, cancel=None):
            del cancel
            passed = (generation / "greeting.txt").read_text() == "hello world\n"
            return ValidationResult(
                profile.profile_id, passed, 0 if passed else 1, "", "", False, False
            )

    class GitHub:
        calls = 0
        attempts = 0
        published_files = None
        fail_once = False

        def base_sha(self, repository_name, branch):
            assert (repository_name, branch) == ("org/repo", "main")
            return revision

        def branch_sha(self, repository_name, branch):
            assert repository_name == "org/repo" and branch

        def find_pr(self, repository_name, account, branch):
            assert (repository_name, account) == ("org/repo", "org") and branch

        def publish(
            self,
            repository_name,
            base,
            base_sha,
            branch,
            account,
            files,
            title,
            marker,
        ):
            assert (repository_name, base, base_sha, account) == (
                "org/repo",
                "main",
                revision,
                "org",
            )
            assert title == "Implement JIRA-101"
            assert marker == "agent-runtime-run:RUN-E2E"
            self.attempts += 1
            if self.fail_once:
                self.fail_once = False
                raise GitHubError("simulated failure after durable intent")
            self.calls += 1
            self.published_files = files
            return PublicationResult(branch, "commit-1", 101, "https://github.test/pr/101")

    class JiraWrite:
        comments = 0
        create_attempts = 0
        body = ""
        fail_once = False

        def find_comment(self, issue_key, marker):
            assert issue_key == "JIRA-101" and marker
            return "comment-101" if marker in self.body else None

        def create_comment(self, issue_key, body):
            assert issue_key == "JIRA-101"
            self.create_attempts += 1
            self.comments += 1
            self.body = body
            if self.fail_once:
                self.fail_once = False
                raise JiraWriteError("simulated failure after remote comment")
            return "comment-101"

    read, github, jira_write = ReadAdapter(), GitHub(), JiraWrite()
    store_path = tmp_path / "runs.db"
    workspace_root = tmp_path / "workspaces"
    store = SQLiteRunStore(store_path)
    store.create_run(
        "RUN-E2E",
        story_hash="pending",
        repository="org/repo",
        base_revision=revision,
        provider="scripted",
        model="model",
        prompt_version="v1",
        policy_version="v1",
    )
    planning_provider = ScriptedProvider(
        (ChatResponse("Update greeting.txt and run unit validation.", "model", "scripted"),)
    )

    def services(active_store):
        manager = WorkspaceManager(workspace_root, active_store, {"org/repo": repository})
        planning = IntakePlanningService(active_store, read, planning_provider)
        validation = ValidationService(
            manager,
            active_store,
            Sandbox(),
            (ValidationProfile("unit", "test", ("trusted-unit",)),),
        )
        publication = PublicationService(
            active_store,
            manager,
            github,
            repository="org/repo",
            base_branch="main",
            target_account="org",
        )
        reporting = JiraReportingService(active_store, jira_write)
        workflow = JiraCodingAgentWorkflow(
            active_store,
            planning,
            ImplementationValidationCoordinator(active_store, validation),
            publication,
            reporting,
        )
        return manager, publication, workflow

    manager, publication, coding_agent = services(store)

    def implement(_attempt):
        manager.apply_patch(
            "RUN-E2E",
            "--- a/greeting.txt\n+++ b/greeting.txt\n@@ -1 +1 @@\n-hello\n+hello world\n",
        )
        return 1, 20, 0.0

    waiting = coding_agent.advance(
        "RUN-E2E",
        "JIRA-101",
        ({"path": "greeting.txt", "symbol": "greeting"},),
        implement,
    )
    assert waiting.awaiting_publication_approval
    assert waiting.state == RunState.AWAITING_PUBLISH_APPROVAL
    assert github.calls == jira_write.comments == 0
    assert manager.diff("RUN-E2E")["changed_files"] == ["greeting.txt"]
    assert store.get_run("RUN-E2E").story_hash == story_hash

    # Reconstruct every service at the durable approval boundary.
    store.close()
    store = SQLiteRunStore(store_path)
    manager, publication, coding_agent = services(store)
    approval = publication.request_approval(
        "RUN-E2E",
        approver="alice",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        title="Implement JIRA-101",
    )
    publication.approve(approval, approver="alice")
    github.fail_once = True
    with pytest.raises(GitHubError, match="durable intent"):
        coding_agent.advance(
            "RUN-E2E",
            "JIRA-101",
            (),
            lambda _attempt: (_ for _ in ()).throw(
                AssertionError("implementation must not repeat")
            ),
            publication_approval_id=approval,
            publication_title="Implement JIRA-101",
        )
    assert store.get_run("RUN-E2E").state == RunState.PUBLISH
    assert (
        store.connection.execute(
            "SELECT status FROM publication_outbox WHERE run_id='RUN-E2E'"
        ).fetchone()["status"]
        == "INTENT"
    )

    # A fresh process resumes the durable publication intent, then crashes after
    # JIRA accepted the marker-bearing comment but before local acknowledgement.
    store.close()
    store = SQLiteRunStore(store_path)
    _manager, _publication, coding_agent = services(store)
    jira_write.fail_once = True
    with pytest.raises(JiraWriteError, match="remote comment"):
        coding_agent.advance(
            "RUN-E2E",
            "JIRA-101",
            (),
            lambda _attempt: (_ for _ in ()).throw(
                AssertionError("implementation must not repeat")
            ),
        )
    assert store.get_run("RUN-E2E").state == RunState.REPORT
    assert (
        store.connection.execute(
            "SELECT status FROM jira_report_outbox WHERE run_id='RUN-E2E'"
        ).fetchone()["status"]
        == "INTENT"
    )

    # Another fresh process reconciles the existing remote marker without a
    # duplicate comment.
    store.close()
    store = SQLiteRunStore(store_path)
    manager, _publication, coding_agent = services(store)
    completed = coding_agent.advance(
        "RUN-E2E",
        "JIRA-101",
        (),
        lambda _attempt: (_ for _ in ()).throw(AssertionError("implementation must not repeat")),
    )
    assert completed.state == RunState.SUCCEEDED
    assert completed.report.comment_id == "comment-101"
    assert github.attempts == 2
    assert github.calls == jira_write.comments == 1
    assert jira_write.create_attempts == 1
    assert github.published_files == {"greeting.txt": b"hello world\n"}
    assert "https://github.test/pr/101" in jira_write.body

    terminal = coding_agent.advance("RUN-E2E", "JIRA-101", (), lambda _attempt: (0, 0, 0.0))
    assert terminal.state == RunState.SUCCEEDED
    assert github.calls == jira_write.comments == 1
    assert read.calls == 1
