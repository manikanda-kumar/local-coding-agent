from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from agent_runtime import GATEWAY_TOOLS, RunState, SQLiteRunStore, WorkspaceManager
from agent_runtime.publication import (
    GitHubAdapter,
    GitHubCredentials,
    GitHubError,
    PublicationDenied,
    PublicationService,
)


def git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()


@dataclass
class GitHubFixture:
    base_sha: str
    calls: list[tuple[str, str]] = field(default_factory=list)
    branch_sha: str | None = None
    branch_marker: str = ""
    pr: dict | None = None
    timeout_create_pr: bool = False
    last_tree: dict | None = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append((request.method, request.url.path))
        method, path = request.method, request.url.path
        if path.endswith("/git/ref/heads/main") and method == "GET":
            return httpx.Response(200, json={"object": {"sha": self.base_sha}})
        if "/git/ref/heads/agent/" in path and method == "GET":
            if self.branch_sha is None:
                return httpx.Response(404, json={"message": "missing"})
            return httpx.Response(200, json={"object": {"sha": self.branch_sha}})
        if path.endswith(f"/git/commits/{self.base_sha}") and method == "GET":
            return httpx.Response(200, json={"sha": self.base_sha, "tree": {"sha": "base-tree"}})
        if self.branch_sha and path.endswith(f"/git/commits/{self.branch_sha}") and method == "GET":
            return httpx.Response(200, json={"sha": self.branch_sha, "message": self.branch_marker})
        if path.endswith("/git/blobs") and method == "POST":
            return httpx.Response(201, json={"sha": "blob-sha"})
        if path.endswith("/git/trees") and method == "POST":
            self.last_tree = json.loads(request.content)
            return httpx.Response(201, json={"sha": "tree-sha"})
        if path.endswith("/git/commits") and method == "POST":
            body = json.loads(request.content)
            self.branch_marker = body["message"]
            self.branch_sha = "published-commit"
            return httpx.Response(201, json={"sha": self.branch_sha})
        if path.endswith("/git/refs") and method == "POST":
            return httpx.Response(201, json={"ref": json.loads(request.content)["ref"]})
        if path.endswith("/pulls") and method == "GET":
            return httpx.Response(200, json=[] if self.pr is None else [self.pr])
        if path.endswith("/pulls") and method == "POST":
            self.pr = {"number": 7, "html_url": "https://github.test/o/r/pull/7"}
            if self.timeout_create_pr:
                self.timeout_create_pr = False
                raise httpx.ReadTimeout("uncertain", request=request)
            return httpx.Response(201, json=self.pr)
        raise AssertionError(f"unexpected request {method} {path}")

    def adapter(self) -> GitHubAdapter:
        return GitHubAdapter(
            GitHubCredentials("token"),
            client=httpx.Client(
                transport=httpx.MockTransport(self.handler), base_url="https://api.github.test"
            ),
        )


@dataclass
class ServiceFixture:
    root: Path
    store: SQLiteRunStore
    manager: WorkspaceManager
    github: GitHubFixture
    service: PublicationService

    def restart(self) -> PublicationService:
        self.store.close()
        self.store = SQLiteRunStore(self.root / "runs.db")
        self.manager = WorkspaceManager(
            self.root / "workspaces", self.store, {"o/r": self.root / "repository"}
        )
        self.service = PublicationService(
            self.store,
            self.manager,
            self.github.adapter(),
            repository="o/r",
            base_branch="main",
            target_account="o",
        )
        return self.service


@pytest.fixture
def publication(tmp_path: Path) -> ServiceFixture:
    repository = tmp_path / "repository"
    repository.mkdir()
    git(repository, "init", "-q")
    git(repository, "config", "user.email", "test@example.test")
    git(repository, "config", "user.name", "Test")
    (repository / "value.txt").write_text("old\n")
    git(repository, "add", ".")
    git(repository, "commit", "-qm", "base")
    revision = git(repository, "rev-parse", "HEAD")
    store = SQLiteRunStore(tmp_path / "runs.db")
    store.create_run(
        "RUN",
        story_hash="story-hash",
        repository="o/r",
        base_revision=revision,
        provider="scripted",
        model="model",
        prompt_version="1",
        policy_version="policy-1",
    )
    store.save_story_snapshot("RUN", "story-hash", {"key": "STORY-1"})
    for state in (
        RunState.INTAKE,
        RunState.ANALYZE,
        RunState.PLAN_READY,
        RunState.IMPLEMENT,
    ):
        store.transition("RUN", state)
    manager = WorkspaceManager(tmp_path / "workspaces", store, {"o/r": repository})
    manager.apply_patch("RUN", "--- a/value.txt\n+++ b/value.txt\n@@ -1 +1 @@\n-old\n+new\n")
    generation, _ = manager._generation(manager.acquire("RUN"))
    store.transition("RUN", RunState.VALIDATE)
    now = datetime.now(UTC).isoformat()
    with store.connection:
        store.connection.execute(
            "INSERT INTO validation_attempts "
            "(run_id,attempt,profile_id,generation,status,result_json,created_at,completed_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            ("RUN", 1, "unit", generation, "PASSED", "{}", now, now),
        )
        store.connection.execute(
            "INSERT INTO validation_checkpoints "
            "(run_id,attempt,passed,results_json,created_at) VALUES (?,?,?,?,?)",
            ("RUN", 1, 1, "{}", now),
        )
    store.transition("RUN", RunState.AWAITING_PUBLISH_APPROVAL)
    github = GitHubFixture(revision)
    service = PublicationService(
        store,
        manager,
        github.adapter(),
        repository="o/r",
        base_branch="main",
        target_account="o",
    )
    return ServiceFixture(tmp_path, store, manager, github, service)


def expiry(**kwargs: int) -> datetime:
    return datetime.now(UTC) + timedelta(**kwargs)


def approved(fixture: ServiceFixture, *, title: str = "PR title") -> str:
    approval = fixture.service.request_approval(
        "RUN", approver="alice", expires_at=expiry(hours=1), title=title
    )
    fixture.service.approve(approval, approver="alice")
    return approval


def test_request_and_approve_are_local_and_publish_requires_approval(
    publication: ServiceFixture,
) -> None:
    approval = publication.service.request_approval(
        "RUN", approver="alice", expires_at=expiry(hours=1), title="PR title"
    )
    publication.service.approve(approval, approver="alice")
    assert publication.github.calls == []

    other = publication.service.request_approval(
        "RUN", approver="bob", expires_at=expiry(hours=1), title="Other"
    )
    with pytest.raises(PublicationDenied):
        publication.service.publish("RUN", other, title="Other")
    assert publication.github.calls == []


def test_success_persists_exact_intent_and_result(publication: ServiceFixture) -> None:
    result = publication.service.publish("RUN", approved(publication), title="PR title")
    assert result.pull_request_id == 7
    assert publication.store.get_run("RUN").state == RunState.REPORT
    outbox = publication.store.connection.execute("SELECT * FROM publication_outbox").fetchone()
    assert outbox["status"] == "SUCCEEDED"
    assert json.loads(outbox["remote_json"])["pull_request_url"] == result.pull_request_url
    assert json.loads(outbox["intent_json"])["title"] == "PR title"
    assert publication.github.last_tree["base_tree"] == "base-tree"


@pytest.mark.parametrize("change", ["title", "diff", "repository", "target", "expiry"])
def test_binding_changes_and_expiry_invalidate(publication: ServiceFixture, change: str) -> None:
    approval = approved(publication)
    title = "PR title"
    if change == "title":
        title = "Changed"
    elif change == "diff":
        _, generation_path = publication.manager._generation(publication.manager.acquire("RUN"))
        (generation_path / "value.txt").write_text("newer\n")
    elif change == "repository":
        publication.store.connection.execute(
            "UPDATE runs SET repository='o/other' WHERE run_id='RUN'"
        )
    elif change == "target":
        publication.service.base_branch = "develop"
    else:
        publication.store.connection.execute(
            "UPDATE publish_approvals SET expires_at=? WHERE approval_id=?",
            (expiry(seconds=-1).isoformat(), approval),
        )
    with pytest.raises(PublicationDenied):
        publication.service.publish("RUN", approval, title=title)
    assert publication.github.calls == []


def test_approval_is_single_use(publication: ServiceFixture) -> None:
    approval = approved(publication)
    publication.service.publish("RUN", approval, title="PR title")
    before = len(publication.github.calls)
    with pytest.raises(PublicationDenied):
        publication.service.publish("RUN", approval, title="PR title")
    assert len(publication.github.calls) == before


def test_resume_returns_persisted_result_after_success(publication: ServiceFixture) -> None:
    expected = publication.service.publish("RUN", approved(publication), title="PR title")
    before = len(publication.github.calls)

    assert publication.service.resume("RUN") == expected
    assert len(publication.github.calls) == before


def test_restart_after_approval_can_publish(publication: ServiceFixture) -> None:
    approval = approved(publication)
    result = publication.restart().publish("RUN", approval, title="PR title")
    assert result.pull_request_id == 7


def test_base_drift_keeps_approval_and_has_only_get(publication: ServiceFixture) -> None:
    approval = approved(publication)
    publication.github.base_sha = "drifted"
    with pytest.raises(PublicationDenied, match="drifted"):
        publication.service.publish("RUN", approval, title="PR title")
    row = publication.store.connection.execute(
        "SELECT status FROM publish_approvals WHERE approval_id=?", (approval,)
    ).fetchone()
    assert row["status"] == "APPROVED"
    assert publication.store.get_run("RUN").state == RunState.AWAITING_PUBLISH_APPROVAL
    assert (
        publication.store.connection.execute("SELECT COUNT(*) FROM publication_outbox").fetchone()[
            0
        ]
        == 0
    )
    assert publication.github.calls == [("GET", "/repos/o/r/git/ref/heads/main")]


def test_timeout_after_pr_creation_reconciles_once(publication: ServiceFixture) -> None:
    publication.github.timeout_create_pr = True
    result = publication.service.publish("RUN", approved(publication), title="PR title")
    assert result.pull_request_id == 7
    assert publication.github.calls.count(("POST", "/repos/o/r/pulls")) == 1
    assert publication.github.calls.count(("GET", "/repos/o/r/pulls")) == 2


def test_preexisting_branch_requires_exact_marker(publication: ServiceFixture) -> None:
    publication.github.branch_sha = "foreign-commit"
    publication.github.branch_marker = "agent-runtime-run:SOMEONE-ELSE"
    with pytest.raises(GitHubError, match="owned by another"):
        publication.service.publish("RUN", approved(publication), title="PR title")
    assert not any(method == "POST" for method, _ in publication.github.calls)


def test_credentials_errors_and_malformed_responses_are_sanitized() -> None:
    secret = "github-secret-token"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == f"Bearer {secret}"
        return httpx.Response(200, json={"message": secret})

    credentials = GitHubCredentials(secret)
    adapter = GitHubAdapter(
        credentials,
        client=httpx.Client(
            transport=httpx.MockTransport(handler), base_url="https://api.github.test"
        ),
    )
    assert secret not in repr(credentials)
    with pytest.raises(GitHubError) as caught:
        adapter.base_sha("o/r", "main")
    assert secret not in str(caught.value) and secret not in repr(caught.value)


def test_gateway_has_no_approval_tools() -> None:
    names = {tool.name for tool in GATEWAY_TOOLS}
    assert not any("approval" in name or "publish" in name for name in names)
