from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from agent_runtime import (
    JiraAuth,
    JiraReportingService,
    JiraWriteAdapter,
    JiraWriteError,
    ReportingDenied,
    RunState,
    SQLiteRunStore,
)


def reporting_store(tmp_path) -> SQLiteRunStore:
    store = SQLiteRunStore(tmp_path / "runs.db")
    store.create_run(
        "RUN-8",
        story_hash="story",
        repository="o/r",
        base_revision="abc",
        provider="p",
        model="m",
        prompt_version="1",
        policy_version="1",
    )
    store.save_story_snapshot("RUN-8", "story", {"issue_key": "ABC-8"})
    now = datetime.now(UTC).isoformat()
    with store.connection:
        store.connection.execute("UPDATE runs SET state='REPORT' WHERE run_id='RUN-8'")
        store.connection.execute(
            "INSERT INTO validation_checkpoints(run_id,attempt,passed,results_json,created_at) "
            "VALUES (?,?,?,?,?)",
            ("RUN-8", 3, 1, '{"unit":"passed"}', now),
        )
        store.connection.execute(
            "INSERT INTO publish_approvals VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "approval",
                "RUN-8",
                "story",
                1,
                "o/r",
                "abc",
                1,
                "diff",
                "cap",
                "act",
                "main",
                "o",
                "title",
                "1",
                "alice",
                now,
                "digest",
                "CONSUMED",
                now,
                now,
                now,
            ),
        )
        store.connection.execute(
            "INSERT INTO publication_outbox VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "RUN-8",
                "approval",
                "digest",
                "branch",
                "SUCCEEDED",
                "{}",
                json.dumps({"pull_request_url": "https://github.test/o/r/pull/8"}),
                now,
                now,
            ),
        )
    return store


def test_exact_report_body_persisted_before_post_and_terminal_after_post(tmp_path) -> None:
    store = reporting_store(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"comments": [], "total": 0})
        row = store.connection.execute("SELECT * FROM jira_report_outbox").fetchone()
        assert row["status"] == "INTENT"
        document = json.loads(request.content)["body"]
        assert document["type"] == "doc"
        assert document["content"][0]["content"][0]["text"] == row["body"]
        assert "https://github.test/o/r/pull/8" in row["body"]
        assert "Validation checkpoint 3: PASSED" in row["body"]
        assert "Run ID: RUN-8" in row["body"]
        assert store.get_run("RUN-8").state == RunState.REPORT
        return httpx.Response(201, json={"id": "44"})

    adapter = JiraWriteAdapter(
        "https://jira.test",
        JiraAuth("bearer", "secret"),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    result = JiraReportingService(store, adapter).report("RUN-8")
    assert result.comment_id == "44"
    assert store.get_run("RUN-8").state == RunState.SUCCEEDED
    assert (
        store.connection.execute("SELECT remote_comment_id FROM jira_report_outbox").fetchone()[0]
        == "44"
    )


def test_timeout_reconciles_marker_without_duplicate(tmp_path) -> None:
    store, comments, posts = reporting_store(tmp_path), [], 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.method == "GET":
            return httpx.Response(200, json={"comments": comments, "total": len(comments)})
        posts += 1
        body = json.loads(request.content)["body"]
        comments.append({"id": "55", "body": body})
        raise httpx.ReadTimeout("secret leaked upstream", request=request)

    adapter = JiraWriteAdapter(
        "https://jira.test",
        JiraAuth("bearer", "secret"),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert JiraReportingService(store, adapter).report("RUN-8").comment_id == "55"
    assert posts == 1


def test_failure_stays_report_and_errors_are_sanitized(tmp_path) -> None:
    store = reporting_store(tmp_path)
    secret = "credential-never-visible"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == f"Bearer {secret}"
        return httpx.Response(500, text=secret)

    adapter = JiraWriteAdapter(
        "https://jira.test",
        JiraAuth("bearer", secret),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(JiraWriteError) as caught:
        JiraReportingService(store, adapter).report("RUN-8")
    assert str(caught.value) == "JIRA write failed"
    assert secret not in repr(adapter._auth)
    assert store.get_run("RUN-8").state == RunState.REPORT


def test_reconstructed_reporting_service_reconciles_persisted_intent(tmp_path) -> None:
    store = reporting_store(tmp_path)
    failing = JiraWriteAdapter(
        "https://jira.test",
        JiraAuth("bearer", "secret"),
        client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(500, request=request))
        ),
    )
    with pytest.raises(JiraWriteError):
        JiraReportingService(store, failing).report("RUN-8")
    assert (
        store.connection.execute("SELECT status FROM jira_report_outbox").fetchone()[0] == "INTENT"
    )

    def recovered(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"comments": [], "total": 0})
        return httpx.Response(201, json={"id": "restart-1"})

    adapter = JiraWriteAdapter(
        "https://jira.test",
        JiraAuth("bearer", "secret"),
        client=httpx.Client(transport=httpx.MockTransport(recovered)),
    )
    assert JiraReportingService(store, adapter).resume("RUN-8").comment_id == "restart-1"
    assert store.get_run("RUN-8").state == RunState.SUCCEEDED


def test_transition_is_unavailable_by_default_allowlisted_exact_and_single_use(tmp_path) -> None:
    store, transitions = reporting_store(tmp_path), []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"comments": [], "total": 0})
        if request.url.path.endswith("/transitions"):
            transitions.append(json.loads(request.content)["transition"]["id"])
            return httpx.Response(204)
        return httpx.Response(201, json={"id": "66"})

    adapter = JiraWriteAdapter(
        "https://jira.test",
        JiraAuth("bearer", "secret"),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ReportingDenied):
        JiraReportingService(store, adapter).request_transition_approval(
            "RUN-8", "31", approver="alice", expires_at=datetime.now(UTC) + timedelta(hours=1)
        )
    service = JiraReportingService(store, adapter, transition_ids={"31"})
    approval = service.request_transition_approval(
        "RUN-8", "31", approver="alice", expires_at=datetime.now(UTC) + timedelta(hours=1)
    )
    service.approve_transition(approval, approver="alice")
    assert service.report("RUN-8", transition_approval_id=approval).transitioned
    assert transitions == ["31"]
    with pytest.raises(ReportingDenied):
        service.report("RUN-8", transition_approval_id=approval)


def test_failed_transition_keeps_approval_and_run_resumable(tmp_path) -> None:
    store = reporting_store(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"comments": [], "total": 0})
        if request.url.path.endswith("/transitions"):
            return httpx.Response(500, text="failure")
        return httpx.Response(201, json={"id": "77"})

    service = JiraReportingService(
        store,
        JiraWriteAdapter(
            "https://jira.test",
            JiraAuth("bearer", "secret"),
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        ),
        transition_ids={"31"},
    )
    approval = service.request_transition_approval(
        "RUN-8", "31", approver="alice", expires_at=datetime.now(UTC) + timedelta(hours=1)
    )
    service.approve_transition(approval, approver="alice")

    with pytest.raises(JiraWriteError):
        service.report("RUN-8", transition_approval_id=approval)

    assert store.get_run("RUN-8").state == RunState.REPORT
    assert (
        store.connection.execute(
            "SELECT status FROM jira_transition_approvals WHERE approval_id=?", (approval,)
        ).fetchone()[0]
        == "APPROVED"
    )
