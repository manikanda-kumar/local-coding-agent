"""Durable, adapter-owned JIRA reporting and explicitly approved transitions."""

from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import httpx

from agent_runtime.durable import RunState, SQLiteRunStore
from agent_runtime.jira import JiraAuth, _text
from agent_runtime.metrics import MetricEvent, MetricsSink, emit_metric


class JiraWriteError(RuntimeError):
    """Sanitized JIRA write failure."""


class JiraWriteTimeout(JiraWriteError):
    pass


class ReportingDenied(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class JiraReportResult:
    comment_id: str
    issue_key: str
    marker: str
    transitioned: bool = False


class JiraWriteAdapter:
    """Owns credentials; callers can only perform the reporting-specific JIRA operations."""

    def __init__(
        self,
        base_url: str,
        auth: JiraAuth,
        *,
        client: httpx.Client | None = None,
        timeout: float = 10,
        page_size: int = 50,
    ) -> None:
        self.base_url, self._auth = base_url.rstrip("/"), auth
        self._client, self._timeout, self.page_size = client or httpx.Client(), timeout, page_size
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        auth = None
        if self._auth.kind == "bearer":
            headers["Authorization"] = f"Bearer {self._auth.secret}"
        else:
            if not self._auth.username:
                raise JiraWriteError("JIRA write failed")
            auth = (self._auth.username, self._auth.secret)
        try:
            response = self._client.request(
                method,
                f"{self.base_url}{path}",
                headers=headers,
                auth=auth,
                timeout=self._timeout,
                **kwargs,
            )
            response.raise_for_status()
            if not response.content:
                return {}
            value = response.json()
            if not isinstance(value, dict):
                raise TypeError("invalid response")
            return value
        except httpx.TimeoutException:
            raise JiraWriteTimeout("JIRA write timed out; outcome may be uncertain") from None
        except (httpx.HTTPError, TypeError, ValueError):
            raise JiraWriteError("JIRA write failed") from None

    def comments(self, issue_key: str) -> Iterable[dict[str, Any]]:
        start = 0
        while True:
            page = self._request(
                "GET",
                f"/rest/api/3/issue/{quote(issue_key, safe='')}/comment",
                params={"startAt": start, "maxResults": self.page_size},
            )
            values = page.get("comments", [])
            if not isinstance(values, list):
                raise JiraWriteError("JIRA write failed")
            yield from (item for item in values if isinstance(item, dict))
            start += len(values)
            total = page.get("total", start)
            if not isinstance(total, int) or isinstance(total, bool):
                raise JiraWriteError("JIRA write failed")
            if not values or start >= total:
                return

    def find_comment(self, issue_key: str, marker: str) -> str | None:
        for comment in self.comments(issue_key):
            if marker in _text(comment.get("body")):
                value = comment.get("id")
                if isinstance(value, (str, int)) and str(value):
                    return str(value)
        return None

    def create_comment(self, issue_key: str, body: str) -> str:
        document = {
            "type": "doc",
            "version": 1,
            "content": [{"type": "paragraph", "content": [{"type": "text", "text": body}]}],
        }
        result = self._request(
            "POST",
            f"/rest/api/3/issue/{quote(issue_key, safe='')}/comment",
            json={"body": document},
        )
        value = result.get("id")
        if not isinstance(value, (str, int)) or not str(value):
            raise JiraWriteError("JIRA write failed")
        return str(value)

    def transition(self, issue_key: str, transition_id: str) -> None:
        self._request(
            "POST",
            f"/rest/api/3/issue/{quote(issue_key, safe='')}/transitions",
            json={"transition": {"id": transition_id}},
        )


class JiraReportingService:
    def __init__(
        self,
        store: SQLiteRunStore,
        adapter: JiraWriteAdapter,
        *,
        transition_ids: Iterable[str] = (),
        metrics: MetricsSink | None = None,
    ) -> None:
        self.store, self.adapter = store, adapter
        self.transition_ids = frozenset(str(value) for value in transition_ids)
        self.metrics = metrics

    def _binding(
        self, run_id: str, issue_key: str, transition_id: str, story_revision: int, expires_at: str
    ) -> str:
        value = [run_id, issue_key, transition_id, story_revision, expires_at]
        return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()

    def request_transition_approval(
        self, run_id: str, transition_id: str, *, approver: str, expires_at: datetime
    ) -> str:
        run, story = self.store.get_run(run_id), self.store.story_snapshot(run_id)
        if run.state != RunState.REPORT or transition_id not in self.transition_ids:
            raise ReportingDenied("transition is unavailable or not configured")
        if expires_at.tzinfo is None:
            raise ValueError("approval expiry must be timezone-aware")
        issue_key = str(story.snapshot.get("issue_key") or story.snapshot.get("key") or "")
        expiry = expires_at.astimezone(UTC).isoformat()
        digest = self._binding(run_id, issue_key, transition_id, story.revision, expiry)
        approval_id, now = secrets.token_urlsafe(24), datetime.now(UTC).isoformat()
        with self.store.connection:
            self.store.connection.execute(
                "INSERT INTO jira_transition_approvals VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    approval_id,
                    run_id,
                    issue_key,
                    transition_id,
                    story.revision,
                    approver,
                    expiry,
                    digest,
                    "PENDING",
                    now,
                    None,
                    None,
                ),
            )
        return approval_id

    def approve_transition(self, approval_id: str, *, approver: str) -> None:
        now = datetime.now(UTC)
        with self.store.connection:
            row = self.store.connection.execute(
                "SELECT * FROM jira_transition_approvals WHERE approval_id=?", (approval_id,)
            ).fetchone()
            if (
                row is None
                or row["status"] != "PENDING"
                or row["approver"] != approver
                or datetime.fromisoformat(row["expires_at"]) <= now
            ):
                raise ReportingDenied("approval is unavailable, mismatched, or expired")
            self.store.connection.execute(
                "UPDATE jira_transition_approvals SET status='APPROVED',approved_at=? WHERE approval_id=?",
                (now.isoformat(), approval_id),
            )

    def _comment(self, run_id: str) -> JiraReportResult:
        run = self.store.get_run(run_id)
        if run.state != RunState.REPORT:
            raise ReportingDenied("durable REPORT state required")
        row = self.store.connection.execute(
            "SELECT * FROM jira_report_outbox WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            story = self.store.story_snapshot(run_id)
            issue_key = str(story.snapshot.get("issue_key") or story.snapshot.get("key") or "")
            publication = self.store.connection.execute(
                "SELECT remote_json,status FROM publication_outbox WHERE run_id=?", (run_id,)
            ).fetchone()
            validation = self.store.connection.execute(
                "SELECT attempt,passed,results_json FROM validation_checkpoints WHERE run_id=? "
                "ORDER BY attempt DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            if (
                not issue_key
                or publication is None
                or publication["status"] != "SUCCEEDED"
                or validation is None
            ):
                raise ReportingDenied(
                    "required persisted publication or validation result is missing"
                )
            pr_url = json.loads(publication["remote_json"])["pull_request_url"]
            marker = f"[agent-runtime-report:{hashlib.sha256(run_id.encode()).hexdigest()}]"
            outcome = "PASSED" if validation["passed"] else "FAILED"
            try:
                raw_results = json.loads(validation["results_json"])
            except (TypeError, ValueError):
                raw_results = []
            if isinstance(raw_results, list):
                summary = ", ".join(
                    f"{item.get('profile_id', 'profile')}={'PASSED' if item.get('passed') else 'FAILED'}"
                    for item in raw_results
                    if isinstance(item, dict)
                )
            else:
                summary = "Persisted validation checkpoint available"
            body = (
                f"Pull request: {pr_url}\nValidation checkpoint {validation['attempt']}: {outcome}\n"
                f"Validation summary: {summary or outcome}\nRun ID: {run_id}\n{marker}"
            )
            now = datetime.now(UTC).isoformat()
            with self.store.connection:
                self.store.connection.execute(
                    "INSERT INTO jira_report_outbox VALUES (?,?,?,?,?,?,?,?,?)",
                    (run_id, issue_key, marker, body, "INTENT", None, None, now, now),
                )
            row = self.store.connection.execute(
                "SELECT * FROM jira_report_outbox WHERE run_id=?", (run_id,)
            ).fetchone()
        if row["status"] == "SUCCEEDED":
            return JiraReportResult(row["remote_comment_id"], row["issue_key"], row["marker"])
        comment_id = self.adapter.find_comment(row["issue_key"], row["marker"])
        if comment_id is None:
            try:
                comment_id = self.adapter.create_comment(row["issue_key"], row["body"])
            except JiraWriteTimeout:
                comment_id = self.adapter.find_comment(row["issue_key"], row["marker"])
                if comment_id is None:
                    raise JiraWriteError("JIRA comment outcome is uncertain") from None
        result = JiraReportResult(comment_id, row["issue_key"], row["marker"])
        now = datetime.now(UTC).isoformat()
        with self.store.connection:
            self.store.connection.execute(
                "UPDATE jira_report_outbox SET status='SUCCEEDED',remote_comment_id=?,result_json=?,updated_at=? WHERE run_id=?",
                (
                    comment_id,
                    json.dumps(asdict(result)),
                    now,
                    run_id,
                ),
            )
        return result

    def report(self, run_id: str, *, transition_approval_id: str | None = None) -> JiraReportResult:
        try:
            result = self._report(run_id, transition_approval_id=transition_approval_id)
        except Exception:
            emit_metric(
                self.metrics, MetricEvent("report", "jira_report", "failure", run_id=run_id)
            )
            raise
        emit_metric(self.metrics, MetricEvent("report", "jira_report", "success", run_id=run_id))
        return result

    def _report(
        self, run_id: str, *, transition_approval_id: str | None = None
    ) -> JiraReportResult:
        result = self._comment(run_id)
        transitioned = False
        if transition_approval_id is not None:
            row = self.store.connection.execute(
                "SELECT * FROM jira_transition_approvals WHERE approval_id=? AND run_id=?",
                (transition_approval_id, run_id),
            ).fetchone()
            story, now = self.store.story_snapshot(run_id), datetime.now(UTC)
            if (
                row is None
                or row["status"] != "APPROVED"
                or row["transition_id"] not in self.transition_ids
                or row["story_revision"] != story.revision
                or row["binding_digest"]
                != self._binding(
                    run_id,
                    result.issue_key,
                    row["transition_id"],
                    story.revision,
                    row["expires_at"],
                )
                or datetime.fromisoformat(row["expires_at"]) <= now
            ):
                raise ReportingDenied("approval is consumed, expired, or no longer exact")
            self.adapter.transition(result.issue_key, row["transition_id"])
            with self.store.connection:
                changed = self.store.connection.execute(
                    "UPDATE jira_transition_approvals SET status='CONSUMED',consumed_at=? "
                    "WHERE approval_id=? AND status='APPROVED'",
                    (datetime.now(UTC).isoformat(), transition_approval_id),
                )
                if changed.rowcount != 1:
                    raise ReportingDenied("approval already consumed")
            transitioned = True
        self.store.transition(run_id, RunState.SUCCEEDED, {"jira_comment_id": result.comment_id})
        return JiraReportResult(result.comment_id, result.issue_key, result.marker, transitioned)

    def resume(self, run_id: str, *, transition_approval_id: str | None = None) -> JiraReportResult:
        return self.report(run_id, transition_approval_id=transition_approval_id)
