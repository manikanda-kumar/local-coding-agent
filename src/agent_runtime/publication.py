"""Runtime-owned, exact-approval GitHub publication boundary."""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

from agent_runtime.durable import RunState, SQLiteRunStore
from agent_runtime.workspace import WorkspaceManager


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class PublicationDenied(RuntimeError):
    pass


class GitHubError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class GitHubTimeout(GitHubError):
    pass


@dataclass(frozen=True, slots=True, repr=False)
class GitHubCredentials:
    token: str

    def __repr__(self) -> str:
        return "GitHubCredentials(token=<redacted>)"


@dataclass(frozen=True, slots=True)
class ApprovalBinding:
    run_id: str
    story_hash: str
    story_revision: int
    repository: str
    base_revision: str
    workspace_generation: int
    diff_digest: str
    capability: str
    action: str
    target_branch: str
    target_account: str
    title: str
    policy_version: str
    approver: str
    expires_at: str

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical(asdict(self)).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class PublicationResult:
    branch: str
    commit_sha: str
    pull_request_id: int
    pull_request_url: str


class GitHubAdapter:
    """Credentials remain inside this adapter and are never included in errors/results."""

    def __init__(
        self, credentials: GitHubCredentials, *, client: httpx.Client | None = None
    ) -> None:
        self._credentials = credentials
        self._client = client or httpx.Client(base_url="https://api.github.com", timeout=20)

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        headers = {
            "Authorization": f"Bearer {self._credentials.token}",
            "Accept": "application/vnd.github+json",
        }
        try:
            response = self._client.request(method, path, headers=headers, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException:
            raise GitHubTimeout("GitHub request timed out; outcome may be uncertain") from None
        except (httpx.HTTPError, ValueError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            label = status if status is not None else "unknown"
            raise GitHubError(
                f"GitHub request failed (status {label})", status_code=status
            ) from None

    @staticmethod
    def _sha(payload: Any, resource: str) -> str:
        try:
            value = payload["sha"]
        except (KeyError, TypeError):
            raise GitHubError(f"GitHub returned a malformed {resource} response") from None
        if not isinstance(value, str) or not value:
            raise GitHubError(f"GitHub returned a malformed {resource} response")
        return value

    @staticmethod
    def _ref_sha(payload: Any) -> str:
        try:
            value = payload["object"]["sha"]
        except (KeyError, TypeError):
            raise GitHubError("GitHub returned a malformed ref response") from None
        if not isinstance(value, str) or not value:
            raise GitHubError("GitHub returned a malformed ref response")
        return value

    def base_sha(self, repository: str, branch: str) -> str:
        return self._ref_sha(self._request("GET", f"/repos/{repository}/git/ref/heads/{branch}"))

    def branch_sha(self, repository: str, branch: str) -> str | None:
        try:
            return self._ref_sha(
                self._request("GET", f"/repos/{repository}/git/ref/heads/{branch}")
            )
        except GitHubError as error:
            if error.status_code == 404:
                return None
            raise

    def find_pr(self, repository: str, account: str, branch: str) -> dict[str, Any] | None:
        rows = self._request(
            "GET",
            f"/repos/{repository}/pulls",
            params={"state": "all", "head": f"{account}:{branch}"},
        )
        if not isinstance(rows, list):
            raise GitHubError("GitHub returned a malformed pull request list response")
        if rows and not isinstance(rows[0], dict):
            raise GitHubError("GitHub returned a malformed pull request list response")
        return rows[0] if rows else None

    @staticmethod
    def _result(branch: str, commit_sha: str, payload: Any) -> PublicationResult:
        try:
            number, url = payload["number"], payload["html_url"]
        except (KeyError, TypeError):
            raise GitHubError("GitHub returned a malformed pull request response") from None
        if not isinstance(number, int) or isinstance(number, bool) or not isinstance(url, str):
            raise GitHubError("GitHub returned a malformed pull request response")
        return PublicationResult(branch, commit_sha, number, url)

    def publish(
        self,
        repository: str,
        base: str,
        base_sha: str,
        branch: str,
        account: str,
        files: dict[str, bytes | None],
        title: str,
        marker: str,
    ) -> PublicationResult:
        existing_sha = self.branch_sha(repository, branch)
        commit_sha = existing_sha
        if commit_sha is None:
            base_commit = self._request("GET", f"/repos/{repository}/git/commits/{base_sha}")
            try:
                base_tree_sha = base_commit["tree"]["sha"]
            except (KeyError, TypeError):
                raise GitHubError("GitHub returned a malformed commit response") from None
            if not isinstance(base_tree_sha, str) or not base_tree_sha:
                raise GitHubError("GitHub returned a malformed commit response")
            tree_entries = []
            for path, content in sorted(files.items()):
                sha = None
                if content is not None:
                    try:
                        text = content.decode("utf-8")
                    except UnicodeDecodeError:
                        raise GitHubError("publication only supports UTF-8 files") from None
                    sha = self._sha(
                        self._request(
                            "POST",
                            f"/repos/{repository}/git/blobs",
                            json={"content": text, "encoding": "utf-8"},
                        ),
                        "blob",
                    )
                tree_entries.append({"path": path, "mode": "100644", "type": "blob", "sha": sha})
            tree = self._sha(
                self._request(
                    "POST",
                    f"/repos/{repository}/git/trees",
                    json={"base_tree": base_tree_sha, "tree": tree_entries},
                ),
                "tree",
            )
            commit_sha = self._sha(
                self._request(
                    "POST",
                    f"/repos/{repository}/git/commits",
                    json={"message": marker, "tree": tree, "parents": [base_sha]},
                ),
                "commit",
            )
            self._request(
                "POST",
                f"/repos/{repository}/git/refs",
                json={"ref": f"refs/heads/{branch}", "sha": commit_sha},
            )
        else:
            existing_commit = self._request("GET", f"/repos/{repository}/git/commits/{commit_sha}")
            if existing_commit.get("message") != marker:
                raise GitHubError("deterministic publication branch is owned by another action")
        pr = self.find_pr(repository, account, branch)
        if pr is None:
            try:
                pr = self._request(
                    "POST",
                    f"/repos/{repository}/pulls",
                    json={"title": title, "head": branch, "base": base, "body": marker},
                )
            except GitHubTimeout:
                try:
                    pr = self.find_pr(repository, account, branch)
                except GitHubTimeout:
                    raise GitHubError("GitHub pull request outcome is uncertain") from None
                if pr is None:
                    raise GitHubError("GitHub pull request outcome is uncertain") from None
        return self._result(branch, commit_sha, pr)


class PublicationService:
    CAPABILITY, ACTION = "source_control.github.publish", "create_pull_request"

    def __init__(
        self,
        store: SQLiteRunStore,
        workspaces: WorkspaceManager,
        adapter: GitHubAdapter,
        *,
        repository: str,
        base_branch: str,
        target_account: str,
    ) -> None:
        self.store, self.workspaces, self.adapter = store, workspaces, adapter
        self.repository, self.base_branch, self.target_account = (
            repository,
            base_branch,
            target_account,
        )
        if repository.split("/", 1)[0] != target_account:
            raise ValueError(
                "first GitHub adapter supports branches in the repository owner account"
            )

    def _current(self, run_id: str, approver: str, expires_at: str, title: str) -> ApprovalBinding:
        run = self.store.get_run(run_id)
        if run.state != RunState.AWAITING_PUBLISH_APPROVAL:
            raise PublicationDenied("durable AWAITING_PUBLISH_APPROVAL state required")
        if run.repository != self.repository:
            raise PublicationDenied("adapter repository mismatch")
        story = self.store.story_snapshot(run_id)
        if run.story_hash != story.content_hash:
            raise PublicationDenied("active story snapshot does not match the run binding")
        workspace = self.workspaces.acquire(run_id)
        generation, _ = self.workspaces._generation(workspace)
        diff = self.workspaces.diff(run_id)["diff"]
        return ApprovalBinding(
            run_id,
            story.content_hash,
            story.revision,
            run.repository,
            run.base_revision,
            generation,
            hashlib.sha256(diff.encode()).hexdigest(),
            self.CAPABILITY,
            self.ACTION,
            self.base_branch,
            self.target_account,
            title,
            run.policy_version,
            approver,
            expires_at,
        )

    def request_approval(
        self, run_id: str, *, approver: str, expires_at: datetime, title: str
    ) -> str:
        if expires_at.tzinfo is None:
            raise ValueError("approval expiry must be timezone-aware")
        binding = self._current(run_id, approver, expires_at.astimezone(UTC).isoformat(), title)
        approval_id, now = secrets.token_urlsafe(24), datetime.now(UTC).isoformat()
        with self.store.connection:
            self.store.connection.execute(
                "INSERT INTO publish_approvals "
                "(approval_id,run_id,story_hash,story_revision,repository,base_revision,"
                "workspace_generation,diff_digest,capability,action,target_branch,target_account,"
                "title,policy_version,approver,expires_at,action_digest,status,created_at,"
                "approved_at,consumed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    approval_id,
                    *asdict(binding).values(),
                    binding.digest,
                    "PENDING",
                    now,
                    None,
                    None,
                ),
            )
        return approval_id

    def approve(self, approval_id: str, *, approver: str) -> None:
        now = datetime.now(UTC)
        with self.store.connection:
            row = self.store.connection.execute(
                "SELECT * FROM publish_approvals WHERE approval_id=?", (approval_id,)
            ).fetchone()
            if (
                row is None
                or row["status"] != "PENDING"
                or row["approver"] != approver
                or datetime.fromisoformat(row["expires_at"]) <= now
            ):
                raise PublicationDenied("approval is unavailable, mismatched, or expired")
            self.store.connection.execute(
                "UPDATE publish_approvals SET status='APPROVED',approved_at=? WHERE approval_id=?",
                (now.isoformat(), approval_id),
            )

    def publish(self, run_id: str, approval_id: str, *, title: str) -> PublicationResult:
        now = datetime.now(UTC)
        row = self.store.connection.execute(
            "SELECT * FROM publish_approvals WHERE approval_id=? AND run_id=?",
            (approval_id, run_id),
        ).fetchone()
        if row is None:
            raise PublicationDenied("exact approval not found")
        binding = self._current(run_id, row["approver"], row["expires_at"], title)
        if (
            row["status"] != "APPROVED"
            or row["action_digest"] != binding.digest
            or datetime.fromisoformat(row["expires_at"]) <= now
        ):
            raise PublicationDenied("approval is consumed, expired, or no longer exact")
        valid = self.store.connection.execute(
            "SELECT attempt,passed FROM validation_checkpoints WHERE run_id=? ORDER BY attempt DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        generations = self.store.connection.execute(
            "SELECT DISTINCT generation FROM validation_attempts WHERE run_id=? AND attempt=? AND status='PASSED'",
            (run_id, valid["attempt"] if valid else -1),
        ).fetchall()
        if (
            valid is None
            or not valid["passed"]
            or {x[0] for x in generations} != {binding.workspace_generation}
        ):
            raise PublicationDenied(
                "latest required validation checkpoint is not passed for this generation"
            )
        branch = f"agent/run-{run_id.lower()}-{binding.diff_digest[:12]}"
        intent = {
            "repository": binding.repository,
            "base": binding.target_branch,
            "base_revision": binding.base_revision,
            "branch": branch,
            "title": binding.title,
        }
        if (
            self.adapter.base_sha(binding.repository, binding.target_branch)
            != binding.base_revision
        ):
            raise PublicationDenied(
                "remote base revision drifted; rebase and revalidation required"
            )
        with self.store.connection:
            cursor = self.store.connection.execute(
                "UPDATE publish_approvals SET status='CONSUMED',consumed_at=? WHERE approval_id=? AND status='APPROVED'",
                (now.isoformat(), approval_id),
            )
            if cursor.rowcount != 1:
                raise PublicationDenied("approval already consumed")
            self.store.connection.execute(
                "INSERT INTO publication_outbox "
                "(run_id,approval_id,action_digest,branch,status,intent_json,remote_json,"
                "created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    approval_id,
                    binding.digest,
                    branch,
                    "INTENT",
                    _canonical(intent),
                    None,
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
            state = self.store.connection.execute(
                "UPDATE runs SET state=? WHERE run_id=? AND state=?",
                (RunState.PUBLISH, run_id, RunState.AWAITING_PUBLISH_APPROVAL),
            )
            if state.rowcount != 1:
                raise PublicationDenied("run state changed before publication intent")
        workspace = self.workspaces.acquire(run_id)
        _, generation_path = self.workspaces._generation(workspace)
        changed = self.workspaces.diff(run_id)["changed_files"]
        files = {
            name: (generation_path / name).read_bytes()
            if (generation_path / name).exists()
            else None
            for name in changed
        }
        result = self.adapter.publish(
            binding.repository,
            binding.target_branch,
            binding.base_revision,
            branch,
            binding.target_account,
            files,
            binding.title,
            f"agent-runtime-run:{run_id}",
        )
        with self.store.connection:
            self.store.connection.execute(
                "UPDATE publication_outbox SET status='SUCCEEDED',remote_json=?,updated_at=? WHERE run_id=?",
                (_canonical(asdict(result)), datetime.now(UTC).isoformat(), run_id),
            )
            self.store.connection.execute(
                "UPDATE runs SET state=? WHERE run_id=? AND state=?",
                (RunState.REPORT, run_id, RunState.PUBLISH),
            )
        return result

    def resume(self, run_id: str) -> PublicationResult:
        """Resume a consumed, persisted intent without consulting a model or issuing a new approval."""
        run = self.store.get_run(run_id)
        outbox = self.store.connection.execute(
            "SELECT * FROM publication_outbox WHERE run_id=?", (run_id,)
        ).fetchone()
        if outbox is None:
            raise PublicationDenied("no resumable publication intent")
        if outbox["status"] == "SUCCEEDED" and run.state in {
            RunState.PUBLISH,
            RunState.REPORT,
        }:
            return PublicationResult(**json.loads(outbox["remote_json"]))
        if run.state != RunState.PUBLISH:
            raise PublicationDenied("no resumable publication intent")
        approval = self.store.connection.execute(
            "SELECT * FROM publish_approvals WHERE approval_id=?", (outbox["approval_id"],)
        ).fetchone()
        intent = json.loads(outbox["intent_json"])
        if (
            approval is None
            or approval["status"] != "CONSUMED"
            or run.repository != self.repository
        ):
            raise PublicationDenied("persisted publication binding is invalid")
        if self.adapter.base_sha(self.repository, intent["base"]) != intent["base_revision"]:
            raise PublicationDenied(
                "remote base revision drifted; rebase and revalidation required"
            )
        workspace = self.workspaces.acquire(run_id)
        generation, generation_path = self.workspaces._generation(workspace)
        diff = self.workspaces.diff(run_id)
        digest = hashlib.sha256(diff["diff"].encode()).hexdigest()
        if generation != approval["workspace_generation"] or digest != approval["diff_digest"]:
            raise PublicationDenied("workspace changed after publication intent")
        files = {
            name: (generation_path / name).read_bytes()
            if (generation_path / name).exists()
            else None
            for name in diff["changed_files"]
        }
        result = self.adapter.publish(
            self.repository,
            intent["base"],
            intent["base_revision"],
            intent["branch"],
            approval["target_account"],
            files,
            intent["title"],
            f"agent-runtime-run:{run_id}",
        )
        with self.store.connection:
            self.store.connection.execute(
                "UPDATE publication_outbox SET status='SUCCEEDED',remote_json=?,updated_at=? WHERE run_id=?",
                (_canonical(asdict(result)), datetime.now(UTC).isoformat(), run_id),
            )
            self.store.connection.execute(
                "UPDATE runs SET state=? WHERE run_id=? AND state=?",
                (RunState.REPORT, run_id, RunState.PUBLISH),
            )
        return result
