from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Any


class RunState(StrEnum):
    NEW = "NEW"
    INTAKE = "INTAKE"
    ANALYZE = "ANALYZE"
    PLAN_READY = "PLAN_READY"
    IMPLEMENT = "IMPLEMENT"
    VALIDATE = "VALIDATE"
    AWAITING_PUBLISH_APPROVAL = "AWAITING_PUBLISH_APPROVAL"
    PUBLISH = "PUBLISH"
    REPORT = "REPORT"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


_FLOW = (
    RunState.NEW,
    RunState.INTAKE,
    RunState.ANALYZE,
    RunState.PLAN_READY,
    RunState.IMPLEMENT,
    RunState.VALIDATE,
    RunState.AWAITING_PUBLISH_APPROVAL,
    RunState.PUBLISH,
    RunState.REPORT,
    RunState.SUCCEEDED,
)
_NEXT = dict(pairwise(_FLOW))
_ALLOWED_TRANSITIONS = {*_NEXT.items(), (RunState.VALIDATE, RunState.IMPLEMENT)}
_TERMINAL = {RunState.SUCCEEDED, RunState.FAILED, RunState.CANCELLED}


class InvalidTransition(ValueError):
    pass


class RunCancelled(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: str
    state: RunState
    story_hash: str
    repository: str
    base_revision: str
    provider: str
    model: str
    prompt_version: str
    policy_version: str
    usage: dict[str, Any]
    limits: dict[str, Any]
    profile_id: str = "provider-default"
    profile_sha256: str = "dde4c0e4b318df885b415f7cd2bbb33c3f4cf544dacb003902ac0cafa8fa1ee4"


@dataclass(frozen=True, slots=True)
class StoredInvocation:
    invocation_id: str
    execution_id: str
    status: str
    capability_id: str = ""
    result: Any = None
    error: str | None = None
    replayed: bool = False
    normalized_version: int | None = None


@dataclass(frozen=True, slots=True)
class StoredStorySnapshot:
    run_id: str
    revision: int
    content_hash: str
    snapshot: dict[str, Any]
    active: bool


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def invocation_identity(
    run_id: str, step: str, capability_id: str, arguments: Mapping[str, Any]
) -> str:
    material = _json([run_id, step, capability_id, arguments]).encode()
    return hashlib.sha256(material).hexdigest()


class ArtifactStore:
    """Local immutable content-addressed storage."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, content: bytes) -> str:
        digest = hashlib.sha256(content).hexdigest()
        target = self.root / digest[:2] / digest[2:]
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            temporary.write_bytes(content)
            try:
                os.link(temporary, target)
            except FileExistsError:
                pass
            finally:
                temporary.unlink(missing_ok=True)
        return digest

    def get(self, digest: str) -> bytes:
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid SHA-256 digest")
        content = (self.root / digest[:2] / digest[2:]).read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise OSError("artifact content hash mismatch")
        return content


class SQLiteRunStore:
    def __init__(self, path: str | Path) -> None:
        self.connection = sqlite3.connect(path, isolation_level="IMMEDIATE")
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
              run_id TEXT PRIMARY KEY, state TEXT NOT NULL, story_hash TEXT NOT NULL,
              repository TEXT NOT NULL, base_revision TEXT NOT NULL, provider TEXT NOT NULL,
              model TEXT NOT NULL, prompt_version TEXT NOT NULL, policy_version TEXT NOT NULL,
              usage_json TEXT NOT NULL, limits_json TEXT NOT NULL, created_at TEXT NOT NULL,
              profile_id TEXT NOT NULL DEFAULT 'provider-default',
              profile_sha256 TEXT NOT NULL DEFAULT
                'dde4c0e4b318df885b415f7cd2bbb33c3f4cf544dacb003902ac0cafa8fa1ee4'
            );
            CREATE TABLE IF NOT EXISTS transitions (
              id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
              from_state TEXT NOT NULL, to_state TEXT NOT NULL, attempted_at TEXT NOT NULL,
              committed INTEGER NOT NULL, detail TEXT
            );
            CREATE TABLE IF NOT EXISTS checkpoints (
              id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
              state TEXT NOT NULL, model TEXT NOT NULL, policy_version TEXT NOT NULL,
              payload_json TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS invocations (
              invocation_id TEXT PRIMARY KEY, execution_id TEXT UNIQUE NOT NULL,
              run_id TEXT NOT NULL REFERENCES runs(run_id), step TEXT NOT NULL,
              capability_id TEXT NOT NULL, arguments_json TEXT NOT NULL, status TEXT NOT NULL,
              result_json TEXT, error TEXT, created_at TEXT NOT NULL, completed_at TEXT,
              normalized_version INTEGER
            );
            CREATE TABLE IF NOT EXISTS approvals (
              approval_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
              action_digest TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audit_events (
              event_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
              timestamp TEXT NOT NULL, outcome TEXT NOT NULL, operation TEXT NOT NULL,
              capability_id TEXT, execution_id TEXT, detail TEXT
            );
            CREATE TABLE IF NOT EXISTS story_snapshots (
              run_id TEXT NOT NULL REFERENCES runs(run_id), revision INTEGER NOT NULL,
              content_hash TEXT NOT NULL, snapshot_json TEXT NOT NULL, active INTEGER NOT NULL,
              created_at TEXT NOT NULL, PRIMARY KEY(run_id, revision),
              UNIQUE(run_id, content_hash)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS one_active_story_snapshot
              ON story_snapshots(run_id) WHERE active=1;
            CREATE TABLE IF NOT EXISTS plans (
              run_id TEXT NOT NULL REFERENCES runs(run_id), story_revision INTEGER NOT NULL,
              content TEXT NOT NULL, created_at TEXT NOT NULL,
              PRIMARY KEY(run_id, story_revision),
              FOREIGN KEY(run_id, story_revision) REFERENCES story_snapshots(run_id, revision)
            );
            CREATE TABLE IF NOT EXISTS analysis_evidence (
              run_id TEXT NOT NULL REFERENCES runs(run_id), story_revision INTEGER NOT NULL,
              evidence_json TEXT NOT NULL, created_at TEXT NOT NULL,
              PRIMARY KEY(run_id, story_revision),
              FOREIGN KEY(run_id, story_revision) REFERENCES story_snapshots(run_id, revision)
            );
            CREATE TABLE IF NOT EXISTS workspaces (
              workspace_id TEXT PRIMARY KEY, run_id TEXT UNIQUE NOT NULL REFERENCES runs(run_id),
              repository TEXT NOT NULL, base_revision TEXT NOT NULL, path TEXT UNIQUE NOT NULL,
              created_at TEXT NOT NULL, current_generation INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS workspace_mutations (
              mutation_id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id),
              sequence INTEGER NOT NULL, capability_id TEXT NOT NULL, arguments_json TEXT NOT NULL,
              status TEXT NOT NULL, result_json TEXT, error TEXT, created_at TEXT NOT NULL,
              completed_at TEXT, UNIQUE(workspace_id, sequence)
            );
            CREATE TABLE IF NOT EXISTS workspace_checkpoints (
              id INTEGER PRIMARY KEY, workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id),
              mutation_id TEXT NOT NULL REFERENCES workspace_mutations(mutation_id),
              diff TEXT NOT NULL, changed_files_json TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS validation_attempts (
              id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
              attempt INTEGER NOT NULL, profile_id TEXT NOT NULL, generation INTEGER NOT NULL,
              status TEXT NOT NULL, result_json TEXT, created_at TEXT NOT NULL,
              completed_at TEXT, UNIQUE(run_id,attempt,profile_id)
            );
            CREATE TABLE IF NOT EXISTS validation_checkpoints (
              id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
              attempt INTEGER NOT NULL, passed INTEGER NOT NULL, results_json TEXT NOT NULL,
              created_at TEXT NOT NULL, UNIQUE(run_id,attempt)
            );
            CREATE TABLE IF NOT EXISTS publish_approvals (
              approval_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
              story_hash TEXT NOT NULL, story_revision INTEGER NOT NULL,
              repository TEXT NOT NULL, base_revision TEXT NOT NULL,
              workspace_generation INTEGER NOT NULL, diff_digest TEXT NOT NULL,
              capability TEXT NOT NULL, action TEXT NOT NULL, target_branch TEXT NOT NULL,
              target_account TEXT NOT NULL, title TEXT NOT NULL, policy_version TEXT NOT NULL,
              approver TEXT NOT NULL, expires_at TEXT NOT NULL, action_digest TEXT NOT NULL UNIQUE,
              status TEXT NOT NULL, created_at TEXT NOT NULL, approved_at TEXT, consumed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS publication_outbox (
              run_id TEXT PRIMARY KEY REFERENCES runs(run_id), approval_id TEXT NOT NULL UNIQUE
                REFERENCES publish_approvals(approval_id), action_digest TEXT NOT NULL,
              branch TEXT NOT NULL, status TEXT NOT NULL, intent_json TEXT NOT NULL,
              remote_json TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS jira_report_outbox (
              run_id TEXT PRIMARY KEY REFERENCES runs(run_id), issue_key TEXT NOT NULL,
              marker TEXT NOT NULL UNIQUE, body TEXT NOT NULL, status TEXT NOT NULL,
              remote_comment_id TEXT, result_json TEXT, created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS jira_transition_approvals (
              approval_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
              issue_key TEXT NOT NULL, transition_id TEXT NOT NULL, story_revision INTEGER NOT NULL,
              approver TEXT NOT NULL, expires_at TEXT NOT NULL, binding_digest TEXT NOT NULL UNIQUE,
              status TEXT NOT NULL, created_at TEXT NOT NULL, approved_at TEXT, consumed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS jira_transition_outbox (
              approval_id TEXT PRIMARY KEY REFERENCES jira_transition_approvals(approval_id),
              run_id TEXT NOT NULL REFERENCES runs(run_id), issue_key TEXT NOT NULL,
              transition_id TEXT NOT NULL, target_status TEXT NOT NULL, status TEXT NOT NULL,
              created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS run_skill_pins (
              run_id TEXT PRIMARY KEY REFERENCES runs(run_id), skill_id TEXT NOT NULL,
              version TEXT NOT NULL, content_sha256 TEXT NOT NULL, signature BLOB NOT NULL,
              signer_id TEXT NOT NULL, pinned_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS continuity_ledgers (
              run_id TEXT PRIMARY KEY REFERENCES runs(run_id), revision INTEGER NOT NULL,
              goal TEXT NOT NULL, constraints_json TEXT NOT NULL,
              completed_json TEXT NOT NULL, next_json TEXT NOT NULL,
              working_set_json TEXT NOT NULL, learnings_json TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS continuity_decisions (
              run_id TEXT NOT NULL REFERENCES continuity_ledgers(run_id), sequence INTEGER NOT NULL,
              decision TEXT NOT NULL, source TEXT NOT NULL, provenance TEXT NOT NULL,
              created_at TEXT NOT NULL, PRIMARY KEY(run_id, sequence)
            );
            CREATE TABLE IF NOT EXISTS continuity_updates (
              run_id TEXT NOT NULL REFERENCES continuity_ledgers(run_id), revision INTEGER NOT NULL,
              update_json TEXT NOT NULL, created_at TEXT NOT NULL,
              PRIMARY KEY(run_id, revision)
            );
            CREATE TRIGGER IF NOT EXISTS immutable_continuity_decision_update
              BEFORE UPDATE ON continuity_decisions
              BEGIN SELECT RAISE(ABORT, 'continuity decisions are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS immutable_continuity_decision_delete
              BEFORE DELETE ON continuity_decisions
              BEGIN SELECT RAISE(ABORT, 'continuity decisions are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS immutable_run_skill_pin_update
              BEFORE UPDATE ON run_skill_pins
              BEGIN SELECT RAISE(ABORT, 'run skill pins are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS immutable_run_skill_pin_delete
              BEFORE DELETE ON run_skill_pins
              BEGIN SELECT RAISE(ABORT, 'run skill pins are immutable'); END;
            """
        )
        from agent_runtime.profiles import (
            PROVIDER_DEFAULT_PROFILE_ID,
            PROVIDER_DEFAULT_PROFILE_SHA256,
        )

        self.connection.execute("BEGIN IMMEDIATE")
        try:
            run_columns = {
                row["name"] for row in self.connection.execute("PRAGMA table_info(runs)")
            }
            if "profile_id" not in run_columns:
                self.connection.execute(
                    "ALTER TABLE runs ADD COLUMN profile_id TEXT NOT NULL DEFAULT "
                    f"'{PROVIDER_DEFAULT_PROFILE_ID}'"
                )
            if "profile_sha256" not in run_columns:
                self.connection.execute(
                    "ALTER TABLE runs ADD COLUMN profile_sha256 TEXT NOT NULL DEFAULT "
                    f"'{PROVIDER_DEFAULT_PROFILE_SHA256}'"
                )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        invocation_columns = {
            row["name"] for row in self.connection.execute("PRAGMA table_info(invocations)")
        }
        if "normalized_version" not in invocation_columns:
            with self.connection:
                self.connection.execute(
                    "ALTER TABLE invocations ADD COLUMN normalized_version INTEGER"
                )

    def close(self) -> None:
        self.connection.close()

    def pin_skill(self, run_id: str, package: Any) -> None:
        """Persist the complete verified identity; an existing pin is immutable."""
        from agent_runtime.skills import SkillPin

        pin = SkillPin.from_package(package)
        with self.connection:
            try:
                self.connection.execute(
                    "INSERT INTO run_skill_pins VALUES (?,?,?,?,?,?,?)",
                    (
                        run_id,
                        pin.skill_id,
                        pin.version,
                        pin.content_sha256,
                        pin.signature,
                        pin.signer_id,
                        datetime.now(UTC).isoformat(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                if self.skill_pin(run_id) != pin:
                    raise ValueError("run already has a different immutable skill pin") from exc

    def skill_pin(self, run_id: str) -> Any:
        from agent_runtime.skills import SkillPin

        row = self.connection.execute(
            "SELECT * FROM run_skill_pins WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return SkillPin(
            row["skill_id"],
            row["version"],
            row["content_sha256"],
            bytes(row["signature"]),
            row["signer_id"],
        )

    def create_run(
        self,
        run_id: str,
        *,
        story_hash: str,
        repository: str,
        base_revision: str,
        provider: str,
        model: str,
        prompt_version: str,
        policy_version: str,
        profile_id: str | None = None,
        model_profile: Any | None = None,
        usage: Mapping[str, Any] | None = None,
        limits: Mapping[str, Any] | None = None,
    ) -> RunRecord:
        from agent_runtime.profiles import (
            MODEL_PROFILES,
            PROVIDER_DEFAULT_PROFILE_ID,
            PROVIDER_DEFAULT_PROFILE_SHA256,
            ModelRequestProfile,
        )

        selected_id = profile_id or PROVIDER_DEFAULT_PROFILE_ID
        if model_profile is not None and not isinstance(model_profile, ModelRequestProfile):
            raise TypeError("model_profile must be a ModelRequestProfile")
        if selected_id == PROVIDER_DEFAULT_PROFILE_ID:
            if model_profile is not None:
                raise ValueError("provider-default cannot pin an explicit model profile")
            profile_sha256 = PROVIDER_DEFAULT_PROFILE_SHA256
        else:
            selected = model_profile or MODEL_PROFILES.get(selected_id)
            if selected is None:
                raise ValueError("unknown model request profile")
            profile_sha256 = selected.operational_fingerprint()
        now = datetime.now(UTC).isoformat()
        with self.connection:
            self.connection.execute(
                "INSERT INTO runs(run_id,state,story_hash,repository,base_revision,provider,model,"
                "prompt_version,policy_version,usage_json,limits_json,created_at,profile_id,"
                "profile_sha256) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    RunState.NEW,
                    story_hash,
                    repository,
                    base_revision,
                    provider,
                    model,
                    prompt_version,
                    policy_version,
                    _json(usage or {}),
                    _json(limits or {}),
                    now,
                    selected_id,
                    profile_sha256,
                ),
            )
            self.connection.execute(
                "INSERT INTO checkpoints(run_id,state,model,policy_version,payload_json,created_at) "
                "VALUES (?,?,?,?,?,?)",
                (run_id, RunState.NEW, model, policy_version, "{}", now),
            )
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> RunRecord:
        row = self.connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return RunRecord(
            row["run_id"],
            RunState(row["state"]),
            row["story_hash"],
            row["repository"],
            row["base_revision"],
            row["provider"],
            row["model"],
            row["prompt_version"],
            row["policy_version"],
            json.loads(row["usage_json"]),
            json.loads(row["limits_json"]),
            row["profile_id"],
            row["profile_sha256"],
        )

    def transition(
        self, run_id: str, target: RunState, payload: Mapping[str, Any] | None = None
    ) -> None:
        now = datetime.now(UTC).isoformat()
        run = self.get_run(run_id)
        valid = run.state not in _TERMINAL and (
            (run.state, target) in _ALLOWED_TRANSITIONS
            or target in {RunState.FAILED, RunState.CANCELLED}
        )
        if not valid:
            with self.connection:
                self.connection.execute(
                    "INSERT INTO transitions(run_id,from_state,to_state,attempted_at,committed,detail) "
                    "VALUES (?,?,?,?,0,'invalid transition')",
                    (run_id, run.state, target, now),
                )
            raise InvalidTransition(f"cannot transition {run.state} to {target}")
        with self.connection:
            changed = self.connection.execute(
                "UPDATE runs SET state=? WHERE run_id=? AND state=?",
                (target, run_id, run.state),
            )
            if changed.rowcount != 1:
                current = self.connection.execute(
                    "SELECT state FROM runs WHERE run_id=?", (run_id,)
                ).fetchone()
                current_state = current[0] if current is not None else "missing"
                self.connection.execute(
                    "INSERT INTO transitions(run_id,from_state,to_state,attempted_at,committed,detail) "
                    "VALUES (?,?,?,?,0,?)",
                    (
                        run_id,
                        run.state,
                        target,
                        now,
                        f"stale state; current state is {current_state}",
                    ),
                )
            else:
                self.connection.execute(
                    "INSERT INTO transitions(run_id,from_state,to_state,attempted_at,committed,detail) "
                    "VALUES (?,?,?,?,?,?)",
                    (run_id, run.state, target, now, 1, None),
                )
                self.connection.execute(
                    "INSERT INTO checkpoints(run_id,state,model,policy_version,payload_json,created_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (run_id, target, run.model, run.policy_version, _json(payload or {}), now),
                )
        if changed.rowcount != 1:
            raise InvalidTransition(f"run state changed from expected {run.state}")

    def checkpoint(self, run_id: str) -> sqlite3.Row:
        row = self.connection.execute(
            "SELECT * FROM checkpoints WHERE run_id=? ORDER BY id DESC LIMIT 1", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(run_id)
        return row

    def save_story_snapshot(
        self, run_id: str, content_hash: str, snapshot: Mapping[str, Any], *, activate: bool = True
    ) -> StoredStorySnapshot:
        """Append an immutable revision; identical content reuses its existing revision."""
        run = self.get_run(run_id)
        if activate and run.state != RunState.NEW and content_hash != run.story_hash:
            raise ValueError("story binding is immutable after intake; start a new run")
        existing = self.connection.execute(
            "SELECT * FROM story_snapshots WHERE run_id=? AND content_hash=?",
            (run_id, content_hash),
        ).fetchone()
        if existing is not None:
            if activate:
                with self.connection:
                    if not existing["active"]:
                        self.connection.execute(
                            "UPDATE story_snapshots SET active=0 WHERE run_id=?", (run_id,)
                        )
                        self.connection.execute(
                            "UPDATE story_snapshots SET active=1 WHERE run_id=? AND revision=?",
                            (run_id, existing["revision"]),
                        )
                    self.connection.execute(
                        "UPDATE runs SET story_hash=? WHERE run_id=? AND state=?",
                        (content_hash, run_id, RunState.NEW),
                    )
            return self.story_snapshot(run_id, existing["revision"])
        row = self.connection.execute(
            "SELECT COALESCE(MAX(revision),0)+1 FROM story_snapshots WHERE run_id=?", (run_id,)
        ).fetchone()
        revision = row[0]
        with self.connection:
            if activate:
                self.connection.execute(
                    "UPDATE story_snapshots SET active=0 WHERE run_id=?", (run_id,)
                )
            self.connection.execute(
                "INSERT INTO story_snapshots VALUES (?,?,?,?,?,?)",
                (
                    run_id,
                    revision,
                    content_hash,
                    _json(snapshot),
                    int(activate),
                    datetime.now(UTC).isoformat(),
                ),
            )
            if activate:
                # Intake may replace a caller's placeholder with the fetched immutable binding.
                self.connection.execute(
                    "UPDATE runs SET story_hash=? WHERE run_id=? AND state=?",
                    (content_hash, run_id, RunState.NEW),
                )
        return self.story_snapshot(run_id, revision)

    def story_snapshot(self, run_id: str, revision: int | None = None) -> StoredStorySnapshot:
        if revision is None:
            row = self.connection.execute(
                "SELECT * FROM story_snapshots WHERE run_id=? AND active=1", (run_id,)
            ).fetchone()
        else:
            row = self.connection.execute(
                "SELECT * FROM story_snapshots WHERE run_id=? AND revision=?", (run_id, revision)
            ).fetchone()
        if row is None:
            raise KeyError(f"no story snapshot for run {run_id}")
        return StoredStorySnapshot(
            row["run_id"],
            row["revision"],
            row["content_hash"],
            json.loads(row["snapshot_json"]),
            bool(row["active"]),
        )

    def save_plan(self, run_id: str, revision: int, content: str) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO plans VALUES (?,?,?,?)",
                (run_id, revision, content, datetime.now(UTC).isoformat()),
            )

    def save_analysis_evidence(
        self, run_id: str, revision: int, evidence: tuple[Mapping[str, Any], ...]
    ) -> None:
        encoded = _json(evidence)
        existing = self.connection.execute(
            "SELECT evidence_json FROM analysis_evidence WHERE run_id=? AND story_revision=?",
            (run_id, revision),
        ).fetchone()
        if existing is not None:
            if existing[0] != encoded:
                raise ValueError("analysis evidence is immutable for a story revision")
            return
        with self.connection:
            self.connection.execute(
                "INSERT INTO analysis_evidence VALUES (?,?,?,?)",
                (run_id, revision, encoded, datetime.now(UTC).isoformat()),
            )

    def analysis_evidence(self, run_id: str, revision: int) -> tuple[dict[str, Any], ...]:
        row = self.connection.execute(
            "SELECT evidence_json FROM analysis_evidence WHERE run_id=? AND story_revision=?",
            (run_id, revision),
        ).fetchone()
        if row is None:
            return ()
        return tuple(json.loads(row[0]))

    def plan(self, run_id: str, revision: int | None = None) -> str:
        snapshot = self.story_snapshot(run_id, revision)
        row = self.connection.execute(
            "SELECT content FROM plans WHERE run_id=? AND story_revision=?",
            (run_id, snapshot.revision),
        ).fetchone()
        if row is None:
            raise KeyError(f"no plan for run {run_id} revision {snapshot.revision}")
        return row[0]

    def begin_invocation(
        self, run_id: str, step: str, capability_id: str, arguments: Mapping[str, Any]
    ) -> StoredInvocation:
        run = self.get_run(run_id)
        if run.state in _TERMINAL:
            if run.state == RunState.CANCELLED:
                raise RunCancelled("run is cancelled")
            raise RuntimeError(f"run is terminal: {run.state}")
        identity = invocation_identity(run_id, step, capability_id, arguments)
        existing = self.connection.execute(
            "SELECT * FROM invocations WHERE invocation_id=?", (identity,)
        ).fetchone()
        if existing is not None:
            return self._invocation(existing, replayed=True)
        execution_id = hashlib.sha256(f"execution:{identity}".encode()).hexdigest()
        with self.connection:
            self.connection.execute(
                "INSERT INTO invocations(invocation_id,execution_id,run_id,step,capability_id,"
                "arguments_json,status,created_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    identity,
                    execution_id,
                    run_id,
                    step,
                    capability_id,
                    _json(arguments),
                    "RUNNING",
                    datetime.now(UTC).isoformat(),
                ),
            )
        return StoredInvocation(identity, execution_id, "RUNNING", capability_id)

    def recover_running_invocations(self, run_id: str | None = None) -> int:
        where = "status='RUNNING'"
        parameters: tuple[Any, ...] = ()
        if run_id is not None:
            where += " AND run_id=?"
            parameters = (run_id,)
        with self.connection:
            cursor = self.connection.execute(
                f"UPDATE invocations SET status='FAILED',error='interrupted before terminal result',"
                f"completed_at=? WHERE {where}",
                (datetime.now(UTC).isoformat(), *parameters),
            )
        return cursor.rowcount

    def restart_interrupted_invocation(self, invocation_id: str) -> StoredInvocation:
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE invocations SET status='RUNNING',error=NULL,completed_at=NULL "
                "WHERE invocation_id=? AND status='FAILED' "
                "AND error='interrupted before terminal result'",
                (invocation_id,),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("invocation is not recoverable")
        row = self.connection.execute(
            "SELECT * FROM invocations WHERE invocation_id=?", (invocation_id,)
        ).fetchone()
        return self._invocation(row)

    def finish_invocation(
        self,
        invocation_id: str,
        *,
        result: Any = None,
        error: str | None = None,
        normalized_version: int | None = None,
    ) -> StoredInvocation:
        status = "FAILED" if error is not None else "SUCCEEDED"
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE invocations SET status=?,result_json=?,error=?,completed_at=?,"
                "normalized_version=? "
                "WHERE invocation_id=? AND status='RUNNING'",
                (
                    status,
                    None if error is not None else _json(result),
                    error,
                    datetime.now(UTC).isoformat(),
                    normalized_version,
                    invocation_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("invocation is already terminal")
        row = self.connection.execute(
            "SELECT * FROM invocations WHERE invocation_id=?", (invocation_id,)
        ).fetchone()
        return self._invocation(row)

    def invocation_by_execution(self, run_id: str, execution_id: str) -> StoredInvocation | None:
        row = self.connection.execute(
            "SELECT * FROM invocations WHERE run_id=? AND execution_id=?", (run_id, execution_id)
        ).fetchone()
        return None if row is None else self._invocation(row)

    @staticmethod
    def _invocation(row: sqlite3.Row, replayed: bool = False) -> StoredInvocation:
        result = json.loads(row["result_json"]) if row["result_json"] is not None else None
        return StoredInvocation(
            row["invocation_id"],
            row["execution_id"],
            row["status"],
            row["capability_id"],
            result,
            row["error"],
            replayed,
            row["normalized_version"],
        )

    def emit_audit(self, event: Any) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO audit_events VALUES (?,?,?,?,?,?,?,?)",
                (
                    event.event_id,
                    event.run_id,
                    event.timestamp.isoformat(),
                    event.outcome,
                    event.operation,
                    event.capability_id,
                    event.execution_id,
                    event.detail,
                ),
            )

    def begin_validation(self, run_id: str, attempt: int, profile_id: str, generation: int) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO validation_attempts(run_id,attempt,profile_id,generation,status,created_at) VALUES (?,?,?,?,?,?)",
                (run_id, attempt, profile_id, generation, "RUNNING", datetime.now(UTC).isoformat()),
            )

    def finish_validation(self, run_id: str, attempt: int, profile_id: str, result: Any) -> None:
        encoded = _json(result)
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE validation_attempts SET status=?,result_json=?,completed_at=? WHERE run_id=? AND attempt=? AND profile_id=? AND status='RUNNING'",
                (
                    "PASSED" if result["passed"] else "FAILED",
                    encoded,
                    datetime.now(UTC).isoformat(),
                    run_id,
                    attempt,
                    profile_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("validation attempt is not running")

    def validation_result(
        self, run_id: str, attempt: int, profile_id: str
    ) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT status,result_json FROM validation_attempts "
            "WHERE run_id=? AND attempt=? AND profile_id=?",
            (run_id, attempt, profile_id),
        ).fetchone()
        if row is None or row["status"] == "RUNNING":
            return None
        return json.loads(row["result_json"])

    def next_validation_attempt(self, run_id: str) -> int:
        checkpoint = self.connection.execute(
            "SELECT COALESCE(MAX(attempt),0) FROM validation_checkpoints WHERE run_id=?", (run_id,)
        ).fetchone()[0]
        running = self.connection.execute(
            "SELECT COALESCE(MAX(attempt),0) FROM validation_attempts "
            "WHERE run_id=? AND status='RUNNING'",
            (run_id,),
        ).fetchone()[0]
        return running if running > checkpoint else checkpoint + 1

    def save_validation_checkpoint(self, run_id: str, attempt: int, results: Any) -> None:
        passed = bool(results) and all(item["passed"] for item in results)
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO validation_checkpoints(run_id,attempt,passed,results_json,created_at) VALUES (?,?,?,?,?)",
                (run_id, attempt, int(passed), _json(results), datetime.now(UTC).isoformat()),
            )


class DurableAuditSink:
    def __init__(self, store: SQLiteRunStore) -> None:
        self.store = store

    def emit(self, event: Any) -> None:
        self.store.emit_audit(event)
