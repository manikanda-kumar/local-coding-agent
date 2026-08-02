from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from agent_runtime.durable import SQLiteRunStore, _json
from agent_runtime.gateway import Capability, CapabilityCard, CapabilityDescriptor, Effect

MAX_STRING = 2_000
MAX_ITEMS = 32
MAX_HISTORY = 32
MAX_ACTIVITY = 32
MAX_REVISION = 1_000_000
MEMORY_WRITE_STATES = ("NEW", "INTAKE", "ANALYZE", "PLAN_READY", "IMPLEMENT", "VALIDATE")


@dataclass(frozen=True, slots=True)
class ContinuityDecision:
    decision: str
    source: str
    provenance: str


@dataclass(frozen=True, slots=True)
class ContinuityLedger:
    run_id: str
    revision: int
    goal: str
    constraints: tuple[str, ...]
    decisions: tuple[ContinuityDecision, ...]
    completed: tuple[str, ...]
    next: tuple[str, ...]
    working_set: tuple[str, ...]
    learnings: tuple[str, ...]


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_STRING:
        raise ValueError(f"{name} must be a non-empty string of at most {MAX_STRING} characters")
    return value


def _items(value: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or len(value) > MAX_ITEMS:
        raise ValueError(f"{name} must contain at most {MAX_ITEMS} items")
    return tuple(_text(item, name) for item in value)


class ContinuityService:
    """Runtime-owned, bounded continuity state; authority is deliberately absent."""

    def __init__(self, store: SQLiteRunStore) -> None:
        self.store = store

    def initialize(
        self, run_id: str, goal: str, constraints: Sequence[str] = ()
    ) -> ContinuityLedger:
        goal, constraints = _text(goal, "goal"), _items(constraints, "constraints")
        now = datetime.now(UTC).isoformat()
        with self.store.connection:
            self.store.connection.execute(
                "INSERT INTO continuity_ledgers VALUES (?,?,?,?,?,?,?,?,?)",
                (run_id, 0, goal, _json(constraints), "[]", "[]", "[]", "[]", now),
            )
        return self.get(run_id)

    def add_decision(
        self, run_id: str, decision: str, *, source: str, provenance: str
    ) -> ContinuityLedger:
        values = (
            _text(decision, "decision"),
            _text(source, "source"),
            _text(provenance, "provenance"),
        )
        with self.store.connection:
            count = self.store.connection.execute(
                "SELECT COUNT(*) FROM continuity_decisions WHERE run_id=?", (run_id,)
            ).fetchone()[0]
            if count >= MAX_ITEMS:
                raise ValueError("decision limit reached")
            self.store.connection.execute(
                "INSERT INTO continuity_decisions VALUES (?,?,?,?,?,?)",
                (run_id, count + 1, *values, datetime.now(UTC).isoformat()),
            )
        return self.get(run_id)

    def update(self, run_id: str, update: Mapping[str, Any], *, revision: int) -> ContinuityLedger:
        allowed = {"completed", "next", "working_set", "learnings"}
        unknown = update.keys() - allowed
        if unknown:
            raise ValueError(f"continuity update has forbidden field: {min(unknown)}")
        if not update:
            raise ValueError("continuity update must not be empty")
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
            raise ValueError("revision must be a non-negative integer")
        clean = {key: _items(value, key) for key, value in update.items()}
        if not any(clean.values()):
            raise ValueError("continuity update must append at least one item")
        encoded_update = _json(clean)
        now = datetime.now(UTC).isoformat()
        with self.store.connection:
            row = self.store.connection.execute(
                "SELECT * FROM continuity_ledgers WHERE run_id=?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(run_id)
            if row["revision"] != revision:
                applied = self.store.connection.execute(
                    "SELECT 1 FROM continuity_updates WHERE run_id=? AND revision=? "
                    "AND update_json=?",
                    (run_id, revision + 1, encoded_update),
                ).fetchone()
                if applied is not None:
                    return self.get(run_id)
                raise ValueError("stale continuity revision")
            if revision >= MAX_REVISION:
                raise ValueError("continuity revision limit reached")
            columns = {
                "completed": "completed_json",
                "next": "next_json",
                "working_set": "working_set_json",
                "learnings": "learnings_json",
            }
            merged = {key: json.loads(row[column]) for key, column in columns.items()}
            for key, additions in clean.items():
                merged[key] = list(_items((*merged[key], *additions), key))
            changed = self.store.connection.execute(
                "UPDATE continuity_ledgers SET revision=?,completed_json=?,next_json=?,"
                "working_set_json=?,learnings_json=?,updated_at=? WHERE run_id=? AND revision=? "
                "AND EXISTS (SELECT 1 FROM runs WHERE runs.run_id=continuity_ledgers.run_id "
                f"AND state IN ({','.join('?' for _ in MEMORY_WRITE_STATES)}))",
                (
                    revision + 1,
                    _json(merged["completed"]),
                    _json(merged["next"]),
                    _json(merged["working_set"]),
                    _json(merged["learnings"]),
                    now,
                    run_id,
                    revision,
                    *MEMORY_WRITE_STATES,
                ),
            )
            if changed.rowcount != 1:
                state = self.store.get_run(run_id).state
                if state not in MEMORY_WRITE_STATES:
                    raise ValueError("continuity writes are denied in the durable run state")
                raise ValueError("stale continuity revision")
            self.store.connection.execute(
                "INSERT INTO continuity_updates VALUES (?,?,?,?)",
                (run_id, revision + 1, encoded_update, now),
            )
            self.store.connection.execute(
                "DELETE FROM continuity_updates WHERE run_id=? AND revision NOT IN "
                "(SELECT revision FROM continuity_updates WHERE run_id=? "
                "ORDER BY revision DESC LIMIT ?)",
                (run_id, run_id, MAX_HISTORY),
            )
        return self.get(run_id)

    def get(self, run_id: str) -> ContinuityLedger:
        row = self.store.connection.execute(
            "SELECT * FROM continuity_ledgers WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(run_id)
        decisions = self.store.connection.execute(
            "SELECT decision,source,provenance FROM continuity_decisions "
            "WHERE run_id=? ORDER BY sequence",
            (run_id,),
        ).fetchall()
        return ContinuityLedger(
            run_id,
            row["revision"],
            row["goal"],
            tuple(json.loads(row["constraints_json"])),
            tuple(ContinuityDecision(*item) for item in decisions),
            tuple(json.loads(row["completed_json"])),
            tuple(json.loads(row["next_json"])),
            tuple(json.loads(row["working_set_json"])),
            tuple(json.loads(row["learnings_json"])),
        )

    def activity(self, run_id: str) -> tuple[dict[str, Any], ...]:
        """Derive activity solely from durable runtime events, in database order."""
        transitions = self.store.connection.execute(
            "SELECT id,from_state,to_state,attempted_at FROM transitions "
            "WHERE run_id=? AND committed=1 ORDER BY id DESC LIMIT ?",
            (run_id, MAX_ACTIVITY),
        ).fetchall()
        invocations = self.store.connection.execute(
            "SELECT rowid,capability_id,status,created_at FROM invocations "
            "WHERE run_id=? ORDER BY rowid DESC LIMIT ?",
            (run_id, MAX_ACTIVITY),
        ).fetchall()
        events = [
            {
                "kind": "transition",
                "sequence": row[0],
                "from": row[1],
                "to": row[2],
                "occurred_at": row[3],
            }
            for row in transitions
        ] + [
            {
                "kind": "capability",
                "sequence": row[0],
                "capability_id": row[1],
                "status": row[2],
                "occurred_at": row[3],
            }
            for row in invocations
        ]
        ordered = sorted(
            events, key=lambda item: (item["occurred_at"], item["kind"], item["sequence"])
        )
        return tuple(ordered[-MAX_ACTIVITY:])


def continuity_memory_capability(service: ContinuityService) -> Capability:
    item = {"type": "string", "maxLength": MAX_STRING, "minLength": 1}
    schema = {
        "type": "object",
        "properties": {
            "revision": {"type": "integer", "minimum": 0, "maximum": MAX_REVISION - 1},
            **{
                key: {"type": "array", "items": item, "maxItems": MAX_ITEMS}
                for key in ("completed", "next", "working_set", "learnings")
            },
        },
        "required": ["revision"],
        "additionalProperties": False,
    }

    def update(arguments: Mapping[str, Any], context: Any) -> dict[str, Any]:
        revision = arguments["revision"]
        ledger = service.update(
            context.run_id,
            {k: v for k, v in arguments.items() if k != "revision"},
            revision=revision,
        )
        return {"revision": ledger.revision}

    return Capability(
        CapabilityDescriptor(
            CapabilityCard(
                "continuity.memory.update",
                "Update continuity memory",
                "Append bounded non-authoritative run memory",
            ),
            schema,
            Effect.TRUSTED_MEMORY_WRITE,
        ),
        update,
    )
