"""Durable, vendor-neutral runtime metrics with bounded, recursively redacted attributes."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import islice
from pathlib import Path
from typing import Any, Protocol

SENSITIVE_KEYS = frozenset({"authorization", "cookie", "password", "secret", "token", "api_key"})
_SENSITIVE_NORMALIZED = frozenset(
    {
        "accesstoken",
        "apikey",
        "authorization",
        "clientsecret",
        "cookie",
        "password",
        "passwd",
        "proxyauthorization",
        "refreshtoken",
        "secret",
        "setcookie",
        "token",
    }
)
_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")


def is_sensitive_key(key: str, configured: frozenset[str] = SENSITIVE_KEYS) -> bool:
    normalized = "".join(character for character in key.lower() if character.isalnum())
    configured_normalized = {
        "".join(character for character in item.lower() if character.isalnum())
        for item in configured
    }
    return (
        normalized in configured_normalized | _SENSITIVE_NORMALIZED
        or normalized.endswith(("password", "passwd", "secret", "token", "apikey", "cookie"))
        or normalized.startswith(("authorization", "proxyauthorization", "cookie"))
    )


def redact_sensitive_text(value: str) -> str:
    """Remove common credential forms from untrusted logs and capability text."""
    value = re.sub(
        r"(?im)([\"']?)\b(proxy-authorization|authorization)\b\1\s*:\s*[^\r\n]*",
        r"\2=[REDACTED]",
        value,
    )
    value = re.sub(
        r"(?im)([\"']?)\b(set-cookie|cookie)\b\1\s*:\s*[^\r\n]*",
        r"\2=[REDACTED]",
        value,
    )
    value = re.sub(
        r"(?i)([\"']?)\b(access_token|refresh_token|client_secret|api[_-]?key|password|passwd|secret|token)"
        r"\b\1\s*[:=]\s*(?:\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s,;]+)",
        r"\2=[REDACTED]",
        value,
    )
    value = re.sub(r"(?i)bearer\s+[^\s,;]+", "Bearer [REDACTED]", value)
    return re.sub(
        r"(?i)\b(?:sk|pk)[-_](?:(?:live|test)[-_])?[A-Za-z0-9_-]{8,}\b",
        "[REDACTED]",
        value,
    )


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    """Metrics retention; security audit records remain indefinitely retained."""

    metric_days: int = 30

    def __post_init__(self) -> None:
        if self.metric_days < 1:
            raise ValueError("metric retention must be positive")


@dataclass(frozen=True, slots=True)
class RedactionPolicy:
    sensitive_keys: frozenset[str] = SENSITIVE_KEYS
    max_depth: int = 6
    max_items: int = 32
    max_string: int = 256
    max_nodes: int = 512

    def __post_init__(self) -> None:
        if self.max_depth < 1 or self.max_items < 1 or self.max_string < 1 or self.max_nodes < 1:
            raise ValueError("redaction bounds must be positive")

    def _sensitive(self, key: str) -> bool:
        return is_sensitive_key(key, self.sensitive_keys)

    def sensitive(self, key: str) -> bool:
        return self._sensitive(key)

    def redact(self, value: Any) -> Any:
        return self._redact(value, depth=0, budget=[self.max_nodes])

    def _redact(self, value: Any, *, depth: int, budget: list[int]) -> Any:
        if budget[0] <= 0:
            return "[TRUNCATED]"
        budget[0] -= 1
        if depth >= self.max_depth:
            return "[TRUNCATED]"
        if isinstance(value, Mapping):
            result = {}
            for key, item in islice(value.items(), self.max_items):
                if not isinstance(key, str):
                    raise TypeError("metric attribute keys must be strings")
                raw_name = key
                name = redact_sensitive_text(raw_name)[: self.max_string]
                if name in result:
                    # Truncated keys can collide; never let a later value replace a redaction.
                    result[name] = "[REDACTED]"
                    continue
                result[name] = (
                    "[REDACTED]"
                    if self._sensitive(raw_name)
                    else self._redact(item, depth=depth + 1, budget=budget)
                )
            return result
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [
                self._redact(item, depth=depth + 1, budget=budget)
                for item in islice(value, self.max_items)
            ]
        if isinstance(value, str):
            return redact_sensitive_text(value)[: self.max_string]
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(value) or abs(value) > 10**15:
                raise ValueError("metric numeric attributes must be finite and bounded")
            return value
        raise ValueError("metric attributes must contain only JSON primitives")


@dataclass(frozen=True, slots=True)
class MetricEvent:
    category: str
    name: str
    outcome: str
    value: float = 1
    run_id: str | None = None
    duration_ms: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0
    attributes: Mapping[str, Any] | None = None
    event_id: int | None = None


class MetricsSink(Protocol):
    """Accepts observations whose names, outcomes, and attributes are bounded labels only."""

    def record(self, event: MetricEvent) -> int: ...


def emit_metric(sink: MetricsSink | None, event: MetricEvent) -> bool:
    """Record best-effort telemetry without changing workflow or authorization semantics."""
    if sink is None:
        return False
    try:
        sink.record(event)
    except Exception:  # noqa: BLE001 - an optional observability sink cannot break the workflow
        return False
    return True


class MetricsExporter(Protocol):
    """At-least-once export; ``event_id`` is stable for downstream deduplication."""

    def export(self, events: Sequence[MetricEvent]) -> None: ...


class SQLiteMetricsCollector:
    """Persists before export; exporter failure cannot lose the local observation."""

    def __init__(
        self, path: str | Path, *, redaction: RedactionPolicy | None = None, batch_size: int = 500
    ) -> None:
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or not 1 <= batch_size <= 10_000
        ):
            raise ValueError("metrics batch_size must be an integer from 1 to 10000")
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.redaction = redaction or RedactionPolicy()
        self.batch_size = batch_size
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS metric_events (id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, "
            "category TEXT NOT NULL, name TEXT NOT NULL, outcome TEXT NOT NULL, value REAL NOT NULL, "
            "run_id TEXT, duration_ms REAL, input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL, "
            "cost_usd REAL NOT NULL, attributes_json TEXT NOT NULL, exported_at TEXT)"
        )
        columns = {
            row["name"] for row in self.connection.execute("PRAGMA table_info(metric_events)")
        }
        if "exported_at" not in columns:
            self.connection.execute("ALTER TABLE metric_events ADD COLUMN exported_at TEXT")
        self.connection.commit()

    def record(self, event: MetricEvent) -> int:
        if event.category not in {
            "model",
            "capability",
            "policy",
            "retry",
            "validation",
            "publication",
            "report",
        }:
            raise ValueError("unknown metric category")
        for field, value in (("name", event.name), ("outcome", event.outcome)):
            if not _LABEL.fullmatch(value):
                raise ValueError(f"metric {field} must be a bounded low-cardinality label")
        if event.run_id is not None and (not event.run_id or len(event.run_id) > 128):
            raise ValueError("metric run_id must contain 1-128 characters")
        numeric = (event.value, event.duration_ms, event.cost_usd)
        if any(
            value is not None
            and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            )
            for value in numeric
        ):
            raise ValueError("metric numeric values must be finite and non-negative")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 10**12
            for value in (event.input_tokens, event.output_tokens)
        ):
            raise ValueError("metric token counts must be exact integers from 0 to 1000000000000")
        attributes = self.redaction.redact(event.attributes or {})
        try:
            attributes_json = json.dumps(
                attributes, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("metric attributes must contain finite JSON primitives") from exc
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO metric_events(timestamp,category,name,outcome,value,run_id,duration_ms,"
                "input_tokens,output_tokens,cost_usd,attributes_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    datetime.now(UTC).isoformat(),
                    event.category,
                    event.name,
                    event.outcome,
                    event.value,
                    event.run_id,
                    event.duration_ms,
                    event.input_tokens,
                    event.output_tokens,
                    event.cost_usd,
                    attributes_json,
                ),
            )
        return int(cursor.lastrowid)

    def events(self, *, run_id: str | None = None, pending_only: bool = False) -> list[MetricEvent]:
        clauses: list[str] = []
        parameters: list[str] = []
        if run_id is not None:
            clauses.append("run_id=?")
            parameters.append(run_id)
        if pending_only:
            clauses.append("exported_at IS NULL")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return [
            MetricEvent(
                row["category"],
                row["name"],
                row["outcome"],
                row["value"],
                row["run_id"],
                row["duration_ms"],
                row["input_tokens"],
                row["output_tokens"],
                row["cost_usd"],
                json.loads(row["attributes_json"]),
                row["id"],
            )
            for row in self.connection.execute(
                f"SELECT * FROM metric_events{where} ORDER BY id", parameters
            )
        ]

    def export(self, exporter: MetricsExporter) -> int:
        """Export one bounded ID snapshot with at-least-once delivery semantics."""
        rows = list(
            self.connection.execute(
                "SELECT * FROM metric_events WHERE exported_at IS NULL ORDER BY id LIMIT ?",
                (self.batch_size,),
            )
        )
        if not rows:
            return 0
        events = [
            MetricEvent(
                row["category"],
                row["name"],
                row["outcome"],
                row["value"],
                row["run_id"],
                row["duration_ms"],
                row["input_tokens"],
                row["output_tokens"],
                row["cost_usd"],
                json.loads(row["attributes_json"]),
                row["id"],
            )
            for row in rows
        ]
        exporter.export(events)
        exported_at = datetime.now(UTC).isoformat()
        with self.connection:
            self.connection.executemany(
                "UPDATE metric_events SET exported_at=? WHERE id=? AND exported_at IS NULL",
                ((exported_at, row["id"]) for row in rows),
            )
        return len(rows)

    def enforce_retention(self, policy: RetentionPolicy, *, now: datetime | None = None) -> int:
        cutoff = (now or datetime.now(UTC)) - timedelta(days=policy.metric_days)
        with self.connection:
            cursor = self.connection.execute(
                "DELETE FROM metric_events WHERE timestamp < ? AND exported_at IS NOT NULL",
                (cutoff.isoformat(),),
            )
        return cursor.rowcount


class OpenTelemetryMetricsExporter:
    """Optional adapter. OpenTelemetry is imported only when this exporter is constructed."""

    def __init__(self, meter_name: str = "agent_runtime") -> None:
        try:
            from opentelemetry import metrics
        except ImportError as exc:
            raise RuntimeError("install opentelemetry-api to use this exporter") from exc
        self.meter = metrics.get_meter(meter_name)
        self.events = self.meter.create_counter("runtime.events")
        self.value = self.meter.create_counter("runtime.value")
        self.duration = self.meter.create_histogram("runtime.duration", unit="ms")
        self.input_tokens = self.meter.create_counter("runtime.input_tokens")
        self.output_tokens = self.meter.create_counter("runtime.output_tokens")
        self.cost = self.meter.create_counter("runtime.cost", unit="USD")

    def export(self, events: Sequence[MetricEvent]) -> None:
        for event in events:
            labels = {"category": event.category, "name": event.name, "outcome": event.outcome}
            self.events.add(1, labels)
            if event.value:
                self.value.add(event.value, labels)
            if event.duration_ms:
                self.duration.record(event.duration_ms, labels)
            if event.input_tokens:
                self.input_tokens.add(event.input_tokens, labels)
            if event.output_tokens:
                self.output_tokens.add(event.output_tokens, labels)
            if event.cost_usd:
                self.cost.add(event.cost_usd, labels)
