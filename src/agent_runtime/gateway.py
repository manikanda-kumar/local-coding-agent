from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from agent_runtime.durable import SQLiteRunStore


class PolicyDecision(StrEnum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"


class ExecutionStatus(StrEnum):
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class Effect(StrEnum):
    READ = "read"
    TRUSTED_READ = "trusted_read"
    TRUSTED_WORKSPACE_READ = "trusted_workspace_read"
    TRUSTED_WORKSPACE_WRITE = "trusted_workspace_write"
    TRUSTED_PROCESS_EXECUTION = "trusted_process_execution"


@dataclass(frozen=True, slots=True)
class CapabilityCard:
    capability_id: str
    name: str
    summary: str
    version: str = "1"


@dataclass(frozen=True, slots=True)
class CapabilityDescriptor:
    card: CapabilityCard
    input_schema: dict[str, Any]
    effect: Effect = Effect.TRUSTED_READ


@dataclass(frozen=True, slots=True)
class InvocationContext:
    principal_id: str
    run_id: str
    stage: str
    story_id: str | None = None
    repository_id: str | None = None
    workspace_id: str | None = None
    policy_version: str = "1"


@dataclass(slots=True)
class ExecutionRecord:
    execution_id: str
    invocation_id: str
    capability_id: str
    status: ExecutionStatus
    result: Any = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class AuditEvent:
    event_id: str
    timestamp: datetime
    outcome: str
    operation: str
    run_id: str
    capability_id: str | None = None
    execution_id: str | None = None
    detail: str | None = None


Handler = Callable[[Mapping[str, Any], InvocationContext], Any]


@dataclass(frozen=True, slots=True)
class Capability:
    descriptor: CapabilityDescriptor
    handler: Handler = field(repr=False, compare=False)


class InMemoryCapabilityCatalog:
    def __init__(self, capabilities: tuple[Capability, ...] = ()) -> None:
        self._capabilities: dict[str, Capability] = {}
        for item in capabilities:
            self.register(item)

    def get(self, capability_id: str) -> Capability | None:
        return self._capabilities.get(capability_id)

    def all(self) -> tuple[Capability, ...]:
        return tuple(self._capabilities.values())

    def register(self, capability: Capability) -> None:
        try:
            effect = Effect(capability.descriptor.effect)
        except ValueError as exc:
            raise ValueError("unknown capability effect") from exc
        capability_id = capability.descriptor.card.capability_id
        reserved_effects = {
            "workspace.patch.apply": Effect.TRUSTED_WORKSPACE_WRITE,
            "workspace.file.create": Effect.TRUSTED_WORKSPACE_WRITE,
            "git.diff.read": Effect.TRUSTED_WORKSPACE_READ,
            "workspace.test.run": Effect.TRUSTED_PROCESS_EXECUTION,
            "workspace.lint.run": Effect.TRUSTED_PROCESS_EXECUTION,
        }
        reserved = capability_id.startswith(("workspace.", "git.diff."))
        if reserved and reserved_effects.get(capability_id) != effect:
            raise ValueError("reserved workspace capability has invalid effect or identifier")
        if not reserved and effect in {
            Effect.TRUSTED_WORKSPACE_READ,
            Effect.TRUSTED_WORKSPACE_WRITE,
            Effect.TRUSTED_PROCESS_EXECUTION,
        }:
            raise ValueError("workspace effect is reserved")
        if capability.descriptor.card.capability_id in self._capabilities:
            raise ValueError("capability is already registered")
        self._capabilities[capability.descriptor.card.capability_id] = capability


class StaticPolicyEngine:
    """An explicit allow/approval list; everything else is denied."""

    def __init__(
        self,
        allowed: frozenset[str] = frozenset(),
        require_approval: frozenset[str] = frozenset(),
    ) -> None:
        self._allowed = allowed
        self._require_approval = require_approval

    def decide(self, capability_id: str, context: InvocationContext) -> PolicyDecision:
        del context
        if capability_id in self._allowed:
            return PolicyDecision.ALLOW
        if capability_id in self._require_approval:
            return PolicyDecision.REQUIRE_APPROVAL
        return PolicyDecision.DENY


class InMemoryAuditSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def emit(self, event: AuditEvent) -> None:
        self.events.append(event)


class GatewayError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _validate(value: Any, schema: Mapping[str, Any], path: str = "arguments") -> None:
    expected = schema.get("type")
    types: dict[str, type | tuple[type, ...]] = {
        "object": dict,
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "null": type(None),
    }
    if expected in types and (
        not isinstance(value, types[expected])
        or expected in {"integer", "number"}
        and isinstance(value, bool)
    ):
        raise GatewayError("invalid_request", f"{path} must be {expected}")
    if expected == "object":
        properties = schema.get("properties", {})
        missing = set(schema.get("required", ())) - value.keys()
        if missing:
            raise GatewayError("invalid_request", f"{path} missing required field: {min(missing)}")
        unknown = value.keys() - properties.keys()
        if unknown and schema.get("additionalProperties", True) is False:
            raise GatewayError("invalid_request", f"{path} has unknown field: {min(unknown)}")
        for key in value.keys() & properties.keys():
            _validate(value[key], properties[key], f"{path}.{key}")
    if expected == "array" and "items" in schema:
        for index, item in enumerate(value):
            _validate(item, schema["items"], f"{path}[{index}]")


class CapabilityGateway:
    def __init__(
        self,
        catalog: InMemoryCapabilityCatalog,
        policy: StaticPolicyEngine,
        audit: InMemoryAuditSink,
        context: InvocationContext,
        store: SQLiteRunStore | None = None,
    ) -> None:
        self.catalog, self.policy, self.audit, self.context = catalog, policy, audit, context
        self.store = store
        self._executions: dict[str, ExecutionRecord] = {}

    def _audit(self, outcome: str, operation: str, **kwargs: Any) -> None:
        self.audit.emit(
            AuditEvent(
                str(uuid4()), datetime.now(UTC), outcome, operation, self.context.run_id, **kwargs
            )
        )

    def audit_invalid(self, operation: str, detail: str) -> None:
        self._audit("invalid_request", operation, detail=detail)

    def search(self, query: str) -> tuple[CapabilityCard, ...]:
        if not isinstance(query, str):
            self._audit("invalid_request", "search", detail="query must be a string")
            raise GatewayError("invalid_request", "query must be a string")
        query = query.casefold()
        cards = tuple(
            capability.descriptor.card
            for capability in self.catalog.all()
            if self.policy.decide(capability.descriptor.card.capability_id, self.context)
            == PolicyDecision.ALLOW
            and query
            in (
                capability.descriptor.card.name + " " + capability.descriptor.card.summary
            ).casefold()
        )
        self._audit("success", "search")
        return cards

    def _authorized(self, capability_id: str, operation: str) -> Capability:
        capability = self.catalog.get(capability_id)
        decision = self.policy.decide(capability_id, self.context)
        if capability is None or decision != PolicyDecision.ALLOW:
            detail = (
                "approval required" if decision == PolicyDecision.REQUIRE_APPROVAL else "denied"
            )
            self._audit("denial", operation, capability_id=capability_id, detail=detail)
            code = "approval_required" if decision == PolicyDecision.REQUIRE_APPROVAL else "denied"
            raise GatewayError(code, detail)
        effect = capability.descriptor.effect
        if capability_id.startswith(("workspace.", "git.diff.")) and effect not in {
            "trusted_workspace_read",
            "trusted_workspace_write",
        }:
            self._audit("denial", operation, capability_id=capability_id, detail="untrusted effect")
            raise GatewayError("denied", "untrusted effect classification")
        if effect == Effect.TRUSTED_WORKSPACE_WRITE and self.context.stage != "IMPLEMENT":
            self._audit("denial", operation, capability_id=capability_id, detail="stage denied")
            raise GatewayError("denied", "workspace writes require IMPLEMENT stage")
        if effect == Effect.TRUSTED_PROCESS_EXECUTION and self.context.stage != "VALIDATE":
            self._audit("denial", operation, capability_id=capability_id, detail="stage denied")
            raise GatewayError("denied", "validation execution requires VALIDATE stage")
        return capability

    def describe(self, capability_id: str) -> CapabilityDescriptor:
        capability = self._authorized(capability_id, "describe")
        self._audit("success", "describe", capability_id=capability_id)
        return capability.descriptor

    def invoke(self, capability_id: str, arguments: Mapping[str, Any]) -> ExecutionRecord:
        capability = self._authorized(capability_id, "invoke")
        try:
            _validate(arguments, capability.descriptor.input_schema)
        except GatewayError as error:
            self._audit("invalid_request", "invoke", capability_id=capability_id, detail=str(error))
            raise
        durable = None
        if self.store is not None:
            try:
                durable = self.store.begin_invocation(
                    self.context.run_id, self.context.stage, capability_id, arguments
                )
            except Exception as error:
                self._audit("failure", "invoke", capability_id=capability_id, detail=str(error))
                raise GatewayError("run_unavailable", str(error)) from error
            record = ExecutionRecord(
                durable.execution_id,
                durable.invocation_id,
                capability_id,
                ExecutionStatus(durable.status),
                durable.result,
                durable.error,
            )
            if durable.replayed:
                if (
                    record.status == ExecutionStatus.FAILED
                    and record.error == "interrupted before terminal result"
                    and capability.descriptor.effect == Effect.TRUSTED_WORKSPACE_WRITE
                ):
                    durable = self.store.restart_interrupted_invocation(record.invocation_id)
                    record.status = ExecutionStatus(durable.status)
                    record.error = None
                else:
                    self._executions[record.execution_id] = record
                    self._audit(
                        "success" if record.status == ExecutionStatus.SUCCEEDED else "failure",
                        "invoke_replay",
                        capability_id=capability_id,
                        execution_id=record.execution_id,
                    )
                    return record
        else:
            record = ExecutionRecord(
                str(uuid4()), str(uuid4()), capability_id, ExecutionStatus.RUNNING
            )
        self._executions[record.execution_id] = record
        try:
            record.result = capability.handler(arguments, self.context)
            record.status = ExecutionStatus.SUCCEEDED
            if self.store is not None:
                self.store.finish_invocation(record.invocation_id, result=record.result)
            self._audit(
                "success", "invoke", capability_id=capability_id, execution_id=record.execution_id
            )
        except Exception as error:  # noqa: BLE001 - capability failures become records
            record.status, record.error = ExecutionStatus.FAILED, str(error)
            if self.store is not None:
                self.store.finish_invocation(record.invocation_id, error=str(error))
            self._audit(
                "failure",
                "invoke",
                capability_id=capability_id,
                execution_id=record.execution_id,
                detail=str(error),
            )
        return record

    def status(self, execution_id: str) -> ExecutionRecord:
        record = self._executions.get(execution_id)
        if record is None and self.store is not None:
            durable = self.store.invocation_by_execution(self.context.run_id, execution_id)
            if durable is not None:
                record = ExecutionRecord(
                    durable.execution_id,
                    durable.invocation_id,
                    durable.capability_id,
                    ExecutionStatus(durable.status),
                    durable.result,
                    durable.error,
                )
        if record is None:
            self._audit(
                "invalid_request", "status", execution_id=execution_id, detail="unknown execution"
            )
            raise GatewayError("invalid_request", "unknown execution")
        self._audit(
            "success", "status", capability_id=record.capability_id, execution_id=execution_id
        )
        return record

    def cancel(self, execution_id: str) -> ExecutionRecord:
        record = self._executions.get(execution_id)
        if record is None:
            self._audit(
                "invalid_request", "cancel", execution_id=execution_id, detail="unknown execution"
            )
            raise GatewayError("invalid_request", "unknown execution")
        if record.status != ExecutionStatus.RUNNING:
            self._audit(
                "invalid_request",
                "cancel",
                capability_id=record.capability_id,
                execution_id=execution_id,
                detail="execution is already terminal",
            )
            raise GatewayError("not_cancellable", "execution is already terminal")
        record.status = ExecutionStatus.CANCELLED
        self._audit(
            "cancellation", "cancel", capability_id=record.capability_id, execution_id=execution_id
        )
        return record


def fixture_read_capability() -> Capability:
    card = CapabilityCard("fixture.read", "Fixture read", "Read a harmless fixture value")
    schema = {
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
        "additionalProperties": False,
    }
    return Capability(
        CapabilityDescriptor(card, schema),
        lambda args, _: {"key": args["key"], "value": "fixture-value"},
    )


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, (tuple, list)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (StrEnum, datetime)):
        return str(value)
    return value
