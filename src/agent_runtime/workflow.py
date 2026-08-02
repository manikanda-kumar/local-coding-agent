from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from agent_runtime.durable import RunState, SQLiteRunStore
from agent_runtime.planning import IntakePlanningService
from agent_runtime.publication import PublicationResult, PublicationService
from agent_runtime.reporting import JiraReportingService, JiraReportResult
from agent_runtime.validation import ImplementationValidationCoordinator

ImplementationDriver = Callable[[int], tuple[int, int, float]]


@dataclass(frozen=True, slots=True)
class CodingAgentOutcome:
    state: RunState
    plan: str | None = None
    publication: PublicationResult | None = None
    report: JiraReportResult | None = None
    awaiting_publication_approval: bool = False
    awaiting_transition_approval: bool = False
    validation_exhausted: bool = False


class JiraCodingAgentWorkflow:
    """Runtime-owned, resumable composition of the existing workflow services."""

    def __init__(
        self,
        store: SQLiteRunStore,
        intake_planning: IntakePlanningService,
        implementation_validation: ImplementationValidationCoordinator,
        publication: PublicationService,
        reporting: JiraReportingService,
        *,
        max_steps: int = 16,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.store = store
        self.intake_planning = intake_planning
        self.implementation_validation = implementation_validation
        self.publication = publication
        self.reporting = reporting
        self.max_steps = max_steps

    def advance(
        self,
        run_id: str,
        issue_key: str,
        repository_evidence: tuple[Mapping[str, Any], ...],
        implement: ImplementationDriver,
        *,
        publication_approval_id: str | None = None,
        publication_title: str | None = None,
        transition_approval_id: str | None = None,
        require_transition_approval: bool = False,
    ) -> CodingAgentOutcome:
        plan = None
        published = None
        report = None
        terminal = {RunState.SUCCEEDED, RunState.FAILED, RunState.CANCELLED}
        for _ in range(self.max_steps):
            state = self.store.get_run(run_id).state
            if state in terminal:
                return CodingAgentOutcome(state, plan, published, report)
            if state == RunState.NEW:
                self.intake_planning.intake(run_id, issue_key)
            elif state in {RunState.INTAKE, RunState.ANALYZE}:
                plan = self.intake_planning.plan(run_id, repository_evidence)
            elif state in {RunState.PLAN_READY, RunState.IMPLEMENT, RunState.VALIDATE}:
                if not self.implementation_validation.run(run_id, implement):
                    return CodingAgentOutcome(
                        self.store.get_run(run_id).state,
                        plan,
                        validation_exhausted=True,
                    )
            elif state == RunState.AWAITING_PUBLISH_APPROVAL:
                if publication_approval_id is None or publication_title is None:
                    return CodingAgentOutcome(state, plan, awaiting_publication_approval=True)
                published = self.publication.publish(
                    run_id, publication_approval_id, title=publication_title
                )
            elif state == RunState.PUBLISH:
                published = self.publication.resume(run_id)
            elif state == RunState.REPORT:
                if require_transition_approval and transition_approval_id is None:
                    return CodingAgentOutcome(
                        state,
                        plan,
                        published,
                        awaiting_transition_approval=True,
                    )
                report = self.reporting.resume(
                    run_id, transition_approval_id=transition_approval_id
                )
            else:  # pragma: no cover - exhaustive guard for future states
                raise RuntimeError(f"unsupported run state: {state}")
        raise RuntimeError("workflow exceeded its bounded step count")

    run = advance
