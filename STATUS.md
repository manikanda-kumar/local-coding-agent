# Local Coding Agent — Implementation Status

Last updated: 2026-08-02

This document summarizes what is implemented and what remains after completing the initial
autonomous JIRA-agent roadmap in [PLAN.md](PLAN.md). The next-stage items incorporate the earlier
local-agent analysis around durable project memory, context composition, local-model operation,
and verification-driven execution.

## Completed

### Fresh-orb lifecycle

- `.agents/setup` installs the Python development environment and verifies Bubblewrap namespace
  isolation. It is idempotent and has been run successfully twice in a fresh orb.
- `.agents/resume` verifies the resumed environment and has been run successfully.
- Both scripts are committed with mode `100755`.

### Provider and model foundation

- Provider-neutral chat, message, usage, reasoning, and tool-call contracts.
- OpenAI-compatible provider support for OpenRouter, vLLM, and similar endpoints.
- Capability-aware model routing and deterministic scripted-provider tests.
- Live OpenRouter smoke tests completed for `google/gemma-4-31b-it` and
  `minimax/minimax-m2.7` without storing credentials in the repository.
- Backend-qualified vLLM request profiles now pin sampling, timeout, reasoning replay, parser,
  and bounded extension behavior for Gemma-4-31B, MiniMax-M2.7, GLM-5.2, and Kimi-K3.
- A checked-in opt-in live contract reproduces a complete gateway loop; all four target models
  passed it against OpenRouter on 2026-08-02.

### Governed capability gateway

- Models receive only five stable gateway operations: capability search, describe, invoke,
  execution status, and cancellation.
- Models never receive direct MCP schemas, endpoints, credentials, unrestricted shell access, or
  external skill installation.
- Runtime-owned identity, stage, workspace, execution, approval, and idempotency context.
- Deny-by-default discovery and invocation policy with durable audit records.
- Reviewed MCP mappings with schema-hash drift detection, bounded output, redaction, and artifact
  spillover.

### Durable autonomous JIRA workflow

- Transactional SQLite run state machine, checkpoints, invocation replay, cancellation, and
  content-addressed artifacts.
- Immutable JIRA story snapshots and revision-bound implementation plans.
- Repository evidence pinned to Git objects without executing repository code.
- Runtime-owned immutable workspace generations with confined typed mutations.
- Bubblewrap-isolated validation profiles with network, environment, process, time, CPU, memory,
  and output controls.
- Bounded implementation/correction loop driven by trusted validation results.
- A runtime-owned `JiraCodingAgentWorkflow` now composes intake, planning, implementation,
  validation, approval/publication, and reporting from persisted state.
- Fresh-process tests reconstruct the workflow across publication and JIRA reporting crash
  windows without duplicate remote effects.

### Controlled external effects

- GitHub publication requires an expiring, single-use approval bound to the exact story,
  repository, base revision, workspace generation, diff, title, branch, account, and policy.
- Durable publication intent and reconciliation prevent duplicate pull requests after restart or
  uncertain remote outcomes.
- JIRA reporting uses idempotent marked comments and durable reconciliation.
- JIRA transitions are allowlisted and require an exact approval; failure reporting cannot move a
  story to Done.

### Internal skills, evaluation, and observability

- Ed25519-signed internal declarative skill packages with expiry, revocation, pinned content, and
  intersected capability policy.
- No URL, repository, or JIRA-based skill installation; skills cannot execute code, select
  endpoints, install packages, access secrets, or broaden authority.
- Checked-in deterministic golden-task evaluation contract and durable-state restart matrix.
- Structured model, policy, capability, retry, validation, publication, and reporting metrics.
- Bounded recursive metric redaction, durable at-least-once batch export, pending-event retention,
  stable event IDs, and optional OpenTelemetry export.
- Security audit records are retained indefinitely by the current local runtime.
- The golden task is executed through the real workflow/services and scored from persisted
  validation checkpoints and the resulting workspace, rather than by editing an in-memory dict.
- Gateway outputs are recursively credential-redacted, byte-bounded, artifact-spilled, and
  normalization-versioned before model exposure or durable persistence; legacy raw results are
  refused on replay.

### Current verification baseline

- `175 passed, 2 skipped` in the complete test suite.
- Ruff lint and formatting checks pass.
- The skipped tests are opt-in live JIRA and OpenRouter checks, not unit-test failures.

## Pending — Next Product Phases

### Phase 11 — Deterministic continuity ledger

- Done: compact revisioned per-run goal, constraints, immutable decisions/provenance, completed
  work, next work, working set, and learnings persist in SQLite.
- Done: bounded activity is derived from durable transitions and invocations, not model prose.
- Done: `continuity.memory.update` can append only bounded non-authoritative fields and is
  deny-by-default, stage-gated, durable-state checked, and crash/replay safe.
- Done: atomic updates, immutable decision triggers, bounded history, exact reopen, stale-CAS, and
  crash-window behavior are tested.

### Phase 12 — Budgeted context composer

- Done: conservative profile-pinned context budgeting preserves the stable prompt prefix and
  newest complete assistant/tool rounds while accounting for output reserve and gateway schemas.
- Done: malformed, orphaned, or duplicate tool correlations and a minimum context that cannot fit
  fail closed instead of sending an invalid model request.
- Done: deterministic task-tag routing selects bounded repository evidence and internal knowledge
  pages in stable order without semantic retrieval or authority coupling.
- Done: inject only the continuity head, active task state, relevant repository evidence, and selected
  internal knowledge pages within a configured token budget.
- Done: treat all retrieved content as untrusted data and keep context selection separate from capability
  authorization.
- Measure context relevance and model success against the golden-task harness.

### Phase 13 — Local and hybrid execution profiles

- Done: explicit local-vLLM request profiles for the four target models without changing workflow
  contracts.
- Pending: deployment-level hybrid frontier/local routing profiles.
- Support quality-triggered compaction and a separately configured summarization model.
- Keep subagent depth shallow; use subagents for bounded perception work and deterministic code for
  aggregation.
- Verify reasoning replay and tool-call behavior for each configured vLLM chat template.

### Phase 14 — Resume packets and knowledge routing

- Done (primitive): bounded resume packets and tagged deterministic knowledge routing are available.
- Done (planning): new and resumed planning calls consume a current resume packet and composed
  context, fail closed on provider/prompt/policy/profile drift, and budget the complete request.
- Pending: build the implementation model loop and enforce the same pins at each model-call boundary.

### Phase 15 — End-to-end JIRA-story dogfooding

- Run representative stories through intake, planning, implementation, validation, approval,
  publication, and reporting in a non-production integration environment.
- Compare local and hybrid profiles on success rate, retries, latency, token use, cost, and human
  review findings.
- Add failure-injection exercises for model timeout, MCP timeout, sandbox termination, base drift,
  GitHub uncertainty, JIRA uncertainty, and process restart.
- Promote a profile only after golden-task and human-review thresholds are defined and met.

## Pending — Deployment Inputs and Explicit Approvals

The following are intentionally not committed or exercised until deployment-specific values and
authorization are provided:

- Production JIRA, GitHub, MCP, vLLM, and telemetry endpoints and credentials.
- Reviewed MCP tool mappings and organization-specific capability/effect policy.
- Trusted validation profiles for each target repository.
- GitHub account, repository, base-branch, and approval identities.
- JIRA project transition allowlists and reporting policy.
- Internal skill signing keys, trusted signer registry, owners, expiry, and revocation process.
- Organization-specific retention periods, SLOs, dashboards, incident response, backup, and
  disaster-recovery requirements.
- Approval for any live pull-request creation or JIRA mutation.

## Deferred Until Measurements Justify Them

- Multi-worker claiming and distributed workflow coordination.
- Semantic or long-term memory and vector retrieval.
- Repository knowledge graphs.
- General plugin frameworks and external skill marketplaces.
- Multiple observability vendors.
- High-availability deployment and database migration beyond the current single-host SQLite
  baseline.

These items should be introduced only when measured workload, reliability, or retrieval quality
demonstrates that the simpler deterministic design is insufficient.
