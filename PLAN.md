# Enterprise Agent Runtime Platform

## Vision

Build a provider-agnostic enterprise agent platform that powers coding assistants, banking assistants, customer support agents and future AI products.

The runtime should be independent of any LLM provider.

Applications become plugins.

---

# Design Principles

- Provider Agnostic
- OpenAI Compatible APIs
- Local-first
- Cloud Ready
- Cost Optimized
- Observable
- Secure
- Enterprise Ready

---

# High-Level Architecture

Applications

- Coding Agent
- Banking Assistant
- Widget Platform
- Contact Center
- Internal Copilot

↓

Enterprise Agent Runtime

↓

Provider SDK

↓

OpenAI
Anthropic
Gemini
OpenRouter
vLLM
SGLang
Enterprise Models

---

# Core Modules

## Workspace Manager

Repository
Files
Artifacts
Plans

---

## Context Composer

Instead of prompt history

Maintain

- Repository Graph
- Active Task
- Recent Changes
- Design Documents
- Open Questions
- Dependencies

Context becomes structured data.

---

## Memory Manager

Short Memory

Long Memory

Semantic Memory

Task Memory

---

## Model Router

Select provider based on

Latency

Cost

Capability

Context Length

Tool Support

Availability

---

## Tool Registry

Every tool described through schemas.

Examples

Filesystem

Git

GitHub

Browser

Database

Kubernetes

Terminal

Documentation

Internal APIs

---

## Execution Engine

Planning

Reasoning

Tool Calls

Retries

Reflection

Checkpointing

Streaming

---

## Provider SDK

Every provider implements

Chat

Stream

Tool Calls

Embeddings

Capabilities

Models

Pricing

Context Limits

---

## Observability

OpenTelemetry

Langfuse

LangSmith

Helicone

Prometheus

Grafana

Every execution produces

- Trace
- Cost
- Latency
- Tokens
- Tool Calls
- Errors
- Success Rate
- User Feedback

---

## Evaluation

Golden Tasks

Regression Tests

Benchmarks

Human Review

Quality Scores

---

## Security

RBAC

Secrets

Audit Logs

Policy Engine

Approval Gates

Sandbox Execution

---

# Coding Agent MVP

Capabilities

Repository Search

Planning

Code Generation

Testing

PR Creation

Documentation

Review

Refactoring

---

# Future Applications

- Banking Assistant
- Experience Platform
- Contact Center
- AI Workflow Engine
- Internal Enterprise Copilot

---

# Active Product Goal

Build an autonomous JIRA agent that can take a story from intake through a tested pull
request while keeping models isolated from credentials, MCP servers, external skills, host
execution, and irreversible actions.

The first complete release must prove this workflow:

1. Read and snapshot a JIRA story.
2. Analyze the story using approved internal code-intelligence capabilities.
3. Produce an implementation plan with repository evidence.
4. Apply a bounded change in an isolated workspace.
5. Run trusted validation profiles in a real sandbox.
6. Pause before publication.
7. Create exactly one pull request after approval.
8. Report the result to JIRA without duplicate side effects.
9. Resume safely from every persisted stage after a restart.

---

# Architecture Decisions

## Central Capability Gateway

Models never connect directly to MCP servers, skills, credentials, or internal APIs. They receive
five stable tools backed by one gateway:

- `capability_search`
- `capability_describe`
- `capability_invoke`
- `execution_status`
- `execution_cancel`

The gateway owns capability discovery, schema validation, policy decisions, execution identity,
idempotency, output limits, redaction, and audit. Runtime-owned context such as principal, run,
story, repository, stage, workspace, policy, and skill version never appears in model-controlled
arguments.

MCP tools are mapped to reviewed internal capability IDs and schema hashes. Dynamic MCP discovery
does not grant authority, and schema drift quarantines a capability until it is reviewed.

## Durable Run State Machine

The runtime, not the model or a skill, owns transitions:

```text
NEW -> INTAKE -> ANALYZE -> PLAN_READY -> IMPLEMENT -> VALIDATE
    -> AWAITING_PUBLISH_APPROVAL -> PUBLISH -> REPORT -> SUCCEEDED
```

`FAILED` and `CANCELLED` are terminal states. Approval is a separate blocking record bound to an
exact action rather than a general run-level permission.

## Internal Skills Only

Skills are immutable, signed, versioned orchestration recipes from an internal registry. A skill
can narrow available capabilities but cannot expand policy, define new execution mechanisms,
select MCP endpoints, install dependencies, read secrets, or own state transitions. URL installs,
JIRA attachment installs, and automatic repository-local skill loading are prohibited.

## Workspace Executor

The workspace executor is an adapter behind the gateway, not an unrestricted super-tool. Typed
operations such as patch application and trusted test profiles are preferred. Arbitrary shell
execution is disabled by default and must never be treated as sandboxed merely because it has a
working directory or subprocess wrapper.

---

# Implementation Roadmap

Status values: `[ ]` pending, `[>]` active, `[x]` complete.

## Phase 0 — Model tool-call contract `[x]`

Goal: support a deterministic multi-turn model/tool loop before implementing the gateway.

Deliverables:

- Typed tool definitions, tool calls, function arguments, and tool results.
- Assistant messages that preserve tool calls in conversation history.
- Tool-result messages correlated by required `tool_call_id`.
- OpenAI-compatible serialization and response parsing.
- Structured errors for malformed arguments, incomplete output, and unsupported tool use.
- A deterministic scripted provider for runtime tests.
- Remove or disable advertised streaming support until the provider contract implements it.

Acceptance criteria:

- A tool call can round-trip through a provider response and the next request unchanged.
- Invalid JSON arguments remain available for a recoverable gateway validation error.
- Tool results are correlated to the originating call.
- Routing cannot select a model without tool support for a tool-required stage.
- Existing OpenRouter and MiniMax reasoning behavior remains covered.

## Phase 1 — Read-only gateway walking skeleton `[x]`

Goal: complete one model-to-gateway loop without external systems or write capabilities.

Deliverables:

- Capability card, descriptor, invocation context, policy decision, execution record, and audit
  event contracts.
- In-memory capability catalog, deny-by-default policy engine, and audit sink.
- Search, describe, invoke, status, and cancel gateway operations.
- One harmless fixture read capability.
- Bounded agent runner with turn and invocation limits.

Acceptance criteria:

- A scripted model performs search, describe, invoke, receives a result, and gives a final answer.
- Denied capabilities are absent from discovery and denied again if guessed.
- Arguments are schema validated and unknown fields are rejected.
- Every success, denial, invalid request, failure, and cancellation is audited.
- Model arguments cannot set identity, stage, workspace, policy, or idempotency keys.
- The loop stops predictably at configured limits.

## Phase 2 — Durable run kernel `[x]`

Goal: make execution resumable and auditable before enabling any write.

Deliverables:

- SQLite run, checkpoint, invocation, approval, and audit persistence.
- Local content-addressed artifact storage for bounded large results.
- Runtime-owned state machine with validated transitions.
- Persisted story hash, repository revision, model, prompt, policy, usage, and limits.
- Runtime-generated invocation identities and replay-safe idempotency.

Acceptance criteria:

- Invalid state transitions fail.
- Restart resumes at the last committed checkpoint.
- Crash-injection tests do not lose completed results or mark incomplete work successful.
- Replaying an invocation returns the stored result without executing it again.
- Cancellation prevents new work.
- Model and policy selection are pinned per checkpoint.

## Phase 3 — JIRA intake and planning `[x]`

Goal: turn a real, read-only JIRA story into a persisted implementation plan.

Deliverables:

- JIRA issue and comment read capabilities behind the gateway.
- Normalized, immutable story snapshot with revision/hash tracking.
- `INTAKE -> ANALYZE -> PLAN_READY` workflow.
- Explicit treatment of story content and attachments as untrusted data.

Acceptance criteria:

- Mock HTTP tests cover authentication, pagination, timeout, normalization, and redaction.
- Credentials never enter model arguments, results, audit payloads, or exceptions.
- Prompt injection in a story cannot expose capabilities or force transitions.
- Resume uses the stored snapshot rather than silently fetching changed content.
- Story changes require an explicit refresh and replan decision.
- Truncated or empty model output cannot become a successful plan.

## Phase 4 — Read-only repository intelligence over MCP `[>]`

Goal: analyze a repository using approved internal MCP capabilities without exposing MCP directly.

Deliverables:

- One MCP transport and one allowlisted internal server integration.
- Reviewed mappings from MCP tools to stable capability IDs, versions, effects, and schema hashes.
- Repository snapshot pinned to the run's base revision.
- Bounded, redacted MCP outputs with artifact spillover.

Acceptance criteria:

- Models see only gateway tools, never MCP schemas, endpoints, or credentials.
- Unmapped tools cannot be discovered or invoked by guessed names.
- MCP annotations cannot lower trusted risk classification.
- Schema drift quarantines the capability.
- Timeout, cancellation, malformed output, and oversized output are structured failures.
- A fixture story reaches `PLAN_READY` with cited file and symbol evidence.

## Phase 5 — Confined workspace edits `[ ]`

Goal: permit reversible edits without executing repository code.

Deliverables:

- Disposable workspace per run, pinned to the recorded base revision.
- Typed patch, file-create, and diff-read capabilities.
- Serialized workspace mutations and durable idempotency.
- Path, file-count, byte, and allowed/denied pattern limits.

Acceptance criteria:

- Absolute paths, traversal, symlink escape, `.git` writes, and cross-run access are denied.
- Replay after restart does not apply a patch twice.
- Concurrent mutations are serialized.
- Base revision drift fails loudly.
- Resulting diff and changed-file list are checkpointed.
- Writes are only allowed during `IMPLEMENT`.
- No repository code is executed in this phase.

## Phase 6 — Sandboxed validation loop `[ ]`

Goal: execute trusted validation profiles and allow bounded correction attempts.

Deliverables:

- `SandboxBackend` boundary and one real isolation implementation.
- Trusted test and lint profiles resolved outside model and repository content.
- Network, environment, process, CPU, memory, output, and timeout controls.
- `PLAN_READY -> IMPLEMENT -> VALIDATE` loop with bounded retries and budgets.

Acceptance criteria:

- A fixture story produces the expected change and passing tests.
- Validation failure can trigger a bounded correction attempt.
- Looping models stop at turn, attempt, time, token, or cost limits.
- Network, host paths, other workspaces, and unapproved environment variables are inaccessible.
- Timeout and cancellation terminate descendant processes.
- Validation cannot pass to publication without required checks or an approved exception.

## Phase 7 — Approval and pull-request publication `[ ]`

Goal: safely perform the first irreversible external effect.

Deliverables:

- Approval records bound to run, story, repository, base, diff, action, target, expiry, approver,
  and policy version.
- Source-control branch, push, and pull-request adapter for one provider.
- Persisted publication intent and uncertain-outcome reconciliation.
- Minimal approval CLI or API.

Acceptance criteria:

- No push or pull-request call occurs before approval.
- A changed diff, repository, target, or expired approval requires new approval.
- Approval is single-use and cannot be supplied by the model.
- Restart resumes publication without recreating the action through the model.
- Timeout after remote creation reconciles to the existing pull request.
- Base drift stops publication or requires explicit rebase and revalidation.

## Phase 8 — JIRA reporting `[ ]`

Goal: report a successful run without duplicate or unsafe workflow mutations.

Deliverables:

- Idempotent JIRA comment capability.
- Optional allowlisted and approval-gated status transitions.
- `REPORT -> SUCCEEDED` terminal workflow.

Acceptance criteria:

- The report contains the pull-request URL, validation summary, and run ID.
- Restart or timeout does not duplicate comments.
- Only configured transitions are available.
- Reporting failure leaves the run resumable rather than falsely successful.
- Failure reporting cannot transition a story to Done.

## Phase 9 — Signed internal skills `[ ]`

Goal: add reusable orchestration only after the code-owned workflow is proven.

Deliverables:

- Immutable skill manifest, version, content hash, signature, signer, owner, expiry, and revocation.
- Skill allowlist of capability IDs and maximum effect level.
- Exact skill version/hash pinned to each run.
- Internal registry with no external installation path.

Acceptance criteria:

- Modified, expired, revoked, or unknown-signer skills are rejected.
- Skill capability access is the intersection of task, stage, skill, and catalog policy.
- A skill can never broaden authority or alter its own manifest.
- Skills cannot execute code, install packages, select endpoints, or access raw secrets.
- Resume uses the exact pinned skill content.

## Phase 10 — Evaluation and operational hardening `[ ]`

Goal: measure reliability and prepare for production based on observed needs.

Deliverables:

- Golden JIRA/repository task corpus and model regression suite.
- Crash/restart matrix for every state.
- Structured metrics for model calls, capability latency, policy decisions, retries, cost, and
  outcomes.
- OpenTelemetry export and retention/redaction policies.
- Multi-worker claiming only when workload demonstrates the need.

Deferred until justified by measurements:

- Semantic capability search.
- Semantic and long-term memory.
- Repository knowledge graphs.
- Distributed queues and workflow engines.
- Multiple observability vendors.
- General application plugin framework.

---

# Mandatory Security Gates

Before any workspace write:

- Runtime-created invocation context and identifiers.
- Deny-by-default catalog and policy at discovery and invocation.
- Trusted internal effect classification.
- Durable audit and invocation records written before execution.
- Input validation, output limits, and redaction.
- Replay-safe idempotency and mutation serialization.
- Workspace isolation and path confinement.

Before any repository code execution:

- A real sandbox boundary; subprocess working directories are not isolation.
- Network, environment, and secrets denied by default.
- Enforced process, time, CPU, memory, and output quotas.

Before any external write:

- Approval bound to the exact action digest.
- Persisted intent and reconciliation for uncertain outcomes.
- Adapter-owned credentials and egress restrictions.
- Immediate stale-story, stale-base, diff, and validation checks.

---

# First Release Success Criteria

- Provider and model can be changed without changing the gateway or workflow contracts.
- The model has no direct path to MCP, credentials, skills, host execution, or external APIs.
- A JIRA story can produce a tested diff and exactly one approved pull request.
- Every state can resume safely after process restart.
- Every capability decision and invocation is auditable.
- Story, repository, MCP, and tool output are consistently treated as untrusted content.
- External skills cannot be installed or loaded.
- Applications remain independent of model providers and infrastructure adapters.
