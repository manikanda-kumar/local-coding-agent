# Continuity Ledger

## Goal (incl. success criteria)

- Remediate `review.md` in priority order and turn the governance chassis into an end-to-end coding
  agent for locally hosted MiniMax-M2.7, Gemma-4-31B, GLM-5.2, and Kimi-K3.
- Success: Tier 1 correctness/security defects closed; interleaved reasoning and local-model
  profiles supported; intake-to-report orchestrator proven by realistic golden tasks.

## Constraints/Assumptions

- Preserve the deny-by-default gateway, exact approval bindings, sandbox boundary, and durable
  replay/reconciliation guarantees while adding runnable product behavior.
- No live JIRA/GitHub mutation without deployment credentials and explicit action approval.
- Target large open-weight local models, not small 7B-class models.

## Key decisions

- 3 parallel review lenses: (1) plan/architecture fit for local models — fable, (2) implementation
  vs claimed guarantees — opus, (3) independent code+test audit via codex gpt-5.5.

## State

### Done

- Read PLAN.md (15 phases, 0–10 marked complete; 11–15 pending), STATUS.md, file inventory.
- Repo shape: ~3.6k LoC src in `src/agent_runtime/`, ~2.3k LoC tests, 15 test files.
  Claimed baseline `95 passed, 1 skipped`.
- Lens 1 (fable, plan/architecture fit for local models) COMPLETE. Verdict: governance platform,
  not local-model harness. Key findings: zero local-model machinery (no constrained decoding,
  no chat-template/parser matrix, no `<think>` hygiene, no context budget, no seed/determinism,
  no prefix-cache awareness); strict tool-call parsing raises on malformed calls
  (`openai_compatible.py:104-121`); 5-tool gateway indirection harder for weak models than flat
  tools; $/M-token routing is cloud concept (`routing.py:44-56`); `timeout=60` + no streaming;
  turn starvation risk (8 turns/4 invocations); planning.py monolithic prompt vs small windows;
  only cloud (OpenRouter) smoke-tested, never a local endpoint; tests all scripted-provider.
  Recommended reorder: 13a model-interface hardening → goldens on real 30B → 12 → 11 → 15 → 14;
  freeze governance. Enterprise vision (banking/contact-center) = scope inflation.
- Lens 3 (codex gpt-5.5, independent skeptical audit via subagent wrapper) COMPLETE. Verdict:
  "security/state primitives are real; the autonomous local-model agent is not." Corroborates
  Lens 1 independently. Key findings:
  - Real/substantive: `durable.py` (SQLite state/replay/crash recovery), `workspace.py` (git
    seeding, locking, patch apply), `validation.py` (real Bubblewrap sandbox w/ PID/net/IPC/UTS/
    cgroup isolation), `skills.py` (real Ed25519 verify/revocation).
  - Thin/disconnected: `models.py` (mostly dataclasses), `routing.py` (static filter only, no
    endpoint probing/enforcement), `evaluation.py` (compares dicts/bools, doesn't run anything),
    `planning.py` (accepts any non-empty text as valid plan, no schema), `gateway.py:419-430`
    fixture capability admittedly a harmless stub. No app-level "JIRA story → PR" orchestrator
    exists — planning/validation/publication/reporting are separate unwired APIs.
  - All adapters (JIRA, GitHub, MCP, OpenAI-compatible) only ever tested via `httpx.MockTransport`;
    live JIRA test is skipped without env vars (`tests/test_jira_planning.py:213-224`). STATUS's
    live-OpenRouter-smoke-test claim not reproducible from repo.
  - Weakest tests named: `test_checked_in_golden_contract_and_regression_details`
    (tests/test_phase10.py:43-59, edits a dict + hardcoded bools, not real execution);
    `test_scripted_search_describe_invoke_final_loop_and_five_safe_tools` and
    `test_runner_and_gateway_emit_model_policy_and_capability_metrics` (fully scripted provider +
    fixture capability only); `test_network_namespace_has_no_interfaces_except_loopback`
    (proves one connection fails, not real isolation). Flimsiest claimed guarantee overall:
    local-model readiness — zero real vLLM/SGLang/Qwen/gpt-oss test coverage.
  - Codex could not itself rerun `95 passed, 1 skipped` (pytest failed on writable tmp dir in its
    sandbox) — claim remains documentary only, unverified independently.
  - Concrete Qwen3-32B/gpt-oss-20b breakage points traced through runner.py/openai_compatible.py:
    tool_calls.arguments must be a string or it fails (openai_compatible.py:104-120); no
    reasoning_content replay (models.py:127-129, no reasoning field in to_dict); `<think>` content
    with no tool call is returned as final answer verbatim (runner.py:88-90); no tool_choice/
    grammar/constrained decoding at all; no token counting/context budget — full history resent
    every turn (runner.py:55-100), MCP outputs up to 32KB inline, diffs up to 1MB; no max_tokens
    set by AgentRunner; 8-turn/4-invocation hard limits are terminal failures, not resumable.
  - Top 10 ranked fixes (impact/effort) largely converge with Lens 1: (1) normalize local-model
    responses (think-tags/reasoning/tool-arg variants), (2) context budgeting, (3) explicit local
    model profiles (Qwen/gpt-oss/vLLM/SGLang), (4) authoritative routing enforcement, (5) build
    the missing intake→plan→implement→validate→publish→report orchestrator, (6) constrained
    decoding support, (7) malformed-tool-call repair policy, (8) wire capabilities into runtime
    (not manual assembly), (9) real local integration test harness, (10) failure-injection e2e
    tests.

- Lens 2 (opus, implementation vs claimed guarantees) COMPLETE. Verdict: primitives largely real
  and adversarially tested (workspace confinement, publication approval binding, Ed25519 skills),
  but PLAN overstates 3 areas. Claim-vs-code:
  - HOLDS: invalid-transition rejection, checkpoint resume, idempotent replay (no re-execute),
    durable cancellation, path confinement (validate-before-join, per-component lstat,
    `O_NOFOLLOW|O_EXCL`, flock+RLock), bwrap net/env/proc/CPU/mem/timeout controls + killpg,
    no silent unsandboxed fallback, approval single-use/digest-bound/expiring, base-drift check,
    Ed25519 payload correctness (domain-sep + length-prefix, signer allowlist/expiry/revocation).
  - HIGH defect `gateway.py:265-270`: prefix check denies every `trusted_process_execution`
    capability → `workspace.test.run`/`workspace.lint.run` can never be described or invoked.
    Phase 6 gateway path dead; stage gate at `:274-276` unreachable. Verified live.
  - HIGH defect `reporting.py:328-336`: JIRA transition executes BEFORE approval consumed →
    timeout+retry double-transitions the story. No transition reconciliation (comments have one).
  - MEDIUM: `durable.py:355-383` transition() read-modify-write TOCTOU, unguarded
    `UPDATE runs SET state=?`; `gateway.py:343-360` audit written AFTER handler, not before
    (violates PLAN:581); `publication.py:419-422,449-452` raw state UPDATEs bypass
    transition()/checkpoints.
  - NOT IMPLEMENTED: gateway-owned output limits + redaction (PLAN:305) — absent from gateway.py,
    pushed into adapters. Durable `arguments_json`/`result_json` + audit `detail` unredacted.
  - Dead/aspirational: `ValidationService.required_passed`, `gateway.cancel` (unreachable —
    invoke is synchronous), sandbox `cancel=` never wired, `RedactionPolicy` never fed real
    attributes (no caller sets `attributes=`).
  - Search/describe asymmetry: `search` filters on policy only, so effect/stage-denied
    capabilities leak into discovery.
  - Coverage gaps: no gateway-mediated validation test (would have caught the HIGH), no cross-run
    workspace test, no credentials-in-audit test, no concurrent-transition test, no transition
    duplicate-on-timeout test.

### Now

- Tier 1 committed as `a18d356`: gateway-mediated validation, conservative at-most-once JIRA
  transitions, state CAS, pre-handler audit, and cross-platform coordinator tests.
- Tier 2 committed as `5054b3b`: exact tagged or structured reasoning replay; bounded
  opaque reasoning; strict request-extension validation; profile timeout/sampling controls; object
  tool-argument normalization; `<think>` leak handling; false streaming support removed.
- Current-vLLM-qualified profiles added for MiniMax-M2.7, GLM-5.2, Kimi-K3 agentic, and
  Gemma-4-31B-it, including parser launch metadata and vendor-recommended sampling baselines.
- Tier 3 end-to-end workflow complete and independently reviewed SHIP: persisted-state dispatch,
  external publication/transition approval pauses, immutable story binding, passed-checkpoint
  recovery, and real-service reconstruction across PUBLISH and REPORT crash windows.
- Executable golden workflow scoring replaces the fake dict-only claim; opt-in real OpenRouter
  search/describe/invoke loops passed Gemma-4-31B, MiniMax-M2.7, GLM-5.2, and Kimi-K3.
- Gateway-owned output normalization now preserves shape until byte sizing, recursively sanitizes
  structured and textual credentials, sanitizes MCP/validation artifacts before spill, persists a
  normalization version, and refuses legacy raw rows on replay.
- Prefix-cache-stable context budgeting uses conservative request accounting, model-profile
  context/output bounds, immutable prompt prefixes, and newest complete assistant/tool groups;
  malformed correlations and unfit minimum context fail closed. Independent review: SHIP.
- Deterministic continuity ledger persists bounded runtime-owned goals/constraints/decisions and
  model-writable non-authoritative progress memory; durable-state CAS, immutable provenance,
  bounded event activity, and interrupted-update replay are tested. Independent review: SHIP.
- Deterministic resume packets pin story, repository, policy, model profile, skill, continuity, and
  workspace generation. The context composer bounds and deterministically ranks untrusted evidence
  and knowledge, escapes its sole envelope, and rejects stale state. Independent review: SHIP.
- Current verification: `170 passed, 2 skipped`; Ruff checks, format check, and diff check pass.

### Next

- Commit the deterministic resume context as a reviewable milestone.
- Wire resume-packet validation and composed context into actual workflow/model-call construction.
- Run the checked-in live contract against the locally hosted vLLM endpoints when deployment URLs
  are supplied; no credentials or endpoints are committed.

## Open questions

- RESOLVED 2026-08-03 (user): target = **large open-weight models hosted locally** — MiniMax-M2.7,
  Gemma-4-31B, soon GLM-5.2, Kimi-K3. NOT small/quantized 7B-class. Consequences:
  - Retract earlier critique: 5-tool gateway indirection is fine; 8k-32k context framing wrong;
    constrained decoding demoted to nice-to-have; no few-shot scaffolding needed.
  - Escalate: interleaved-thinking round-trip is now the #1 defect (all targets except Gemma-4 are
    thinking models); per-model chat-template/parser profiles mandatory; 60s timeout + no
    streaming untenable; context budget matters for prefill latency/GPU-seconds not overflow;
    MoE + continuous batching = non-deterministic at temp 0, goldens need tolerant scoring.
  - Gemma-4-31B is the outlier: dense, no native tool-call format (needs pythonic/prompted parser).
- UNCONFIRMED: real JIRA/GitHub/MCP endpoints never exercised (deployment inputs pending).
- UNCONFIRMED: single shared vLLM/SGLang node or per-model deployments? affects prefix-cache and
  concurrency design.

## Verified model-interface defects (read by me, 2026-08-03)

- `models.py:127-129` `to_assistant_message()` drops `reasoning` and `reasoning_details`;
  `ChatMessage` (`models.py:67-93`) has no field to carry them. Parsed at
  `openai_compatible.py:94-95` then discarded. Breaks M2.7/GLM/Kimi interleaved thinking.
- `ChatRequest.metadata` (`models.py:102`) never read by `chat()` (`openai_compatible.py:40-49`).
  Dead field → no `chat_template_kwargs` (thinking on/off), no `extra_body`, no `guided_json`,
  no `seed`/`top_p`/`stop`.
- `runner.py:62` sends no temperature/max_tokens/seed. Server-default sampling every call.
- `openai_compatible.py:20` fixed `timeout=60.0`, no streaming. Provider has no stream method yet
  `ModelCapabilities.supports_streaming` (`models.py:136`) + `RoutingRequirements.streaming`
  (`routing.py:22`) let routing select on an unusable capability — contradicts PLAN Phase 0.
- `openai_compatible.py:82` `content = message.get("content") or ""` → unparsed `<think>` blocks
  land in content and are returned as the final answer via `runner.py:88-90`.
- `openai_compatible.py:118-119` non-string tool arguments raise and kill the run; no repair.
- `runner.py:94,100` limit breaches raise bare `RuntimeError` — not checkpointed, not resumable.
- `routing.py:44-56` sorts by $/M-token; local deployments have no cost → all tie at `inf` → dead
  key, routing degenerates to `priority`. Capabilities self-declared, never probed against the
  server.

## Working set (files/ids/commands)

- `PLAN.md`, `STATUS.md`, `src/agent_runtime/*.py`, `tests/*.py`, `.agents/setup`, `.agents/resume`
- biggest modules: `durable.py` 681, `publication.py` 526, `workspace.py` 517, `gateway.py` 442,
  `validation.py` 390
- prior commands: `git remote add origin https://github.com/manikanda-kumar/local-coding-agent.git`

## Project learnings

- Python 3.11+, setuptools src-layout, deps: `cryptography`, `httpx`; dev: `pytest`, `ruff`.
- Sandbox isolation = Bubblewrap (`bwrap`), verified by `.agents/setup`.
- `.gitignore` excludes *.md except explicitly exempted ones.
- System python3 (3.14 homebrew) has no pytest — use `uv venv --python 3.13` + `uv pip install -e ".[dev]"`.
- Test baseline on macOS: **8 failed, 87 passed, 1 skipped**. All 8 are `tests/test_validation.py`
  raising `SandboxUnavailable` (`validation.py:98`) — bwrap is Linux-only and the tests do NOT
  skip. `skipif os.name != "posix"` (`test_validation.py:102`) is wrong; macOS is posix.
  The claimed `95 passed, 1 skipped` baseline is Linux-only.
- Coordinator tests (Phase 6 retry/budget/resume logic) are coupled to the real bwrap backend via
  `coordinator_fixture` (`test_validation.py:151`) — untestable off Linux despite needing no
  real isolation.
