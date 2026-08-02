# Implementation & Plan Review

Date: 2026-08-03
Scope: `PLAN.md`, `STATUS.md`, `src/agent_runtime/**`, `tests/**` at commit `ed54fd0`
Method: three independent review passes (architecture fit, claims-vs-code, independent skeptical
audit) plus direct source verification of every HIGH finding.

Target deployment (confirmed with owner): **large open-weight models hosted locally** —
MiniMax-M2.7, Gemma-4-31B, and soon GLM-5.2, Kimi-K3. Not small or heavily quantized 7B-class
models. This materially shapes the findings below.

---

## 1. Verdict

The security and durability primitives are real, carefully built, and adversarially tested. The
local-model coding agent is not built yet.

Phases 0–10 are marked complete in `PLAN.md`. What exists is a governance chassis: a capability
gateway, a durable run kernel, a confined workspace, a real sandbox, signed skills, an approval
gate, and publication/reporting adapters. What does not exist is (a) any machinery specific to
serving open-weight models locally, and (b) an orchestrator that connects the phases into the
nine-step workflow the plan calls the "Active Product Goal".

Two HIGH-severity defects make declared functionality unreachable or unsafe. Both were verified
directly in source, not inferred.

---

## 2. What is genuinely real

Independent passes agreed on this list. These modules contain working machinery, not scaffolding.

| Module | Substance |
|---|---|
| `durable.py` | SQLite run state machine, checkpoints, invocation replay that returns stored results without re-execution, crash recovery, cancellation |
| `workspace.py` | Path confinement done correctly: validate-before-join, per-component `lstat`, `O_NOFOLLOW\|O_EXCL` on create, `flock` + in-process `RLock` serialization, git-seeded disposable generations |
| `validation.py` | A real Bubblewrap sandbox — PID/net/IPC/UTS/cgroup namespaces, CPU/memory/time/output quotas, `killpg` on timeout, and **no silent fallback to unsandboxed execution** |
| `skills.py` | Ed25519 verification over a domain-separated, length-prefixed payload; signer allowlist, expiry, revocation all enforced |
| `publication.py` | Single-use, digest-bound, expiring approval; base-revision drift check; durable intent and reconciliation against uncertain remote outcomes |

Path confinement and approval binding in particular are covered by adversarial tests, not just
happy-path ones. This is above the norm for agent codebases.

---

## 3. HIGH-severity defects

### 3.1 `gateway.py:265-270` — Phase 6 is unreachable through the gateway

```python
# gateway.py:118-119
"workspace.test.run": Effect.TRUSTED_PROCESS_EXECUTION,
"workspace.lint.run": Effect.TRUSTED_PROCESS_EXECUTION,

# gateway.py:265-270
if capability_id.startswith(("workspace.", "git.diff.")) and effect not in {
    "trusted_workspace_read",
    "trusted_workspace_write",
}:
    self._audit("denial", operation, capability_id=capability_id, detail="untrusted effect")
    raise GatewayError("denied", "untrusted effect classification")
```

Both validation capabilities are named `workspace.*` and carry `trusted_process_execution`. The
prefix guard therefore denies them unconditionally, at `describe` and at `invoke`. The VALIDATE
stage gate immediately below (`gateway.py:274-276`) is dead code — no capability can ever reach it.

`Effect` is a `StrEnum` (`gateway.py:31`), so the set-membership comparison itself is correct. The
bug is the prefix-based classification, which conflates "lives in the workspace" with "reads or
writes the workspace".

**Failure scenario:** a model in VALIDATE stage calls `capability_describe("workspace.test.run")`
and receives `denied: untrusted effect classification`. It can never run a test. The bounded
implement/correct loop that Phase 6 describes cannot be driven by a model at all.

**Why tests miss it:** every validation test calls `ValidationService` directly. There is no test
that exercises a validation capability *through* the gateway.

**Fix:** classify on `effect` alone, or add `trusted_process_execution` to the permitted set for
the `workspace.` prefix and let the stage gate at `:274-276` do its job.

### 3.2 `reporting.py:328-337` — JIRA transition can fire twice

```python
:328  self.adapter.transition(result.issue_key, row["transition_id"])   # external effect
:329  with self.store.connection:
:330      changed = self.store.connection.execute(
:331          "UPDATE jira_transition_approvals SET status='CONSUMED',consumed_at=? "
:332          "WHERE approval_id=? AND status='APPROVED'", ...
```

The irreversible external effect happens before the approval is consumed.

**Failure scenario:** the transition succeeds remotely, then the process crashes or the HTTP call
times out after the server committed. On resume (`reporting.py:341-342` calls `report` again) the
approval row is still `APPROVED`, all binding checks pass, and the story is transitioned a second
time. Violates `PLAN.md:522` ("Restart or timeout does not duplicate comments") and the Mandatory
Security Gate requiring "persisted intent and reconciliation for uncertain outcomes".

The comment path has an idempotency marker. The transition path has none.

**Fix:** consume the approval first (or write a durable intent record first), then transition, then
reconcile by reading current issue status on resume.

---

## 4. MEDIUM findings

| # | Location | Issue |
|---|---|---|
| 4.1 | `durable.py:355-383` | `transition()` is read-modify-write with an unguarded `UPDATE runs SET state=?`. Two concurrent transitions can both read the old state and both commit. No `WHERE state=?` predicate, no row lock. |
| 4.2 | `gateway.py:343-360` | Audit record is written *after* the handler runs. `PLAN.md:581` requires "durable audit and invocation records written before execution". A crash mid-execution leaves no audit trail of the attempt. |
| 4.3 | `publication.py:419-422, 449-452` | Raw `UPDATE` statements mutate run state directly, bypassing `transition()` and its checkpoint writes. State can advance without a checkpoint. |
| 4.4 | `gateway.py` (search path) | `search` filters on policy only; `describe`/`invoke` additionally filter on effect and stage. Effect- and stage-denied capabilities therefore appear in discovery results. `PLAN.md:380` claims denied capabilities are "absent from discovery". |
| 4.5 | `gateway.py` (absent) | Gateway-owned output limits and redaction (`PLAN.md:305`) are not implemented in the gateway. They are pushed into individual adapters. `arguments_json` / `result_json` in the durable store and the `detail` field on audit events are written unredacted. |
| 4.6 | `gateway.py` `RedactionPolicy` | Never receives real attributes — no caller passes `attributes=`. The redaction path is effectively inert. |

### Dead / aspirational code

- `gateway.cancel` is unreachable: `invoke` is synchronous, so no execution is ever in flight when
  a `execution_cancel` tool call could arrive.
- The sandbox `cancel=` parameter is never wired to anything.
- `ValidationService.required_passed` is defined and never used.
- `ModelCapabilities.supports_streaming` (`models.py:136`) and `RoutingRequirements.streaming`
  (`routing.py:22`) exist while the provider has no streaming method — routing can select on a
  capability that cannot be used. `PLAN.md:354` explicitly required removing advertised streaming
  until the provider contract implements it.

---

## 5. Local-model readiness

This is the largest gap relative to the stated purpose, and it is entirely absent from phases 0–10.
"Local-first" is design principle #3 in `PLAN.md:18`; the local execution profile is Phase 13,
unstarted.

The target set — MiniMax-M2.7, GLM-5.2, Kimi-K3 — is **interleaved-thinking** models. Gemma-4-31B
is the outlier: dense, and with no native tool-call format.

### 5.1 Reasoning is parsed and then discarded — highest-impact defect

```python
# openai_compatible.py:94-95   (captured)
reasoning=message.get("reasoning") or message.get("reasoning_content"),
reasoning_details=tuple(message.get("reasoning_details") or ()),

# models.py:127-129            (dropped)
def to_assistant_message(self) -> ChatMessage:
    content = self.content or None if self.tool_calls else self.content
    return ChatMessage(role="assistant", content=content, tool_calls=self.tool_calls)
```

`ChatMessage` (`models.py:67-93`) has no field to carry reasoning, and `to_dict` (`:85-93`) cannot
emit one. So on every agentic turn the model's thinking is captured into `ChatResponse` and then
silently dropped before the next request.

MiniMax-M2 documents that prior-turn thinking must be fed back; quality degrades measurably
otherwise. The same applies to GLM and Kimi thinking variants. `reasoning_details` — the mechanism
for round-tripping opaque or encrypted reasoning blocks — is discarded identically.

For a multi-turn tool-using coding agent on these models, this is the single biggest quality lever
in the repository. It is a type change (add fields to `ChatMessage`, emit in `to_dict`, preserve in
`to_assistant_message`), not a one-line fix.

### 5.2 No way to configure thinking, sampling, or guided decoding

`ChatRequest.metadata` (`models.py:102`) is never read by `chat()` (`openai_compatible.py:40-49`).
The payload is built from `model`, `messages`, `temperature`, `max_tokens`, `tools` and nothing
else. Consequences:

- No `chat_template_kwargs` → cannot toggle thinking mode on GLM / MiniMax.
- No `extra_body` → no `guided_json`, no vLLM/SGLang-specific options.
- No `seed`, `top_p`, `stop`, `presence_penalty`.

And `runner.py:62` constructs `ChatRequest(tuple(messages), tools=GATEWAY_TOOLS)` — no temperature,
no `max_tokens`, no seed. Every model call runs on server-default sampling.

### 5.3 Timeout and streaming

`openai_compatible.py:20` hardcodes `timeout=60.0` and there is no streaming path. A single
M2.7 or K3 thinking turn can emit thousands of reasoning tokens; on a local node, 60 seconds is
routinely exceeded. Long turns die on timeout rather than completing.

### 5.4 Unparsed `<think>` becomes the final answer

`openai_compatible.py:82` takes `message.get("content") or ""` verbatim. If the serving stack lacks
the correct `--reasoning-parser`, thinking tags land in `content`, and `runner.py:88-90` returns
that content as the agent's final answer when no tool call is present. Nothing detects this. The
run reports success with a thinking monologue as its output.

### 5.5 No per-model profile matrix

Each target model needs a specific vLLM/SGLang `--tool-call-parser`, `--reasoning-parser`, chat
template, thinking toggle, timeout, and sampling defaults. Gemma-4-31B needs pythonic or prompted
function calling rather than a native tool format. None of this is encoded anywhere. There is no
registry, no config surface, and no test per model.

### 5.6 Malformed tool arguments kill the run

`openai_compatible.py:118-119` raises `InvalidModelOutputError` when `arguments` is not a string.
Several serving stacks emit a dict. There is no repair loop and no bounded retry — the exception
propagates out of `AgentRunner.run`.

### 5.7 Limit breaches are not resumable

`runner.py:94` and `runner.py:100` raise bare `RuntimeError` on invocation and turn limits. Nothing
is checkpointed, so a long-running thinking-model task that exhausts its budget is simply lost
rather than resumable. The limits themselves (8 turns, 4 invocations) are low for agentic coding
work, though that is a tuning matter rather than a defect.

### 5.8 Routing is dead weight for an all-local fleet

`routing.py:44-56` sorts candidates by average `$/M-token`. Local deployments have no per-token
cost, so `input_cost_per_million` is `None`, `average_cost` becomes `float("inf")`, every candidate
ties, and selection degenerates to `priority`. The cost key is inert.

The axes that matter for local serving — queue depth, GPU-seconds, warm vs. cold weights, KV-cache
residency — are not modeled. Capabilities (`context_window`, `supports_tools`) are self-declared in
`ModelDeployment` and never probed against the server.

### 5.9 Context discipline, correctly framed

`runner.py:55-100` resends the full message history every turn; MCP outputs up to 32KB and diffs up
to 1MB are inlined. With 128k–1M context windows this is not an overflow risk — it is a **prefill
latency and GPU-seconds** problem, and it is the dominant cost on a local node. Prefix-cache-stable
prompt ordering (so the KV cache is reused across turns) matters more than aggressive trimming,
and nothing in the prompt construction is written with cache reuse in mind.

### 5.10 Determinism

MoE models under continuous batching are not deterministic even at `temperature=0` — expert
routing and kernel selection are batch-dependent. Any golden-task harness must use tolerant
scoring rather than exact match. This is currently moot because the harness does not execute
anything (see §6).

---

## 6. No end-to-end orchestrator

`planning.py`, `validation.py`, `publication.py`, and `reporting.py` are separate, unwired APIs.
There is no code path that takes a JIRA story to a pull request. The nine-step "Active Product
Goal" at `PLAN.md:275-285` has no single entry point implementing it; the phases were built as
parallel silos and assembled only inside individual tests.

---

## 7. Test honesty

The claimed baseline `95 passed, 1 skipped` (`STATUS.md:74`) is **Linux-only**.

Measured on macOS: **8 failed, 87 passed, 1 skipped**. All 8 failures are
`tests/test_validation.py` raising `SandboxUnavailable` (`validation.py:98`) because Bubblewrap is
Linux-only. The guard at `test_validation.py:102` is `skipif os.name != "posix"` — but macOS *is*
posix, so the tests run and fail rather than skipping. Wrong predicate.

Related: the Phase 6 coordinator tests (retry, budget, resume logic) are coupled to the real
Bubblewrap backend through `coordinator_fixture` (`test_validation.py:151`) despite testing
control-flow that needs no real isolation. They are untestable off Linux for no good reason.

### Weakest tests

- `test_checked_in_golden_contract_and_regression_details` (`tests/test_phase10.py:43-59`) — mutates
  a dict and asserts hardcoded booleans. Executes nothing. This is the "deterministic regression
  harness" of Phase 10.
- `test_scripted_search_describe_invoke_final_loop_and_five_safe_tools` and
  `test_runner_and_gateway_emit_model_policy_and_capability_metrics` — scripted provider plus the
  fixture capability only. They verify the harness, not the system.
- `test_network_namespace_has_no_interfaces_except_loopback` — proves one connection fails, which
  is weaker than proving isolation.

### Adapters are never exercised against anything real

JIRA, GitHub, MCP, and the OpenAI-compatible provider are only ever driven through
`httpx.MockTransport`. The live JIRA test skips without env vars
(`tests/test_jira_planning.py:213-224`). `STATUS.md:25-26` claims live OpenRouter smoke tests
against `google/gemma-4-31b-it` and `minimax/minimax-m2.7`, but nothing in the repository
reproduces them.

`planning.py` accepts any non-empty text as a valid plan — no schema validation — while
`PLAN.md:425` claims "truncated or empty model output cannot become a successful plan". It can.

### Acceptance criteria with no test

- Validation capability invoked *through* the gateway (would have caught §3.1).
- Cross-run workspace access denial.
- Credentials absent from audit payloads and exception text.
- Concurrent run-state transitions.
- JIRA transition duplication after timeout (would have caught §3.2).

---

## 8. Scope

`PLAN.md:1-53` opens with an "Enterprise Agent Runtime Platform" vision spanning banking
assistants, contact centers, and a widget platform. The repository is a local coding agent.

The provider abstraction that vision motivated is worth having regardless. The rest — the
generalized application-plugin framing, the multi-vendor observability list, the breadth of the
capability taxonomy — justified governance work that competed directly with making a locally
hosted model complete a task. Recommend cutting the multi-application vision from `PLAN.md` and
restating the goal as the coding agent it is.

---

## 9. Recommended order

**Tier 1 — correctness, contained**

1. `gateway.py:265-270` effect classification (§3.1).
2. `reporting.py:328-337` consume-before-transition + reconciliation (§3.2).
3. `test_validation.py:102` skip predicate; decouple coordinator tests from the bwrap backend (§7).
4. Guard `durable.py` `transition()` with a state predicate; move gateway audit before execution
   (§4.1, §4.2).

**Tier 2 — model interface, highest quality-per-effort for the stated purpose**

5. Carry `reasoning` and `reasoning_details` through `ChatMessage` → `to_dict` →
   `to_assistant_message` (§5.1).
6. Wire `ChatRequest.metadata` into `extra_body` / `chat_template_kwargs`; add `seed`, `top_p`,
   `stop`; set `temperature` and `max_tokens` from a profile in `runner.py` (§5.2).
7. Streaming plus per-model timeout; either implement streaming or remove the advertised
   capability (§5.3, dead-code list).
8. Per-model profile registry: template, tool parser, reasoning parser, thinking toggle, timeout,
   sampling defaults. Gemma-4 gets its own branch (§5.5).
9. `<think>`-leak detection and a bounded malformed-tool-argument repair loop instead of raising
   (§5.4, §5.6).
10. Checkpoint limit breaches so long runs resume (§5.7).

**Tier 3 — make it actually run**

11. Build the intake → analyze → plan → implement → validate → publish → report orchestrator (§6).
12. Golden tasks against real MiniMax-M2.7 and Gemma-4-31B endpoints, with tolerant scoring
    (§5.10).
13. Gateway-owned output limits and redaction; feed `RedactionPolicy` real attributes (§4.5, §4.6).
14. Prefix-cache-stable prompt construction and a context budget measured in prefill cost (§5.9).
15. Only then: context composer (Phase 12) and continuity ledger (Phase 11).

Freeze new governance work — signed-skill tooling, additional approval surfaces, further
observability vendors — until one target model completes a golden task end to end.

---

## 10. Open questions

- Single shared vLLM/SGLang node serving all models, or per-model deployments? Determines whether
  prefix-cache residency and queue-depth routing need to be first-class.
- Is hybrid frontier/local still a goal, or is the target local-only? `PLAN.md` says
  provider-agnostic; `STATUS.md` Phase 13 still lists hybrid profiles as pending.
- Which serving stack version is authoritative for the parser matrix? Tool-call and reasoning
  parsers change between vLLM releases and the profile registry needs a pinned target.
