"""Trusted validation profiles executed behind a Linux Bubblewrap boundary."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from agent_runtime.durable import ArtifactStore, RunState, SQLiteRunStore
from agent_runtime.gateway import (
    Capability,
    CapabilityCard,
    CapabilityDescriptor,
    Effect,
    InvocationContext,
)
from agent_runtime.metrics import MetricEvent, MetricsSink, emit_metric
from agent_runtime.workspace import WorkspaceManager


class SandboxUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ValidationProfile:
    profile_id: str
    kind: str
    argv: tuple[str, ...]
    required: bool = True
    environment: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if (
            self.kind not in {"test", "lint"}
            or not self.argv
            or any(not isinstance(x, str) or not x or "\0" in x for x in self.argv)
        ):
            raise ValueError("invalid runtime-owned validation profile")
        if any(
            key not in {"LANG", "LC_ALL", "TZ"} or "\0" in value for key, value in self.environment
        ):
            raise ValueError("profile environment is not allowlisted")


@dataclass(frozen=True, slots=True)
class SandboxLimits:
    wall_seconds: float = 30
    cpu_seconds: int = 20
    address_space_bytes: int = 512 * 1024 * 1024
    processes: int = 64
    file_size_bytes: int = 16 * 1024 * 1024
    output_bytes: int = 64 * 1024
    full_output_bytes: int = 1024 * 1024


@dataclass(frozen=True, slots=True)
class ValidationResult:
    profile_id: str
    passed: bool
    exit_code: int | None
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    timed_out: bool = False
    cancelled: bool = False
    duration_seconds: float = 0
    output_artifact: str | None = None


class SandboxBackend(Protocol):
    def run(
        self, profile: ValidationProfile, generation: Path, *, cancel: threading.Event | None = None
    ) -> ValidationResult: ...


class BubblewrapSandboxBackend:
    def __init__(
        self,
        scratch_root: str | Path,
        *,
        limits: SandboxLimits | None = None,
        artifacts: ArtifactStore | None = None,
    ) -> None:
        self.root = Path(scratch_root).absolute()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.limits, self.artifacts = limits or SandboxLimits(), artifacts
        if shutil.which("bwrap") is None:
            raise SandboxUnavailable("Bubblewrap is not installed")

    def _command(self, profile: ValidationProfile, copy: Path) -> list[str]:
        maximum_file = min(self.limits.file_size_bytes, self.limits.full_output_bytes)
        command = [
            "/usr/bin/prlimit",
            f"--cpu={self.limits.cpu_seconds}",
            f"--as={self.limits.address_space_bytes}",
            f"--nproc={self.limits.processes}",
            f"--fsize={maximum_file}",
            "--core=0",
            "--",
            "/usr/bin/bwrap",
            "--die-with-parent",
            "--new-session",
            "--unshare-user",
            "--unshare-pid",
            "--unshare-net",
            "--unshare-ipc",
            "--unshare-uts",
            "--unshare-cgroup",
            "--clearenv",
        ]
        # Debian merged-/usr systems need only /usr; real library directories are mounted when present.
        for item in ("/usr", "/lib", "/lib64", "/bin", "/sbin"):
            path = Path(item)
            if path.is_symlink():
                command += ["--symlink", os.readlink(path), item]
            elif path.exists():
                command += ["--ro-bind", item, item]
        command += [
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--bind",
            str(copy),
            "/workspace",
            "--chdir",
            "/workspace",
            "--setenv",
            "PATH",
            "/usr/bin:/bin",
            "--setenv",
            "HOME",
            "/tmp",
            "--setenv",
            "TMPDIR",
            "/tmp",
        ]
        for key, value in profile.environment:
            command += ["--setenv", key, value]
        return [*command, "--", *profile.argv]

    def run(
        self, profile: ValidationProfile, generation: Path, *, cancel: threading.Event | None = None
    ) -> ValidationResult:
        if not generation.is_dir() or generation.is_symlink():
            raise ValueError("generation must be a real directory")
        cancel = cancel or threading.Event()
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="validation-", dir=self.root) as temporary:
            copy = Path(temporary) / "workspace"
            shutil.copytree(generation, copy, symlinks=False)
            stdout_path, stderr_path = Path(temporary) / "stdout", Path(temporary) / "stderr"
            with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
                try:
                    process = subprocess.Popen(
                        self._command(profile, copy),
                        stdin=subprocess.DEVNULL,
                        stdout=out,
                        stderr=err,
                        env={},
                        start_new_session=True,
                    )
                except OSError as exc:
                    raise SandboxUnavailable("Bubblewrap could not start") from exc
                timed_out = cancelled = False
                while process.poll() is None:
                    if cancel.is_set():
                        cancelled = True
                    if time.monotonic() - started >= self.limits.wall_seconds:
                        timed_out = True
                    if cancelled or timed_out:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        break
                    time.sleep(0.02)
                process.wait()
            raw_out, raw_err = stdout_path.read_bytes(), stderr_path.read_bytes()
            full = raw_out + b"\n--- stderr ---\n" + raw_err
            artifact = (
                self.artifacts.put(full)
                if self.artifacts
                and (
                    len(raw_out) > self.limits.output_bytes
                    or len(raw_err) > self.limits.output_bytes
                )
                else None
            )
            limit = self.limits.output_bytes
            result = ValidationResult(
                profile.profile_id,
                process.returncode == 0 and not timed_out and not cancelled,
                process.returncode,
                raw_out[:limit].decode(errors="replace"),
                raw_err[:limit].decode(errors="replace"),
                len(raw_out) > limit,
                len(raw_err) > limit,
                timed_out,
                cancelled,
                time.monotonic() - started,
                artifact,
            )
            return result


class ValidationService:
    def __init__(
        self,
        manager: WorkspaceManager,
        store: SQLiteRunStore,
        backend: SandboxBackend,
        profiles: Sequence[ValidationProfile],
    ) -> None:
        self.manager, self.store, self.backend = manager, store, backend
        self.profiles = {profile.profile_id: profile for profile in profiles}

    def run(
        self,
        run_id: str,
        profile_id: str,
        attempt: int = 1,
        *,
        cancel: threading.Event | None = None,
    ) -> ValidationResult:
        if self.store.get_run(run_id).state != RunState.VALIDATE:
            raise RuntimeError("validation requires durable VALIDATE stage")
        profile = self.profiles.get(profile_id)
        if profile is None:
            raise KeyError("unknown trusted validation profile")
        workspace = self.manager.acquire(run_id)
        generation, path = self.manager._generation(workspace)
        existing = self.store.validation_result(run_id, attempt, profile_id)
        if existing is not None:
            return ValidationResult(**existing)
        self.store.begin_validation(run_id, attempt, profile_id, generation)
        result = self.backend.run(profile, path, cancel=cancel)
        self.store.finish_validation(run_id, attempt, profile_id, asdict(result))
        return result

    def required_passed(self, run_id: str, attempt: int) -> bool:
        required = {p.profile_id for p in self.profiles.values() if p.required}
        rows = self.store.connection.execute(
            "SELECT profile_id,status FROM validation_attempts WHERE run_id=? AND attempt=?",
            (run_id, attempt),
        ).fetchall()
        return (
            bool(required)
            and {row["profile_id"] for row in rows if row["status"] == "PASSED"} >= required
        )


def validation_capabilities(service: ValidationService) -> tuple[Capability, ...]:
    schema = {
        "type": "object",
        "properties": {"profile": {"type": "string"}},
        "required": ["profile"],
        "additionalProperties": False,
    }

    def capability(kind: str) -> Capability:
        identifier = f"workspace.{kind}.run"

        def invoke(args: Mapping[str, object], context: InvocationContext) -> dict[str, object]:
            profile = service.profiles.get(str(args["profile"]))
            if profile is None or profile.kind != kind:
                raise KeyError("unknown trusted profile for capability")
            return asdict(service.run(context.run_id, profile.profile_id))

        return Capability(
            CapabilityDescriptor(
                CapabilityCard(
                    identifier, f"Run trusted {kind}", f"Run a runtime-owned {kind} profile"
                ),
                schema,
                Effect.TRUSTED_PROCESS_EXECUTION,
            ),
            invoke,
        )

    return capability("test"), capability("lint")


@dataclass(frozen=True, slots=True)
class CoordinatorLimits:
    attempts: int = 3
    turns: int = 24
    elapsed_seconds: float = 900
    tokens: int = 100_000
    cost: float = 10.0


class ImplementationValidationCoordinator:
    """Code-owned bounded correction loop; callbacks cannot select validation commands."""

    def __init__(
        self,
        store: SQLiteRunStore,
        validation: ValidationService,
        limits: CoordinatorLimits | None = None,
        *,
        metrics: MetricsSink | None = None,
    ) -> None:
        self.store, self.validation, self.limits = store, validation, limits or CoordinatorLimits()
        self.metrics = metrics

    def run(self, run_id: str, implement: Callable[[int], tuple[int, int, float]]) -> bool:
        started, turns, tokens, cost = time.monotonic(), 0, 0, 0.0
        if self.store.get_run(run_id).state == RunState.PLAN_READY:
            self.store.transition(run_id, RunState.IMPLEMENT)
        elif self.store.get_run(run_id).state == RunState.VALIDATE:
            last_checkpoint = self.store.connection.execute(
                "SELECT attempt,passed FROM validation_checkpoints "
                "WHERE run_id=? ORDER BY attempt DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            running = self.store.connection.execute(
                "SELECT 1 FROM validation_attempts WHERE run_id=? AND status='RUNNING' LIMIT 1",
                (run_id,),
            ).fetchone()
            if last_checkpoint is not None and running is None:
                if last_checkpoint["passed"]:
                    self.store.transition(
                        run_id,
                        RunState.AWAITING_PUBLISH_APPROVAL,
                        {"validation_attempt": last_checkpoint["attempt"]},
                    )
                    emit_metric(
                        self.metrics,
                        MetricEvent("validation", "coordinator", "success", run_id=run_id),
                    )
                    return True
                self.store.transition(run_id, RunState.IMPLEMENT)
        for attempt in range(self.store.next_validation_attempt(run_id), self.limits.attempts + 1):
            if attempt > 1:
                emit_metric(
                    self.metrics,
                    MetricEvent("retry", "validation_attempt", "started", run_id=run_id),
                )
            if time.monotonic() - started > self.limits.elapsed_seconds:
                break
            state = self.store.get_run(run_id).state
            if state == RunState.IMPLEMENT:
                used_turns, used_tokens, used_cost = implement(attempt)
                turns, tokens, cost = turns + used_turns, tokens + used_tokens, cost + used_cost
                if (
                    turns > self.limits.turns
                    or tokens > self.limits.tokens
                    or cost > self.limits.cost
                ):
                    break
                self.store.transition(run_id, RunState.VALIDATE)
            elif state != RunState.VALIDATE:
                raise RuntimeError("run is not resumable by validation coordinator")
            results = [
                asdict(self.validation.run(run_id, p.profile_id, attempt))
                for p in self.validation.profiles.values()
                if p.required
            ]
            self.store.save_validation_checkpoint(run_id, attempt, results)
            if results and all(result["passed"] for result in results):
                self.store.transition(
                    run_id, RunState.AWAITING_PUBLISH_APPROVAL, {"validation_attempt": attempt}
                )
                emit_metric(
                    self.metrics,
                    MetricEvent(
                        "validation",
                        "coordinator",
                        "success",
                        run_id=run_id,
                        input_tokens=tokens,
                        cost_usd=cost,
                    ),
                )
                return True
            if attempt < self.limits.attempts:
                self.store.transition(run_id, RunState.IMPLEMENT, {"validation_attempt": attempt})
        emit_metric(
            self.metrics,
            MetricEvent(
                "validation",
                "coordinator",
                "failure",
                run_id=run_id,
                input_tokens=tokens,
                cost_usd=cost,
            ),
        )
        return False
