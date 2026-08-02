"""Immutable, non-executing workspace generations."""

from __future__ import annotations

import difflib
import fcntl
import fnmatch
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from agent_runtime.durable import RunState, SQLiteRunStore, invocation_identity
from agent_runtime.gateway import (
    Capability,
    CapabilityCard,
    CapabilityDescriptor,
    Effect,
    InvocationContext,
)


class WorkspaceError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class WorkspaceLimits:
    maximum_files: int = 100
    maximum_bytes: int = 2_000_000
    maximum_patch_bytes: int = 1_000_000
    maximum_diff_bytes: int = 1_000_000
    allowed_paths: tuple[str, ...] = ()
    denied_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Workspace:
    workspace_id: str
    run_id: str
    repository: str
    base_revision: str
    path: Path


_guard = threading.Lock()
_locks: dict[str, threading.RLock] = {}


def _thread_lock(key: str) -> threading.RLock:
    with _guard:
        return _locks.setdefault(key, threading.RLock())


class WorkspaceManager:
    def __init__(
        self,
        trusted_root: str | Path,
        store: SQLiteRunStore,
        repositories: Mapping[str, str | Path],
        limits: WorkspaceLimits | None = None,
    ) -> None:
        self.root = Path(trusted_root).absolute()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.root.is_symlink() or not stat.S_ISDIR(self.root.lstat().st_mode):
            raise WorkspaceError("trusted root must be a real directory")
        self._root_fd = os.open(
            self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        )
        self.store = store
        self.limits = limits or WorkspaceLimits()
        self.repositories = {key: Path(value).absolute() for key, value in repositories.items()}

    def close(self) -> None:
        if self._root_fd >= 0:
            os.close(self._root_fd)
            self._root_fd = -1

    def __del__(self) -> None:
        try:
            self.close()
        except (AttributeError, OSError):
            pass

    @contextmanager
    def _locked(self, workspace_id: str):
        with _thread_lock(workspace_id):
            lock_path = self.root / f".{workspace_id}.lock"
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)

    def acquire(self, run_id: str) -> Workspace:
        run = self.store.get_run(run_id)
        workspace_id = hashlib.sha256(
            f"workspace:{run_id}:{run.repository}:{run.base_revision}".encode()
        ).hexdigest()
        with self._locked(workspace_id):
            row = self.store.connection.execute(
                "SELECT * FROM workspaces WHERE run_id=?", (run_id,)
            ).fetchone()
            if row is None:
                source = self.repositories.get(run.repository)
                if source is None:
                    raise WorkspaceError("repository is not trusted runtime configuration")
                canonical = self._canonical(source, run.base_revision)
                if canonical != run.base_revision:
                    raise WorkspaceError("base revision is not canonical")
                target = self.root / workspace_id
                if target.exists():
                    shutil.rmtree(target)
                target.mkdir(mode=0o700)
                self._seed(source, canonical, target / "gen-0")
                now = datetime.now(UTC).isoformat()
                try:
                    with self.store.connection:
                        self.store.connection.execute(
                            "INSERT INTO workspaces(workspace_id,run_id,repository,base_revision,path,created_at,current_generation) VALUES (?,?,?,?,?,?,0)",
                            (workspace_id, run_id, run.repository, canonical, str(target), now),
                        )
                except Exception:
                    shutil.rmtree(target, ignore_errors=True)
                    raise
                row = self.store.connection.execute(
                    "SELECT * FROM workspaces WHERE run_id=?", (run_id,)
                ).fetchone()
            workspace = self._from_row(row)
            self._verify(workspace)
            self._cleanup(workspace)
            return workspace

    @staticmethod
    def _git(source: Path, *args: str) -> bytes:
        env = {
            "PATH": "/usr/bin:/bin",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_COUNT": "3",
            "GIT_CONFIG_KEY_0": "core.hooksPath",
            "GIT_CONFIG_VALUE_0": "/dev/null",
            "GIT_CONFIG_KEY_1": "core.fsmonitor",
            "GIT_CONFIG_VALUE_1": "false",
            "GIT_CONFIG_KEY_2": "core.attributesFile",
            "GIT_CONFIG_VALUE_2": "/dev/null",
        }
        try:
            return subprocess.run(
                ["git", *args], cwd=source, env=env, capture_output=True, check=True, timeout=30
            ).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            raise WorkspaceError("trusted Git object read failed") from exc

    def _canonical(self, source: Path, revision: str) -> str:
        return (
            self._git(source, "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}")
            .decode()
            .strip()
        )

    def _seed(self, source: Path, revision: str, target: Path) -> None:
        target.mkdir(mode=0o700)
        count = total = 0
        try:
            listing = self._git(source, "ls-tree", "-r", "-z", "--full-tree", revision)
            for entry in listing.split(b"\0"):
                if not entry:
                    continue
                metadata, raw_path = entry.split(b"\t", 1)
                mode, kind, object_id = metadata.decode("ascii").split()
                if mode not in {"100644", "100755"} or kind != "blob":
                    raise WorkspaceError("repository contains unsupported non-regular entry")
                path = self._valid_path(raw_path.decode("utf-8"))
                content = self._git(source, "cat-file", "blob", object_id)
                count += 1
                total += len(content)
                self._check_limits(count, total)
                destination = target / path
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                with open(destination, "xb") as output:
                    output.write(content)
        except Exception:
            shutil.rmtree(target, ignore_errors=True)
            raise

    def _from_row(self, row: Any) -> Workspace:
        path = Path(row["path"])
        if path.parent != self.root or path.name != row["workspace_id"]:
            raise WorkspaceError("persisted workspace escaped trusted root")
        return Workspace(
            row["workspace_id"], row["run_id"], row["repository"], row["base_revision"], path
        )

    def _generation(self, workspace: Workspace) -> tuple[int, Path]:
        row = self.store.connection.execute(
            "SELECT current_generation FROM workspaces WHERE workspace_id=?",
            (workspace.workspace_id,),
        ).fetchone()
        if row is None:
            raise WorkspaceError("workspace metadata missing")
        return row[0], workspace.path / f"gen-{row[0]}"

    def _verify(self, workspace: Workspace) -> None:
        run = self.store.get_run(workspace.run_id)
        if (run.repository, run.base_revision) != (workspace.repository, workspace.base_revision):
            raise WorkspaceError("workspace configuration drift")
        number, path = self._generation(workspace)
        if number < 0 or not path.is_dir() or path.is_symlink():
            raise WorkspaceError("current generation missing or unsafe")
        self._scan(path)

    def _cleanup(self, workspace: Workspace) -> None:
        current, _ = self._generation(workspace)
        for path in workspace.path.glob("candidate-*"):
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
        for path in workspace.path.glob("gen-*"):
            if (
                path.name not in {"gen-0", f"gen-{current}"}
                and path.is_dir()
                and not path.is_symlink()
            ):
                shutil.rmtree(path)

    def _valid_path(self, raw: str) -> str:
        if not isinstance(raw, str) or not raw or "\\" in raw or "\0" in raw:
            raise WorkspaceError("invalid workspace-relative path")
        pure = PurePosixPath(raw)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
            raise WorkspaceError("invalid workspace-relative path")
        normalized = pure.as_posix()
        if raw.rstrip("/") != normalized or any(part.casefold() == ".git" for part in pure.parts):
            raise WorkspaceError("path is not normalized or accesses Git metadata")
        if self.limits.allowed_paths and not any(
            fnmatch.fnmatchcase(normalized, p) for p in self.limits.allowed_paths
        ):
            raise WorkspaceError("path is not allowed")
        if any(fnmatch.fnmatchcase(normalized, p) for p in self.limits.denied_paths):
            raise WorkspaceError("path is denied")
        return normalized

    def _safe_target(self, root: Path, raw: str, *, creating: bool = False) -> Path:
        path = self._valid_path(raw)
        current = root
        for part in PurePosixPath(path).parts[:-1]:
            current /= part
            if not current.exists():
                if creating:
                    current.mkdir(mode=0o700)
                else:
                    raise WorkspaceError("path does not exist")
            if current.is_symlink() or not stat.S_ISDIR(current.lstat().st_mode):
                raise WorkspaceError("unsafe parent traversal")
        target = root / path
        if target.exists() and (target.is_symlink() or not stat.S_ISREG(target.lstat().st_mode)):
            raise WorkspaceError("target is not a regular file")
        return target

    def _scan(self, root: Path) -> tuple[int, int]:
        if root.is_symlink() or not stat.S_ISDIR(root.lstat().st_mode):
            raise WorkspaceError("generation root is unsafe")
        count = total = 0
        for directory, dirs, files in os.walk(root, followlinks=False):
            base = Path(directory)
            for name in dirs:
                item = base / name
                if item.is_symlink() or not stat.S_ISDIR(item.lstat().st_mode):
                    raise WorkspaceError("generation contains unsafe directory")
            for name in files:
                item = base / name
                if item.is_symlink() or not stat.S_ISREG(item.lstat().st_mode):
                    raise WorkspaceError("generation contains non-regular file")
                self._valid_path(item.relative_to(root).as_posix())
                count += 1
                total += item.stat().st_size
                self._check_limits(count, total)
        return count, total

    def _check_limits(self, count: int, total: int) -> None:
        if count > self.limits.maximum_files or total > self.limits.maximum_bytes:
            raise WorkspaceError("workspace file or byte limit exceeded")

    def read_file(
        self,
        run_id: str,
        path: str,
        *,
        start_line: int = 1,
        max_lines: int = 200,
        maximum_page_bytes: int = 32_000,
    ) -> dict[str, Any]:
        if (
            not isinstance(start_line, int)
            or isinstance(start_line, bool)
            or start_line < 1
            or not isinstance(max_lines, int)
            or isinstance(max_lines, bool)
            or not 1 <= max_lines <= 400
            or not 1 <= maximum_page_bytes <= 32_000
        ):
            raise WorkspaceError("file read range is invalid")
        workspace = self.acquire(run_id)
        with self._locked(workspace.workspace_id):
            self._verify(workspace)
            generation, root = self._generation(workspace)
            del root
            normalized = self._valid_path(path)
            parts = PurePosixPath(normalized).parts
            directory_fd = os.dup(self._root_fd)
            try:
                for part in (workspace.workspace_id, f"gen-{generation}", *parts[:-1]):
                    next_fd = os.open(
                        part,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                        dir_fd=directory_fd,
                    )
                    os.close(directory_fd)
                    directory_fd = next_fd
                fd = os.open(
                    parts[-1],
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=directory_fd,
                )
            except OSError as error:
                raise WorkspaceError("workspace file cannot be opened safely") from error
            finally:
                os.close(directory_fd)
            try:
                metadata = os.fstat(fd)
                if not stat.S_ISREG(metadata.st_mode):
                    raise WorkspaceError("target is not a regular file")
                chunks: list[bytes] = []
                total = 0
                while True:
                    chunk = os.read(fd, min(65_536, self.limits.maximum_bytes + 1 - total))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
                    if total > self.limits.maximum_bytes:
                        raise WorkspaceError("workspace file exceeds byte limit")
            finally:
                os.close(fd)
            data = b"".join(chunks)
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError as error:
                raise WorkspaceError("workspace file is not UTF-8 text") from error
            lines = text.splitlines(keepends=True)
            if start_line > len(lines) + 1:
                raise WorkspaceError("file read starts beyond end of file")
            selected: list[str] = []
            selected_bytes = 0
            for line in lines[start_line - 1 : start_line - 1 + max_lines]:
                line_bytes = len(line.encode("utf-8"))
                if selected_bytes + line_bytes > maximum_page_bytes:
                    if not selected:
                        raise WorkspaceError("one file line exceeds the page byte limit")
                    break
                selected.append(line)
                selected_bytes += line_bytes
            end_line = start_line + len(selected) - 1
            eof = end_line >= len(lines)
            return {
                "path": normalized,
                "generation": generation,
                "start_line": start_line,
                "end_line": end_line,
                "content": "".join(selected),
                "next_start_line": None if eof else end_line + 1,
                "eof": eof,
                "content_sha256": hashlib.sha256(data).hexdigest(),
            }

    def create_file(self, run_id: str, path: str, content: str) -> dict[str, Any]:
        data = content.encode()
        return self._mutate(
            run_id,
            "workspace.file.create",
            {"path": path, "content": content},
            lambda root: self._create(root, path, data),
        )

    def _create(self, root: Path, path: str, data: bytes) -> None:
        target = self._safe_target(root, path, creating=True)
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)

    def apply_patch(self, run_id: str, patch: str) -> dict[str, Any]:
        if len(patch.encode()) > self.limits.maximum_patch_bytes or "\0" in patch:
            raise WorkspaceError("patch is binary or exceeds byte limit")
        return self._mutate(
            run_id, "workspace.patch.apply", {"patch": patch}, lambda root: self._apply(root, patch)
        )

    def _apply(self, root: Path, patch: str) -> None:
        forbidden = r"^(GIT binary patch|Binary files |rename |copy |deleted file mode|new file mode|old mode|new mode|index )"
        if re.search(forbidden, patch, re.MULTILINE):
            raise WorkspaceError(
                "binary, create, delete, rename, copy, and mode patches are unsupported"
            )
        lines = patch.splitlines(keepends=True)
        index = 0
        while index < len(lines):
            if not lines[index].startswith("--- a/"):
                raise WorkspaceError("malformed patch")
            old = lines[index][6:].rstrip("\r\n").split("\t", 1)[0]
            index += 1
            if index >= len(lines) or not lines[index].startswith("+++ b/"):
                raise WorkspaceError("malformed patch")
            new = lines[index][6:].rstrip("\r\n").split("\t", 1)[0]
            if old != new:
                raise WorkspaceError("patch target mismatch")
            target = self._safe_target(root, old)
            original = target.read_text()
            source = original.splitlines(keepends=True)
            output: list[str] = []
            cursor = 0
            index += 1
            while index < len(lines) and lines[index].startswith("@@"):
                match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", lines[index])
                if not match:
                    raise WorkspaceError("malformed hunk")
                start = int(match.group(1)) - 1
                if start < cursor:
                    raise WorkspaceError("overlapping hunk")
                output.extend(source[cursor:start])
                cursor = start
                index += 1
                while index < len(lines) and not lines[index].startswith(("@@", "--- a/")):
                    marker, text = lines[index][:1], lines[index][1:]
                    if marker in {" ", "-"}:
                        if cursor >= len(source) or source[cursor] != text:
                            raise WorkspaceError("patch context mismatch")
                        if marker == " ":
                            output.append(text)
                        cursor += 1
                    elif marker == "+":
                        output.append(text)
                    elif lines[index].startswith("\\ No newline at end of file"):
                        pass
                    else:
                        raise WorkspaceError("malformed hunk line")
                    index += 1
            output.extend(source[cursor:])
            target.write_text("".join(output))

    def _snapshot(self, base: Path, current: Path) -> dict[str, Any]:
        base_files = {p.relative_to(base).as_posix(): p for p in base.rglob("*") if p.is_file()}
        current_files = {
            p.relative_to(current).as_posix(): p for p in current.rglob("*") if p.is_file()
        }
        changed: list[str] = []
        chunks: list[str] = []
        for name in sorted(base_files.keys() | current_files.keys()):
            before = base_files.get(name)
            after = current_files.get(name)
            old_bytes = b"" if before is None else before.read_bytes()
            new_bytes = b"" if after is None else after.read_bytes()
            if old_bytes == new_bytes:
                continue
            try:
                old = old_bytes.decode().splitlines(keepends=True)
                new = new_bytes.decode().splitlines(keepends=True)
            except UnicodeDecodeError:
                raise WorkspaceError("binary file changes are unsupported") from None
            changed.append(name)
            chunks.extend(difflib.unified_diff(old, new, f"a/{name}", f"b/{name}"))
        diff = "".join(chunks)
        if len(diff.encode()) > self.limits.maximum_diff_bytes:
            raise WorkspaceError("diff byte limit exceeded")
        _, total = self._scan(current)
        return {"diff": diff, "changed_files": changed, "changed_bytes": total}

    def diff(self, run_id: str) -> dict[str, Any]:
        workspace = self.acquire(run_id)
        with self._locked(workspace.workspace_id):
            self._verify(workspace)
            _, current = self._generation(workspace)
            return self._snapshot(workspace.path / "gen-0", current)

    def _mutate(
        self, run_id: str, capability: str, arguments: Mapping[str, Any], operation: Any
    ) -> dict[str, Any]:
        workspace = self.acquire(run_id)
        with self._locked(workspace.workspace_id):
            self._verify(workspace)
            mutation_id = invocation_identity(run_id, "IMPLEMENT", capability, arguments)
            existing = self.store.connection.execute(
                "SELECT * FROM workspace_mutations WHERE mutation_id=? AND workspace_id=?",
                (mutation_id, workspace.workspace_id),
            ).fetchone()
            if existing:
                if existing["status"] == "SUCCEEDED":
                    return json.loads(existing["result_json"])
                raise WorkspaceError(existing["error"] or "mutation did not complete")
            run = self.store.get_run(run_id)
            if run.state != RunState.IMPLEMENT:
                raise WorkspaceError("workspace writes require durable IMPLEMENT stage")
            generation, current = self._generation(workspace)
            candidate = workspace.path / f"candidate-{mutation_id}"
            shutil.copytree(current, candidate, symlinks=False)
            try:
                operation(candidate)
                self._scan(candidate)
                result = self._snapshot(workspace.path / "gen-0", candidate)
                published = workspace.path / f"gen-{generation + 1}"
                os.rename(candidate, published)
                now = datetime.now(UTC).isoformat()
                with self.store.connection:
                    state = self.store.connection.execute(
                        "SELECT state FROM runs WHERE run_id=?", (run_id,)
                    ).fetchone()[0]
                    current_db = self.store.connection.execute(
                        "SELECT current_generation FROM workspaces WHERE workspace_id=?",
                        (workspace.workspace_id,),
                    ).fetchone()[0]
                    if state != RunState.IMPLEMENT or current_db != generation:
                        raise WorkspaceError("run state or generation changed before publication")
                    sequence = self.store.connection.execute(
                        "SELECT COALESCE(MAX(sequence),0)+1 FROM workspace_mutations WHERE workspace_id=?",
                        (workspace.workspace_id,),
                    ).fetchone()[0]
                    self.store.connection.execute(
                        "INSERT INTO workspace_mutations VALUES (?,?,?,?,?,'SUCCEEDED',?,NULL,?,?)",
                        (
                            mutation_id,
                            workspace.workspace_id,
                            sequence,
                            capability,
                            json.dumps(arguments, sort_keys=True),
                            json.dumps(result, sort_keys=True),
                            now,
                            now,
                        ),
                    )
                    self.store.connection.execute(
                        "INSERT INTO workspace_checkpoints(workspace_id,mutation_id,diff,changed_files_json,created_at) VALUES (?,?,?,?,?)",
                        (
                            workspace.workspace_id,
                            mutation_id,
                            result["diff"],
                            json.dumps(result["changed_files"]),
                            now,
                        ),
                    )
                    self.store.connection.execute(
                        "UPDATE workspaces SET current_generation=? WHERE workspace_id=? AND current_generation=?",
                        (generation + 1, workspace.workspace_id, generation),
                    )
                if current != workspace.path / "gen-0":
                    shutil.rmtree(current)
                return result
            except Exception:
                shutil.rmtree(candidate, ignore_errors=True)
                published = workspace.path / f"gen-{generation + 1}"
                row = self.store.connection.execute(
                    "SELECT current_generation FROM workspaces WHERE workspace_id=?",
                    (workspace.workspace_id,),
                ).fetchone()
                if published.exists() and (row is None or row[0] != generation + 1):
                    shutil.rmtree(published, ignore_errors=True)
                raise


def workspace_capabilities(manager: WorkspaceManager) -> tuple[Capability, ...]:
    strict = {"type": "object", "additionalProperties": False}
    read_schema = strict | {
        "properties": {
            "path": {"type": "string", "minLength": 1, "maxLength": 512},
            "start_line": {"type": "integer", "minimum": 1, "maximum": 1_000_000},
            "max_lines": {"type": "integer", "minimum": 1, "maximum": 400},
        },
        "required": ["path"],
    }
    patch_schema = strict | {"properties": {"patch": {"type": "string"}}, "required": ["patch"]}
    create_schema = strict | {
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    }

    def checked(context: InvocationContext) -> str:
        workspace = manager.acquire(context.run_id)
        if context.workspace_id != workspace.workspace_id:
            raise WorkspaceError("cross-run workspace identity denied")
        return context.run_id

    return (
        Capability(
            CapabilityDescriptor(
                CapabilityCard(
                    "workspace.file.read",
                    "Read workspace file",
                    "Read bounded UTF-8 lines from the current confined workspace generation",
                ),
                read_schema,
                Effect.TRUSTED_WORKSPACE_READ,
            ),
            lambda a, c: manager.read_file(
                checked(c),
                a["path"],
                start_line=a.get("start_line", 1),
                max_lines=a.get("max_lines", 200),
            ),
        ),
        Capability(
            CapabilityDescriptor(
                CapabilityCard(
                    "workspace.patch.apply", "Apply patch", "Apply a confined text patch"
                ),
                patch_schema,
                Effect.TRUSTED_WORKSPACE_WRITE,
            ),
            lambda a, c: manager.apply_patch(checked(c), a["patch"]),
        ),
        Capability(
            CapabilityDescriptor(
                CapabilityCard(
                    "workspace.file.create", "Create file", "Create one confined text file"
                ),
                create_schema,
                Effect.TRUSTED_WORKSPACE_WRITE,
            ),
            lambda a, c: manager.create_file(checked(c), a["path"], a["content"]),
        ),
        Capability(
            CapabilityDescriptor(
                CapabilityCard("git.diff.read", "Read diff", "Read confined workspace diff"),
                strict | {"properties": {}},
                Effect.TRUSTED_WORKSPACE_READ,
            ),
            lambda _a, c: manager.diff(checked(c)),
        ),
    )
