"""Read-only repository snapshots selected entirely by trusted runtime configuration."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


class RepositorySnapshotError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RepositorySnapshot:
    path: Path
    origin: str
    base_revision: str

    def read_file(self, relative_path: str, *, maximum_bytes: int = 1_000_000) -> bytes:
        candidate = Path(relative_path)
        if candidate.is_absolute() or ".." in candidate.parts or relative_path.startswith(".git/"):
            raise RepositorySnapshotError("invalid repository path")
        result = _git(self.path, "show", f"{self.base_revision}:{candidate.as_posix()}")
        if len(result) > maximum_bytes:
            raise RepositorySnapshotError("repository file is too large")
        return result


def _git(path: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", *arguments],
            cwd=path,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            env={"PATH": "/usr/bin:/bin", "GIT_CONFIG_NOSYSTEM": "1"},
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise RepositorySnapshotError("repository snapshot validation failed") from error


class RepositorySnapshotProvider:
    def __init__(self, trusted_path: str | Path, trusted_origin: str) -> None:
        self.path = Path(trusted_path).resolve()
        self.origin = trusted_origin

    def acquire(self, recorded_base_revision: str) -> RepositorySnapshot:
        actual_origin = _git(self.path, "remote", "get-url", "origin").decode().strip()
        if actual_origin != self.origin:
            raise RepositorySnapshotError("repository origin mismatch")
        canonical = _git(
            self.path,
            "rev-parse",
            "--verify",
            "--end-of-options",
            f"{recorded_base_revision}^{{commit}}",
        )
        canonical_revision = canonical.decode().strip()
        if canonical_revision != recorded_base_revision:
            raise RepositorySnapshotError("repository base revision is not canonical")
        # Pin to the recorded object. A moving checkout is intentionally irrelevant to reads.
        return RepositorySnapshot(self.path, self.origin, canonical_revision)
