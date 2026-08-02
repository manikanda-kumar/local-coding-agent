"""Deterministic contracts and harness for checked-in golden JIRA/repository tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MAX_TEXT = 100_000


def _string_map(value: Any, field: str) -> dict[str, str]:
    if not isinstance(value, dict) or len(value) > 1_000:
        raise ValueError(f"{field} must be a bounded object")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key or len(key) > 500:
            raise ValueError(f"{field} keys must be bounded non-empty strings")
        if not isinstance(item, str) or len(item) > _MAX_TEXT:
            raise ValueError(f"{field} values must be bounded strings")
        result[key] = item
    return result


@dataclass(frozen=True, slots=True)
class GoldenTask:
    task_id: str
    jira: dict[str, Any]
    repository: dict[str, str]
    expected_files: dict[str, str]
    required_validations: tuple[str, ...]

    @classmethod
    def load(cls, path: str | Path) -> GoldenTask:
        data = json.loads(Path(path).read_text())
        expected = {"task_id", "jira", "repository", "expected_files", "required_validations"}
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError("golden task must contain exactly the supported fields")
        if not isinstance(data["task_id"], str) or not 1 <= len(data["task_id"]) <= 128:
            raise ValueError("task_id must be a bounded non-empty string")
        jira = data["jira"]
        if not isinstance(jira, dict) or not jira or len(jira) > 100:
            raise ValueError("jira must be a bounded non-empty object")
        jira = _string_map(jira, "jira")
        validations = data["required_validations"]
        if not isinstance(validations, list) or not 1 <= len(validations) <= 100:
            raise ValueError("required_validations must be a bounded non-empty list")
        if any(not isinstance(item, str) or not 1 <= len(item) <= 128 for item in validations):
            raise ValueError("validation names must be bounded non-empty strings")
        if len(set(validations)) != len(validations):
            raise ValueError("validation names must be unique")
        return cls(
            data["task_id"],
            jira,
            _string_map(data["repository"], "repository"),
            _string_map(data["expected_files"], "expected_files"),
            tuple(validations),
        )


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    task_id: str
    success: bool
    quality_score: float
    regressions: tuple[str, ...]


def evaluate(
    task: GoldenTask, changed_files: dict[str, str], validations: dict[str, bool]
) -> EvaluationResult:
    checks: list[tuple[str, bool]] = []
    for path, content in task.expected_files.items():
        checks.append((f"file:{path}", changed_files.get(path) == content))
    checks.append(("unexpected-files", set(changed_files) == set(task.expected_files)))
    checks.extend(
        (f"validation:{name}", validations.get(name) is True) for name in task.required_validations
    )
    regressions = tuple(name for name, passed in checks if not passed)
    score = round(100 * sum(passed for _, passed in checks) / len(checks), 2) if checks else 100.0
    return EvaluationResult(task.task_id, not regressions, score, regressions)
