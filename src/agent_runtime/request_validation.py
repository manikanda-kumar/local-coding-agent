from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

MAX_JSON_BYTES = 16_384
MAX_DEPTH = 6
MAX_ITEMS = 256
MAX_STRING = 8_192
MAX_TIMEOUT = 600.0
ALLOWED_EXTENSIONS = frozenset(
    {"chat_template_kwargs", "guided_json", "guided_regex", "guided_choice", "reasoning_effort"}
)
_SECRET_WORDS = ("token", "secret", "password", "api_key", "authorization", "credential")


def validate_json(
    value: Any,
    *,
    label: str,
    secret_keys: bool = False,
    max_bytes: int = MAX_JSON_BYTES,
    max_depth: int = MAX_DEPTH,
    max_items: int = MAX_ITEMS,
    max_string: int = MAX_STRING,
) -> Any:
    """Validate, bound, and detach an untrusted JSON value."""

    items = 0

    def walk(item: Any, depth: int) -> None:
        nonlocal items
        if depth > max_depth:
            raise ValueError(f"{label} is nested too deeply")
        items += 1
        if items > max_items:
            raise ValueError(f"{label} contains too many items")
        if item is None or isinstance(item, bool):
            return
        if isinstance(item, str):
            if len(item) > max_string:
                raise ValueError(f"{label} contains an oversized string")
            return
        if isinstance(item, int):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(f"{label} contains a non-finite number")
            return
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise TypeError(f"{label} keys must be strings")
                if len(key) > 256:
                    raise ValueError(f"{label} contains an oversized key")
                if secret_keys and any(word in key.lower() for word in _SECRET_WORDS):
                    raise ValueError(f"{label} cannot contain secret-bearing fields")
                walk(child, depth + 1)
            return
        if isinstance(item, Sequence) and not isinstance(item, (bytes, bytearray)):
            for child in item:
                walk(child, depth + 1)
            return
        raise ValueError(f"{label} must contain only JSON values")

    walk(value, 0)
    try:
        encoded = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain only strict JSON values") from error
    if len(encoded) > max_bytes:
        raise ValueError(f"{label} exceeds its byte limit")
    return deepcopy(value)


def validate_extensions(values: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(values, Mapping) or not all(isinstance(key, str) for key in values):
        raise ValueError("request extensions must be an object with string keys")
    unknown = set(values) - ALLOWED_EXTENSIONS
    if unknown:
        raise ValueError(f"unsupported request extensions: {', '.join(sorted(unknown, key=str))}")
    return validate_json(values, label="request extensions", secret_keys=True)


def validate_request_options(
    *,
    temperature: float | None,
    max_tokens: int | None,
    top_p: float | None,
    top_k: int | None,
    seed: int | None,
    stop: tuple[str, ...],
    timeout: float | None,
    reasoning_mode: str,
    reasoning_field: str,
    extensions: Mapping[str, Any],
) -> dict[str, Any]:
    if temperature is not None and (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or not 0 <= temperature <= 2
    ):
        raise ValueError("temperature must be finite and between 0 and 2")
    if max_tokens is not None and (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or not 1 <= max_tokens <= 1_000_000
    ):
        raise ValueError("max_tokens is out of range")
    if top_p is not None and (
        isinstance(top_p, bool)
        or not isinstance(top_p, (int, float))
        or not math.isfinite(top_p)
        or not 0 <= top_p <= 1
    ):
        raise ValueError("top_p must be finite and between 0 and 1")
    if top_k is not None and (
        isinstance(top_k, bool) or not isinstance(top_k, int) or not 0 <= top_k <= 1_000_000
    ):
        raise ValueError("top_k is out of range")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or not -(2**63) <= seed < 2**63
    ):
        raise ValueError("seed is out of range")
    if (
        not isinstance(stop, tuple)
        or len(stop) > 16
        or any(not isinstance(value, str) or not value or len(value) > 256 for value in stop)
    ):
        raise ValueError("stop sequences are invalid or unbounded")
    if timeout is not None and (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or not 0 < timeout <= MAX_TIMEOUT
    ):
        raise ValueError("timeout must be finite and between 0 and 600 seconds")
    if reasoning_mode not in {"preserve", "structured", "tagged_content"}:
        raise ValueError("unsupported reasoning mode")
    if reasoning_field not in {"reasoning", "reasoning_content"}:
        raise ValueError("unsupported reasoning field")
    return validate_extensions(extensions)
