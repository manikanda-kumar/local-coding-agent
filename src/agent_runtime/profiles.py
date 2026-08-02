from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from agent_runtime.request_validation import validate_request_options

ReasoningField = Literal["reasoning", "reasoning_content"]
ReasoningMode = Literal["structured", "tagged_content"]


def _freeze(value: Any, depth: int = 0) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item, depth + 1) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item, depth + 1) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError("request extensions must contain only JSON values")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ModelRequestProfile:
    temperature: float
    max_output_tokens: int
    top_p: float
    top_k: int | None = None
    seed: int | None = None
    stop: tuple[str, ...] = ()
    timeout: float = 300.0
    reasoning_field: ReasoningField = "reasoning"
    reasoning_mode: ReasoningMode = "structured"
    extensions: Mapping[str, Any] = field(default_factory=dict)
    tool_parser: str | None = None
    reasoning_parser: str | None = None
    deployment_notes: str = ""
    context_window_tokens: int = 131_072

    def __post_init__(self) -> None:
        extensions = validate_request_options(
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            seed=self.seed,
            stop=self.stop,
            timeout=self.timeout,
            reasoning_mode=self.reasoning_mode,
            reasoning_field=self.reasoning_field,
            extensions=self.extensions,
        )
        if (
            not isinstance(self.context_window_tokens, int)
            or isinstance(self.context_window_tokens, bool)
            or self.context_window_tokens <= self.max_output_tokens
            or self.context_window_tokens > 10_000_000
            or any(
                value is not None and (not isinstance(value, str) or not value or len(value) > 128)
                for value in (self.tool_parser, self.reasoning_parser)
            )
            or len(self.deployment_notes) > 1024
        ):
            raise ValueError("deployment metadata or context window is invalid or unbounded")
        object.__setattr__(self, "extensions", _freeze(extensions))

    def request_extensions(self) -> dict[str, Any]:
        return _thaw(self.extensions)

    def chat_request_options(self) -> dict[str, Any]:
        """Return the runtime-owned options shared by all model call sites."""
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "seed": self.seed,
            "stop": self.stop,
            "timeout": self.timeout,
            "reasoning_field": self.reasoning_field,
            "reasoning_mode": self.reasoning_mode,
            "extensions": self.request_extensions(),
        }

    def operational_fingerprint(self) -> str:
        """Hash every setting that can affect a request or its context contract."""
        value = {
            "context_window_tokens": self.context_window_tokens,
            "extensions": self.request_extensions(),
            "max_output_tokens": self.max_output_tokens,
            "reasoning_field": self.reasoning_field,
            "reasoning_mode": self.reasoning_mode,
            "reasoning_parser": self.reasoning_parser,
            "seed": self.seed,
            "stop": list(self.stop),
            "temperature": self.temperature,
            "timeout": self.timeout,
            "tool_parser": self.tool_parser,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


PROVIDER_DEFAULT_PROFILE_ID = "provider-default"
PROVIDER_DEFAULT_PROFILE_SHA256 = hashlib.sha256(
    b"agent-runtime:model-request-profile:provider-default:v1"
).hexdigest()


MODEL_PROFILES: Mapping[str, ModelRequestProfile] = MappingProxyType(
    {
        "minimax-m2.7-vllm": ModelRequestProfile(
            1.0,
            8192,
            0.95,
            top_k=40,
            tool_parser="minimax_m2",
            reasoning_parser="minimax_m2",
            deployment_notes="Current vLLM; older builds may require minimax_m2_append_think.",
            context_window_tokens=131_072,
        ),
        "glm-5.2-vllm": ModelRequestProfile(
            1.0,
            8192,
            0.95,
            extensions={"chat_template_kwargs": {"clear_thinking": False}},
            tool_parser="glm47",
            reasoning_parser="glm45",
            context_window_tokens=131_072,
        ),
        "kimi-k3-vllm-agentic": ModelRequestProfile(
            1.0,
            8192,
            1.0,
            reasoning_field="reasoning_content",
            extensions={"reasoning_effort": "max"},
            tool_parser="kimi_k3",
            reasoning_parser="kimi_k3",
            deployment_notes="Vendor wire contract uses reasoning_content; current vLLM may expose reasoning.",
            context_window_tokens=131_072,
        ),
        "gemma-4-31b-it-vllm": ModelRequestProfile(
            1.0,
            8192,
            0.95,
            top_k=64,
            extensions={"chat_template_kwargs": {"enable_thinking": True}},
            tool_parser="gemma4",
            reasoning_parser="gemma4",
            deployment_notes="Reasoning replay is supported only for tool continuation, not ordinary history.",
            context_window_tokens=131_072,
        ),
    }
)
