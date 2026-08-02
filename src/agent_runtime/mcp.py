"""Reviewed MCP tools exposed through the capability gateway (never to a model directly)."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx

from agent_runtime.durable import ArtifactStore
from agent_runtime.gateway import Capability, CapabilityCard, CapabilityDescriptor


def canonical_schema_hash(schema: Mapping[str, Any]) -> str:
    encoded = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


class MCPFailure(RuntimeError):
    """A deliberately detail-free, machine classifiable adapter failure."""

    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = code
        super().__init__(message or code.replace("_", " "))


@dataclass(frozen=True, slots=True, repr=False)
class MCPCredentials:
    headers: Mapping[str, str] = field(default_factory=dict)

    def __repr__(self) -> str:
        return "MCPCredentials(headers=<redacted>)"


class StreamableHTTPTransport:
    """Synchronous MCP Streamable HTTP JSON-RPC client with session handling."""

    def __init__(
        self,
        endpoint: str,
        credentials: MCPCredentials | None = None,
        *,
        client: httpx.Client | None = None,
        timeout: float = 10,
        maximum_response_bytes: int = 8_000_000,
    ) -> None:
        self._endpoint = endpoint
        self._credentials = credentials or MCPCredentials()
        self._client = client or httpx.Client()
        self._timeout = timeout
        self._maximum_response_bytes = maximum_response_bytes
        self._session_id: str | None = None
        self._next_id = 0
        self._initialized = False

    def _request(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        self._next_id += 1
        request_id = self._next_id
        headers = {"Accept": "application/json, text/event-stream", **self._credentials.headers}
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        body = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            body["params"] = dict(params)
        try:
            response = self._client.post(
                self._endpoint, json=body, headers=headers, timeout=self._timeout
            )
            response.raise_for_status()
        except httpx.TimeoutException:
            raise MCPFailure("timeout") from None
        except httpx.HTTPError:
            raise MCPFailure("server_unavailable") from None
        if len(response.content) > self._maximum_response_bytes:
            raise MCPFailure("output_too_large")
        session = response.headers.get("Mcp-Session-Id")
        if session:
            self._session_id = session
        try:
            payload = response.json()
        except (ValueError, UnicodeError):
            raise MCPFailure("malformed_response") from None
        if (
            not isinstance(payload, dict)
            or payload.get("jsonrpc") != "2.0"
            or payload.get("id") != request_id
        ):
            raise MCPFailure("malformed_response")
        if "error" in payload:
            raise MCPFailure("server_error")
        if "result" not in payload:
            raise MCPFailure("malformed_response")
        return payload["result"]

    def initialize(self) -> None:
        if self._initialized:
            return
        result = self._request(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "enterprise-agent-runtime", "version": "1"},
            },
        )
        if not isinstance(result, dict) or not isinstance(result.get("protocolVersion"), str):
            raise MCPFailure("malformed_response")
        headers = {"Content-Type": "application/json", **self._credentials.headers}
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        try:
            response = self._client.post(
                self._endpoint,
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                headers=headers,
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.TimeoutException:
            raise MCPFailure("timeout") from None
        except httpx.HTTPError:
            raise MCPFailure("server_unavailable") from None
        self._initialized = True

    def list_tools(self) -> tuple[dict[str, Any], ...]:
        self.initialize()
        result = self._request("tools/list", {})
        if not isinstance(result, dict) or not isinstance(result.get("tools"), list):
            raise MCPFailure("malformed_response")
        if not all(isinstance(tool, dict) for tool in result["tools"]):
            raise MCPFailure("malformed_response")
        return tuple(result["tools"])

    def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        cancelled: threading.Event | None = None,
    ) -> Any:
        self.initialize()
        if cancelled is not None and cancelled.is_set():
            raise MCPFailure("cancelled")
        result = self._request("tools/call", {"name": name, "arguments": dict(arguments)})
        if cancelled is not None and cancelled.is_set():
            raise MCPFailure("cancelled")
        if not isinstance(result, dict) or "content" not in result:
            raise MCPFailure("malformed_result")
        if result.get("isError") is True:
            raise MCPFailure("tool_failed")
        return result


_SECRET_KEYS = ("secret", "token", "password", "passwd", "api_key", "apikey", "authorization")


def redact_untrusted(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]"
            if any(part in str(key).casefold() for part in _SECRET_KEYS)
            else redact_untrusted(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_untrusted(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ReviewedToolMapping:
    server_name: str
    tool_name: str
    capability_id: str
    capability_name: str
    summary: str
    version: str
    input_schema: Mapping[str, Any]
    expected_schema_sha256: str
    effect: str = "read"

    def __post_init__(self) -> None:
        if self.effect != "read":
            raise ValueError("Phase 4 MCP mappings must have trusted effect=read")
        if canonical_schema_hash(self.input_schema) != self.expected_schema_sha256:
            raise ValueError("reviewed input schema does not match its approved hash")


class MCPAllowlistAdapter:
    def __init__(
        self,
        server_name: str,
        transport: StreamableHTTPTransport,
        mappings: tuple[ReviewedToolMapping, ...],
        *,
        artifact_store: ArtifactStore | None = None,
        inline_bytes: int = 32_000,
        hard_bytes: int = 8_000_000,
    ) -> None:
        self.server_name, self.transport = server_name, transport
        self.mappings = mappings
        self.artifact_store = artifact_store
        self.inline_bytes, self.hard_bytes = inline_bytes, hard_bytes
        self.quarantined: dict[str, str] = {}

    def capabilities(self) -> tuple[Capability, ...]:
        discovered = {tool.get("name"): tool for tool in self.transport.list_tools()}
        approved: list[Capability] = []
        for mapping in self.mappings:
            if mapping.server_name != self.server_name:
                continue
            tool = discovered.get(mapping.tool_name)
            schema = tool.get("inputSchema") if isinstance(tool, dict) else None
            if tool is None:
                self.quarantined[mapping.capability_id] = "missing"
                continue
            if (
                not isinstance(schema, dict)
                or canonical_schema_hash(schema) != mapping.expected_schema_sha256
            ):
                self.quarantined[mapping.capability_id] = "schema_drift"
                continue
            self.quarantined.pop(mapping.capability_id, None)
            approved.append(self._capability(mapping))
        return tuple(approved)

    def _capability(self, mapping: ReviewedToolMapping) -> Capability:
        descriptor = CapabilityDescriptor(
            CapabilityCard(
                mapping.capability_id, mapping.capability_name, mapping.summary, mapping.version
            ),
            dict(mapping.input_schema),
            # Never trust MCP annotations such as readOnlyHint; classification is reviewed here.
            mapping.effect,
        )

        def invoke(arguments: Mapping[str, Any], _context: Any) -> Any:
            output = redact_untrusted(self.transport.call_tool(mapping.tool_name, arguments))
            encoded = json.dumps(output, ensure_ascii=False, separators=(",", ":")).encode()
            if len(encoded) > self.hard_bytes:
                raise MCPFailure("output_too_large")
            if len(encoded) <= self.inline_bytes:
                return output
            if self.artifact_store is None:
                raise MCPFailure("output_too_large")
            digest = self.artifact_store.put(encoded)
            return {
                "truncated": True,
                "summary": f"MCP output was {len(encoded)} bytes and stored as an artifact",
                "artifact_sha256": digest,
            }

        return Capability(descriptor, invoke)
