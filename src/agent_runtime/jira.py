from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal
from urllib.parse import quote

import httpx

from agent_runtime.gateway import Capability, CapabilityCard, CapabilityDescriptor


class JiraReadError(RuntimeError):
    """A deliberately detail-free external service error."""


@dataclass(frozen=True, slots=True)
class JiraAuth:
    kind: Literal["bearer", "basic"]
    secret: str
    username: str | None = None

    def __repr__(self) -> str:
        return f"JiraAuth(kind={self.kind!r}, secret=<redacted>, username=<redacted>)"


@dataclass(frozen=True, slots=True)
class Comment:
    comment_id: str
    author: str
    body: str
    created: str
    updated: str


@dataclass(frozen=True, slots=True)
class StorySnapshot:
    issue_key: str
    summary: str
    description: str
    status: str
    issue_type: str
    updated: str
    comments: tuple[Comment, ...]
    content_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_text(item) for item in value)
    if isinstance(value, dict):
        text = value.get("text", "")
        children = _text(value.get("content", []))
        suffix = "\n" if value.get("type") in {"paragraph", "heading", "listItem"} else ""
        return f"{text}{children}{suffix}"
    return str(value)


class JiraReadAdapter:
    def __init__(
        self,
        base_url: str,
        auth: JiraAuth,
        *,
        timeout: float = 10.0,
        client: httpx.Client | None = None,
        page_size: int = 50,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._auth = auth
        self._timeout = timeout
        self._client = client or httpx.Client()
        self._owns_client = client is None
        self.page_size = page_size

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _headers(self) -> dict[str, str]:
        if self._auth.kind == "bearer":
            return {"Authorization": f"Bearer {self._auth.secret}", "Accept": "application/json"}
        return {"Accept": "application/json"}

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        basic = None
        if self._auth.kind == "basic":
            if not self._auth.username:
                raise ValueError("basic authentication requires a username")
            basic = (self._auth.username, self._auth.secret)
        try:
            response = self._client.get(
                f"{self.base_url}{path}",
                params=params,
                headers=self._headers(),
                auth=basic,
                timeout=self._timeout,
            )
            response.raise_for_status()
            value = response.json()
            if not isinstance(value, dict):
                raise TypeError("non-object response")
            return value
        except (httpx.HTTPError, TypeError, ValueError):
            raise JiraReadError("JIRA read failed") from None

    def issue(self, issue_key: str) -> dict[str, Any]:
        return self._get(
            f"/rest/api/3/issue/{quote(issue_key, safe='')}",
            {"fields": "summary,description,status,issuetype,updated"},
        )

    def comments(self, issue_key: str) -> tuple[dict[str, Any], ...]:
        start, comments = 0, []
        while True:
            page = self._get(
                f"/rest/api/3/issue/{quote(issue_key, safe='')}/comment",
                {"startAt": start, "maxResults": self.page_size},
            )
            values = page.get("comments", [])
            if not isinstance(values, list):
                raise JiraReadError("JIRA read failed")
            comments.extend(value for value in values if isinstance(value, dict))
            start += len(values)
            total = page.get("total", start)
            if not values or start >= total:
                return tuple(comments)

    def snapshot(self, issue_key: str) -> StorySnapshot:
        issue, raw_comments = self.issue(issue_key), self.comments(issue_key)
        fields = issue.get("fields", {})
        comments = tuple(
            Comment(
                str(item.get("id", "")),
                str(item.get("author", {}).get("displayName", "")),
                _text(item.get("body")).strip(),
                str(item.get("created", "")),
                str(item.get("updated", "")),
            )
            for item in raw_comments
        )
        content = {
            "issue_key": str(issue.get("key", issue_key)),
            "summary": str(fields.get("summary", "")),
            "description": _text(fields.get("description")).strip(),
            "status": str(fields.get("status", {}).get("name", "")),
            "issue_type": str(fields.get("issuetype", {}).get("name", "")),
            "updated": str(fields.get("updated", "")),
            "comments": [asdict(comment) for comment in comments],
        }
        digest = hashlib.sha256(
            json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
        return StorySnapshot(
            issue_key=content["issue_key"],
            summary=content["summary"],
            description=content["description"],
            status=content["status"],
            issue_type=content["issue_type"],
            updated=content["updated"],
            comments=comments,
            content_hash=digest,
        )


def jira_read_capabilities(adapter: JiraReadAdapter) -> tuple[Capability, Capability]:
    schema = {
        "type": "object",
        "properties": {"issue_key": {"type": "string"}},
        "required": ["issue_key"],
        "additionalProperties": False,
    }
    issue = Capability(
        CapabilityDescriptor(
            CapabilityCard("jira.issue.read", "JIRA issue read", "Read a JIRA issue"), schema
        ),
        lambda args, _context: adapter.issue(args["issue_key"]),
    )
    comments = Capability(
        CapabilityDescriptor(
            CapabilityCard("jira.comments.read", "JIRA comments read", "Read all JIRA comments"),
            schema,
        ),
        lambda args, _context: {"comments": adapter.comments(args["issue_key"])},
    )
    return issue, comments
