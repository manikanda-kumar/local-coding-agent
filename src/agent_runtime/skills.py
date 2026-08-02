"""Signed, declarative, internal-only orchestration skills.

This module deliberately has no network, package-install, import, or execution interface.
Registries are constructed from package objects or an explicitly selected trusted directory.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from agent_runtime.gateway import Effect, InMemoryCapabilityCatalog

_SEMVER = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
_EFFECT_RANK = {effect: rank for rank, effect in enumerate(Effect)}


class SkillVerificationError(ValueError):
    pass


@dataclass(frozen=True, slots=True, order=True)
class SkillCapability:
    capability_id: str
    version: str


@dataclass(frozen=True, slots=True)
class RevocationMetadata:
    revoked_at: datetime
    reason: str

    def __post_init__(self) -> None:
        if self.revoked_at.tzinfo is None or not self.reason:
            raise ValueError("revocation timestamp must be timezone-aware and include a reason")


@dataclass(frozen=True, slots=True)
class SkillManifest:
    skill_id: str
    version: str
    owner: str
    signer_id: str
    content_sha256: str
    capabilities: tuple[SkillCapability, ...]
    maximum_effect: Effect
    repositories: tuple[str, ...]
    expires_at: datetime
    revocation: RevocationMetadata | None = None

    def __post_init__(self) -> None:
        if not self.skill_id or not self.owner or not self.signer_id:
            raise ValueError("skill identity, owner, and signer are required")
        if not _SEMVER.fullmatch(self.version):
            raise ValueError("skill version must be semantic version")
        if len(self.content_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.content_sha256
        ):
            raise ValueError("content_sha256 must be lowercase SHA-256")
        if not self.repositories or any(not repository for repository in self.repositories):
            raise ValueError("at least one explicit repository is required")
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("duplicate capability grant")
        if any(not item.capability_id or not item.version for item in self.capabilities):
            raise ValueError("capability ID and version are required")
        if self.expires_at.tzinfo is None:
            raise ValueError("expiry must be timezone-aware")


def canonical_manifest(manifest: SkillManifest) -> bytes:
    value = asdict(manifest)
    value["maximum_effect"] = manifest.maximum_effect.value
    value["expires_at"] = manifest.expires_at.astimezone(UTC).isoformat()
    if manifest.revocation is not None:
        value["revocation"]["revoked_at"] = manifest.revocation.revoked_at.astimezone(
            UTC
        ).isoformat()
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def signing_payload(manifest: SkillManifest, content: bytes) -> bytes:
    encoded = canonical_manifest(manifest)
    return b"agent-runtime-skill-v1\0" + len(encoded).to_bytes(8, "big") + encoded + content


@dataclass(frozen=True, slots=True)
class SkillPackage:
    manifest: SkillManifest
    content: bytes
    signature: bytes

    def __post_init__(self) -> None:
        try:
            self.content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("skill content must be UTF-8 prompt/template text") from exc
        if not self.signature:
            raise ValueError("skill signature is required")

    @property
    def text(self) -> str:
        return self.content.decode("utf-8")


@dataclass(frozen=True, slots=True)
class TrustedSigner:
    signer_id: str
    public_key: Ed25519PublicKey
    expires_at: datetime | None = None
    revoked_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.signer_id:
            raise ValueError("signer ID is required")
        if any(
            value is not None and value.tzinfo is None
            for value in (self.expires_at, self.revoked_at)
        ):
            raise ValueError("signer timestamps must be timezone-aware")


class TrustedSignerRegistry:
    def __init__(self, signers: tuple[TrustedSigner, ...]) -> None:
        self._signers = {signer.signer_id: signer for signer in signers}
        if len(self._signers) != len(signers):
            raise ValueError("duplicate signer ID")

    def verify(self, package: SkillPackage, now: datetime) -> None:
        signer = self._signers.get(package.manifest.signer_id)
        if signer is None:
            raise SkillVerificationError("unknown signer")
        if signer.revoked_at is not None and signer.revoked_at <= now:
            raise SkillVerificationError("signer is revoked")
        if signer.expires_at is not None and signer.expires_at <= now:
            raise SkillVerificationError("signer is expired")
        try:
            signer.public_key.verify(
                package.signature, signing_payload(package.manifest, package.content)
            )
        except InvalidSignature as exc:
            raise SkillVerificationError("invalid skill signature") from exc


class InternalSkillRegistry:
    def __init__(self, packages: tuple[SkillPackage, ...], signers: TrustedSignerRegistry) -> None:
        self._signers = signers
        self._packages: dict[tuple[str, str], SkillPackage] = {}
        for package in packages:
            key = (package.manifest.skill_id, package.manifest.version)
            if key in self._packages:
                raise ValueError("duplicate skill ID/version")
            self._packages[key] = package

    @classmethod
    def from_trusted_directory(
        cls, root: str | Path, signers: TrustedSignerRegistry
    ) -> InternalSkillRegistry:
        """Load only direct package directories in the explicitly supplied runtime directory."""
        root = Path(root).resolve(strict=True)
        packages: list[SkillPackage] = []
        for child in sorted(root.iterdir()):
            if child.is_symlink() or not child.is_dir():
                continue
            files = {
                item.name for item in child.iterdir() if item.is_file() and not item.is_symlink()
            }
            if files != {"manifest.json", "content.txt", "signature.ed25519"}:
                raise SkillVerificationError(
                    "skill directory must contain only declarative package files"
                )
            raw = json.loads((child / "manifest.json").read_text(encoding="utf-8"))
            allowed = {
                "skill_id",
                "version",
                "owner",
                "signer_id",
                "content_sha256",
                "capabilities",
                "maximum_effect",
                "repositories",
                "expires_at",
                "revocation",
            }
            if set(raw) - allowed:
                raise SkillVerificationError("unknown manifest field")
            revocation = raw.get("revocation")
            packages.append(
                SkillPackage(
                    SkillManifest(
                        skill_id=raw["skill_id"],
                        version=raw["version"],
                        owner=raw["owner"],
                        signer_id=raw["signer_id"],
                        content_sha256=raw["content_sha256"],
                        capabilities=tuple(SkillCapability(**item) for item in raw["capabilities"]),
                        maximum_effect=Effect(raw["maximum_effect"]),
                        repositories=tuple(raw["repositories"]),
                        expires_at=datetime.fromisoformat(raw["expires_at"]),
                        revocation=None
                        if revocation is None
                        else RevocationMetadata(
                            datetime.fromisoformat(revocation["revoked_at"]), revocation["reason"]
                        ),
                    ),
                    (child / "content.txt").read_bytes(),
                    (child / "signature.ed25519").read_bytes(),
                )
            )
        return cls(tuple(packages), signers)

    def resolve(
        self, skill_id: str, version: str, repository: str, *, now: datetime | None = None
    ) -> SkillPackage:
        package = self._packages.get((skill_id, version))
        if package is None:
            raise SkillVerificationError("unknown skill ID/version")
        now = now or datetime.now(UTC)
        manifest = package.manifest
        if hashlib.sha256(package.content).hexdigest() != manifest.content_sha256:
            raise SkillVerificationError("skill content hash mismatch")
        self._signers.verify(package, now)
        if manifest.expires_at <= now:
            raise SkillVerificationError("skill is expired")
        if manifest.revocation is not None and manifest.revocation.revoked_at <= now:
            raise SkillVerificationError("skill is revoked")
        if repository not in manifest.repositories:
            raise SkillVerificationError("skill does not apply to repository")
        return package

    def resume(
        self, pin: SkillPin, repository: str, *, now: datetime | None = None
    ) -> SkillPackage:
        package = self.resolve(pin.skill_id, pin.version, repository, now=now)
        if (package.manifest.content_sha256, package.signature, package.manifest.signer_id) != (
            pin.content_sha256,
            pin.signature,
            pin.signer_id,
        ):
            raise SkillVerificationError("pinned skill was replaced")
        return package

    def pin_run(
        self,
        store: object,
        run_id: str,
        skill_id: str,
        version: str,
        repository: str,
        *,
        now: datetime | None = None,
    ) -> SkillPackage:
        package = self.resolve(skill_id, version, repository, now=now)
        store.pin_skill(run_id, package)  # type: ignore[attr-defined]
        return package


@dataclass(frozen=True, slots=True)
class SkillPin:
    skill_id: str
    version: str
    content_sha256: str
    signature: bytes
    signer_id: str

    @classmethod
    def from_package(cls, package: SkillPackage) -> SkillPin:
        manifest = package.manifest
        return cls(
            manifest.skill_id,
            manifest.version,
            manifest.content_sha256,
            package.signature,
            manifest.signer_id,
        )


def effective_capabilities(
    *,
    task: frozenset[SkillCapability],
    stage: frozenset[SkillCapability],
    skill: SkillManifest,
    catalog: InMemoryCapabilityCatalog,
) -> frozenset[SkillCapability]:
    """Return grants present, at the exact version, in every authority source."""
    signed = frozenset(skill.capabilities)
    narrowed = task & stage & signed
    result = set()
    for grant in narrowed:
        capability = catalog.get(grant.capability_id)
        if (
            capability is not None
            and capability.descriptor.card.version == grant.version
            and _EFFECT_RANK[Effect(capability.descriptor.effect)]
            <= _EFFECT_RANK[skill.maximum_effect]
        ):
            result.add(grant)
    return frozenset(result)


def encode_signature(signature: bytes) -> str:
    return base64.b64encode(signature).decode("ascii")
