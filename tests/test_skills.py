import dataclasses
import hashlib
from datetime import UTC, datetime, timedelta

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from agent_runtime import (
    Capability,
    CapabilityCard,
    CapabilityDescriptor,
    Effect,
    InMemoryCapabilityCatalog,
    InternalSkillRegistry,
    RevocationMetadata,
    SkillCapability,
    SkillManifest,
    SkillPackage,
    SkillVerificationError,
    SQLiteRunStore,
    TrustedSigner,
    TrustedSignerRegistry,
    canonical_manifest,
    effective_capabilities,
    signing_payload,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def package(*, signer_id="signer", expires=None, revocation=None, content=b"Plan {{ task }}"):
    private = Ed25519PrivateKey.generate()
    manifest = SkillManifest(
        "planning",
        "1.2.3",
        "platform",
        signer_id,
        hashlib.sha256(content).hexdigest(),
        (SkillCapability("read", "1"), SkillCapability("workspace.patch.apply", "2")),
        Effect.TRUSTED_WORKSPACE_WRITE,
        ("org/repo",),
        expires or NOW + timedelta(days=1),
        revocation,
    )
    item = SkillPackage(manifest, content, private.sign(signing_payload(manifest, content)))
    return private, item


def registry(private, item, **signer_changes):
    signer = TrustedSigner("signer", private.public_key(), **signer_changes)
    return InternalSkillRegistry((item,), TrustedSignerRegistry((signer,)))


def test_signature_hash_manifest_and_content_tamper_are_rejected():
    private, item = package()
    good = registry(private, item)
    assert good.resolve("planning", "1.2.3", "org/repo", now=NOW).text.startswith("Plan")

    changed_content = dataclasses.replace(item, content=b"malicious prompt")
    with pytest.raises(SkillVerificationError, match="hash"):
        registry(private, changed_content).resolve("planning", "1.2.3", "org/repo", now=NOW)
    changed_manifest = dataclasses.replace(
        item, manifest=dataclasses.replace(item.manifest, owner="attacker")
    )
    with pytest.raises(SkillVerificationError, match="signature"):
        registry(private, changed_manifest).resolve("planning", "1.2.3", "org/repo", now=NOW)


def test_unknown_revoked_expired_signer_and_revoked_expired_package():
    private, item = package()
    unknown = InternalSkillRegistry((item,), TrustedSignerRegistry(()))
    with pytest.raises(SkillVerificationError, match="unknown signer"):
        unknown.resolve("planning", "1.2.3", "org/repo", now=NOW)
    for changes, message in (
        ({"revoked_at": NOW - timedelta(seconds=1)}, "revoked"),
        ({"expires_at": NOW}, "expired"),
    ):
        with pytest.raises(SkillVerificationError, match=message):
            registry(private, item, **changes).resolve("planning", "1.2.3", "org/repo", now=NOW)

    for kwargs, message in (
        ({"expires": NOW}, "expired"),
        ({"revocation": RevocationMetadata(NOW, "incident")}, "revoked"),
    ):
        key, bad = package(**kwargs)
        with pytest.raises(SkillVerificationError, match=message):
            registry(key, bad).resolve("planning", "1.2.3", "org/repo", now=NOW)


def test_repository_mismatch_and_capability_version_effect_narrowing():
    private, item = package()
    with pytest.raises(SkillVerificationError, match="repository"):
        registry(private, item).resolve("planning", "1.2.3", "other/repo", now=NOW)
    schema = {"type": "object"}
    catalog = InMemoryCapabilityCatalog(
        (
            Capability(
                CapabilityDescriptor(CapabilityCard("read", "read", "", "1"), schema, Effect.READ),
                lambda _a, _c: None,
            ),
            Capability(
                CapabilityDescriptor(
                    CapabilityCard("workspace.patch.apply", "write", "", "3"),
                    schema,
                    Effect.TRUSTED_WORKSPACE_WRITE,
                ),
                lambda _a, _c: None,
            ),
        )
    )
    both = frozenset(item.manifest.capabilities)
    assert effective_capabilities(task=both, stage=both, skill=item.manifest, catalog=catalog) == {
        SkillCapability("read", "1")
    }
    read_ceiling = dataclasses.replace(item.manifest, maximum_effect=Effect.READ)
    assert effective_capabilities(task=both, stage=both, skill=read_ceiling, catalog=catalog) == {
        SkillCapability("read", "1")
    }
    assert (
        effective_capabilities(
            task=both,
            stage=frozenset({SkillCapability("workspace.patch.apply", "2")}),
            skill=item.manifest,
            catalog=catalog,
        )
        == frozenset()
    )


def test_pinned_resume_and_replacement_rejection(tmp_path):
    private, item = package()
    trusted = registry(private, item)
    store = SQLiteRunStore(tmp_path / "runs.db")
    store.create_run(
        "run",
        story_hash="s",
        repository="org/repo",
        base_revision="b",
        provider="p",
        model="m",
        prompt_version="p",
        policy_version="p",
    )
    trusted.pin_run(store, "run", "planning", "1.2.3", "org/repo", now=NOW)
    pin = store.skill_pin("run")
    assert trusted.resume(pin, "org/repo", now=NOW) == item
    store.close()

    resumed = SQLiteRunStore(tmp_path / "runs.db")
    assert resumed.skill_pin("run") == pin
    other_key, replacement = package(content=b"changed")
    with pytest.raises(SkillVerificationError):
        registry(other_key, replacement).resume(pin, "org/repo", now=NOW)


def test_registry_exposes_no_external_install_or_execution_api():
    forbidden = {"install", "download", "load_url", "load_jira", "execute", "select_endpoint"}
    assert forbidden.isdisjoint(dir(InternalSkillRegistry))


def test_trusted_directory_accepts_only_declarative_package_files(tmp_path):
    private, item = package()
    package_dir = tmp_path / "planning-1.2.3"
    package_dir.mkdir()
    (package_dir / "manifest.json").write_bytes(canonical_manifest(item.manifest))
    (package_dir / "content.txt").write_bytes(item.content)
    (package_dir / "signature.ed25519").write_bytes(item.signature)
    signers = TrustedSignerRegistry((TrustedSigner("signer", private.public_key()),))

    loaded = InternalSkillRegistry.from_trusted_directory(tmp_path, signers)
    assert loaded.resolve("planning", "1.2.3", "org/repo", now=NOW).text == item.text

    (package_dir / "execute.py").write_text("raise SystemExit('must never load')")
    with pytest.raises(SkillVerificationError, match="only declarative"):
        InternalSkillRegistry.from_trusted_directory(tmp_path, signers)


def test_security_timestamps_must_be_timezone_aware():
    naive = datetime(2026, 1, 1)  # noqa: DTZ001 - intentionally invalid test input
    with pytest.raises(ValueError, match="timezone-aware"):
        RevocationMetadata(naive, "incident")
    private = Ed25519PrivateKey.generate()
    with pytest.raises(ValueError, match="timezone-aware"):
        TrustedSigner("signer", private.public_key(), expires_at=naive)
