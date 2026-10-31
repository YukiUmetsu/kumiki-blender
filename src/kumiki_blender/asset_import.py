"""Source-neutral, license-aware intake for static architectural GLB models.

This module does not fetch URLs or execute Blender. The caller supplies a
reviewed local GLB and explicit rights/provenance metadata. Third-party models
cannot silently become redistributed source artifacts.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .glb_audit import AuditError, audit_glb


class AssetImportError(ValueError):
    """The source model or declared redistribution rights are not suitable."""


MAX_PROVENANCE_BYTES = 32 * 1024
MAX_GLB_BYTES = 512 * 1024 * 1024
_ASSET_ID = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_LICENSE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]{1,79}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class AssetProvenance:
    asset_id: str
    creator: str
    source_url: str
    license: str
    attribution: str
    redistribution_allowed: bool
    expected_sha256: str | None = None

    def to_manifest(self, sha256: str) -> dict[str, Any]:
        return {
            "creator": self.creator,
            "source_url": self.source_url,
            "license": self.license,
            "attribution": self.attribution,
            "redistribution_allowed": self.redistribution_allowed,
            "source_glb_sha256": sha256,
        }


def _label(value: Any, field: str, *, limit: int) -> str:
    if (not isinstance(value, str) or not 1 <= len(value.strip()) <= limit
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise AssetImportError(f"{field} must be printable nonempty text (max {limit} characters)")
    return value.strip()


def load_asset_provenance(path: str | Path) -> AssetProvenance:
    """Validate a small declarative source/license record, without I/O side effects."""
    path = Path(path)
    try:
        if path.stat().st_size > MAX_PROVENANCE_BYTES:
            raise AssetImportError("Provenance JSON exceeds 32 KiB limit")
        raw = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (OSError, UnicodeError, ValueError) as error:
        if isinstance(error, AssetImportError):
            raise
        raise AssetImportError(f"Invalid provenance JSON: {error}") from error
    if not isinstance(raw, dict):
        raise AssetImportError("Provenance must be a JSON object")
    required = {"schema_version", "asset_id", "creator", "source_url", "license", "attribution",
                "redistribution_allowed"}
    optional = {"expected_sha256"}
    if set(raw) - optional != required or set(raw) - (required | optional):
        raise AssetImportError("Provenance has missing or unexpected fields")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise AssetImportError("Provenance schema_version must be 1")
    asset_id = _label(raw["asset_id"], "asset_id", limit=96)
    if not _ASSET_ID.fullmatch(asset_id):
        raise AssetImportError("asset_id must be a lowercase hyphenated slug")
    creator = _label(raw["creator"], "creator", limit=200)
    source_url = _label(raw["source_url"], "source_url", limit=2048)
    parts = urlsplit(source_url)
    if (parts.scheme != "https" or not parts.hostname or parts.username or parts.password
            or parts.fragment or parts.query or not parts.path
            or any(char.isspace() for char in source_url)):
        raise AssetImportError("source_url must be a stable HTTPS asset permalink without credentials, queries or fragments")
    license_name = _label(raw["license"], "license", limit=80)
    if not _LICENSE.fullmatch(license_name):
        raise AssetImportError("license must be a short SPDX-style identifier")
    attribution = raw["attribution"]
    if not isinstance(attribution, str) or len(attribution) > 500 or any(ord(c) < 32 for c in attribution):
        raise AssetImportError("attribution must be printable text (max 500 characters)")
    if license_name.upper().startswith("CC-BY") and not attribution.strip():
        raise AssetImportError("CC-BY models require attribution text")
    if type(raw["redistribution_allowed"]) is not bool or not raw["redistribution_allowed"]:
        raise AssetImportError("Exporting a GLB revision requires redistribution_allowed: true and verified rights")
    expected_sha256 = raw.get("expected_sha256")
    if expected_sha256 is not None and (not isinstance(expected_sha256, str)
                                        or not _SHA256.fullmatch(expected_sha256)):
        raise AssetImportError("expected_sha256 must be a lowercase SHA-256 digest")
    return AssetProvenance(asset_id, creator, source_url, license_name, attribution.strip(),
                           True, expected_sha256)


def inspect_source_glb(path: str | Path, provenance: AssetProvenance | None = None) -> dict[str, Any]:
    """Fail closed on malformed, animated, oversized or unexpectedly changed GLBs."""
    try:
        source = Path(path).resolve(strict=True)
    except OSError as error:
        raise AssetImportError(f"Input GLB is unavailable: {error}") from error
    if not source.is_file() or source.suffix.lower() != ".glb":
        raise AssetImportError("Input must be a local .glb file")
    size = source.stat().st_size
    if size > MAX_GLB_BYTES:
        raise AssetImportError("Input GLB exceeds the 512 MiB intake limit")
    try:
        glb_report = audit_glb(source)
        with source.open("rb") as stream:
            stream.seek(12)
            chunk_length, kind = struct.unpack("<I4s", stream.read(8))
            if kind != b"JSON":
                raise AssetImportError("GLB must begin with a JSON chunk")
            doc = json.loads(stream.read(chunk_length).decode("utf-8"))
    except (AuditError, OSError, ValueError, UnicodeError, struct.error) as error:
        if isinstance(error, AssetImportError):
            raise
        raise AssetImportError(f"Invalid input GLB: {error}") from error
    if doc.get("animations") or doc.get("skins"):
        raise AssetImportError("Animated or rigged GLBs are not supported by the static asset importer")
    if not any(isinstance(node, dict) and type(node.get("mesh")) is int for node in doc.get("nodes", [])):
        raise AssetImportError("Input GLB has no instanced mesh nodes")
    materials = doc.get("materials", [])
    if (not isinstance(materials, list) or any(not isinstance(material, dict)
                                                or not isinstance(material.get("name", ""), str)
                                                for material in materials)):
        raise AssetImportError("GLB material definitions must be objects with optional text names")
    material_names = [material.get("name", "") for material in materials]
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    sha256 = digest.hexdigest()
    if provenance and provenance.expected_sha256 and provenance.expected_sha256 != sha256:
        raise AssetImportError("Input GLB SHA-256 differs from the provenance pin")
    return {"input_bytes": size, "input_sha256": sha256, "glb_audit": glb_report,
            "material_names": material_names,
            "static_mesh_only": True}
