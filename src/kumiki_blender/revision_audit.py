"""Verify a generated Blender asset revision as a portable, complete package."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .export_audit import ExportAuditError, audit_export_files
from .glb_audit import AuditError, audit_glb


class RevisionAuditError(ValueError):
    """A revision cannot be trusted as a complete portable asset."""


def audit_revision(root: str | Path) -> dict[str, Any]:
    """Cross-check the manifest, file signatures, lengths and GLB metadata.

    This checks portability and package consistency, not visual quality or
    architectural accuracy. It never evaluates code from the manifest.
    """
    directory = Path(root).resolve()
    manifest_path = directory / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RevisionAuditError("Missing or unsafe manifest.json")
    try:
        if manifest_path.stat().st_size > 1024 * 1024:
            raise RevisionAuditError("Manifest exceeds 1 MiB limit")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RevisionAuditError(f"Malformed or unreadable manifest: {error}") from error
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise RevisionAuditError("Manifest must declare schema_version 1")
    asset_id = manifest.get("asset_id")
    if not isinstance(asset_id, str) or not re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", asset_id):
        raise RevisionAuditError("Manifest has an invalid asset_id")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise RevisionAuditError("Manifest files must be an object")
    source, model, preview, views = (files.get("source"), files.get("model"),
                                     files.get("preview"), files.get("quality_views"))
    if (not isinstance(source, str) or not source.endswith(".blend")
            or not isinstance(model, str) or not model.endswith(".glb")
            or not isinstance(preview, str) or not preview.endswith(".png")
            or not isinstance(views, list) or not 1 <= len(views) <= 16
            or any(not isinstance(view, str) or not view.endswith(".png") for view in views)
            or views[0] != preview):
        raise RevisionAuditError("Manifest has invalid source/model/preview entries")
    expected_names = [source, model, *views]
    if len(set(expected_names)) != len(expected_names):
        raise RevisionAuditError("Manifest repeats an output path")
    try:
        sizes = audit_export_files(directory, expected_names)
        glb = audit_glb(directory / model)
    except (ExportAuditError, AuditError) as error:
        raise RevisionAuditError(f"Invalid artifact: {error}") from error
    if glb["semantic_nodes"] == 0:
        raise RevisionAuditError("GLB has no semantic part tags")
    recorded_sizes = manifest.get("output_bytes")
    if not isinstance(recorded_sizes, dict) or set(recorded_sizes) != set(sizes):
        raise RevisionAuditError("Manifest output_bytes does not enumerate the expected files")
    for filename, actual_size in sizes.items():
        expected_size = recorded_sizes[filename]
        if type(expected_size) is not int or expected_size != actual_size:
            raise RevisionAuditError(f"Output size mismatch: {filename}")
    if manifest.get("glb_audit") != glb:
        raise RevisionAuditError("GLB audit in manifest does not match the exported model")
    named_parts = manifest.get("named_parts")
    if (not isinstance(named_parts, dict) or not named_parts
            or any(not isinstance(key, str) or type(value) is not int or value < 0
                   for key, value in named_parts.items())
            or sum(named_parts.values()) == 0):
        raise RevisionAuditError("Manifest is missing valid named part counts")
    return {"verified": True, "asset_id": asset_id, "output_count": len(sizes),
            "output_bytes": sizes, "glb_audit": glb}
