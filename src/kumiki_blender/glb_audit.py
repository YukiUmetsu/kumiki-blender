"""Fast, dependency-free GLB envelope and portability checks.

This is not a full glTF accessor/mesh validator or visual acceptance test.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any


class AuditError(ValueError):
    """A generated artifact is missing, malformed, or not self-contained."""


def audit_glb(path: str | Path, *, require_pbr_textures: bool = False) -> dict[str, Any]:
    """Check the GLB envelope and optionally require portable PBR texture paths."""
    source = Path(path)
    try:
        size = source.stat().st_size
        with source.open("rb") as stream:
            header = stream.read(12)
            if len(header) != 12:
                raise AuditError("Not a GLB: incomplete header")
            magic, version, declared_length = struct.unpack("<4sII", header)
            if magic != b"glTF" or version != 2:
                raise AuditError("Not a GLB 2.0 file")
            if declared_length != size:
                raise AuditError("GLB declared length does not match the file")
            chunk_header = stream.read(8)
            if len(chunk_header) != 8:
                raise AuditError("GLB JSON chunk header is incomplete")
            chunk_size, kind = struct.unpack("<I4s", chunk_header)
            if kind != b"JSON" or chunk_size > 16 * 1024 * 1024 or 20 + chunk_size > size:
                raise AuditError("GLB JSON chunk is invalid or too large")
            payload = stream.read(chunk_size)
            if len(payload) != chunk_size:
                raise AuditError("GLB JSON chunk is truncated")
    except OSError as error:
        raise AuditError(f"Cannot read GLB {source}: {error}") from error

    try:
        data = json.loads(payload.decode("utf-8").rstrip(" \t\r\n\x00"))
    except (ValueError, UnicodeError) as error:
        raise AuditError(f"GLB JSON is malformed: {error}") from error
    if not isinstance(data, dict) or not isinstance(data.get("asset"), dict) or data["asset"].get("version") != "2.0":
        raise AuditError("GLB asset metadata must declare glTF 2.0")
    meshes, nodes = data.get("meshes"), data.get("nodes")
    if not isinstance(meshes, list) or not meshes:
        raise AuditError("GLB contains no mesh definitions")
    if not isinstance(nodes, list):
        raise AuditError("GLB has no node list")
    if any(not isinstance(mesh, dict) or not isinstance(mesh.get("primitives"), list)
           or not mesh["primitives"] or any(not isinstance(p, dict) for p in mesh["primitives"])
           for mesh in meshes):
        raise AuditError("GLB contains a mesh with invalid primitives")

    images, buffers = data.get("images", []), data.get("buffers", [])
    materials, textures = data.get("materials", []), data.get("textures", [])
    if (not isinstance(images, list) or not isinstance(buffers, list)
            or not isinstance(materials, list) or not isinstance(textures, list)):
        raise AuditError("GLB image/buffer/material/texture collections must be arrays")
    dependencies = [*images, *buffers]
    if any(not isinstance(item, dict) for item in dependencies):
        raise AuditError("GLB has invalid image/buffer metadata")
    external = [item["uri"] for item in dependencies if "uri" in item and
                (not isinstance(item["uri"], str) or not item["uri"].startswith("data:"))]
    if external:
        raise AuditError(f"GLB has {len(external)} external image/buffer references; export portable embedded assets")

    if require_pbr_textures:
        textured_materials = (
            material for material in materials if isinstance(material, dict)
            and isinstance(material.get("pbrMetallicRoughness"), dict)
            and isinstance(material["pbrMetallicRoughness"].get("baseColorTexture"), dict)
            and isinstance(material["pbrMetallicRoughness"].get("metallicRoughnessTexture"), dict)
            and isinstance(material.get("normalTexture"), dict)
        )
        if (not images or not textures or not any(textured_materials)
                or not any(isinstance(primitive.get("attributes"), dict)
                           and "TEXCOORD_0" in primitive["attributes"]
                           for mesh in meshes for primitive in mesh["primitives"])):
            raise AuditError("GLB is missing embedded albedo/roughness/normal PBR textures or UV0")

    semantic_nodes = sum(
        1 for node in nodes if isinstance(node, dict) and isinstance(node.get("extras"), dict)
        and isinstance(node["extras"].get("kumiki_part"), str)
    )
    return {
        "meshes": len(meshes),
        "nodes": len(nodes),
        "materials": len(materials),
        "semantic_nodes": semantic_nodes,
        "external_references": 0,
        "bytes": size,
    }
