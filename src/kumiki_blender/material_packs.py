"""Validated local PBR texture packs for reusable architectural materials.

No automatic downloads, guessed licenses, hidden absolute paths, or Blender
dependency. A caller must supply files and explicitly confirm export rights.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .asset_import import AssetImportError


MAX_RECIPE_BYTES = 64 * 1024
MAX_MAP_BYTES = 32 * 1024 * 1024
_MAP_NAMES = {"base_color", "roughness", "normal", "metallic"}
_REQUIRED_MAPS = {"base_color", "roughness", "normal"}
_ID = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_LICENSE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]{1,79}\Z")


@dataclass(frozen=True)
class TextureMap:
    name: str
    relative_path: str
    file_path: Path
    sha256: str
    size_bytes: int

    def to_manifest(self) -> dict[str, Any]:
        return {"file": self.relative_path, "sha256": self.sha256, "bytes": self.size_bytes}


@dataclass(frozen=True)
class TexturePack:
    pack_id: str
    creator: str
    source_url: str
    license: str
    attribution: str
    tile_size_m: float
    metallic: float
    normal_strength: float
    maps: tuple[TextureMap, ...]

    def file_for(self, channel: str) -> TextureMap | None:
        return next((entry for entry in self.maps if entry.name == channel), None)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "id": self.pack_id,
            "creator": self.creator,
            "source_url": self.source_url,
            "license": self.license,
            "attribution": self.attribution,
            "redistribution_allowed": True,
            "tile_size_m": self.tile_size_m,
            "metallic": self.metallic,
            "normal_strength": self.normal_strength,
            "maps": {entry.name: entry.to_manifest() for entry in self.maps},
        }


@dataclass(frozen=True)
class TextureAssignment:
    target: str
    pack: TexturePack

    def to_manifest(self) -> dict[str, Any]:
        return {"target": self.target, "pack": self.pack.to_manifest()}


def _printable(value: Any, name: str, limit: int, *, allow_empty: bool = False) -> str:
    if (not isinstance(value, str) or len(value) > limit
            or (not allow_empty and (not value or value != value.strip()))
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise AssetImportError(f"Texture pack {name} must be printable text (max {limit} chars)")
    return value


def _number(value: Any, name: str, minimum: float, maximum: float) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not minimum <= value <= maximum):
        raise AssetImportError(f"Texture pack {name} must be a finite number in [{minimum}, {maximum}]")
    return float(value)


def _validate_image(path: Path, name: str) -> tuple[str, int]:
    size = path.stat().st_size
    if size < 16 or size > MAX_MAP_BYTES:
        raise AssetImportError(f"Texture {name} must be 16 bytes–32 MiB")
    with path.open("rb") as stream:
        header = stream.read(32)
        if path.suffix.lower() == ".png":
            if (not header.startswith(b"\x89PNG\r\n\x1a\n") or header[12:16] != b"IHDR"
                    or int.from_bytes(header[16:20], "big") == 0
                    or int.from_bytes(header[20:24], "big") == 0):
                raise AssetImportError(f"Texture {name} is not a valid PNG header")
        elif path.suffix.lower() in (".jpg", ".jpeg"):
            if not header.startswith(b"\xff\xd8\xff"):
                raise AssetImportError(f"Texture {name} is not a JPEG")
        else:
            raise AssetImportError(f"Texture {name} must be PNG or JPEG for portable glTF")
        digest = hashlib.sha256()
        digest.update(header)
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest(), size


def _load_map(directory: Path, name: str, entry: Any, expected: Any) -> TextureMap:
    relative = _printable(entry, f"maps.{name}", 240)
    candidate = Path(relative)
    if (candidate.is_absolute() or "\\" in relative or not candidate.parts
            or any(part in (".", "..") for part in candidate.parts)):
        raise AssetImportError(f"Texture maps.{name} must be a relative path inside the recipe directory")
    current = directory
    for part in candidate.parts:
        current = current / part
        if current.is_symlink():
            raise AssetImportError(f"Texture maps.{name} may not follow a symlink")
    try:
        path = current.resolve(strict=True)
        if not path.is_file() or not path.is_relative_to(directory):
            raise AssetImportError(f"Texture maps.{name} is outside the recipe directory")
        digest, size = _validate_image(path, name)
    except (OSError, ValueError) as error:
        if isinstance(error, AssetImportError):
            raise
        raise AssetImportError(f"Cannot read texture maps.{name}: {error}") from error
    if expected is not None and (not isinstance(expected, str) or not _DIGEST.fullmatch(expected)
                                 or digest != expected):
        raise AssetImportError(f"Texture maps.{name} SHA-256 pin mismatch")
    return TextureMap(name, relative, path, digest, size)


def load_texture_packs(path: str | Path) -> tuple[TextureAssignment, ...]:
    """Load a strict, rights-declared recipe and hash every source texture."""
    recipe_file = Path(path)
    try:
        if recipe_file.stat().st_size > MAX_RECIPE_BYTES:
            raise AssetImportError("Texture pack recipe exceeds 64 KiB")
        raw = json.loads(recipe_file.read_text(encoding="utf-8"),
                         parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (OSError, UnicodeError, ValueError) as error:
        if isinstance(error, AssetImportError):
            raise
        raise AssetImportError(f"Invalid texture pack recipe: {error}") from error
    if (not isinstance(raw, dict) or set(raw) != {"schema_version", "packs", "assignments"}
            or type(raw["schema_version"]) is not int or raw["schema_version"] != 1):
        raise AssetImportError("Texture pack recipe requires schema_version 1, packs and assignments")
    packs_raw, assignments_raw = raw["packs"], raw["assignments"]
    if (not isinstance(packs_raw, list) or not 1 <= len(packs_raw) <= 16
            or not isinstance(assignments_raw, list) or not 1 <= len(assignments_raw) <= 32):
        raise AssetImportError("Texture pack recipe needs 1–16 packs and 1–32 assignments")
    directory = recipe_file.resolve().parent
    packs: dict[str, TexturePack] = {}
    for item in packs_raw:
        required = {"id", "creator", "source_url", "license", "attribution", "redistribution_allowed", "maps"}
        optional = {"sha256", "tile_size_m", "metallic", "normal_strength", "normal_convention"}
        if not isinstance(item, dict) or not required <= set(item) or set(item) - (required | optional):
            raise AssetImportError("Texture pack has missing or unexpected fields")
        pack_id = _printable(item["id"], "id", 64)
        if not _ID.fullmatch(pack_id) or pack_id in packs:
            raise AssetImportError(f"Texture pack id must be a unique hyphenated slug: {pack_id}")
        if type(item["redistribution_allowed"]) is not bool or not item["redistribution_allowed"]:
            raise AssetImportError(f"Texture pack {pack_id} requires verified redistribution_allowed: true")
        creator = _printable(item["creator"], "creator", 200)
        url = _printable(item["source_url"], "source_url", 2048)
        parts = urlsplit(url)
        if (parts.scheme != "https" or not parts.hostname or parts.username or parts.password
                or parts.query or parts.fragment or not parts.path or any(c.isspace() for c in url)):
            raise AssetImportError(f"Texture pack {pack_id} needs a stable HTTPS source permalink")
        license_name = _printable(item["license"], "license", 80)
        if not _LICENSE.fullmatch(license_name):
            raise AssetImportError(f"Texture pack {pack_id} needs a short SPDX-style license identifier")
        attribution = _printable(item["attribution"], "attribution", 500, allow_empty=True)
        if license_name.upper().startswith("CC-BY") and not attribution.strip():
            raise AssetImportError(f"Texture pack {pack_id} requires CC-BY attribution")
        if item.get("normal_convention", "opengl") != "opengl":
            raise AssetImportError("Texture pack normals must use OpenGL (+Y) convention")
        maps_raw, hashes = item["maps"], item.get("sha256", {})
        if (not isinstance(maps_raw, dict) or not _REQUIRED_MAPS <= set(maps_raw)
                or set(maps_raw) - _MAP_NAMES or not isinstance(hashes, dict)
                or set(hashes) - set(maps_raw)):
            raise AssetImportError(f"Texture pack {pack_id} needs base_color, roughness, normal PNG/JPEG maps")
        maps = tuple(_load_map(directory, name, file, hashes.get(name))
                     for name, file in sorted(maps_raw.items()))
        packs[pack_id] = TexturePack(
            pack_id, creator, url, license_name, attribution,
            _number(item.get("tile_size_m", 1.5), "tile_size_m", 0.05, 100.0),
            _number(item.get("metallic", 0.0), "metallic", 0.0, 1.0),
            _number(item.get("normal_strength", 1.0), "normal_strength", 0.0, 2.0),
            maps,
        )
    results: list[TextureAssignment] = []
    used_targets: set[str] = set()
    used_packs: set[str] = set()
    for item in assignments_raw:
        if not isinstance(item, dict) or set(item) != {"target", "pack"}:
            raise AssetImportError("Texture assignments require exact target and pack fields")
        target = _printable(item["target"], "assignment.target", 128)
        pack_id = _printable(item["pack"], "assignment.pack", 64)
        if target in used_targets or pack_id not in packs:
            raise AssetImportError(f"Duplicate target or missing pack for texture assignment: {target}")
        used_targets.add(target)
        used_packs.add(pack_id)
        results.append(TextureAssignment(target, packs[pack_id]))
    if used_packs != set(packs):
        raise AssetImportError("Texture pack recipe contains unassigned packs")
    return tuple(results)


def verify_texture_targets(names: list[str], assignments: tuple[TextureAssignment, ...]) -> None:
    """Fail closed before Blender if a target is missing or ambiguous."""
    for item in assignments:
        count = names.count(item.target)
        if count != 1:
            raise AssetImportError(f"Texture pack target '{item.target}' matches {count} materials; expected one")
