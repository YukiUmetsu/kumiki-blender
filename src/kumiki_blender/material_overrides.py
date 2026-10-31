"""Strict, source-independent PBR restyling recipes for reviewed GLB assets.

Only explicitly named materials are replaced. The Blender adapter performs
the actual shader changes; parsing and source-name preflight need no Blender.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .asset_import import AssetImportError
from .surfaces import PBRSurface, SurfaceError, get_surface


MAX_OVERRIDE_BYTES = 32 * 1024


@dataclass(frozen=True)
class MaterialOverride:
    material_name: str
    preset: str
    surface: PBRSurface

    def to_manifest(self) -> dict[str, Any]:
        return {
            "source_material": self.material_name,
            "preset": self.preset,
            "color": self.surface.color,
            "roughness": self.surface.roughness,
        }


def load_material_overrides(path: str | Path) -> tuple[MaterialOverride, ...]:
    """Load an explicit set of named overrides, rejecting typos and duplicates."""
    try:
        source = Path(path)
        if source.stat().st_size > MAX_OVERRIDE_BYTES:
            raise AssetImportError("Material overrides exceed 32 KiB")
        raw = json.loads(source.read_text(encoding="utf-8"),
                         parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (OSError, UnicodeError, ValueError) as error:
        if isinstance(error, AssetImportError):
            raise
        raise AssetImportError(f"Invalid material overrides JSON: {error}") from error
    if (not isinstance(raw, dict) or set(raw) != {"schema_version", "overrides"}
            or type(raw["schema_version"]) is not int or raw["schema_version"] != 1):
        raise AssetImportError("Material overrides require schema_version 1 and overrides")
    entries = raw["overrides"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= 32:
        raise AssetImportError("Material overrides must contain 1–32 entries")
    result: list[MaterialOverride] = []
    names: set[str] = set()
    for index, entry in enumerate(entries):
        if (not isinstance(entry, dict)
                or not {"material_name", "preset"} <= set(entry)
                or set(entry) - {"material_name", "preset", "color", "roughness"}):
            raise AssetImportError(f"Material override #{index} has missing or unexpected fields")
        name, preset = entry["material_name"], entry["preset"]
        if (not isinstance(name, str) or not 1 <= len(name.strip()) <= 128
                or name != name.strip() or any(ord(c) < 32 or ord(c) == 127 for c in name)):
            raise AssetImportError(f"Material override #{index} needs a printable exact material_name")
        if name in names:
            raise AssetImportError(f"Duplicate material override: {name}")
        if not isinstance(preset, str):
            raise AssetImportError(f"Material override #{index} needs a preset name")
        try:
            surface = get_surface(preset, color=entry.get("color"), roughness=entry.get("roughness"))
        except SurfaceError as error:
            raise AssetImportError(f"Invalid override for {name}: {error}") from error
        names.add(name)
        result.append(MaterialOverride(name, preset, surface))
    return tuple(result)


def verify_material_overrides(material_names: list[str], overrides: tuple[MaterialOverride, ...]) -> None:
    """Require an unambiguous exact match in source GLB, before Blender work."""
    for entry in overrides:
        matches = sum(name == entry.material_name for name in material_names)
        if matches != 1:
            raise AssetImportError(
                f"Material override '{entry.material_name}' matches {matches} GLB materials; expected exactly one"
            )
