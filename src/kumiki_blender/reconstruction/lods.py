"""Source-rights-aware high-resolution statue-scan LOD and PBR bake contract."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kumiki_blender.asset_import import AssetImportError, AssetProvenance, inspect_source_glb


@dataclass(frozen=True)
class ScanLODSettings:
    target_triangles: int = 100_000
    map_size: int = 1024
    ray_distance_m: float = 0.05
    render_previews: bool = False

    def validate(self) -> "ScanLODSettings":
        if type(self.target_triangles) is not int or not 300 <= self.target_triangles <= 5_000_000:
            raise AssetImportError("target_triangles must be 300–5,000,000")
        if self.map_size not in (256, 512, 1024, 2048, 4096) or type(self.map_size) is not int:
            raise AssetImportError("map_size must be 256, 512, 1024, 2048 or 4096")
        if (type(self.ray_distance_m) not in (int, float) or not math.isfinite(self.ray_distance_m)
                or not 0.0001 <= self.ray_distance_m <= 2.0):
            raise AssetImportError("ray_distance_m must be finite in [0.0001, 2] metres")
        if type(self.render_previews) is not bool:
            raise AssetImportError("render_previews must be boolean")
        return self


def plan_scan_lods(source: str | Path, provenance: AssetProvenance,
                   output: str | Path, settings: ScanLODSettings) -> dict[str, Any]:
    """Preflight rights and expected work without modifying any input/output."""
    settings.validate()
    destination = Path(output).resolve()
    if Path(output).is_symlink() or destination.exists():
        raise AssetImportError(f"LOD output already exists: {destination}")
    source_path = Path(source).resolve()
    inspected = inspect_source_glb(source_path, provenance)
    return {
        "schema_version": 1, "asset_id": provenance.asset_id,
        "source_glb": str(source_path), "source_glb_sha256": inspected["input_sha256"],
        "input_bytes": inspected["input_bytes"],
        "source_provenance": provenance.to_manifest(inspected["input_sha256"]),
        "target_triangles": settings.target_triangles,
        "map_size": settings.map_size,
        "ray_distance_m": settings.ray_distance_m,
        "render_previews": settings.render_previews,
        "output": str(destination),
        "status": "preflight-only / bake-and-artistic-review-required",
        "limitations": "Source geometry and derivative textures require verified redistribution rights; LOD normal baking preserves some relief but cannot invent missing anatomy.",
    }
