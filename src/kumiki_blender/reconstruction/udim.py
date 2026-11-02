"""Private UDIM packaging of verified multi-atlas OBJ photo reconstructions.

Per-material original UVs are copied into numbered tiles, duplicating shared
texture-coordinate indices as needed. Geometry and input assets are untouched.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .dense import _json, _sha256
from .references import ReferenceError


def remap_obj_uv_for_udim(source: str,
                          material_tiles: dict[str, int]) -> tuple[str, dict[str, int]]:
    """Rewrite material-specific texture *indices*, not mesh vertex positions."""
    if not material_tiles or len(material_tiles) > 100:
        raise ReferenceError("UDIM conversion needs 1–100 material tile assignments")
    for tile in material_tiles.values():
        if type(tile) is not int or not 1001 <= tile <= 1999:
            raise ReferenceError("UDIM tile must be a supported 1001+ integer")
    lines = source.splitlines()
    original_uv: list[tuple[float, float, tuple[str, ...]]] = []
    for line in lines:
        if line.startswith("vt "):
            args = line.split()[1:]
            if not 2 <= len(args) <= 3:
                raise ReferenceError("UDIM source UV needs two or three coordinates")
            try:
                u, v = float(args[0]), float(args[1])
            except ValueError as error:
                raise ReferenceError("UDIM source UV must be numeric") from error
            if (not math.isfinite(u) or not math.isfinite(v)
                    or not -.00001 <= u <= 1.00001 or not -.00001 <= v <= 1.00001):
                raise ReferenceError("UDIM source UV is outside original 0–1 atlas tile")
            original_uv.append((u, v, tuple(args[2:])))
    if not original_uv:
        raise ReferenceError("UDIM source OBJ has no UV texture coordinates")
    if len(original_uv) > 5_000_000:
        raise ReferenceError("UDIM UV index budget exceeded")
    # Every source OBJ created by OpenMVS declares its UVs before the first
    # polygon. Refuse unusual interleaving instead of corrupting relative IDs.
    first_face = next((i for i, line in enumerate(lines) if line.startswith("f ")), None)
    if first_face is None or any(line.startswith("vt ") for line in lines[first_face:]):
        raise ReferenceError("UDIM requires UV coordinates declared before all faces")
    remapped: dict[tuple[int, int], int] = {}
    new_uv: list[str] = []
    output: list[str] = []
    active_material: str | None = None
    faces = 0
    for line in lines:
        if line.startswith("mtllib "):
            output.append("mtllib udim.mtl")
        elif line.startswith("usemtl "):
            active_material = line[7:].strip()
            if active_material not in material_tiles:
                raise ReferenceError("UDIM OBJ uses an unknown material")
            output.append(line)
        elif line.startswith("f "):
            if active_material is None:
                raise ReferenceError("UDIM face has no assigned material")
            tile = material_tiles[active_material]
            col, row = (tile - 1001) % 10, (tile - 1001) // 10
            tokens = []
            for token in line.split()[1:]:
                parts = token.split("/")
                if len(parts) not in (2, 3) or not parts[1]:
                    raise ReferenceError("UDIM face is missing a UV index")
                try:
                    old_idx = int(parts[1])
                except ValueError as error:
                    raise ReferenceError("UDIM face has invalid UV index") from error
                # Negative OBJ indices are relative to the original UV table,
                # never to newly inserted UDIM entries.
                idx = old_idx - 1 if old_idx > 0 else len(original_uv) + old_idx
                if not 0 <= idx < len(original_uv):
                    raise ReferenceError("UDIM face UV index is out of range")
                key = (idx, tile)
                if key not in remapped:
                    u, v, extras = original_uv[idx]
                    # Positive indices point after the entire original vt block.
                    remapped[key] = len(original_uv) + len(new_uv) + 1
                    suffix = (" " + " ".join(extras)) if extras else ""
                    new_uv.append(f"vt {u + col:.8g} {v + row:.8g}{suffix}")
                parts[1] = str(remapped[key])
                tokens.append("/".join(parts))
            if len(tokens) < 3:
                raise ReferenceError("UDIM polygon must have at least three vertices")
            faces += 1
            output.append("f " + " ".join(tokens))
        else:
            output.append(line)
    if not faces:
        raise ReferenceError("UDIM source OBJ has no material-assigned faces")
    output[first_face:first_face] = new_uv
    return "\n".join(output) + "\n", {"remapped_faces": faces,
                                            "new_texture_coordinates": len(new_uv),
                                            "source_texture_coordinates": len(original_uv)}


def package_finished_udim(texture_finish: str | Path, output: str | Path) -> dict[str, Any]:
    """Create a separate, hashed 1001+ tile package for editable Blender use."""
    from PIL import Image
    from .blender_dense import _finished_texture_plan
    from .uv_coverage import _material_textures

    finish, target = Path(texture_finish).resolve(), Path(output).resolve()
    if (Path(texture_finish).is_symlink() or not finish.is_dir()
            or Path(output).is_symlink() or target.exists()
            or target == finish or target.is_relative_to(finish)):
        raise ReferenceError("UDIM packaging requires an existing finished revision and unused external output")
    record = _json(finish / "texture-finish-report.json")
    root = Path(record["source_dense_revision"])
    if root.is_symlink() or not root.is_dir():
        raise ReferenceError("UDIM original dense revision is missing")
    dense_record = _json(root / "dense-report.json")
    plan = _finished_texture_plan(root, dense_record, finish, max_faces=10_000_000)
    atlas_paths = {Path(item["path"]).name: Path(item["path"])
                   for item in plan["texture_atlases"]}
    materials = _material_textures(Path(plan["source_mtl"]), set(atlas_paths))
    tile_by_atlas = {name: 1001 + i for i, name in enumerate(sorted(atlas_paths))}
    material_tiles = {name: tile_by_atlas[atlas] for name, atlas in materials.items()}
    source_obj = Path(plan["source_obj"]).read_text(encoding="utf-8")
    converted, diagnostics = remap_obj_uv_for_udim(source_obj, material_tiles)
    original_mtl = Path(plan["source_mtl"]).read_text(encoding="utf-8")
    # Every MTL page is bound to a material above. Replace only those map_Kd
    # paths while retaining all other material parameters.
    material_lines = ["map_Kd albedo.<UDIM>.png" if line.strip().startswith("map_Kd ")
                      else line for line in original_mtl.splitlines()]
    by_atlas = {entry["atlas"]: entry for entry in plan["surface_maps"]}
    if any(entry["height"] is not None for entry in by_atlas.values()) and any(
            entry["height"] is None for entry in by_atlas.values()):
        raise ReferenceError("UDIM height tiles must be supplied for every atlas, not a subset")
    target.mkdir(parents=True, exist_ok=False)
    (target / "udim.obj").write_text(converted, encoding="utf-8")
    (target / "udim.mtl").write_text("\n".join(material_lines) + "\n", encoding="utf-8")
    report: dict[str, Any] = {
        "schema_version": 1, "status": "private-udim-artistic-review-required",
        "distribution_allowed": False, "source_texture_finish_revision": str(finish),
        "source_texture_finish_report_sha256": _sha256(finish / "texture-finish-report.json"),
        "obj": {"path": "udim.obj", "sha256": _sha256(target / "udim.obj")},
        "mtl": {"path": "udim.mtl", "sha256": _sha256(target / "udim.mtl")},
        "material_tiles": material_tiles, "uv_remap": diagnostics, "tiles": [],
        "limitations": ("Blender supports tiled images, but glTF has no native UDIM; "
                        "re-export via Blender's split-material handling after visual review. "
                        "Height tiles are Blender/Cycles-only, not native GLB displacement."),
    }
    for atlas_name, tile in tile_by_atlas.items():
        page: dict[str, Any] = {"source_atlas": atlas_name, "tile": tile}
        if by_atlas.get(atlas_name, {}).get("height") is not None:
            page["displacement_scale_m"] = by_atlas[atlas_name]["displacement_scale_m"]
        for channel in ("albedo", "roughness", "height"):
            path = (atlas_paths[atlas_name] if channel == "albedo" else
                    by_atlas.get(atlas_name, {}).get(channel, {}).get("path")
                    if by_atlas.get(atlas_name, {}).get(channel) is not None else None)
            if path is None:
                continue
            destination = target / f"{channel}.{tile}.png"
            with Image.open(path) as image:
                image.save(destination, "PNG")
            page[channel] = {"path": destination.name, "sha256": _sha256(destination)}
        report["tiles"].append(page)
    (target / "udim-report.json").write_text(json.dumps(report, indent=2) + "\n",
                                               encoding="utf-8")
    return report
