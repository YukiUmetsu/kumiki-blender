"""Bake signed scan-to-UV geometry offsets; never infer height from albedo.

The high-density PLY and textured OBJ must be registered in the same COLMAP
coordinate frame. Their absolute scale is unknown until an artist supplies a
measured metres-per-unit conversion. Output height uses a signed 0.5 midlevel,
with a separate coverage mask and explicit candidate-only provenance.
"""

from __future__ import annotations

import json
import math
import struct
import tempfile
import zlib
from pathlib import Path
from typing import Any

from .blender_dense import _png_metadata, plan_dense_blender_review
from .dense import _sha256
from .photo_uv_projection import _read_obj_triangles
from .references import ReferenceError


def plan_scan_displacement(dense_revision: str | Path, output: str | Path, *,
                           meters_per_unit: float, max_offset_m: float = .05,
                           max_side: int = 2048,
                           max_mesh_faces: int = 5_000_000,
                           max_uv_triangles: int = 500_000) -> dict[str, Any]:
    """Check exact geometry pins, units, output isolation and raster budgets."""
    original = Path(dense_revision).resolve()
    target = Path(output).resolve()
    if (original.is_relative_to(target) or target.is_relative_to(original)
            or Path(output).is_symlink()):
        raise ReferenceError("Scan displacement output must be outside the dense revision")
    if (type(meters_per_unit) not in (float, int) or not math.isfinite(meters_per_unit)
            or not 1e-8 <= meters_per_unit <= 1000):
        raise ReferenceError("meters_per_unit requires a finite measured conversion in [1e-8, 1000]")
    if (type(max_offset_m) not in (float, int) or not math.isfinite(max_offset_m)
            or not 1e-5 <= max_offset_m <= .25):
        raise ReferenceError("max_offset_m must be in [0.00001, 0.25] metres")
    if type(max_side) is not int or not 128 <= max_side <= 4096:
        raise ReferenceError("max_side must be in [128, 4096]")
    if type(max_uv_triangles) is not int or not 100 <= max_uv_triangles <= 2_000_000:
        raise ReferenceError("max_uv_triangles must be in [100, 2000000]")
    plan = plan_dense_blender_review(original, target, textured=True,
                                     max_mesh_faces=max_mesh_faces)
    atlas_names = {Path(entry["path"]).name for entry in plan["texture_atlases"]}
    triangles = _read_obj_triangles(Path(plan["source_obj"]),
                                    Path(plan["source_mtl"]), atlas_names, max_uv_triangles)
    sizes = {}
    total = 0
    for entry in plan["texture_atlases"]:
        path = Path(entry["path"])
        if path.suffix.lower() != ".png":
            raise ReferenceError("Measured scan displacement requires PNG diffuse atlases")
        (width, height), _depth, _color = _png_metadata(path)
        if width * height > 67_108_864:
            raise ReferenceError("Scan displacement atlas exceeds 64 million texels")
        total += width * height
        factor = min(1, max_side / max(width, height))
        sizes[path.name] = {"original": [width, height],
                            "sampled": [max(1, round(width * factor)),
                                        max(1, round(height * factor))]}
    if total > 134_217_728:
        raise ReferenceError("Scan displacement output exceeds 128 million texels")
    return {**plan, "uv_triangle_count": len(triangles), "atlas_dimensions": sizes,
            "meters_per_unit": float(meters_per_unit),
            "max_offset_m": float(max_offset_m), "max_side": max_side,
            "max_uv_triangles": max_uv_triangles,
            "displacement_scale_m": 2 * float(max_offset_m),
            "status": "preflight-only / Blender-scan-bake-required"}


def _png_chunk(kind: bytes, body: bytes) -> bytes:
    return (struct.pack(">I", len(body)) + kind + body +
            struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF))


def _write_gray_png(path: Path, data: Any, *, bits: int, np: Any) -> None:
    """Write genuine 16-bit grayscale height or 8-bit confidence without PIL."""
    if bits not in (8, 16):
        raise ValueError("PNG grayscale depth must be 8 or 16")
    height, width = data.shape
    if bits == 16:
        values = np.asarray(data, dtype=">u2")
    else:
        values = np.asarray(data, dtype=np.uint8)
    rows = b"".join(b"\x00" + values[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, bits, 0, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", header) +
                     _png_chunk(b"IDAT", zlib.compress(rows, 6)) + _png_chunk(b"IEND", b""))


def _sample_triangle(triangle: Any, size: tuple[int, int], bvh: Any,
                     limit_units: float, *, np: Any, Vector: Any) -> Any:
    """Yield (x, y, signed offset or None); retain missing-hit evidence."""
    xyz = np.asarray(triangle.xyz, dtype=np.float64)
    normal = np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])
    magnitude = float(np.linalg.norm(normal))
    if magnitude <= 1e-12:
        return
    normal /= magnitude
    nvec = Vector(normal)
    uv = np.asarray([(u * size[0], (1 - v) * size[1]) for u, v in triangle.uv])
    from .photo_uv_projection import _raster_cells

    for ys, xs, weights in _raster_cells(uv, size, np, tile_side=32):
        points = weights @ xyz
        for y, x, point in zip(ys, xs, points):
            origin = Vector(point)
            hits = []
            # Cast from either side, then keep the *nearest* intersection.
            # This avoids choosing a back layer of a statue by accident.
            for offset, direction in ((-nvec, nvec), (nvec, -nvec)):
                hit, high_normal, _face, _distance = bvh.ray_cast(
                    origin + offset * limit_units, direction, 2 * limit_units)
                if hit is None or high_normal is None:
                    continue
                signed = (hit - origin).dot(nvec)
                if (abs(signed) <= limit_units + 1e-8
                        and abs(high_normal.dot(nvec)) >= .35):
                    hits.append(signed)
            if hits:
                yield int(x), int(y), min(hits, key=abs)
            else:
                yield int(x), int(y), None


def bake_scanned_displacement(bpy: Any, dense_revision: str | Path, output: str | Path,
                              *, meters_per_unit: float, max_offset_m: float = .05,
                              max_side: int = 2048,
                              max_mesh_faces: int = 5_000_000,
                              max_uv_triangles: int = 500_000) -> dict[str, Any]:
    """Project real dense PLY geometry onto UV sheets using native Blender BVH."""
    import bmesh
    import numpy as np
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    plan = plan_scan_displacement(dense_revision, output,
                                  meters_per_unit=meters_per_unit, max_offset_m=max_offset_m,
                                  max_side=max_side, max_mesh_faces=max_mesh_faces,
                                  max_uv_triangles=max_uv_triangles)
    if tuple(bpy.app.version) < (4, 2, 0):
        raise ReferenceError("Scan displacement baking requires Blender 4.2+")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if "FINISHED" not in bpy.ops.wm.ply_import(filepath=plan["source_ply"],
                                                import_colors="SRGB"):
        raise ReferenceError("Blender could not load verified dense PLY")
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if (len(meshes) != 1 or len(meshes[0].data.vertices) != plan["vertices"]
            or len(meshes[0].data.polygons) != plan["faces"]):
        raise ReferenceError("Blender PLY geometry differs from pinned scan counts")
    mesh = bmesh.new()
    mesh.from_mesh(meshes[0].data)
    mesh.transform(meshes[0].matrix_world)
    mesh.normal_update()
    bvh = BVHTree.FromBMesh(mesh)
    triangles = _read_obj_triangles(Path(plan["source_obj"]),
                                    Path(plan["source_mtl"]), set(plan["atlas_dimensions"]),
                                    max_uv_triangles)
    by_atlas = {name: [] for name in plan["atlas_dimensions"]}
    for triangle in triangles:
        by_atlas[triangle.atlas].append(triangle)
    limit = plan["max_offset_m"] / plan["meters_per_unit"]
    final_target = Path(plan["output"])
    final_target.parent.mkdir(parents=True, exist_ok=True)
    # Build on a private sibling path first. An interrupted or invalid bake
    # must not leave half-written maps at the final revision name.
    staging = tempfile.TemporaryDirectory(prefix=".scan-displacement-",
                                          dir=final_target.parent)
    target = Path(staging.name)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "private-scan-displacement-candidate / scale-and-artistic-review-required",
        "distribution_allowed": False,
        "source_dense_revision": plan["source_revision"],
        "source_dense_report_sha256": _sha256(Path(plan["source_revision"]) / "dense-report.json"),
        "source_scan_sha256": plan["source_sha256"],
        "source_obj_sha256": plan["source_obj_sha256"],
        "meters_per_unit": plan["meters_per_unit"],
        "max_offset_m": plan["max_offset_m"],
        "displacement_scale_m": plan["displacement_scale_m"],
        "atlas_dimensions": plan["atlas_dimensions"],
        "uv_triangle_count": len(triangles), "atlases": {},
        "limitations": ("Ray-cast normal offsets require matched PLY/OBJ geometry; missing intersections "
                        "are neutral with separate validity masks. Physical scale is user-measured, "
                        "not recovered from COLMAP. No silhouettes or missing anatomy are fabricated."),
    }
    recipe = {"schema_version": 1, "atlases": {}}
    try:
        for index, (name, info) in enumerate(plan["atlas_dimensions"].items(), 1):
            sampled_width, sampled_height = info["sampled"]
            original_width, original_height = info["original"]
            offsets = np.zeros((sampled_height, sampled_width), dtype=np.float64)
            coverage = np.zeros((sampled_height, sampled_width), dtype=np.uint8)
            mapped = np.zeros((sampled_height, sampled_width), dtype=bool)
            collisions = np.zeros((sampled_height, sampled_width), dtype=bool)
            samples = conflicts = 0
            for triangle in by_atlas[name]:
                for x, y, delta in _sample_triangle(triangle, (sampled_width, sampled_height),
                                                     bvh, limit, np=np, Vector=Vector):
                    mapped[y, x] = True
                    if delta is None:
                        continue
                    if collisions[y, x]:
                        continue
                    if coverage[y, x]:
                        if abs(offsets[y, x] - delta) > limit * .01:
                            collisions[y, x] = True
                            coverage[y, x] = 0
                            conflicts += 1
                        continue
                    offsets[y, x] = delta
                    coverage[y, x] = 255
                    samples += 1
            coverage[collisions] = 0
            observed = int(np.count_nonzero(coverage))
            mapped_count = int(np.count_nonzero(mapped))
            if not observed:
                raise ReferenceError(f"No valid scan-to-UV ray hits for atlas {name}; check registration")
            observed_metres = offsets[coverage > 0] * plan["meters_per_unit"]
            relief_statistics = {
                "min": round(float(np.min(observed_metres)), 7),
                "median": round(float(np.median(observed_metres)), 7),
                "max": round(float(np.max(observed_metres)), 7),
                "p95_absolute": round(float(np.percentile(np.abs(observed_metres), 95)), 7),
            }
            # Height 0.5 is zero offset. 16-bit integer map scales to ±max_offset_m.
            height = np.uint16(np.clip(np.rint((.5 + offsets / (2 * limit)) * 65535), 0, 65535))
            height[coverage == 0] = 32768
            preview = np.uint8(np.clip(np.rint(128 + offsets / limit * 127), 0, 255))
            preview[coverage == 0] = 0
            if (sampled_width, sampled_height) != (original_width, original_height):
                yy = np.minimum((np.arange(original_height) * sampled_height /
                                 original_height).astype(int), sampled_height - 1)
                xx = np.minimum((np.arange(original_width) * sampled_width /
                                 original_width).astype(int), sampled_width - 1)
                height, coverage = height[np.ix_(yy, xx)], coverage[np.ix_(yy, xx)]
                preview = preview[np.ix_(yy, xx)]
            page = f"atlas-{index:02d}"
            (target / page).mkdir()
            height_file, mask_file = f"{page}/scan-height-16bit.png", f"{page}/coverage.png"
            preview_file = f"{page}/signed-relief-preview.png"
            _write_gray_png(target / height_file, height, bits=16, np=np)
            _write_gray_png(target / mask_file, coverage, bits=8, np=np)
            _write_gray_png(target / preview_file, preview, bits=8, np=np)
            recipe["atlases"][name] = {
                "regions": [{"id": "unclassified-surface", "mask": mask_file, "roughness": .85}],
                "height_map": height_file,
                "displacement_scale_m": plan["displacement_scale_m"],
            }
            manifest["atlases"][name] = {
                "height": {"path": height_file, "sha256": _sha256(target / height_file)},
                "coverage": {"path": mask_file, "sha256": _sha256(target / mask_file)},
                "signed_preview": {"path": preview_file,
                                   "sha256": _sha256(target / preview_file)},
                "sampled_texels": int(sampled_width * sampled_height),
                "ray_hits": samples, "overlap_conflicts": conflicts,
                "mapped_uv_texels": mapped_count,
                "valid_uv_texels": observed,
                "missing_or_conflicting_uv_texels": mapped_count - observed,
                "valid_fraction_of_uv": round(observed / mapped_count, 6),
                "valid_fraction": round(observed / (sampled_width * sampled_height), 6),
                "signed_offset_m": relief_statistics,
                "inspection_flags": (["sparse-ray-coverage"] if observed / mapped_count < .8 else [])
                + (["near-zero-relief"] if relief_statistics["p95_absolute"] < .0001 else []),
            }
    finally:
        mesh.free()
    if (_sha256(Path(plan["source_ply"])) != plan["source_sha256"]
            or _sha256(Path(plan["source_obj"])) != plan["source_obj_sha256"]
            or any(_sha256(Path(entry["path"])) != entry["sha256"]
                   for entry in plan["texture_atlases"])):
        raise ReferenceError("Dense scan/OBJ/atlas input changed during displacement baking")
    (target / "surface-recipe.example.json").write_text(json.dumps(recipe, indent=2) + "\n")
    manifest["example_surface_recipe"] = "surface-recipe.example.json"
    manifest["warning"] = ("Masks and roughness in this recipe are review placeholders. "
                           "Verify measured object scale, ray coverage and scan registration "
                           "before final Cycles displacement; GLB does not carry this height.")
    (target / "scan-displacement-report.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if final_target.exists() or final_target.is_symlink():
        raise ReferenceError("Scan displacement destination was created during baking")
    target.rename(final_target)
    staging.cleanup()
    return manifest
