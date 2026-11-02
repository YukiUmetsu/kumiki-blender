"""Reproject verified COLMAP photos onto OpenMVS UVs with depth visibility.

Host-side Pillow/NumPy implementation: no Blender or CUDA requirement. It is
deliberately bounded and emits masked *observations*, not fabricated material
maps. Coarse depth rasterization rejects camera-occluded UV samples. Inspect
coverage/occlusion and refine masks before multi-view albedo fusion.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .blender_dense import _matched_scan_cameras, _textured_obj_plan
from .dense import _json, _sha256
from .references import ReferenceError
from .uv_coverage import _material_textures


@dataclass(frozen=True)
class _Triangle:
    atlas: str
    xyz: tuple[tuple[float, float, float], ...]
    uv: tuple[tuple[float, float], ...]


def _read_obj_triangles(obj: Path, mtl: Path, atlas_names: set[str],
                        max_triangles: int) -> list[_Triangle]:
    materials = _material_textures(mtl, atlas_names)
    vertices: list[tuple[float, float, float]] = []
    texture: list[tuple[float, float]] = []
    triangles: list[_Triangle] = []
    current = None

    def point_index(value: str, count: int) -> int:
        try:
            number = int(value)
        except ValueError as error:
            raise ReferenceError("Projected OBJ has a noninteger vertex/UV index") from error
        index = number - 1 if number > 0 else count + number
        if number == 0 or not 0 <= index < count:
            raise ReferenceError("Projected OBJ references an unknown vertex/UV")
        return index

    with obj.open(encoding="utf-8") as stream:
        for line in stream:
            words = line.split()
            if not words:
                continue
            if words[0] == "v":
                if len(words) < 4:
                    raise ReferenceError("Projected OBJ vertex is incomplete")
                try:
                    point = tuple(float(value) for value in words[1:4])
                except ValueError as error:
                    raise ReferenceError("Projected OBJ contains invalid vertices") from error
                if not all(math.isfinite(v) and abs(v) <= 1e8 for v in point):
                    raise ReferenceError("Projected OBJ contains out-of-range vertices")
                vertices.append(point)
                if len(vertices) > 5_000_000:
                    raise ReferenceError("Projected OBJ vertex budget exceeded")
            elif words[0] == "vt":
                if len(words) < 3:
                    raise ReferenceError("Projected OBJ UV is incomplete")
                try:
                    uv = tuple(float(value) for value in words[1:3])
                except ValueError as error:
                    raise ReferenceError("Projected OBJ contains invalid UV values") from error
                if not all(math.isfinite(v) and -1e-5 <= v <= 1 + 1e-5 for v in uv):
                    raise ReferenceError("Projected OBJ UV extends outside its original atlas")
                texture.append(uv)
                if len(texture) > 5_000_000:
                    raise ReferenceError("Projected OBJ UV budget exceeded")
            elif words[0] == "usemtl":
                current = materials.get(" ".join(words[1:]))
                if current is None:
                    raise ReferenceError("Projected OBJ has an unpinned material")
            elif words[0] == "f":
                if current is None or not 3 <= len(words) - 1 <= 32:
                    raise ReferenceError("Projected OBJ face has an invalid material/vertex count")
                face = []
                for token in words[1:]:
                    parts = token.split("/")
                    if len(parts) < 2 or not parts[1]:
                        raise ReferenceError("Projected OBJ face lacks texture coordinates")
                    face.append((vertices[point_index(parts[0], len(vertices))],
                                 texture[point_index(parts[1], len(texture))]))
                for index in range(1, len(face) - 1):
                    selected = (face[0], face[index], face[index + 1])
                    triangles.append(_Triangle(current, tuple(item[0] for item in selected),
                                               tuple(item[1] for item in selected)))
                    if len(triangles) > max_triangles:
                        raise ReferenceError("UV projection triangle budget exceeded; supply a verified proxy mesh")
    if not triangles:
        raise ReferenceError("UV projection requires a textured triangulated surface")
    return triangles


def _camera_coords(points: Any, camera: dict[str, Any], scaled_size: tuple[int, int],
                   np: Any) -> tuple[Any, Any]:
    """Match the precise COLMAP camera-to-world frame in blender_dense."""
    rotation = np.asarray(camera["rotation_3x3"], dtype=np.float64)
    center = np.asarray(camera["location_colmap"], dtype=np.float64)
    local = (np.asarray(points, dtype=np.float64) - center) @ rotation
    depth = -local[..., 2]
    intrinsic = camera["intrinsics"]
    sx = scaled_size[0] / intrinsic["width"]
    sy = scaled_size[1] / intrinsic["height"]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (intrinsic["fx_px"] * (local[..., 0] / depth) + intrinsic["cx_px"]) * sx
        v = (intrinsic["cy_px"] - intrinsic["fy_px"] * (local[..., 1] / depth)) * sy
    return np.stack((u, v), axis=-1), depth


def _raster_cells(points: Any, size: tuple[int, int], np: Any,
                  tile_side: int = 64) -> Iterator[tuple[Any, Any, Any]]:
    """Yield center-sampled triangle barycentrics in bounded 64px chunks."""
    if not np.isfinite(points).all():
        return
    x0 = max(0, math.floor(float(np.min(points[:, 0]))))
    y0 = max(0, math.floor(float(np.min(points[:, 1]))))
    x1 = min(size[0], math.ceil(float(np.max(points[:, 0])) + 1))
    y1 = min(size[1], math.ceil(float(np.max(points[:, 1])) + 1))
    if x1 <= x0 or y1 <= y0:
        return
    (ax, ay), (bx, by), (cx, cy) = points
    denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if not math.isfinite(float(denom)) or abs(denom) < 1e-10:
        return
    for top in range(y0, y1, tile_side):
        for left in range(x0, x1, tile_side):
            yy, xx = np.mgrid[top:min(y1, top + tile_side),
                              left:min(x1, left + tile_side)]
            px, py = xx + .5, yy + .5
            w0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denom
            w1 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denom
            w2 = 1 - w0 - w1
            inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
            if np.any(inside):
                yield yy[inside], xx[inside], np.stack(
                    (w0[inside], w1[inside], w2[inside]), axis=1)


def _front_facing(triangle: _Triangle, camera: dict[str, Any], np: Any) -> bool:
    p = np.asarray(triangle.xyz, dtype=np.float64)
    normal = np.cross(p[1] - p[0], p[2] - p[0])
    direction = np.asarray(camera["location_colmap"]) - np.mean(p, axis=0)
    return float(np.dot(normal, direction)) > 1e-9


def _project_view(triangles: list[_Triangle], camera: dict[str, Any], atlas_sizes: dict[str, tuple[int, int]],
                  *, camera_side: int, np: Any, Image: Any) -> tuple[dict[str, tuple[Any, Any]], dict[str, int]]:
    with Image.open(camera["source_photo_path"]) as original:
        original.load()
        factor = min(1, camera_side / max(original.size))
        image_size = (max(1, round(original.width * factor)),
                      max(1, round(original.height * factor)))
        photograph = np.asarray(original.convert("RGB").resize(
            image_size, Image.Resampling.BILINEAR), dtype=np.uint8)
    depth_buffer = np.full((image_size[1], image_size[0]), np.inf, dtype=np.float64)

    # Camera-space triangle depth buffer (not just a face-center heuristic).
    for triangle in triangles:
        if not _front_facing(triangle, camera, np):
            continue
        projected, depth = _camera_coords(np.asarray(triangle.xyz), camera, image_size, np)
        if np.any(depth <= 1e-6):
            continue
        for ys, xs, weights in _raster_cells(projected, image_size, np):
            # Correct interpolation for perspective depth (1/z is affine).
            sampled_depth = 1 / (weights @ (1 / depth))
            depth_buffer[ys, xs] = np.minimum(depth_buffer[ys, xs], sampled_depth)

    outputs = {name: (np.zeros((size[1], size[0], 3), dtype=np.uint8),
                      np.zeros((size[1], size[0]), dtype=np.uint8))
               for name, size in atlas_sizes.items()}
    conflicts = {name: np.zeros((size[1], size[0]), dtype=bool)
                 for name, size in atlas_sizes.items()}
    rejected = {"occluded_samples": 0, "visible_samples": 0, "backfacing_faces": 0}
    for triangle in triangles:
        if not _front_facing(triangle, camera, np):
            rejected["backfacing_faces"] += 1
            continue
        xyz = np.asarray(triangle.xyz, dtype=np.float64)
        normal = np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])
        normal /= np.maximum(np.linalg.norm(normal), 1e-12)
        eye = np.asarray(camera["location_colmap"], dtype=np.float64)
        uv_size = atlas_sizes[triangle.atlas]
        uv_coords = np.asarray([(u * uv_size[0], (1 - v) * uv_size[1])
                                for u, v in triangle.uv], dtype=np.float64)
        color, mask = outputs[triangle.atlas]
        for yy, xx, weights in _raster_cells(uv_coords, uv_size, np):
            world = weights @ xyz
            projected, depth = _camera_coords(world, camera, image_size, np)
            finite = np.isfinite(projected).all(axis=1)
            bounded = finite & (np.abs(projected).max(axis=1) < 1e8)
            px = np.floor(np.where(bounded, projected[:, 0], -1)).astype(np.int64)
            py = np.floor(np.where(bounded, projected[:, 1], -1)).astype(np.int64)
            visible = ((depth > 1e-6) & (px >= 0) & (px < image_size[0])
                       & (py >= 0) & (py < image_size[1]) & bounded)
            if not np.any(visible):
                continue
            xs, ys, px, py, depth = (array[visible] for array in (xx, yy, px, py, depth))
            z = depth_buffer[py, px]
            accepted = np.isfinite(z) & (depth <= z * 1.025 + 1e-4)
            rejected["occluded_samples"] += int(np.count_nonzero(~accepted))
            if not np.any(accepted):
                continue
            xs, ys, px, py = (array[accepted] for array in (xs, ys, px, py))
            sampled_color = photograph[py, px]
            accepted_world = world[visible][accepted]
            towards_eye = eye - accepted_world
            incidence = np.clip((towards_eye @ normal) /
                                np.maximum(np.linalg.norm(towards_eye, axis=1), 1e-12), 0, 1)
            view_confidence = np.uint8(incidence * 255 + .5)
            # Overlapping UV islands with contradictory colors are ambiguous.
            conflict = ((mask[ys, xs] > 0) &
                        (np.max(np.abs(color[ys, xs].astype(np.int16)
                                       - sampled_color.astype(np.int16)), axis=1) > 24))
            conflicts[triangle.atlas][ys[conflict], xs[conflict]] = True
            safe = ~conflicts[triangle.atlas][ys, xs]
            color[ys[safe], xs[safe]] = sampled_color[safe]
            mask[ys[safe], xs[safe]] = view_confidence[safe]
            rejected["visible_samples"] += int(np.count_nonzero(safe))
    for name, (_color, mask) in outputs.items():
        mask[conflicts[name]] = 0
    return outputs, rejected


def project_registered_photos_to_uv(dense_revision: str | Path, output: str | Path,
                                    *, views: int = 3, atlas_side: int = 2048,
                                    camera_side: int = 1536,
                                    max_triangles: int = 500_000) -> dict[str, Any]:
    """Create independent masked UV observations suitable for surface recipes."""
    from PIL import Image
    import numpy as np

    source, target = Path(dense_revision).resolve(), Path(output).resolve()
    if (Path(dense_revision).is_symlink() or not source.is_dir()
            or target.exists() or Path(output).is_symlink()
            or target == source or target.is_relative_to(source)
            or source.is_relative_to(target)):
        raise ReferenceError("UV photo projection needs an existing dense revision and new external directory")
    for name, value, low, high in (("views", views, 3, 8),
                                   ("atlas_side", atlas_side, 128, 4096),
                                   ("camera_side", camera_side, 128, 2048),
                                   ("max_triangles", max_triangles, 100, 2_000_000)):
        if type(value) is not int or not low <= value <= high:
            raise ReferenceError(f"{name} must be an integer in [{low}, {high}]")
    report = _json(source / "dense-report.json")
    pinned = _textured_obj_plan(source, report, 10_000_000)
    cameras = _matched_scan_cameras(report, views)
    # Never write an observation output into either immutable reconstruction
    # revision (dense mesh or its linked sparse/undistorted photo evidence).
    sparse_source = Path(report["source_revision"]).resolve()
    if target.is_relative_to(sparse_source) or sparse_source.is_relative_to(target):
        raise ReferenceError("UV projection output must be outside both source revisions")
    atlases = {Path(item["path"]).name: Path(item["path"])
               for item in pinned["texture_atlases"]}
    original_sizes = {}
    sampled_sizes = {}
    for name, path in atlases.items():
        with Image.open(path) as image:
            original_sizes[name] = image.size
        factor = min(1, atlas_side / max(original_sizes[name]))
        sampled_sizes[name] = tuple(max(1, round(axis * factor))
                                    for axis in original_sizes[name])
    if sum(width * height for width, height in original_sizes.values()) * views > 256_000_000:
        raise ReferenceError("UV photo projection exceeds 256 million output texels; reduce views or atlas size")
    # Detect malformed or mismatched camera evidence before creating the output.
    # The source index pins its bytes, but a valid hash is not a decodable photo.
    for camera in cameras:
        try:
            with Image.open(camera["source_photo_path"]) as image:
                if image.size != (camera["intrinsics"]["width"],
                                  camera["intrinsics"]["height"]):
                    raise ReferenceError("Registered photo dimensions differ from COLMAP intrinsics")
                image.verify()
        except (OSError, ValueError) as error:
            raise ReferenceError("Registered photograph is not a decodable image") from error
    triangles = _read_obj_triangles(Path(pinned["source_obj"]),
                                    Path(pinned["source_mtl"]), set(atlases), max_triangles)
    target.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "schema_version": 1, "status": "private-masked-projection / artistic-review-required",
        "distribution_allowed": False,
        "source_dense_revision": str(source),
        "source_dense_report_sha256": _sha256(source / "dense-report.json"),
        "source_obj_sha256": pinned["source_obj_sha256"],
        "triangle_count": len(triangles), "views": views,
        "atlas_sample_side": atlas_side, "photo_sample_side": camera_side,
        "atlases": {},
        "limitations": ("Coarse photo-space z-buffer and sampled UV projection; inspect "
                        "occlusion and UV island masks. View-specific sun/shadow and "
                        "reflectance remain uncalibrated. Upscaling cannot invent detail."),
    }
    recipe: dict[str, Any] = {"schema_version": 1, "atlases": {}}
    for index, (name, size) in enumerate(original_sizes.items(), start=1):
        page = f"atlas-{index:02d}"
        (target / page).mkdir()
        Image.new("L", size, 255).save(target / page / "region.png")
        recipe["atlases"][name] = {
            "regions": [{"id": "unclassified-surface", "mask": f"{page}/region.png",
                         "roughness": .85}],
            "observations": [],
        }
        manifest["atlases"][name] = {"original_dimensions": list(size),
                                     "projected_dimensions": list(sampled_sizes[name]),
                                     "views": []}

    for index, camera in enumerate(cameras):
        result, quality = _project_view(triangles, camera, sampled_sizes,
                                        camera_side=camera_side, np=np, Image=Image)
        with Path(camera["source_photo_path"]).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != camera["source_photo_sha256"]:
                raise ReferenceError("Rectified photo changed during UV projection")
        for page_index, (name, size) in enumerate(original_sizes.items(), start=1):
            page = f"atlas-{page_index:02d}"
            sampled, confidence = result[name]
            # Pillow resamples RGBA in premultiplied-alpha space. Transparent
            # unobserved texels must NOT bleed black into a valid UV boundary.
            resampled = Image.fromarray(np.dstack((sampled, confidence)), "RGBA").resize(
                size, Image.Resampling.BILINEAR)
            projected = resampled.convert("RGB")
            visible = resampled.getchannel("A")
            photo_name = f"{page}/view-{index + 1:02d}.png"
            mask_name = f"{page}/visible-{index + 1:02d}.png"
            projected.save(target / photo_name)
            visible.save(target / mask_name)
            view_id = f"camera-{camera['image_id']}"
            recipe["atlases"][name]["observations"].append(
                {"id": view_id, "photo": photo_name, "confidence": mask_name})
            manifest["atlases"][name]["views"].append({
                "id": view_id, "source_photo": camera["filename"],
                "source_photo_sha256": camera["source_photo_sha256"],
                "photo": {"path": photo_name, "sha256": _sha256(target / photo_name)},
                "confidence": {"path": mask_name, "sha256": _sha256(target / mask_name)},
                "visible_fraction": round(float(np.count_nonzero(confidence)) /
                                          confidence.size, 6),
                "diagnostics": quality,
            })
    (target / "surface-recipe.example.json").write_text(json.dumps(recipe, indent=2) + "\n")
    manifest["example_surface_recipe"] = "surface-recipe.example.json"
    manifest["warning"] = ("The example roughness prior and all-white region masks are placeholders. "
                           "Review/edit region classification and visibility before enabling fusion.")
    (target / "uv-projection-report.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
