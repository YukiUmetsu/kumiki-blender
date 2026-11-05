"""Conservative, object-agnostic coverage QA for a real dense MVS mesh.

Samples actual PLY triangles against ALL registered COLMAP cameras by default.
Projected visibility without ray occlusion is an *upper bound*, not evidence
that the visible material is accurate. If even this optimistic test finds weak
coverage, a dense reconstruction needs more registered views. Separately,
open mesh boundaries identify holes that cannot be found by sampling faces.

The colored point clouds are private Blender overlays in COLMAP coordinates;
they never add geometry to, or certify, the reconstructed object.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .blender_dense import _matched_scan_cameras
from .dense import _json, _ply_counts, _sha256
from .references import ReferenceError

_PLY_SCALARS = {
    "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
    "short": "i2", "ushort": "u2", "int16": "i2", "uint16": "u2",
    "int": "i4", "uint": "u4", "int32": "i4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def _mesh_source(dense: Path, report: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    record = report.get("mesh")
    if not isinstance(record, dict):
        raise ReferenceError("Dense revision has no recorded mesh")
    relative, expected = record.get("ply"), record.get("sha256")
    if (not isinstance(relative, str) or Path(relative).is_absolute()
            or not Path(relative).parts or ".." in Path(relative).parts
            or not isinstance(expected, str) or len(expected) != 64):
        raise ReferenceError("Dense mesh must have a safe SHA-pinned workspace path")
    mesh = dense / relative
    if (mesh.is_symlink() or not mesh.is_file()
            or not mesh.resolve().is_relative_to(dense)):
        raise ReferenceError("Dense mesh is missing or linked")
    if _sha256(mesh) != expected:
        raise ReferenceError("Dense mesh changed after its recorded reconstruction")
    counts = _ply_counts(mesh)
    if any(record.get(key) != counts[key] for key in ("vertices", "faces")):
        raise ReferenceError("Dense mesh counts differ from the pinned report")
    return mesh, counts


def plan_surface_coverage(dense_revision: str | Path, output: str | Path, *,
                          max_samples: int = 40_000, min_views: int = 2,
                          max_cameras: int = 256, max_mesh_faces: int = 5_000_000,
                          use_masks: bool = True) -> dict[str, Any]:
    """Read-only preflight for a camera-coverage and open-boundary audit."""
    for key, value, low, high in (
            ("max_samples", max_samples, 100, 200_000),
            ("min_views", min_views, 1, 8),
            ("max_cameras", max_cameras, 2, 256),
            ("max_mesh_faces", max_mesh_faces, 1_000, 5_000_000)):
        if type(value) is not int or not low <= value <= high:
            raise ReferenceError(f"{key} must be an integer in [{low}, {high}]")
    if type(use_masks) is not bool:
        raise ReferenceError("use_masks must be a boolean")
    dense, target = Path(dense_revision).resolve(), Path(output).resolve()
    if (Path(dense_revision).is_symlink() or not dense.is_dir()
            or Path(output).is_symlink() or target.exists()
            or target == dense or target.is_relative_to(dense)
            or dense.is_relative_to(target)):
        raise ReferenceError("Surface QA requires a dense revision and a new external output directory")
    report = _json(dense / "dense-report.json")
    if not str(report.get("status", "")).startswith("dense-mesh-candidate"):
        raise ReferenceError("Surface QA requires a completed dense-mesh candidate")
    mesh, counts = _mesh_source(dense, report)
    if counts["faces"] > max_mesh_faces:
        raise ReferenceError("Dense mesh exceeds the bounded surface QA face budget")
    registered = report.get("registered_images")
    if type(registered) is not int or registered < 2:
        raise ReferenceError("Surface QA needs multiple registered camera poses")
    cameras = _matched_scan_cameras(report, min(registered, max_cameras))
    masks = {}
    mask_records = report.get("rectified_masks", [])
    if use_masks and mask_records:
        if not isinstance(mask_records, list) or len(mask_records) != registered:
            raise ReferenceError("Surface QA requires a mask for every registered photo")
        for item in mask_records:
            if not isinstance(item, dict) or not isinstance(item.get("filename"), str):
                raise ReferenceError("Invalid dense mask evidence")
            name, expected = item["filename"], item.get("sha256")
            mask = dense / "workspace/images" / (name + ".mask.png")
            if (mask.is_symlink() or not mask.is_file() or not mask.resolve().is_relative_to(dense)
                    or not isinstance(expected, str) or _sha256(mask) != expected):
                raise ReferenceError(f"Surface QA rectified mask changed: {name}")
            masks[name] = str(mask)
        if any(camera["filename"] not in masks for camera in cameras):
            raise ReferenceError("Surface QA masks do not match registered cameras")
    return {
        "schema_version": 1, "source_revision": str(dense), "output": str(target),
        "mesh": str(mesh), "mesh_sha256": report["mesh"]["sha256"],
        "mesh_faces": counts["faces"], "mesh_vertices": counts["vertices"],
        "cameras": cameras, "registered_cameras": registered,
        "audited_cameras": len(cameras), "mask_paths": masks,
        "max_samples": max_samples, "min_views": min_views,
        "max_mesh_faces": max_mesh_faces,
        "status": "preflight-only / geometric-coverage-not-verified",
        "distribution_allowed": False,
    }


def _load_triangular_ply(path: Path, *, expected_vertices: int,
                         expected_faces: int, np: Any) -> tuple[Any, Any]:
    """Read OpenMVS's triangular binary/ascii PLY with bounded NumPy storage.

    Reject unexpected variable-length formats rather than silently misread a
    point cloud or strip with different face properties.
    """
    header: list[str] = []
    with path.open("rb") as stream:
        first = stream.readline()
        if first.strip() != b"ply":
            raise ReferenceError("Surface QA expects a PLY mesh")
        consumed = len(first)
        while consumed < 65_000:
            line = stream.readline()
            consumed += len(line)
            if not line:
                raise ReferenceError("Truncated PLY header")
            if line.strip() == b"end_header":
                break
            try:
                header.append(line.decode("ascii").strip())
            except UnicodeDecodeError as error:
                raise ReferenceError("PLY header is not ASCII") from error
        else:
            raise ReferenceError("Oversized PLY header")
        offset = stream.tell()
    fmt = next((line.split()[1] for line in header if line.startswith("format ")), None)
    if fmt not in ("ascii", "binary_little_endian"):
        raise ReferenceError("Surface QA supports ASCII or little-endian PLY meshes")
    vertex_props: list[tuple[str, str]] = []
    face_props = []
    active = ""
    for line in header:
        words = line.split()
        if len(words) == 3 and words[0] == "element":
            active = words[1]
        elif words and words[0] == "property":
            if active == "vertex":
                if len(words) != 3 or words[1] not in _PLY_SCALARS:
                    raise ReferenceError("Unsupported PLY vertex property")
                vertex_props.append((words[2], words[1]))
            elif active == "face":
                face_props.append(words)
    # OpenMVS 2.4 emits `property list uint8 uint32 vertex_indices`.
    # `uchar uint` is the same binary layout but these are different legal
    # PLY type spellings. Older and third-party exporters also use int32 and
    # the singular `vertex_index` property name.
    face_index_kind = (face_props[0][3] if len(face_props) == 1
                       and len(face_props[0]) == 5 else None)
    if (not {"x", "y", "z"}.issubset({name for name, _ in vertex_props})
            or _PLY_SCALARS.get(face_index_kind) not in ("i4", "u4")
            or not face_props or face_props[0][:2] != ["property", "list"]
            or _PLY_SCALARS.get(face_props[0][2]) != "u1"
            or face_props[0][4] not in ("vertex_indices", "vertex_index")):
        raise ReferenceError(f"Surface QA requires triangular PLY face indices; found: {face_props!r}")
    if fmt == "binary_little_endian":
        dtype = np.dtype([(name, "<" + _PLY_SCALARS[kind]) for name, kind in vertex_props])
        vertices = np.fromfile(path, dtype=dtype, count=expected_vertices, offset=offset)
        face_offset = offset + expected_vertices * dtype.itemsize
        index_dtype = "<u4" if _PLY_SCALARS[face_index_kind] == "u4" else "<i4"
        face_dtype = np.dtype([("size", "u1"), ("indices", index_dtype, (3,))])
        faces_raw = np.fromfile(path, dtype=face_dtype, count=expected_faces, offset=face_offset)
        if (len(vertices) != expected_vertices or len(faces_raw) != expected_faces
                or np.any(faces_raw["size"] != 3)
                or face_offset + expected_faces * face_dtype.itemsize != path.stat().st_size):
            raise ReferenceError("Unsupported or truncated triangular PLY data")
        xyz = np.column_stack([vertices[axis] for axis in "xyz"]).astype(np.float64)
        faces = faces_raw["indices"].copy()
    else:
        xyz = np.empty((expected_vertices, 3), dtype=np.float64)
        faces = np.empty((expected_faces, 3), dtype=np.int32)
        coordinate_cols = [next(i for i, prop in enumerate(vertex_props) if prop[0] == axis)
                           for axis in "xyz"]
        with path.open("r", encoding="ascii") as stream:
            for line in stream:
                if line.strip() == "end_header":
                    break
            try:
                for i in range(expected_vertices):
                    values = stream.readline().split()
                    xyz[i] = [float(values[k]) for k in coordinate_cols]
                for i in range(expected_faces):
                    values = stream.readline().split()
                    if len(values) != 4 or values[0] != "3":
                        raise ReferenceError("Surface QA requires triangle faces")
                    faces[i] = [int(value) for value in values[1:]]
            except (IndexError, ValueError, OverflowError) as error:
                raise ReferenceError("Malformed ASCII triangular PLY") from error
    if (not np.isfinite(xyz).all() or np.max(np.abs(xyz)) > 1e9
            or np.any(faces < 0) or np.any(faces >= expected_vertices)):
        raise ReferenceError("PLY contains invalid coordinates or indices")
    return xyz, faces


def _boundary_edges(xyz: Any, faces: Any, *, max_points: int, np: Any) -> tuple[Any, dict[str, int]]:
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges.sort(axis=1)
    packed = np.ascontiguousarray(edges).view(
        np.dtype([("a", "<i4"), ("b", "<i4")])).reshape(-1)
    unique, counts = np.unique(packed, return_counts=True)
    boundary = unique[counts == 1].view("<i4").reshape(-1, 2)
    selected = boundary[np.linspace(0, len(boundary) - 1, min(len(boundary), max_points),
                                    dtype=np.int64)] if len(boundary) else boundary
    midpoints = (xyz[selected[:, 0]] + xyz[selected[:, 1]]) / 2 if len(selected) else xyz[:0]
    return midpoints, {"boundary_edges": int(len(boundary)),
                       "nonmanifold_edges": int(np.count_nonzero(counts > 2)),
                       "sampled_boundary_points": int(len(selected))}


def _potential_camera_counts(xyz: Any, normals: Any, cameras: list[dict[str, Any]],
                             masks: dict[str, str], *, np: Any, Image: Any) -> tuple[Any, Any, list[dict[str, Any]]]:
    from .photo_uv_projection import _camera_coords

    possible = np.zeros(len(xyz), dtype=np.uint16)
    front_facing = np.zeros(len(xyz), dtype=np.uint16)
    camera_stats = []
    for camera in cameras:
        w, h = camera["intrinsics"]["width"], camera["intrinsics"]["height"]
        projected, depth = _camera_coords(xyz, camera, (w, h), np)
        finite = np.isfinite(projected).all(axis=1)
        valid = finite & (depth > 1e-6)
        inside = valid & (projected[:, 0] >= 0) & (projected[:, 0] < w) & (
            projected[:, 1] >= 0) & (projected[:, 1] < h)
        if camera["filename"] in masks:
            with Image.open(masks[camera["filename"]]) as image:
                if image.format != "PNG" or image.mode not in ("L", "1") or image.size != (w, h):
                    raise ReferenceError(f"Mask geometry does not match {camera['filename']}")
                binary = np.asarray(image.convert("L"), dtype=np.uint8)
                if not np.isin(binary, (0, 255)).all():
                    raise ReferenceError(f"Mask is not binary: {camera['filename']}")
            indices = np.flatnonzero(inside)
            inside[indices] = binary[projected[indices, 1].astype(int),
                                      projected[indices, 0].astype(int)] == 255
        possible += inside
        towards = np.asarray(camera["location_colmap"]) - xyz
        angle = np.sum(normals * towards, axis=1) / np.maximum(
            np.linalg.norm(towards, axis=1), 1e-12)
        front_facing += inside & (angle > 0.15)
        camera_stats.append({"filename": camera["filename"],
                             "projected_subject_samples": int(np.count_nonzero(inside)),
                             "front_facing_samples": int(np.count_nonzero(inside & (angle > 0.15)))})
    return possible, front_facing, camera_stats


def _write_ply_points(path: Path, xyz: Any, rgb: Any) -> None:
    with path.open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n"
                     "comment QA overlay, not a reconstruction mesh\n"
                     f"element vertex {len(xyz)}\n"
                     "property float x\nproperty float y\nproperty float z\n"
                     "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point, color in zip(xyz, rgb):
            stream.write(f"{point[0]:.7g} {point[1]:.7g} {point[2]:.7g} "
                         f"{int(color[0])} {int(color[1])} {int(color[2])}\n")


def audit_surface_coverage(dense_revision: str | Path, output: str | Path, *,
                           max_samples: int = 40_000, min_views: int = 2,
                           max_cameras: int = 256, max_mesh_faces: int = 5_000_000,
                           use_masks: bool = True) -> dict[str, Any]:
    """Emit hash-pinned upper-bound view support and open-boundary overlays."""
    import numpy as np
    from PIL import Image

    plan = plan_surface_coverage(dense_revision, output, max_samples=max_samples,
                                 min_views=min_views, max_cameras=max_cameras,
                                 max_mesh_faces=max_mesh_faces, use_masks=use_masks)
    xyz, faces = _load_triangular_ply(Path(plan["mesh"]),
                                      expected_vertices=plan["mesh_vertices"],
                                      expected_faces=plan["mesh_faces"], np=np)
    indices = np.linspace(0, len(faces) - 1, min(len(faces), max_samples), dtype=np.int64)
    selected = xyz[faces[indices]]
    sampled_xyz = selected.mean(axis=1)
    cross = np.cross(selected[:, 1] - selected[:, 0], selected[:, 2] - selected[:, 0])
    area = np.linalg.norm(cross, axis=1) / 2
    normals = cross / np.maximum(area[:, None] * 2, 1e-12)
    possible, front_facing, views = _potential_camera_counts(
        sampled_xyz, normals, plan["cameras"], plan["mask_paths"], np=np, Image=Image)
    degenerate = area <= 1e-12
    possible[degenerate] = 0
    front_facing[degenerate] = 0
    boundary, topology = _boundary_edges(xyz, faces, max_points=50_000, np=np)
    missing = possible < min_views
    region_min = np.min(xyz, axis=0)
    span = np.maximum(np.max(xyz, axis=0) - region_min, 1e-9)
    grid = np.minimum(3, ((sampled_xyz - region_min) / span * 4).astype(np.int32))
    grouped = {}
    for i, (cell, insufficient) in enumerate(zip(grid, missing)):
        key = tuple(map(int, cell))
        metrics = grouped.setdefault(key, [0, 0, 0.0])
        metrics[0] += 1
        metrics[1] += int(insufficient)
        metrics[2] += float(area[i])
    regions = sorted(({"normalized_grid_cell": list(cell), "samples": metrics[0],
                       "weak_support_samples": metrics[1],
                       "weak_support_fraction": round(metrics[1] / metrics[0], 4),
                       "sampled_surface_area": round(metrics[2], 5)}
                      for cell, metrics in grouped.items() if metrics[0] >= 5),
                     key=lambda item: (-item["weak_support_samples"], -item["samples"]))
    colors = np.empty((len(possible), 3), dtype=np.uint8)
    colors[possible == 0] = (235, 47, 50)
    colors[possible == 1] = (255, 141, 37)
    colors[possible == 2] = (255, 215, 48)
    colors[possible >= 3] = (49, 175, 92)
    # A stronger front-facing metric is reported but never used as the hard
    # gate: OpenMVS triangle winding can be inconsistent on damaged stone.
    target = Path(plan["output"])
    target.mkdir(parents=True, exist_ok=False)
    _write_ply_points(target / "view-support-points.ply", sampled_xyz, colors)
    _write_ply_points(target / "open-boundary-points.ply", boundary,
                      np.tile((240, 50, 225), (len(boundary), 1)))
    weak_fraction = float(np.count_nonzero(missing) / len(missing))
    complete_camera_set = plan["audited_cameras"] == plan["registered_cameras"]
    evidence = {
        "schema_version": 1,
        "status": ("incomplete-view-audit / subset-of-registered-cameras" if not complete_camera_set
                   else "coverage-gap-detected / review-required" if weak_fraction > 0.05
                   else "coverage-candidate / photographic-fidelity-not-verified"),
        "source_revision": plan["source_revision"], "mesh_sha256": plan["mesh_sha256"],
        "registered_cameras": plan["registered_cameras"],
        "audited_cameras": plan["audited_cameras"],
        "foreground_masks_applied": bool(plan["mask_paths"]),
        "mesh_faces": plan["mesh_faces"], "sampled_faces": int(len(indices)),
        "degenerate_sampled_faces": int(np.count_nonzero(degenerate)),
        "min_views": min_views,
        "possible_view_support": {
            "zero": int(np.count_nonzero(possible == 0)),
            "one": int(np.count_nonzero(possible == 1)),
            "two": int(np.count_nonzero(possible == 2)),
            "three_or_more": int(np.count_nonzero(possible >= 3)),
            "weak_fraction": round(weak_fraction, 5),
            "front_facing_weak_fraction": round(float(np.mean(front_facing < min_views)), 5),
        },
        "topology": topology, "weak_regions": regions[:24],
        "per_camera": views,
        "overlays": ["view-support-points.ply", "open-boundary-points.ply"],
        "distribution_allowed": False, "photographic_fidelity_verified": False,
        "limitations": (
            "View counts are optimistic projected camera/mask intersections, not true ray-occluded "
            "photographic observations. Low counts indicate definite coverage gaps; high counts "
            "do not guarantee geometry accuracy. Missing faces have no sample points; inspect "
            "magenta open-boundary overlay for holes. COLMAP axes are arbitrary; color clouds "
            "do not automatically label face, paws, top, or back-left."),
    }
    (target / "surface-coverage-report.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence
