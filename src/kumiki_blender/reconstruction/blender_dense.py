"""Import a reviewed dense PLY into a private editable Blender QA scene."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from kumiki_blender.blender_import import imported_bounds
from kumiki_blender.blender_output import aim_at
from kumiki_blender.glb_audit import audit_glb

from .dense import _json, _ply_counts
from .references import ReferenceError


def _matched_scan_cameras(report: dict[str, Any], count: int) -> list[dict[str, Any]]:
    """Camera poses and the dense PLY share COLMAP coordinates: no sculpt anchors."""
    from .cameras import (_matmul, _matvec, _quat_matrix, _read_intrinsics,
                          _read_poses)

    source_value = report.get("source_revision")
    if not isinstance(source_value, str):
        raise ReferenceError("Matched views need the source sparse reconstruction revision")
    source = Path(source_value).resolve()
    if (Path(source_value).is_symlink() or not source.is_dir()
            or (source / "undistorted").is_symlink()):
        raise ReferenceError("Matched-view sparse revision is missing or linked")
    sparse_report = _json(source / "reconstruction-report.json")
    if not str(sparse_report.get("status", "")).startswith("candidate"):
        raise ReferenceError("Matched-view sparse reconstruction is not a candidate")
    model = source / "undistorted/text-model"
    photo_root = source / "undistorted/images"
    if model.is_symlink() or not model.is_dir() or photo_root.is_symlink():
        raise ReferenceError("Rectified camera model or images are missing or linked")
    index = _json(source / "undistorted-reference-index.json")
    entries = {entry["filename"]: entry for entry in index.get("images", [])
               if isinstance(entry, dict) and isinstance(entry.get("filename"), str)}
    poses = _read_poses(model / "images.txt")
    intrinsics = _read_intrinsics(model / "cameras.txt")
    if (len(poses) < count or len(entries) != len(index.get("images", []))
            or len(poses) != sparse_report.get("registered_images")):
        raise ReferenceError("Matched-view image and camera counts differ from sparse report")
    # Even spacing through COLMAP image IDs, with no subjective view labels.
    chosen = [poses[i * len(poses) // count] for i in range(count)]
    matched = []
    for image_id, camera_id, filename, quaternion, tvec in chosen:
        entry = entries.get(filename)
        intr = intrinsics.get(camera_id)
        photo = photo_root / filename
        if (entry is None or intr is None or photo.is_symlink() or not photo.is_file()
                or not photo.resolve().is_relative_to(source)
                or entry.get("dimensions") != [intr["width"], intr["height"]]):
            raise ReferenceError(f"Rectified camera/photo mismatch for {filename}")
        with photo.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != entry.get("sha256"):
            raise ReferenceError(f"Rectified photo changed since reconstruction: {filename}")
        world_to_camera = _quat_matrix(quaternion)
        camera_to_world = [[world_to_camera[j][i] for j in range(3)] for i in range(3)]
        center = _matvec(camera_to_world, tuple(-value for value in tvec))
        orientation = _matmul(camera_to_world, [[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        matched.append({"image_id": image_id, "filename": filename,
                        "source_photo_path": str(photo), "source_photo_sha256": digest,
                        "intrinsics": intr, "location_colmap": center,
                        "rotation_3x3": orientation})
    return matched


def plan_dense_blender_review(dense_revision: str | Path, output: str | Path,
                              *, max_mesh_faces: int = 5_000_000,
                              matched_views: int = 0,
                              matched_resolution_percent: int = 100) -> dict[str, Any]:
    """Check a dense mesh has not changed since its reconstruction report."""
    root = Path(dense_revision).resolve()
    target = Path(output).resolve()
    if Path(dense_revision).is_symlink() or not root.is_dir():
        raise ReferenceError("Dense reconstruction revision is missing or linked")
    if target.exists() or Path(output).is_symlink() or target.is_relative_to(root):
        raise ReferenceError("Blender review needs an unused output outside dense revision")
    if type(max_mesh_faces) is not int or not 100 <= max_mesh_faces <= 10_000_000:
        raise ReferenceError("max_mesh_faces must be 100–10,000,000")
    if type(matched_views) is not int or not 0 <= matched_views <= 12:
        raise ReferenceError("matched_views must be 0–12")
    if (type(matched_resolution_percent) is not int
            or not 10 <= matched_resolution_percent <= 100):
        raise ReferenceError("matched_resolution_percent must be 10–100")
    report = _json(root / "dense-report.json")
    mesh = report.get("mesh")
    if (not isinstance(report.get("status"), str)
            or not report["status"].startswith("dense-mesh-candidate")
            or not isinstance(mesh, dict)):
        raise ReferenceError("Dense revision is not a candidate mesh")
    filename = mesh.get("ply")
    expected = mesh.get("sha256")
    if (not isinstance(filename, str) or not filename.endswith(".ply")
            or not isinstance(expected, str) or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)):
        raise ReferenceError("Dense mesh report lacks a safe PLY path and SHA-256")
    relative = Path(filename)
    if relative.is_absolute() or any(part in (".", "..") for part in relative.parts):
        raise ReferenceError("Dense mesh path is not a safe relative path")
    path = root / relative
    if (path.is_symlink() or not path.resolve().is_relative_to(root)
            or not path.is_file()):
        raise ReferenceError("Dense PLY file is missing or linked outside the revision")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ReferenceError("Dense mesh PLY changed after reconstruction")
    counts = _ply_counts(path)
    if counts["faces"] != mesh.get("faces") or counts["vertices"] != mesh.get("vertices"):
        raise ReferenceError("Dense mesh header no longer matches audited geometry")
    if counts["faces"] > max_mesh_faces:
        raise ReferenceError("Dense mesh exceeds Blender review's face budget; split the scan")
    cameras = _matched_scan_cameras(report, matched_views) if matched_views else []
    return {"source_revision": str(root), "source_ply": str(path), "source_sha256": actual,
            "output": str(target), "faces": counts["faces"], "vertices": counts["vertices"],
            "matched_cameras": cameras,
            "matched_resolution_percent": matched_resolution_percent,
            "status": "preflight-only / Blender-review-required",
            "distribution_allowed": False}


def _preview_axes(bpy: Any, obj: Any, target: Path) -> list[str]:
    scene = bpy.context.scene
    bounds_min, bounds_max = imported_bounds([obj])
    center = tuple((bounds_min[i]+bounds_max[i])/2 for i in range(3))
    size = max(bounds_max[i]-bounds_min[i] for i in range(3))
    if not math.isfinite(size) or size <= 0:
        raise ReferenceError("Dense mesh has invalid or degenerate bounds")
    camera_data = bpy.data.cameras.new("Private dense QA / orthographic views")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = max(0.5, size*1.75)
    camera = bpy.data.objects.new(camera_data.name, camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    world = bpy.data.worlds.new("Private dense QA / neutral world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.35, 0.35, 0.35, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.8
    scene.world = world
    light_data = bpy.data.lights.new("Dense QA / raking key", "AREA")
    light_data.energy = 900.0
    light_data.shape = "DISK"
    light_data.size = max(size*1.3, 0.2)
    light = bpy.data.objects.new(light_data.name, light_data)
    scene.collection.objects.link(light)
    light.location = (center[0]+size, center[1]-size, center[2]+size*1.5)
    aim_at(light, center)
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 8
    scene.render.resolution_x = 320
    scene.render.resolution_y = 320
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    views = [("axis-negative-y", (0, -1.8, 0.4)),
             ("axis-negative-x", (-1.8, 0, 0.4)),
             ("axis-three-quarter", (1.5, -1.5, 0.9))]
    paths = []
    for name, direction in views:
        camera.location = tuple(center[i]+size*direction[i] for i in range(3))
        aim_at(camera, center)
        filename = "renders/" + name + ".png"
        scene.render.filepath = str(target / filename)
        bpy.ops.render.render(write_still=True)
        paths.append(filename)
    return paths


def _preview_matched(bpy: Any, target: Path, entries: list[dict[str, Any]],
                     resolution_percent: int) -> list[str]:
    """Clay renders from rectified photo cameras, in the unscaled scan frame."""
    from mathutils import Matrix  # type: ignore[import-not-found]

    scene = bpy.context.scene
    scene.render.resolution_percentage = resolution_percent
    output = []
    for index, entry in enumerate(entries):
        intr = entry["intrinsics"]
        camera_data = bpy.data.cameras.new("Scan/photo registration / " + entry["filename"])
        camera_data.type = "PERSP"
        camera_data.sensor_fit = "HORIZONTAL"
        camera_data.sensor_width = 36.0
        camera_data.lens = intr["fx_px"] * 36.0 / intr["width"]
        camera_data.shift_x = (intr["width"] / 2 - intr["cx_px"]) / intr["width"]
        camera_data.shift_y = (intr["cy_px"] - intr["height"] / 2) / intr["width"]
        camera_data.show_background_images = True
        background = camera_data.background_images.new()
        background.image = bpy.data.images.load(entry["source_photo_path"], check_existing=True)
        background.alpha = 0.45
        camera = bpy.data.objects.new(camera_data.name, camera_data)
        scene.collection.objects.link(camera)
        rotation = entry["rotation_3x3"]
        location = entry["location_colmap"]
        camera.matrix_world = Matrix((
            (*rotation[0], location[0]), (*rotation[1], location[1]),
            (*rotation[2], location[2]), (0, 0, 0, 1),
        ))
        scene.camera = camera
        scene.render.resolution_x = intr["width"]
        scene.render.resolution_y = intr["height"]
        name = f"renders/photo-matched-{index + 1:02d}.png"
        scene.render.filepath = str(target / name)
        bpy.ops.render.render(write_still=True)
        output.append(name)
    return output


def build_dense_blender_review(dense_revision: str | Path, output: str | Path,
                               *, bpy_module: Any, render_previews: bool = True,
                               max_mesh_faces: int = 5_000_000,
                               matched_views: int = 0,
                               matched_resolution_percent: int = 100) -> dict[str, Any]:
    """Save a private editable .blend and GLB; neither clears photo rights."""
    plan = plan_dense_blender_review(dense_revision, output,
                                     max_mesh_faces=max_mesh_faces,
                                     matched_views=matched_views,
                                     matched_resolution_percent=matched_resolution_percent)
    bpy = bpy_module
    if tuple(bpy.app.version) < (4, 2, 0):
        raise ReferenceError("Dense Blender review requires Blender 4.2+")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    result = bpy.ops.wm.ply_import(filepath=plan["source_ply"], import_colors="SRGB")
    if "FINISHED" not in result:
        raise ReferenceError("Blender could not import the private PLY")
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if len(meshes) != 1 or not meshes[0].data.polygons:
        raise ReferenceError("Dense review expects exactly one nonempty PLY mesh")
    obj = meshes[0]
    if len(obj.data.polygons) != plan["faces"] or len(obj.data.vertices) != plan["vertices"]:
        raise ReferenceError("Blender geometry differs from audited PLY counts")
    obj.name = "Private dense scan / unverified scale and axes"
    material = bpy.data.materials.new("Dense statue / neutral review material")
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (0.53, 0.49, 0.44, 1.0)
    shader.inputs["Roughness"].default_value = 0.92
    if obj.data.color_attributes:
        color = obj.data.color_attributes.active_color
        if color is not None:
            attribute = material.node_tree.nodes.new("ShaderNodeAttribute")
            attribute.attribute_name = color.name
            material.node_tree.links.new(attribute.outputs["Color"], shader.inputs["Base Color"])
    obj.data.materials.clear()
    obj.data.materials.append(material)
    target = Path(plan["output"])
    target.mkdir(parents=True, exist_ok=False)
    record = {**plan, "status": "review-required",
              "historical_fidelity_verified": False, "previews": [],
              "photo_matched_previews": []}

    def checkpoint() -> None:
        (target / "blender-review-report.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8")

    checkpoint()
    try:
        if render_previews:
            (target / "renders").mkdir()
            record["previews"] = _preview_axes(bpy, obj, target)
            if plan["matched_cameras"]:
                record["photo_matched_previews"] = _preview_matched(
                    bpy, target, plan["matched_cameras"], matched_resolution_percent)
        bpy.ops.wm.save_as_mainfile(filepath=str(target / "private-dense-review.blend"))
        for item in bpy.context.selected_objects:
            item.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        result = bpy.ops.export_scene.gltf(filepath=str(target / "private-dense-review.glb"),
                                           export_format="GLB", use_selection=True,
                                           export_animations=False)
        if "FINISHED" not in result:
            raise ReferenceError("Blender failed to export the private review GLB")
        record["glb_audit"] = audit_glb(target / "private-dense-review.glb")
        record["status"] = "private-Blender-mesh-candidate / artistic-and-rights-review-required"
    except (ReferenceError, RuntimeError, OSError, ValueError) as error:
        record.update(failed_stage="Blender-review", error=str(error))
    checkpoint()
    return record
