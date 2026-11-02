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


def plan_dense_blender_review(dense_revision: str | Path, output: str | Path,
                              *, max_mesh_faces: int = 5_000_000) -> dict[str, Any]:
    """Check a dense mesh has not changed since its reconstruction report."""
    root = Path(dense_revision).resolve()
    target = Path(output).resolve()
    if Path(dense_revision).is_symlink() or not root.is_dir():
        raise ReferenceError("Dense reconstruction revision is missing or linked")
    if target.exists() or Path(output).is_symlink() or target.is_relative_to(root):
        raise ReferenceError("Blender review needs an unused output outside dense revision")
    if type(max_mesh_faces) is not int or not 100 <= max_mesh_faces <= 10_000_000:
        raise ReferenceError("max_mesh_faces must be 100–10,000,000")
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
    return {"source_revision": str(root), "source_ply": str(path), "source_sha256": actual,
            "output": str(target), "faces": counts["faces"], "vertices": counts["vertices"],
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


def build_dense_blender_review(dense_revision: str | Path, output: str | Path,
                               *, bpy_module: Any, render_previews: bool = True,
                               max_mesh_faces: int = 5_000_000) -> dict[str, Any]:
    """Save a private editable .blend and GLB; neither clears photo rights."""
    plan = plan_dense_blender_review(dense_revision, output, max_mesh_faces=max_mesh_faces)
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
              "historical_fidelity_verified": False, "previews": []}

    def checkpoint() -> None:
        (target / "blender-review-report.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8")

    checkpoint()
    try:
        if render_previews:
            (target / "renders").mkdir()
            record["previews"] = _preview_axes(bpy, obj, target)
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
