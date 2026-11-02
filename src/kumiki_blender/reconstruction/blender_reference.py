"""Private Blender working scene with authentic SfM cameras and clay QA renders.

This scene is deliberately NOT an asset export: it may link to third-party
photographs, and a registered camera does not validate the imported sculpt.
"""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Any

from .references import ReferenceError


def load_camera_plan(path: str | Path) -> dict[str, Any]:
    """Revalidate externally supplied plan before passing paths into Blender."""
    plan_path = Path(path)
    try:
        if plan_path.stat().st_size > 512*1024:
            raise ReferenceError("Camera plan exceeds 512 KiB")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if (type(plan.get("schema_version")) is not int or plan["schema_version"] != 1
                or plan.get("status") != "alignment-candidate / artistic-review-required"
                or not isinstance(plan.get("cameras"), list)
                or not 3 <= len(plan["cameras"]) <= 256):
            raise ValueError("invalid camera plan header")
        seen = set()
        for camera in plan["cameras"]:
            image_id = camera["image_id"]
            if (not isinstance(image_id, str) or not image_id.isascii()
                    or not image_id.replace("-", "").isalnum() or image_id in seen):
                raise ValueError("duplicate or unsafe camera image_id")
            seen.add(image_id)
            intrinsics = camera["intrinsics"]
            width, height = intrinsics["width"], intrinsics["height"]
            if (type(width) is not int or type(height) is not int
                    or min(width, height) < 64 or width*height > 40_000_000):
                raise ValueError("bad camera dimensions")
            for key in ("fx_px", "fy_px", "cx_px", "cy_px"):
                value = intrinsics[key]
                if type(value) not in (float, int) or not math.isfinite(value):
                    raise ValueError("bad camera intrinsics")
            if (intrinsics["fx_px"] <= 0 or intrinsics["fy_px"] <= 0
                    or abs(intrinsics["fx_px"]-intrinsics["fy_px"])/intrinsics["fx_px"] > 0.005):
                raise ValueError("non-square pixel focal lengths")
            position = camera["translation_m"]
            rotation = camera["rotation_3x3"]
            if len(position) != 3 or len(rotation) != 3 or any(len(row) != 3 for row in rotation):
                raise ValueError("invalid camera transform shape")
            values = [*position, *(v for row in rotation for v in row)]
            if any(type(v) not in (float, int) or not math.isfinite(v) or abs(v) > 1e8 for v in values):
                raise ValueError("invalid camera transform values")
            for i in range(3):
                for j in range(3):
                    dot = sum(rotation[k][i]*rotation[k][j] for k in range(3))
                    if abs(dot - float(i == j)) > 1e-3:
                        raise ValueError("camera rotation is not orthonormal")
            photo = Path(camera["source_photo_path"])
            if not photo.is_absolute() or not photo.is_file():
                raise ValueError("registered source photograph unavailable")
            if "source_photo_sha256" in camera:
                with photo.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != camera["source_photo_sha256"]:
                        raise ValueError("registered source photograph changed after camera planning")
            if camera["source_mask_path"] is not None:
                mask = Path(camera["source_mask_path"])
                if not mask.is_absolute() or not mask.is_file():
                    raise ValueError("registered source mask unavailable")
                if "source_mask_sha256" in camera:
                    with mask.open("rb") as stream:
                        if hashlib.file_digest(stream, "sha256").hexdigest() != camera["source_mask_sha256"]:
                            raise ValueError("registered silhouette mask changed after camera planning")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        if isinstance(error, ReferenceError):
            raise
        raise ReferenceError(f"Invalid camera plan: {error}") from error
    return plan


def build_reference_scene(plan_file: str | Path, source_glb: str | Path,
                          output: str | Path, *, render: bool = False,
                          max_cameras: int = 8, bpy_module: Any = None) -> dict[str, Any]:
    """Import local sculpt GLB, attach SfM photo cameras, render alpha clay masks.

    The caller owns the source GLB's placement in the survey/anchor coordinate
    frame. Nothing is automatically moved or claimed to be historically correct.
    """
    if bpy_module is None:
        raise RuntimeError("Reference scenes require Blender bpy")
    bpy = bpy_module
    if tuple(bpy.app.version) < (4, 2, 0):
        raise ReferenceError("Reference scenes need Blender 4.2+")
    if type(max_cameras) is not int or not 1 <= max_cameras <= 32:
        raise ReferenceError("max_cameras must be 1–32")
    target = Path(output).resolve()
    if Path(output).is_symlink() or target.exists():
        raise ReferenceError(f"Reference scene output already exists: {target}")
    plan = load_camera_plan(plan_file)
    source = Path(source_glb).resolve(strict=True)
    if not source.is_file() or source.suffix.lower() != ".glb":
        raise ReferenceError("Reference scene source must be a local GLB")
    from kumiki_blender.asset_import import inspect_source_glb
    inspect_source_glb(source)  # Static mesh validation; private scene, no export permission inferred.
    from mathutils import Matrix  # type: ignore[import-not-found]

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    bpy.ops.import_scene.gltf(filepath=str(source))
    mesh_count = sum(obj.type == "MESH" for obj in scene.objects)
    if mesh_count == 0:
        raise ReferenceError("Reference scene GLB contains no importable mesh")
    for obj in scene.objects:
        if obj.type == "MESH":
            obj.select_set(False)

    review = bpy.data.collections.new("Review / cameras and lighting (not GLB)")
    scene.collection.children.link(review)
    clay = bpy.data.materials.new("Review-only neutral clay")
    clay.diffuse_color = (0.65, 0.67, 0.65, 1.0)
    clay.use_nodes = True
    bsdf = clay.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.65, 0.67, 0.65, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.9
    scene.view_layers[0].material_override = clay
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 8
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1
    world = bpy.data.worlds.new("Review-only ambient")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.6, 0.6, 0.6, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.8
    lamp_data = bpy.data.lights.new("Review sun", type="SUN")
    lamp_data.energy = 1.5
    lamp = bpy.data.objects.new("Review sun", lamp_data)
    review.objects.link(lamp)
    lamp.rotation_euler = (0.6, -0.4, -0.4)

    camera_objects = []
    for entry in plan["cameras"]:
        intr = entry["intrinsics"]
        cam_data = bpy.data.cameras.new("COLMAP / " + entry["image_id"])
        cam_data.type = "PERSP"
        cam_data.sensor_fit = "HORIZONTAL"
        cam_data.sensor_width = 36.0
        cam_data.lens = intr["fx_px"] * 36.0 / intr["width"]
        cam_data.shift_x = (intr["width"] / 2 - intr["cx_px"]) / intr["width"]
        cam_data.shift_y = (intr["cy_px"] - intr["height"] / 2) / intr["width"]
        cam_data.show_background_images = True
        background = cam_data.background_images.new()
        background.image = bpy.data.images.load(entry["source_photo_path"], check_existing=True)
        background.alpha = 0.4
        camera = bpy.data.objects.new(cam_data.name, cam_data)
        review.objects.link(camera)
        rotation = entry["rotation_3x3"]
        center = entry["translation_m"]
        camera.matrix_world = Matrix((
            (*rotation[0], center[0]), (*rotation[1], center[1]),
            (*rotation[2], center[2]), (0, 0, 0, 1),
        ))
        camera_objects.append((entry, camera))
    scene.camera = camera_objects[0][1]
    target.mkdir(parents=True, exist_ok=False)
    bpy.ops.wm.save_as_mainfile(filepath=str(target / "reference-review.blend"))
    rendered = []
    if render:
        (target / "renders").mkdir()
        for entry, camera in camera_objects[:max_cameras]:
            scene.camera = camera
            scene.render.resolution_x = entry["intrinsics"]["width"]
            scene.render.resolution_y = entry["intrinsics"]["height"]
            frame = "renders/" + entry["image_id"] + ".png"
            scene.render.filepath = str(target / frame)
            bpy.ops.render.render(write_still=True)
            rendered.append(frame)
    summary = {
        "schema_version": 1, "asset_id": plan["asset_id"],
        "status": "private-clay-review / artistic-review-required",
        "source_glb": str(source), "mesh_objects": mesh_count,
        "registered_cameras": len(camera_objects), "rendered_clay_views": rendered,
        "review_blend": "reference-review.blend",
        "historical_fidelity_verified": False,
        "limitations": "GLB vertices must already use the anchor target coordinate frame; Blender clay masks and linked reference photos are for private artistic review only.",
    }
    (target / "review-report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
