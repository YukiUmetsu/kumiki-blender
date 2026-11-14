"""Native Blender: frame-locked clay orthographic turnarounds and part close-ups.

Renders actual evaluated rig geometry without modifying its saved .blend.
References are intentionally NOT compared via arbitrary image registration:
no IoU or likeness score is reported for unmatched camera/pose drawings.
"""
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kumiki_blender.character_build.blender_common import arguments, write_report  # noqa: E402
from kumiki_blender.character_build.character_review_plan import (  # noqa: E402
    confirm_review_sources,
)

# +Z is up and -Y is the front of canonical character construction space.
_CAMERAS = {
    "front": (0, -1, 0),
    "right": (1, 0, 0),
    "back": (0, 1, 0),
    "left": (-1, 0, 0),
    "front_right": (1, -1, 0),
    "front_left": (-1, -1, 0),
    "back_right": (1, 1, 0),
    "back_left": (-1, 1, 0),
    "top": (0, 0, 1),
}


def _world_points(obj, *, group_name=None):
    """Exact evaluated mesh points; vertex group requires preserved topology."""
    dg = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(dg)
    mesh = evaluated.to_mesh()
    try:
        if not 3 <= len(mesh.vertices) <= 500_000:
            raise ValueError(f"{obj.name}: evaluated mesh exceeds review vertex safety limits")
        selected = None
        if group_name is not None:
            group = obj.vertex_groups.get(group_name)
            if group is None:
                raise ValueError(f"{obj.name}: missing focus region group {group_name}")
            if len(obj.data.vertices) != len(mesh.vertices):
                raise ValueError(
                    f"{obj.name}/{group_name}: topology-changing modifier prevents "
                    "reliable vertex-group focus; create a same-topology LOD")
            selected = set()
            for vertex in obj.data.vertices:
                try:
                    weight = group.weight(vertex.index)
                except RuntimeError:
                    weight = 0
                if weight > 0.1:
                    selected.add(vertex.index)
            if len(selected) < 3:
                raise ValueError(
                    f"{obj.name}/{group_name}: fewer than 3 weighted focus vertices")
        result = []
        world = evaluated.matrix_world
        for vertex in mesh.vertices:
            if selected is None or vertex.index in selected:
                point = world @ vertex.co
                if not all(math.isfinite(v) for v in point):
                    raise ValueError(f"{obj.name}: nonfinite evaluated vertex coordinates")
                result.append(point)
        return result
    finally:
        evaluated.to_mesh_clear()


def _extents(points):
    if not points:
        raise ValueError("No evaluated character points to frame")
    bottom = Vector([min(p[i] for p in points) for i in range(3)])
    top = Vector([max(p[i] for p in points) for i in range(3)])
    span = top - bottom
    if max(span) <= 1e-7:
        raise ValueError("Character review geometry has negligible 3D extent")
    return bottom, top


def _bounds_union(existing, current):
    if existing is None:
        return current
    lo = Vector([min(existing[0][i], current[0][i]) for i in range(3)])
    hi = Vector([max(existing[1][i], current[1][i]) for i in range(3)])
    return lo, hi


def _review_bounds(plan, parts):
    """Lock each full/part ROI across the entire requested animated frame set."""
    bounds = {"full": None}
    for region in plan["regions"]:
        bounds[region["id"]] = None
    budget = 0
    for frame in plan["frames"]:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        evaluated = {}
        for obj in parts:
            points = _world_points(obj)
            evaluated[obj.name] = points
            budget += len(points)
            if budget > 2_000_000:
                raise ValueError("Review exceeded 2 million evaluated vertices across frames")
        bounds["full"] = _bounds_union(
            bounds["full"],
            _extents([p for item in evaluated.values() for p in item]))
        for region in plan["regions"]:
            obj = next(p for p in parts if p.name == region["object"])
            points = (_world_points(obj, group_name=region["vertex_group"])
                      if region["vertex_group"] else evaluated[obj.name])
            bounds[region["id"]] = _bounds_union(
                bounds[region["id"]], _extents(points))
    return bounds


def _place_camera(camera_obj, bounds, direction, margin: float) -> dict:
    lo, hi = bounds
    center = (lo + hi) * 0.5
    toward_camera = Vector(direction).normalized()
    rotation = (-toward_camera).to_track_quat("-Z", "Y")
    camera_obj.rotation_euler = rotation.to_euler()
    span = max((hi - lo).length, 0.1)
    camera_obj.location = center + toward_camera * (span * 4 + 2)
    right = rotation @ Vector((1, 0, 0))
    up = rotation @ Vector((0, 1, 0))
    projected = [
        corner - center for x in (lo.x, hi.x) for y in (lo.y, hi.y)
        for z in (lo.z, hi.z) for corner in [Vector((x, y, z))]
    ]
    projected_width = max(p.dot(right) for p in projected) - min(
        p.dot(right) for p in projected)
    projected_height = max(p.dot(up) for p in projected) - min(
        p.dot(up) for p in projected)
    camera_obj.data.type = "ORTHO"
    camera_obj.data.ortho_scale = max(
        .1, projected_width * margin, projected_height * margin)
    camera_obj.data.clip_start = .01
    camera_obj.data.clip_end = max(100, span * 12)
    return {
        "center_m": [round(v, 6) for v in center],
        "camera_xyz_m": [round(v, 6) for v in camera_obj.location],
        "ortho_scale_m": round(camera_obj.data.ortho_scale, 6),
        "bounded_bbox_m": [
            [round(v, 6) for v in lo],
            [round(v, 6) for v in hi],
        ],
    }


def _configure_clay(plan):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "SINGLE"
    scene.display.shading.single_color = (.73, .76, .8)
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "WORLD"
    scene.display.shading.background_type = "WORLD"
    if scene.world is not None:
        scene.world.color = (.045, .055, .07)
    scene.render.film_transparent = False
    scene.render.resolution_x = plan["resolution"]
    scene.render.resolution_y = plan["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    return scene


def main():
    args = arguments()
    plan = args["plan"]
    confirm_review_sources(plan)
    source = Path(plan["source_blend"]).resolve()
    output = Path(args["output"]).resolve()
    if output.exists():
        raise FileExistsError(f"Review output revision already exists: {output}")
    if not bpy.data.filepath or Path(bpy.data.filepath).resolve() != source:
        raise ValueError("Review worker must open the exact SHA-locked source .blend")
    rig = bpy.data.objects.get(plan["armature"])
    if rig is None or rig.type != "ARMATURE":
        raise ValueError(f"Named source armature is missing: {plan['armature']}")
    parts = sorted([
        obj for obj in bpy.data.objects
        if obj.type == "MESH" and not obj.hide_render and
        any(mod.type == "ARMATURE" and mod.object == rig for mod in obj.modifiers)
    ], key=lambda obj: obj.name)
    if not 1 <= len(parts) <= 64:
        raise ValueError("Expected 1-64 visible mesh parts skinned to the review rig")
    for region in plan["regions"]:
        if region["object"] not in {obj.name for obj in parts}:
            raise ValueError(
                f"Focus region {region['id']} targets unknown/hidden skinned part: "
                f"{region['object']}")
    bound = _review_bounds(plan, parts)
    scene = _configure_clay(plan)
    camera = bpy.data.cameras.new("KumikiReviewCamera")
    camera_obj = bpy.data.objects.new("KumikiReviewCamera", camera)
    scene.collection.objects.link(camera_obj)
    scene.camera = camera_obj

    output.parent.mkdir(parents=True, exist_ok=True)
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=False)
    entries = []
    scopes = [{"id": "full", "object": None, "vertex_group": None}, *plan["regions"]]
    for scope in scopes:
        for view in plan["views"]:
            pose = _place_camera(
                camera_obj, bound[scope["id"]],
                _CAMERAS[view], plan["margin"])
            for frame in plan["frames"]:
                scene.frame_set(frame)
                scene.render.filepath = str(images_dir /
                                            f"{scope['id']}-{view}-f{frame:05d}.png")
                bpy.ops.render.render(write_still=True)
                image = Path(scene.render.filepath)
                if not image.is_file() or image.stat().st_size < 100:
                    raise RuntimeError(f"Blender did not produce expected review PNG: {image}")
                entries.append({
                    "scope": scope["id"], "view": view, "frame": frame,
                    "render": f"images/{image.name}", "camera": pose,
                    "reference": (
                        plan["reference_views"].get(view)
                        if scope["id"] == "full" else scope["references"].get(view)
                    ),
                })
    confirm_review_sources(plan)
    report = {
        "schema_version": 1, "status": "source-visual-review-required",
        "source_blend": str(source),
        "source_sha256": plan["source_sha256"],
        "armature": rig.name,
        "render_style": "workbench-neutral-clay",
        "source_camera_frame": "Blender metres; +Z up, front -Y",
        "camera_alignment": plan["reference_alignment"],
        "frame_locked": True,
        "views": plan["views"], "frames": plan["frames"],
        "visible_skinned_parts": [obj.name for obj in parts],
        "regions": plan["regions"],
        "images": entries,
        "contact_sheet": None,
        "limitations": [
            "Reference artwork is NOT camera/pose registered; pixel IoU and silhouette matching are deliberately omitted.",
            "Workbench clay ignores final character PBR, UV textures and color fidelity.",
            "Camera target and ortho scale are locked across animation frames for each review scope.",
            "Region focus uses exact existing object or vertex group, not inferred anatomy.",
            "Passing screenshot generation is NOT approval of source fidelity; compare by eye with validated front/side/back art.",
            "Character source .blend and reference PNG/JPEG/WEBP files are SHA-locked and unchanged.",
        ],
    }
    write_report(output, report)


main()
