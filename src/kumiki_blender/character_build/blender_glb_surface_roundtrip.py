"""Native Blender QA: compare evaluated rig and reimported GLB geometry.

Samples the original saved baked.blend at reviewed clip frames, then opens a
fresh Blender scene and reimports animated.glb. Compares world-space vertex
point clouds and their bounds in BOTH directions using KD-trees, avoiding
invalid assumptions about vertex ordering after glTF triangulation/UV splits.

A passing test verifies evaluated mesh surfaces for the sampled frames, not
materials, subframe motion, artistic identity, or render-engine equivalence.
"""
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector
from mathutils.kdtree import KDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kumiki_blender.character_build.blender_common import arguments, write_report  # noqa: E402
from kumiki_blender.character_build.glb_animation import read_single_animation  # noqa: E402
from kumiki_blender.character_build.roundtrip_plan import (  # noqa: E402
    file_sha256, frame_samples, validate_roundtrip_plan,
)


def _points(obj, limit: int) -> list[tuple[float, float, float]]:
    graph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(graph)
    mesh = eval_obj.to_mesh()
    try:
        if not 3 <= len(mesh.vertices) <= limit:
            raise ValueError(
                f"{obj.name}: evaluated vertex count {len(mesh.vertices)} outside [3, {limit}]")
        world = eval_obj.matrix_world
        result = [tuple(world @ v.co) for v in mesh.vertices]
        if not all(math.isfinite(c) for point in result for c in point):
            raise ValueError(f"{obj.name}: evaluated vertex coordinates are nonfinite")
        return result
    finally:
        eval_obj.to_mesh_clear()


def _bounds(points):
    return [(min(v[axis] for v in points), max(v[axis] for v in points))
            for axis in range(3)]


def _nearest_distances(first, second):
    tree = KDTree(len(second))
    for index, point in enumerate(second):
        tree.insert(Vector(point), index)
    tree.balance()
    return [float(tree.find(Vector(point))[2]) for point in first]


def _p95(values):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(.95 * len(ordered)) - 1)]


def compare_world_surfaces(source, imported) -> dict:
    """Measured 3D point-cloud parity with symmetric nearest-neighbor audit."""
    if not source or not imported:
        raise ValueError("Both compared meshes need evaluated vertices")
    forward = _nearest_distances(source, imported)
    backward = _nearest_distances(imported, source)
    distances = forward + backward
    bounds_before, bounds_after = _bounds(source), _bounds(imported)
    bounds_error = max(
        abs(bounds_before[axis][end] - bounds_after[axis][end])
        for axis in range(3) for end in (0, 1))
    return {
        "source_vertices": len(source),
        "imported_vertices": len(imported),
        "max_nearest_surface_vertex_error_m": round(max(distances), 8),
        "p95_nearest_surface_vertex_error_m": round(_p95(distances), 8),
        "max_bounds_error_m": round(bounds_error, 8),
        "source_bounds_m": [[round(v, 6) for v in pair] for pair in bounds_before],
        "glb_bounds_m": [[round(v, 6) for v in pair] for pair in bounds_after],
        "note": "Bidirectional nearest evaluated vertices, not continuous triangle surface Hausdorff distance",
    }


def _record_source(plan, frames):
    rig = bpy.data.objects.get(plan["armature"])
    if rig is None or rig.type != "ARMATURE":
        raise ValueError(f"Source rig not found: {plan['armature']}")
    if bpy.context.scene.render.fps != plan["fps"]:
        raise ValueError(
            f"Source FPS {bpy.context.scene.render.fps} differs from requested {plan['fps']}")
    objects = sorted((
        obj for obj in bpy.data.objects if obj.type == "MESH" and any(
            modifier.type == "ARMATURE" and modifier.object == rig
            for modifier in obj.modifiers)
    ), key=lambda obj: obj.name)
    if not objects or len(objects) > 32:
        raise ValueError("Expected 1-32 skinned source character mesh parts")
    if any(obj.name.endswith(".001") for obj in objects):
        raise ValueError("Ambiguous duplicate source skinned mesh names")
    recorded = {obj.name: {} for obj in objects}
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        for obj in objects:
            recorded[obj.name][frame] = _points(obj, plan["max_vertices_per_mesh"])
    return recorded


def main():
    args = arguments()
    plan = args["plan"]
    validate_roundtrip_plan(plan)
    output = Path(args["output"]).resolve()
    if output.exists():
        raise FileExistsError(f"Roundtrip QA revision already exists: {output}")
    source = Path(plan["source_blend"]).resolve()
    glb = Path(plan["exported_glb"]).resolve()
    if (not source.is_file() or not glb.is_file() or
            file_sha256(source) != plan["source_sha256"] or
            file_sha256(glb) != plan["glb_sha256"]):
        raise ValueError("Baked Blender source or exported GLB changed after plan lock")
    if not bpy.data.filepath or Path(bpy.data.filepath).resolve() != source:
        raise ValueError("Load the exact baked Blender source named by the roundtrip plan")
    document = read_single_animation(glb)
    if not document.get("skins"):
        raise ValueError("Exported character GLB lacks skeleton skinning")
    frames = frame_samples(
        plan["first_frame"], plan["last_frame"], plan["samples"])
    original = _record_source(plan, frames)

    # A genuine fresh re-import is essential. Evaluating the exporter-side
    # Blender scene twice cannot establish GLB transport parity.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = plan["fps"]
    if "FINISHED" not in bpy.ops.import_scene.gltf(filepath=str(glb)):
        raise RuntimeError("Blender did not import the exported binary GLB")
    imported = {obj.name: obj for obj in bpy.data.objects if obj.type == "MESH"}
    missing = sorted(set(original) - set(imported))
    extras = sorted(set(imported) - set(original))
    problems = []
    if missing or extras:
        problems.append(f"Reimported GLB mesh identities differ: missing={missing}, extra={extras}")
    results = []
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        for name, poses in sorted(original.items()):
            obj = imported.get(name)
            if obj is None:
                continue
            try:
                current = _points(obj, plan["max_vertices_per_mesh"])
                measurement = compare_world_surfaces(poses[frame], current)
            except (ValueError, RuntimeError) as exc:
                problems.append(f"{name} frame {frame}: {exc}")
                continue
            measurement.update({"mesh": name, "frame": frame})
            results.append(measurement)
            if (measurement["p95_nearest_surface_vertex_error_m"] >
                    plan["max_p95_error_m"] + 1e-8):
                problems.append(
                    f"{name} frame {frame}: P95 {measurement['p95_nearest_surface_vertex_error_m']:.6f}m "
                    f"> {plan['max_p95_error_m']:.6f}m")
            if (measurement["max_nearest_surface_vertex_error_m"] >
                    plan["max_max_error_m"] + 1e-8 or
                    measurement["max_bounds_error_m"] >
                    plan["max_max_error_m"] + 1e-8):
                problems.append(
                    f"{name} frame {frame}: max vertex/bounds drift exceeds "
                    f"{plan['max_max_error_m']:.6f}m")
    if file_sha256(source) != plan["source_sha256"] or file_sha256(glb) != plan["glb_sha256"]:
        raise ValueError("Baked .blend or GLB bytes changed during read-only QA")
    expected_count = len(original) * len(frames)
    if len(results) != expected_count:
        problems.append(
            f"Only compared {len(results)} of {expected_count} expected mesh/frame combinations")
    report = {
        "schema_version": 1,
        "status": "structural-fail" if problems else "evaluated-glb-roundtrip-review-required",
        "source_blend": str(source),
        "source_blend_sha256": plan["source_sha256"],
        "exported_glb": str(glb),
        "exported_glb_sha256": plan["glb_sha256"],
        "armature": plan["armature"],
        "fps": plan["fps"],
        "sampled_frames": frames,
        "source_meshes": sorted(original),
        "checked_mesh_frames": len(results),
        "max_p95_error_m": plan["max_p95_error_m"],
        "max_max_error_m": plan["max_max_error_m"],
        "global_max_nearest_error_m": max((
            r["max_nearest_surface_vertex_error_m"] for r in results), default=None),
        "global_p95_worst_frame_m": max((
            r["p95_nearest_surface_vertex_error_m"] for r in results), default=None),
        "poses": results,
        "problems": problems,
        "limitations": [
            "Evaluated vertex nearest-neighbor distances are not continuous triangle Hausdorff distances.",
            "glTF reimport may split vertices at material/UV seams; vertex indices are intentionally not compared.",
            "Only explicitly sampled integer frames are checked, not subframe interpolation or every frame.",
            "Clay-shape parity cannot establish material/texture/color parity or original-art resemblance.",
            "A green structural result still requires 2D reference-based visual review and actual ContentGen playback.",
            "Original saved Blender rig and exported GLB are SHA-locked and never modified.",
        ],
    }
    write_report(output, report)


main()
