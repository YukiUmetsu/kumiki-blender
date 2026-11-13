"""Blender adapter: immutable, bounded pre-rig surface-fairing candidates.

A source .blend is loaded by the worker runner. No source save, modifier
application, topology rebuild, or implicit material/UV edits are permitted.
A separate Python orchestration layer may reject against registered masks.
"""
from __future__ import annotations

import hashlib
import struct
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kumiki_blender.character_build.blender_common import (  # noqa: E402
    activate, arguments, export_interchange_obj, export_registered_obj, write_report)
from kumiki_blender.character_build.surface_fairing import SurfacePolicy, fair_surface  # noqa: E402


def _mesh_digest(obj):
    sha = hashlib.sha256()
    for vertex in obj.data.vertices:
        sha.update(struct.pack("<3d", *vertex.co))
    for poly in obj.data.polygons:
        sha.update(struct.pack("<I", len(poly.vertices)))
        for idx in poly.vertices:
            sha.update(struct.pack("<I", idx))
        sha.update(struct.pack("<I", poly.material_index))
    for layer in obj.data.uv_layers:
        sha.update(layer.name.encode("utf-8"))
        for item in layer.data:
            sha.update(struct.pack("<2d", *item.uv))
    for material in obj.data.materials:
        sha.update((material.name if material else "(none)").encode("utf-8"))
    return sha.hexdigest()


def _file_digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def _group_weights(obj, name):
    group = obj.vertex_groups.get(name)
    if group is None:
        raise ValueError(f"{obj.name}: no vertex group named {name!r}")
    weights = np.zeros(len(obj.data.vertices), dtype=float)
    for vertex in obj.data.vertices:
        for assignment in vertex.groups:
            if assignment.group == group.index:
                weights[vertex.index] = assignment.weight
    if not np.any(weights >= 0.5):
        raise ValueError(f"{obj.name}: vertex group {name!r} has no weight >= 0.5")
    return weights


def _coordinates(obj):
    world = obj.matrix_world
    return np.asarray([tuple(world @ vertex.co) for vertex in obj.data.vertices], dtype=float)


def _triangles(obj):
    obj.data.calc_loop_triangles()
    return np.asarray([tuple(triangle.vertices) for triangle in obj.data.loop_triangles], dtype=np.int64)


def _world_bvh(obj, points=None):
    vertices = ([Vector(row) for row in points] if points is not None else
                [obj.matrix_world @ v.co for v in obj.data.vertices])
    polygons = [tuple(poly.vertices) for poly in obj.data.polygons]
    if not polygons:
        raise ValueError(f"{obj.name}: collision guard has no polygonal faces")
    return BVHTree.FromPolygons(vertices, polygons)


def main():
    args = arguments()
    policy = SurfacePolicy(**args["plan"])
    source = Path(args["source_blend"]).resolve()
    out = Path(args["output"]).resolve()
    if not source.is_file() or source.suffix.lower() != ".blend":
        raise ValueError("Expected an existing .blend source")
    if Path(bpy.data.filepath).resolve() != source:
        raise ValueError("Refusing to refine a different Blender file")
    if out.exists() or source.is_relative_to(out):
        raise FileExistsError("Surface refinement requires a new independent output revision")
    source_sha = _file_digest(source)
    if source_sha != args["source_sha256"]:
        raise ValueError("Source .blend changed after the command planned its refinement")
    obj = bpy.data.objects.get(policy.mesh)
    if obj is None or obj.type != "MESH":
        raise ValueError(f"Missing exact pre-rig mesh object: {policy.mesh}")
    if obj.library or obj.data.library or obj.data.shape_keys:
        raise ValueError("Linked or shape-key meshes require an explicit morph-aware workflow")
    if obj.modifiers or obj.find_armature() is not None:
        raise ValueError("Surface fairing is pre-rig only; remove modifiers/armature only on a source copy")
    if obj.matrix_world.determinant() <= 0:
        raise ValueError("Mirrored/singular transforms unsupported")
    guard_objects = []
    for first, second in policy.guard_pairs:
        name = second if first == policy.mesh else first
        neighbour = bpy.data.objects.get(name)
        if neighbour is None or neighbour.type != "MESH":
            raise ValueError(f"Collision guard requires an existing mesh object {name}")
        if neighbour.modifiers:
            raise ValueError(f"Collision guard {name} has modifiers; evaluated mesh QA is required")
        guard_objects.append((name, neighbour))
    originals = {entry.name: _mesh_digest(entry) for entry in bpy.data.objects if entry.type == "MESH"}
    protected = np.zeros(len(obj.data.vertices), dtype=bool)
    for name in policy.protected_vertex_groups:
        protected |= _group_weights(obj, name) >= 0.5
    editable = (_group_weights(obj, policy.editable_vertex_group)
                if policy.editable_vertex_group else None)
    original_positions = _coordinates(obj)
    candidate, report = fair_surface(
        original_positions, _triangles(obj), policy, protected=protected, editable=editable)
    collision_guards = []
    if guard_objects and report["status"] != "no-safe-improvement":
        previous_tree = _world_bvh(obj, original_positions)
        updated_tree = _world_bvh(obj, candidate)
        for name, neighbour in guard_objects:
            fixed_tree = _world_bvh(neighbour)
            previous_count = len(previous_tree.overlap(fixed_tree))
            revised_count = len(updated_tree.overlap(fixed_tree))
            collision_guards.append({
                "objects": [obj.name, name],
                "before_triangle_overlap_pairs": previous_count,
                "after_triangle_overlap_pairs": revised_count,
                "pass": revised_count <= previous_count,
            })
        if any(not entry["pass"] for entry in collision_guards):
            report["status"] = "no-safe-improvement"
            report["reason"] = "adjacent_part_collision_regression"
            report["rejected_candidate_metrics"] = {k: report[k] for k in (
                "baseline_roughness", "final_roughness", "maximum_displacement_m")}
            candidate = original_positions.copy()
            report["changed_vertices"] = 0
            report["accepted_iterations"] = 0
            report["improvement_fraction"] = 0.0
            report["volume_change_fraction"] = 0.0
            report["maximum_displacement_m"] = 0.0
            report["final_roughness"] = report["baseline_roughness"]
    report["collision_guards"] = collision_guards
    if obj.data.users > 1 and report["status"] != "no-safe-improvement":
        obj.data = obj.data.copy()  # Never edit another object sharing the same datablock.

    outputs = []
    if report["status"] != "no-safe-improvement":
        out.mkdir(parents=True)
        activate(obj)
        export_registered_obj(out / "baseline-source.obj")
        inverse = obj.matrix_world.inverted_safe()
        for vertex, position in zip(obj.data.vertices, candidate):
            vertex.co = inverse @ Vector(position)
        obj.data.update()
        bpy.context.view_layer.update()
        export_registered_obj(out / "surface-candidate-source.obj")
        export_interchange_obj(out / "surface-candidate.obj")
        bpy.ops.wm.save_as_mainfile(filepath=str(out / "surface-candidate.blend"))
        outputs = ["baseline-source.obj", "surface-candidate-source.obj",
                   "surface-candidate.obj", "surface-candidate.blend"]
    others_after = {entry.name: _mesh_digest(entry) for entry in bpy.data.objects
                    if entry.type == "MESH" and entry.name != obj.name}
    if any(value != others_after.get(name) for name, value in originals.items()
           if name != obj.name):
        raise ValueError("Surface refinement unexpectedly modified unrelated mesh geometry")
    if _file_digest(source) != source_sha:
        raise ValueError("Source .blend was modified during refinement")
    report.update({
        "schema_version": 1, "source_blend": str(source), "source_sha256": source_sha,
        "object": obj.name, "baseline_mesh_sha256": originals[obj.name],
        "revised_mesh_sha256": _mesh_digest(obj), "other_meshes_preserved": True,
        "source_blend_preserved": True,
        "coordinate_frame": "Blender world metres, Z up",
        "files": outputs,
    })
    write_report(out, report)


main()
