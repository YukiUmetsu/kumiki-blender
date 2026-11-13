"""Native Blender adapter for explicit, bounded object-pair penetration repair.

No heuristic chooses which object should move. The user supplies directed
(moving, fixed) collision pairs. Fixed meshes must be closed/outward, moving
meshes must have stable editable topology and no modifiers/shape keys.
Every pair is evaluated both by signed vertex clearance AND exact BVH
triangle-overlap candidates. Outputs are immutable Blender revisions.
"""
from __future__ import annotations

import hashlib
import math
import struct
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kumiki_blender.character_build.blender_common import arguments, write_report  # noqa: E402
from kumiki_blender.character_build.penetration_repair import CollisionPolicy, repair_penetration  # noqa: E402


def _mesh_digest(obj) -> str:
    digest = hashlib.sha256()
    for v in obj.data.vertices:
        digest.update(struct.pack("<3d", *v.co))
    for polygon in obj.data.polygons:
        digest.update(struct.pack("<I", len(polygon.vertices)))
        for vertex_id in polygon.vertices:
            digest.update(struct.pack("<I", vertex_id))
    return digest.hexdigest()


def _mesh_world(obj):
    matrix = obj.matrix_world
    return np.array([tuple(matrix @ vertex.co) for vertex in obj.data.vertices], dtype=float)


def _triangles(obj):
    obj.data.calc_loop_triangles()
    return np.asarray([tuple(tri.vertices) for tri in obj.data.loop_triangles], dtype=np.int64)


def _bvh(obj):
    vertices = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    polygons = [tuple(polygon.vertices) for polygon in obj.data.polygons]
    return BVHTree.FromPolygons(vertices, polygons)


def _closed_outward(obj) -> None:
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        if not bm.faces or any(not edge.is_manifold for edge in bm.edges):
            raise ValueError(f"{obj.name}: fixed collision mesh must be closed/manifold")
        if bm.calc_volume(signed=True) <= 1e-10:
            raise ValueError(f"{obj.name}: fixed collision mesh must be outward oriented")
        if obj.matrix_world.determinant() <= 0:
            raise ValueError(f"{obj.name}: mirrored/singular fixed object transform unsupported")
    finally:
        bm.free()


def _editable(obj) -> None:
    if obj is None or obj.type != "MESH":
        raise ValueError("Collision object missing or not a mesh")
    if obj.library is not None or obj.data.library is not None:
        raise ValueError(f"{obj.name}: linked-library mesh cannot be modified")
    if obj.data.shape_keys or obj.modifiers:
        raise ValueError(f"{obj.name}: modifiers/shape keys require deformation-aware repair")
    if obj.matrix_world.determinant() <= 0:
        raise ValueError(f"{obj.name}: moving object transform must not be mirrored/singular")


def _nearest_signed(tree, aabb: tuple[np.ndarray, np.ndarray], clearance: float):
    """Avoid BVH searches for vertices demonstrably outside the blocker AABB."""
    low, high = aabb

    def query(points: np.ndarray):
        signed = np.empty(len(points), dtype=float)
        normals = np.zeros_like(points)
        # The AABB separation is a conservative lower bound for distance to
        # a closed mesh. Vertices farther than clearance cannot penetrate.
        gap = np.maximum(np.maximum(low - points, points - high), 0.0)
        outside = np.linalg.norm(gap, axis=1)
        far = outside > clearance + 1e-8
        signed[far] = outside[far]
        normals[far, 0] = 1.0
        for i in np.flatnonzero(~far):
            p = points[i]
            loc, norm, _face, distance = tree.find_nearest(Vector(p))
            if loc is None or norm is None or norm.length < 1e-9:
                raise ValueError("Fixed mesh has no nearest surface")
            offset = Vector(p) - loc
            signed[i] = math.copysign(distance, offset.dot(norm)) if distance > 0 else 0
            normals[i] = tuple(norm)
        return signed, normals
    return query


def _apply_world(obj, points: np.ndarray):
    inverse = obj.matrix_world.inverted_safe()
    for vertex, position in zip(obj.data.vertices, points):
        vertex.co = inverse @ Vector(position)
    obj.data.update()
    bpy.context.view_layer.update()


def main() -> None:
    args = arguments()
    out = Path(args["output"]).resolve()
    source = Path(args["source_blend"]).resolve()
    if not source.is_file() or not bpy.data.filepath or Path(bpy.data.filepath).resolve() != source:
        raise ValueError("Must load exactly the supplied source .blend")
    if out.exists() or out == source.parent or source.is_relative_to(out):
        raise FileExistsError("Output revision must be new and separate from source .blend")
    plan = args["plan"]
    plan["pairs"] = [CollisionPolicy(**pair) for pair in plan["pairs"]]
    objects = {obj.name: obj for obj in bpy.data.objects if obj.type == "MESH"}
    targets = {p.moving for p in plan["pairs"]}
    before_all = {name: _mesh_digest(obj) for name, obj in objects.items()}
    for policy in plan["pairs"]:
        if policy.moving not in objects or policy.fixed not in objects:
            raise ValueError(f"Missing pair: {policy.moving} / {policy.fixed}")
        _editable(objects[policy.moving])
        _closed_outward(objects[policy.fixed])
        if objects[policy.fixed].modifiers:
            raise ValueError("Fixed collision mesh with modifiers is unsupported")
    # Copy-on-write avoids editing other scene objects using the same datablock.
    for name in targets:
        obj = objects[name]
        if obj.data.users > 1:
            obj.data = obj.data.copy()
    results = []
    accepted_pairs = {}
    for policy in plan["pairs"]:
        movable, fixed = objects[policy.moving], objects[policy.fixed]
        original = _mesh_world(movable)
        before_tree = _bvh(movable)
        fixed_tree = _bvh(fixed)
        before_overlap = len(before_tree.overlap(fixed_tree))
        blocker = _mesh_world(fixed)
        bounds = (blocker.min(axis=0), blocker.max(axis=0))
        candidate, metrics = repair_penetration(original, _triangles(movable), policy,
                                                _nearest_signed(fixed_tree, bounds, policy.clearance_m))
        applied = False
        after_overlap = before_overlap
        blocked_reason = None
        if metrics["status"] == "geometric-candidate-review-required":
            _apply_world(movable, candidate)
            after_overlap = len(_bvh(movable).overlap(fixed_tree))
            # A reduction in vertex penetration with more crossing triangles
            # is NOT a safe improvement. Check all previously accepted pairs
            # as well, because a repair must not break another relationship.
            if after_overlap > before_overlap:
                blocked_reason = "triangle_overlap_regression"
            for (_, prev_moving, prev_fixed), previous_count in accepted_pairs.items():
                if (prev_moving == policy.moving or prev_fixed == policy.moving):
                    current_count = len(_bvh(objects[prev_moving]).overlap(_bvh(objects[prev_fixed])))
                    if current_count > previous_count:
                        blocked_reason = "earlier_collision_pair_regression"
            if blocked_reason:
                _apply_world(movable, original)
                after_overlap = before_overlap
            else:
                applied = True
                accepted_pairs[(len(results), policy.moving, policy.fixed)] = after_overlap
        entry = {
            "moving": policy.moving, "fixed": policy.fixed,
            "before_triangle_overlap_pairs": before_overlap,
            "after_triangle_overlap_pairs": after_overlap,
            "accepted": applied,
            "rejection": blocked_reason,
            **metrics,
        }
        if blocked_reason:
            entry["status"] = "rejected-rolled-back"
        if applied and (after_overlap or metrics["candidate_penetrating_vertices"]):
            entry["status"] = "partial-candidate-review-required"
        results.append(entry)
    after_all = {name: _mesh_digest(obj) for name, obj in objects.items()}
    if any(before_all[name] != after_all[name] for name in objects if name not in targets):
        raise ValueError("Non-target geometry was unexpectedly modified")
    changed = [name for name in targets if before_all[name] != after_all[name]]
    out.mkdir(parents=True)
    files = []
    if changed:
        bpy.ops.wm.save_as_mainfile(filepath=str(out / "collision-candidate.blend"))
        files.append("collision-candidate.blend")
    write_report(out, {
        "schema_version": 1,
        "status": "geometric-candidate-review-required" if changed else "no-safe-improvement",
        "source_blend": str(source),
        "changed_objects": sorted(changed),
        "pairs": results,
        "files": files,
        "limitations": [
            "Only explicitly named parts may move; no inferred object or anatomical intent.",
            "Closest-surface normals approximate penetration only for closed/outward fixed meshes.",
            "BVH triangle-overlap candidates may include allowed contacts, and cannot prove volume penetration.",
            "A mesh with crossing faces but no penetrating vertices may remain unresolved.",
            "No silhouette, facial fidelity, skinning, or animated pose certification is implied.",
            "Only a separate .blend candidate is written; the input .blend is unchanged."
        ],
    })


main()
