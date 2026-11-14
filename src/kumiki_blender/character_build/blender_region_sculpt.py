"""Blender worker: non-destructive region-local geometry correction.

Runs on an existing .blend, modifies only the named mesh in memory, and writes
a *different* .blend and an OBJ candidate. No voxel remeshing, UV loss, or
automatic changes to unrelated parts. Sculpt outcomes need independent QA.
"""
from __future__ import annotations

import hashlib
import math
import struct
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kumiki_blender.character_build.blender_common import arguments, export_interchange_obj, export_registered_obj, write_report  # noqa: E402
from kumiki_blender.character_build.surface_distance import surface_distances  # noqa: E402


def mesh_digest(obj) -> str:
    """Hash mesh topology, local shape AND world transform; include loose edges."""
    sha = hashlib.sha256(b"kumiki-character-sculpt-world-mesh-v2:")
    sha.update(struct.pack("<3Q", len(obj.data.vertices),
                           len(obj.data.edges), len(obj.data.polygons)))
    for row in obj.matrix_world:
        if any(not math.isfinite(v) for v in row):
            raise ValueError(f"{obj.name}: nonfinite object transform")
        sha.update(struct.pack("<4d", *row))
    for vertex in obj.data.vertices:
        if any(not math.isfinite(v) for v in vertex.co):
            raise ValueError(f"{obj.name}: nonfinite vertex coordinates")
        sha.update(struct.pack("<3d", *vertex.co))
    for edge in obj.data.edges:
        sha.update(struct.pack("<2I", *edge.vertices))
    for polygon in obj.data.polygons:
        sha.update(struct.pack("<I", len(polygon.vertices)))
        for index in polygon.vertices:
            sha.update(struct.pack("<I", index))
    return sha.hexdigest()


def _inspect(args: dict, out: Path) -> None:
    """Read-only source mesh metadata for safely authoring an indexed sculpt."""
    obj = bpy.data.objects.get(args["object"])
    if obj is None or obj.type != "MESH":
        raise ValueError("Inspect source requires a named mesh object")
    anchors = []
    for raw in args.get("anchors", []):
        if len(raw) != 3 or any(not math.isfinite(float(x)) for x in raw):
            raise ValueError("Anchor query must have three finite coordinates")
        center = Vector(raw)
        index, distance = min(
            ((vertex.index, (obj.matrix_world @ vertex.co - center).length)
             for vertex in obj.data.vertices),
            key=lambda row: row[1],
            default=(None, None),
        )
        if index is None:
            raise ValueError("Cannot find a seed on an empty mesh")
        anchors.append({
            "requested_center_m": list(raw),
            "nearest_vertex": index,
            "distance_m": round(distance, 9),
            "status": "candidate-seed-review-required",
        })
    write_report(out, {
        "schema_version": 1,
        "source_blend": str(Path(bpy.data.filepath).resolve()),
        "object": obj.name,
        "expected_vertex_count": len(obj.data.vertices),
        "expected_mesh_sha256": mesh_digest(obj),
        "anchor_candidates": anchors,
        "status": "source-probe-only",
        "limitations": [
            "Nearest vertex candidates require visual confirmation of the correct surface.",
            "Mesh checksum covers shape, edges, polygons and world transform, not materials or UVs.",
            "Source .blend is not modified.",
        ],
    })


def main() -> None:
    args = arguments()
    out = Path(args["output"]).resolve()
    if out.exists():
        raise FileExistsError(f"Output revision already exists: {out}")
    if not bpy.data.filepath:
        raise ValueError("refine-region must load a source .blend, not factory startup")
    if Path(bpy.data.filepath).resolve() != Path(args["source_blend"]).resolve():
        raise ValueError("Loaded Blender file differs from requested source .blend")
    if Path(bpy.data.filepath).resolve().is_relative_to(out):
        raise ValueError("Source .blend cannot be inside the output revision")
    if args.get("inspect_only", False):
        _inspect(args, out)
        return
    plan = args["plan"]
    obj = bpy.data.objects.get(plan["object"])
    if obj is None or obj.type != "MESH":
        raise ValueError(f"Expected named Blender mesh object: {plan['object']}")
    if obj.data.shape_keys is not None:
        raise ValueError("Vertex-level refinement on shape-key meshes requires a dedicated shape-key-aware workflow")
    if obj.library is not None or obj.data.library is not None:
        raise ValueError("Refuse to modify linked library mesh data")
    if obj.data.users > 1:
        # Avoid changing all objects that share the same mesh datablock.
        obj.data = obj.data.copy()
    if plan.get("expected_vertex_count") is not None and len(obj.data.vertices) != plan["expected_vertex_count"]:
        raise ValueError("Mesh changed since the refinement plan was authored")
    before_other = {o.name: mesh_digest(o) for o in bpy.data.objects
                    if o.type == "MESH" and o != obj}
    before_mesh = mesh_digest(obj)
    if (plan.get("expected_mesh_sha256") is not None and
            plan["expected_mesh_sha256"] != before_mesh):
        raise ValueError("Source mesh has changed; regenerate the geodesic sculpt plan")
    pinned = frozenset(plan.get("protected_vertex_indices", []))
    if any(index >= len(obj.data.vertices) for index in pinned):
        raise ValueError("Protected vertex ID exceeds the source mesh vertex count")
    before_normals = [polygon.normal.copy() for polygon in obj.data.polygons]
    transform = obj.matrix_world
    inverse_basis = transform.inverted_safe().to_3x3()
    initial = [v.co.copy() for v in obj.data.vertices]
    changed = set()
    operation_reports = []
    mesh_edges = [tuple(edge.vertices) for edge in obj.data.edges]
    for operation in plan["operations"]:
        center = Vector(operation["center_m"])
        delta = Vector(operation["move_m"])
        radius = float(operation["radius_m"])
        world_positions = [tuple(transform @ vertex.co) for vertex in obj.data.vertices]
        mode = operation.get("distance_mode", "euclidean")
        if mode == "geodesic":
            seed = operation["seed_vertex"]
            seed_offset = (Vector(world_positions[seed]) - center).length
            if seed_offset > operation["max_seed_distance_m"]:
                raise ValueError(
                    f"Geodesic seed {seed} is {seed_offset:.6f}m from the expected "
                    "source anchor; mesh or source alignment changed")
            distances = surface_distances(world_positions, mesh_edges, seed, radius,
                                          protected=pinned)
        else:
            distances = {
                i: distance for i, position in enumerate(world_positions)
                if i not in pinned
                for distance in ((Vector(position) - center).length,)
                if distance < radius
            }
        edited_this_operation = 0
        for index, distance in distances.items():
            alpha = max(0.0, 1 - distance / radius)
            if operation["falloff"] == "smoothstep":
                alpha = alpha * alpha * (3 - 2 * alpha)
            if alpha <= 0:
                continue
            obj.data.vertices[index].co += inverse_basis @ (delta * alpha)
            changed.add(index)
            edited_this_operation += 1
        operation_reports.append({
            "distance_mode": mode,
            "seed_vertex": operation.get("seed_vertex"),
            "edited_vertices": edited_this_operation,
        })
    if not changed:
        raise ValueError("No vertex fell inside the requested sculpt radius")
    obj.data.update()
    # Guard against collapsing or flipping faces. This checks local geometry, not
    # proof of correct silhouette or reference appearance.
    flipped = 0
    for original, polygon in zip(before_normals, obj.data.polygons):
        if original.length > 1e-8 and polygon.normal.length > 1e-8 and original.dot(polygon.normal) <= 0:
            flipped += 1
    if flipped:
        raise ValueError(f"Sculpt rejected: {flipped} faces flipped orientation")
    displacement = [((transform @ vertex.co) - (transform @ initial[vertex.index])).length
                    for vertex in obj.data.vertices]
    maximum = max(displacement)
    if maximum > plan.get("max_displacement_m", .05) + 1e-6:
        raise ValueError(f"Displacement {maximum:.6f} exceeds the plan budget")
    untouched = [i for i in range(len(displacement)) if i not in changed]
    if any(displacement[i] > 1e-8 for i in untouched):
        raise ValueError("Out-of-region vertices changed unexpectedly")
    after_other = {o.name: mesh_digest(o) for o in bpy.data.objects
                   if o.type == "MESH" and o != obj}
    if before_other != after_other:
        raise ValueError("A non-target mesh changed during region refinement")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "refined.blend"))
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    export_interchange_obj(out / "region.obj")
    export_registered_obj(out / "region-source.obj")
    write_report(out, {
        "schema_version": 1, "status": "geometric-candidate-art-review-required",
        "source_blend": str(Path(args["source_blend"]).resolve()),
        "object": obj.name,
        "source_mesh_sha256": before_mesh,
        "refined_mesh_sha256": mesh_digest(obj),
        "untouched_mesh_sha256": before_other,
        "expected_mesh_sha256_verified": plan.get("expected_mesh_sha256") == before_mesh
            if plan.get("expected_mesh_sha256") else None,
        "protected_vertex_count": len(pinned),
        "operations": operation_reports,
        "edited_vertices": len(changed),
        "untouched_vertices": len(untouched),
        "total_vertices": len(obj.data.vertices),
        "maximum_displacement_m": round(maximum, 7),
        "flipped_faces": flipped,
        "files": ["refined.blend", "region.obj", "region-source.obj"],
        "limitations": [
            "This is a bounded positional sculpt adjustment, not semantic anatomy synthesis.",
            "Geodesic neighborhoods follow existing mesh edges; they cannot repair missing topology.",
            "Protected vertices are immutable traversal barriers in geodesic mode.",
            "Evaluate full-frame verified source masks and animated deformation before accepting the candidate.",
            "Source .blend is never saved; the generated refined.blend is an independent revision."
        ],
    })


main()
