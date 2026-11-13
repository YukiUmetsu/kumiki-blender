"""Blender worker: replace a box region of the base mesh with the same region of the donor mesh.

Arguments (JSON): base, donor, box_min, box_max (metres), ellipsoid (optional
[cx, cy, cz, rx, ry, rz], used instead of the box), voxel_m, output.
Each mesh is first rebuilt as a closed solid by a voxel remesh (generator meshes
have non-manifold edges, which exact booleans reject). The base loses the box
(boolean difference), the donor keeps only the box (boolean intersect), and the
two closed pieces are fused by one more voxel remesh. Cutting open surfaces
instead leaves holes, and a voxel remesh of an open surface makes a thin double
shell, not a solid.
"""

from __future__ import annotations

import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kumiki_blender.character_build.blender_common import (activate, arguments, clear_startup_scene,  # noqa: E402
                                                           export_registered_obj, import_mesh, mesh_stats, remove_debris, write_report)


def voxel_solid(obj, size: float) -> None:
    remesh = obj.modifiers.new("Solid", "REMESH")
    remesh.mode, remesh.voxel_size = "VOXEL", size
    activate(obj)
    bpy.ops.object.modifier_apply(modifier=remesh.name)
    remove_debris(obj)


def boolean(obj, cutter, operation: str) -> None:
    modifier = obj.modifiers.new("Region", "BOOLEAN")
    modifier.operation, modifier.object, modifier.solver = operation, cutter, "EXACT"
    activate(obj)
    bpy.ops.object.modifier_apply(modifier=modifier.name)


def main() -> None:
    args = arguments()
    output = Path(args["output"]).resolve()
    clear_startup_scene()
    base, donor = import_mesh(args["base"], "Base"), import_mesh(args["donor"], "Donor")
    before = {"base": mesh_stats(base), "donor": mesh_stats(donor)}
    size = float(args["voxel_m"])
    for obj in (base, donor):
        voxel_solid(obj, size)
    low, high = args["box_min"], args["box_max"]
    if args.get("ellipsoid"):  # centre and radii: a round cut follows a head; a box cuts flat walls
        cx, cy, cz, rx, ry, rz = args["ellipsoid"]
        bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=1, location=(cx, cy, cz))
        box = bpy.context.view_layer.objects.active
        box.scale = (rx, ry, rz)
    else:
        bpy.ops.mesh.primitive_cube_add(size=1, location=[(a + b) / 2 for a, b in zip(low, high)])
        box = bpy.context.view_layer.objects.active
        box.scale = [b - a for a, b in zip(low, high)]
    bpy.ops.object.transform_apply(scale=True)
    boolean(base, box, "DIFFERENCE")
    boolean(donor, box, "INTERSECT")
    pieces = {"base_kept": mesh_stats(base), "donor_taken": mesh_stats(donor)}
    bpy.data.objects.remove(box)
    activate(base)
    donor.select_set(True)
    bpy.ops.object.join()
    voxel_solid(base, size)
    base.name = "Swapped"
    output.mkdir(parents=True, exist_ok=True)
    activate(base)
    export_registered_obj(output / "swapped.obj")
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "swapped.blend"))
    write_report(output, {"input": before, "pieces": pieces, "result": mesh_stats(base), "voxel_m": size,
                          "box_min": low, "box_max": high, "files": ["swapped.obj", "swapped.blend"]})


main()
