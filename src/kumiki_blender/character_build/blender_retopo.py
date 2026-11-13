"""Blender worker: quad retopology, UVs and a normal-map bake from the high mesh.

Arguments (JSON): mesh, output, target_faces, bake_size, cage_m.
QuadriFlow needs a closed, manifold mesh. If the input is not, a voxel remesh
runs first; the report says so. The normal map bakes the high mesh's detail
onto the low mesh (Cycles, CPU), so a light mesh keeps the sculpted look.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kumiki_blender.character_build.blender_common import (activate, arguments, clear_startup_scene,  # noqa: E402
                                                           import_mesh, mesh_stats, write_report)


def uv_coverage(obj) -> float:
    """Sum of UV face areas. 1.0 fills the whole UV square."""
    uv = obj.data.uv_layers.active.data
    total = 0.0
    for poly in obj.data.polygons:
        points = [uv[i].uv for i in poly.loop_indices]
        total += abs(sum(points[i].x * points[i - 1].y - points[i - 1].x * points[i].y for i in range(len(points)))) / 2
    return total


def main() -> None:
    args = arguments()
    output = Path(args["output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    clear_startup_scene()
    high = import_mesh(args["mesh"], "High")
    high_stats = mesh_stats(high)
    activate(high)
    bpy.ops.object.duplicate()
    low = bpy.context.view_layer.objects.active
    low.name = low.data.name = "Low"
    steps = []

    def voxel_remesh(reason: str) -> None:
        size = max(high.dimensions) / 300
        remesh = low.modifiers.new("Close", "REMESH")
        remesh.mode, remesh.voxel_size = "VOXEL", size
        activate(low)
        bpy.ops.object.modifier_apply(modifier=remesh.name)
        steps.append(f"voxel remesh {size:.5f} m ({reason})")

    def quadriflow() -> bool:
        started = time.time()
        activate(low)
        # QuadriFlow refuses small meshes (a 0.35 m figure) with a misleading "not manifold"
        # warning; the same mesh at 1 m or more works. Scale up for the remesh, then back.
        factor = 1.5 / max(low.dimensions)
        low.scale = (factor,) * 3
        bpy.ops.object.transform_apply(scale=True)
        result = bpy.ops.object.quadriflow_remesh(target_faces=int(args["target_faces"]), use_preserve_sharp=False,
                                                  use_mesh_symmetry=False, seed=0)
        low.scale = (1 / factor,) * 3
        bpy.ops.object.transform_apply(scale=True)
        ok = "FINISHED" in result and len(low.data.polygons) > 0
        steps.append(f"quadriflow {'finished' if ok else 'refused'} {time.time() - started:.1f} s")
        return ok

    if high_stats["non_manifold_edges"]:
        voxel_remesh("input not manifold")
    # QuadriFlow also refuses meshes with self-intersections or flipped normals, which a
    # silhouette fit can create. A voxel remesh rebuilds a clean closed surface.
    if not quadriflow():
        voxel_remesh("QuadriFlow refused the mesh")
        if not quadriflow():
            raise RuntimeError(f"QuadriFlow failed after a voxel remesh: {steps}")
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.01)
    bpy.ops.object.mode_set(mode="OBJECT")

    size = int(args.get("bake_size", 2048))
    image = bpy.data.images.new("LowNormal", size, size, float_buffer=False)
    image.colorspace_settings.name = "Non-Color"
    material = bpy.data.materials.new("LowMaterial")
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = image
    normal_map = nodes.new("ShaderNodeNormalMap")
    links.new(texture.outputs["Color"], normal_map.inputs["Color"])
    links.new(normal_map.outputs["Normal"], nodes["Principled BSDF"].inputs["Normal"])
    nodes.active = texture
    low.data.materials.clear()
    low.data.materials.append(material)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    bpy.ops.object.select_all(action="DESELECT")
    high.select_set(True)
    low.select_set(True)
    bpy.context.view_layer.objects.active = low
    started = time.time()
    bpy.ops.object.bake(type="NORMAL", use_selected_to_active=True, cage_extrusion=float(args.get("cage_m", 0.005)),
                        normal_space="TANGENT", margin=8)
    steps.append(f"normal bake {size}px {time.time() - started:.1f} s")
    image.filepath_raw = str(output / "low_normal.png")
    image.file_format = "PNG"
    image.save()
    high.hide_render = True
    high.hide_set(True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "retopo.blend"))
    activate(low)
    bpy.ops.export_scene.gltf(filepath=str(output / "low.glb"), export_format="GLB", use_selection=True)
    # glTF stores triangles only. The OBJ keeps the quads (and UVs) for rigging and sculpting.
    bpy.ops.wm.obj_export(filepath=str(output / "low.obj"), export_selected_objects=True, export_materials=False)
    write_report(output, {"high": high_stats, "low": mesh_stats(low), "uv_coverage": round(uv_coverage(low), 4),
                          "steps": steps, "files": ["retopo.blend", "low.obj", "low.glb", "low_normal.png"]})


main()
