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
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kumiki_blender.character_build.blender_common import (activate, arguments, clear_startup_scene,  # noqa: E402
                                                           import_mesh, mesh_stats, remove_debris, write_report)


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
        removed = remove_debris(low)  # voxel remesh leaves small floating pieces; QuadriFlow rejects them
        steps.append(f"voxel remesh {size:.5f} m ({reason}), {removed} debris pieces removed")

    def reduce_for_quadriflow(limit: int = 150_000) -> None:
        """QuadriFlow accepted every mesh near 250k faces and refused larger generator meshes.
        A 15k-face target needs no fine input, so decimate a large mesh first."""
        faces = len(low.data.polygons)
        if faces <= limit:
            return
        decimate = low.modifiers.new("ForQuadriFlow", "DECIMATE")
        decimate.ratio = limit / faces
        activate(low)
        bpy.ops.object.modifier_apply(modifier=decimate.name)
        steps.append(f"decimated {faces} -> {len(low.data.polygons)} faces for QuadriFlow")

    def quadriflow() -> bool:
        """QuadriFlow, tried at several sizes.

        The main causes of a refusal are folds, slivers and debris (cleaned before this).
        On meshes that still had them, acceptance also changed with the absolute size, not
        in one direction. So try the largest dimension at 1.5, 3, 6, 0.8 and 12 m, then
        scale back.
        """
        reduce_for_quadriflow()
        activate(low)
        for size in (1.5, 3.0, 6.0, 0.8, 12.0):
            started = time.time()
            factor = size / max(low.dimensions)
            low.scale = (factor,) * 3
            bpy.ops.object.transform_apply(scale=True)
            result = bpy.ops.object.quadriflow_remesh(target_faces=int(args["target_faces"]),
                                                      use_preserve_sharp=False, use_mesh_symmetry=False, seed=0)
            low.scale = (1 / factor,) * 3
            bpy.ops.object.transform_apply(scale=True)
            ok = "FINISHED" in result and len(low.data.polygons) > 0
            steps.append(f"quadriflow {'finished' if ok else 'refused'} at {size} m {time.time() - started:.1f} s")
            if ok:
                return True
        return False

    # QuadriFlow refuses floating debris, sliver triangles and folded faces with one misleading
    # "not manifold" message. Remove debris, merge vertices closer than a tenth of the median
    # edge (slivers), and make the normals consistent.
    removed = remove_debris(low)
    coords = np.empty(len(low.data.vertices) * 3)
    low.data.vertices.foreach_get("co", coords)
    ends = np.empty(len(low.data.edges) * 2, dtype=np.int64)
    low.data.edges.foreach_get("vertices", ends)
    coords, ends = coords.reshape(-1, 3), ends.reshape(-1, 2)
    median_edge = float(np.median(np.linalg.norm(coords[ends[:, 0]] - coords[ends[:, 1]], axis=1)))
    activate(low)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=0.1 * median_edge)
    bpy.ops.mesh.dissolve_degenerate(threshold=0.1 * median_edge)
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    steps.append(f"merged vertices closer than {0.1 * median_edge * 1000:.3f} mm")
    if removed:
        steps.append(f"{removed} debris pieces removed")
    if high_stats["non_manifold_edges"]:
        voxel_remesh("input not manifold")
    # QuadriFlow also refuses meshes with self-intersections or flipped normals, which a
    # silhouette fit can create. A voxel remesh rebuilds a clean closed surface.
    warnings = []
    if not quadriflow():
        voxel_remesh("QuadriFlow refused the mesh")
        if not quadriflow():
            # Last resort: a voxel remesh sized to the target face count. OpenVDB outputs quads,
            # grid-aligned (no flowing edge loops). Better for a rig than decimated triangles.
            area = sum(p.area for p in low.data.polygons)
            size = (area / int(args["target_faces"])) ** 0.5
            remesh = low.modifiers.new("GridQuads", "REMESH")
            remesh.mode, remesh.voxel_size = "VOXEL", size
            activate(low)
            bpy.ops.object.modifier_apply(modifier=remesh.name)
            remove_debris(low)
            smooth = low.modifiers.new("Relax", "SMOOTH")
            smooth.factor, smooth.iterations = 0.5, 4
            bpy.ops.object.modifier_apply(modifier=smooth.name)
            steps.append(f"grid quads: voxel remesh {size * 1000:.2f} mm (QuadriFlow refused every size)")
            warnings.append("QuadriFlow refused the mesh at every size. low.obj has grid-aligned quads "
                            "without edge loops; retopologize joints by hand for animation.")
    # QuadriFlow follows the voxel stair-steps of a remeshed input. Relax the quads, then snap
    # them back onto the high mesh, so the low mesh lies on the real surface.
    relax = low.modifiers.new("Relax", "SMOOTH")
    relax.factor, relax.iterations = 0.5, 6
    snap = low.modifiers.new("OntoHigh", "SHRINKWRAP")
    snap.target, snap.wrap_method = high, "NEAREST_SURFACEPOINT"
    activate(low)
    for modifier in (relax.name, snap.name):
        bpy.ops.object.modifier_apply(modifier=modifier)
    steps.append("relaxed and shrink-wrapped onto the high mesh")
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
                          "steps": steps, "warnings": warnings, "files": ["retopo.blend", "low.obj", "low.glb", "low_normal.png"]})


main()
