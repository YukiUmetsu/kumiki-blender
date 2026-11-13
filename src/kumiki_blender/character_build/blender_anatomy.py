"""Native Blender anatomical part builder (separate topology, source-locked).

Actual Blender meshes, non-destructive and independently editable; no
voxel unions, symmetry assumptions or automatic reference validation.
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import bpy
from mathutils import Matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kumiki_blender.character_build.anatomy_geometry import (  # noqa: E402
    contour_loft, ocular_globe, swept_tube,
)
from kumiki_blender.character_build.blender_common import arguments, write_report  # noqa: E402


def _baseline_mesh(obj):
    """Hash original mesh geometry, shape keys, UVs, weights, materials and transform."""
    sha = hashlib.sha256()
    sha.update(obj.name.encode())
    for row in obj.matrix_world:
        sha.update(struct.pack("<4d", *row))
    for vertex in obj.data.vertices:
        sha.update(struct.pack("<3d", *vertex.co))
        for group in vertex.groups:
            sha.update(struct.pack("<Id", group.group, group.weight))
    for poly in obj.data.polygons:
        sha.update(struct.pack("<I", len(poly.vertices)))
        for vertex_index in poly.vertices:
            sha.update(struct.pack("<I", vertex_index))
        sha.update(struct.pack("<I", poly.material_index))
    for uvmap in obj.data.uv_layers:
        sha.update(uvmap.name.encode())
        for loop in uvmap.data:
            sha.update(struct.pack("<2d", *loop.uv))
    for material in obj.data.materials:
        sha.update((material.name if material else "<missing>").encode())
    if obj.data.shape_keys:
        for key in obj.data.shape_keys.key_blocks:
            sha.update(key.name.encode())
            for vertex in key.data:
                sha.update(struct.pack("<3d", *vertex.co))
    for modifier in obj.modifiers:
        sha.update((modifier.name + ":" + modifier.type).encode())
    return sha.hexdigest()


def _rig_pose(rig):
    return {name: {
        "head": tuple(bone.head_local),
        "tail": tuple(bone.tail_local),
        "matrix": tuple(tuple(row) for row in rig.pose.bones[name].matrix),
    } for name, bone in [(b.name, b) for b in rig.data.bones]}


def _material(name, recipe):
    material = bpy.data.materials.new(name)
    material.diffuse_color = recipe["rgba"]
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    if shader:
        shader.inputs["Base Color"].default_value = recipe["rgba"]
        shader.inputs["Roughness"].default_value = recipe.get("roughness", .55)
        shader.inputs["Metallic"].default_value = recipe.get("metallic", 0.0)
        shader.inputs["Alpha"].default_value = recipe["rgba"][3]
    return material


def _create_mesh(collection, name, geometry, recipe):
    bpy_mesh = bpy.data.meshes.new(name)
    bpy_mesh.from_pydata(geometry["vertices"], [], geometry["faces"])
    bpy_mesh.update(calc_edges=True)
    obj = bpy.data.objects.new(name, bpy_mesh)
    collection.objects.link(obj)
    mat = _material(name + ".material", recipe)
    obj.data.materials.append(mat)
    for polygon in bpy_mesh.polygons:
        polygon.use_smooth = len(polygon.vertices) <= 4
    layer = bpy_mesh.uv_layers.new(name="Anatomy UV")
    coords = geometry["uvs"]
    for polygon in bpy_mesh.polygons:
        uvals = [coords[bpy_mesh.loops[index].vertex_index][0]
                 for index in polygon.loop_indices]
        seam = max(uvals) - min(uvals) > .5
        for index in polygon.loop_indices:
            u, v = coords[bpy_mesh.loops[index].vertex_index]
            if seam and u < .5:
                u += 1
            layer.data[index].uv = (u, v)
    return obj


def _attach(obj, attachment):
    if attachment["mode"] == "separate":
        return
    rig = bpy.data.objects.get(attachment["armature"])
    if rig is None or rig.type != "ARMATURE" or attachment["bone"] not in rig.data.bones:
        raise ValueError(f"Missing approved armature/bone attachment for {obj.name}")
    old_world = obj.matrix_world.copy()
    obj.parent = rig
    obj.parent_type = "BONE"
    obj.parent_bone = attachment["bone"]
    obj.matrix_world = old_world
    bpy.context.view_layer.update()
    if any(abs(a-b) > 1e-6 for row_a, row_b in zip(obj.matrix_world, old_world)
           for a, b in zip(row_a, row_b)):
        raise ValueError(f"{obj.name}: attaching to bone changed approved world coordinates")


def _parts_for(part):
    kind = part["kind"]
    if kind == "ocular_globe":
        shapes = ocular_globe(
            part["center_m"], part["forward_m"], part["up_m"],
            part["radii_m"], iris_fraction=part["iris_fraction"],
            pupil_fraction=part["pupil_fraction"],
            radial_segments=part["radial_segments"], latitudes=part["latitudes"])
        return [(part["id"] + "." + subpart, shape, part["materials"][subpart])
                for subpart, shape in shapes.items()]
    if kind == "swept_tube":
        return [(part["id"], swept_tube(
            part["control_points_m"], part["radii_m"],
            radial_segments=part["radial_segments"],
            subdivisions=part["subdivisions"],
            reference_up=part["reference_up"]), part["material"])]
    return [(part["id"], contour_loft(
        part["sections_m"], section_subdivisions=part["section_subdivisions"],
        contour_subdivisions=part["contour_subdivisions"]), part["material"])]


def main():
    args = arguments()
    plan = args["plan"]
    source = Path(plan["source_blend"]).resolve()
    folder = Path(args["output"]).resolve()
    if folder.exists():
        raise FileExistsError(f"Anatomy revision exists: {folder}")
    if not bpy.data.filepath or Path(bpy.data.filepath).resolve() != source:
        raise ValueError("Open the exact original .blend specified by the anatomy plan")
    # Validate names, attachments, and geometry before creating any output.
    existing = {obj.name for obj in bpy.data.objects}
    planned = []
    total_mesh_vertices = 0
    for part in plan["parts"]:
        components = _parts_for(part)
        for component, mesh, material in components:
            name = "ANATOMY." + component
            if name in existing or any(item[1] == name for item in planned):
                raise ValueError(f"Anatomy object name collision: {name}")
            if mesh["stats"]["boundary_edges"] or mesh["stats"]["nonmanifold_edges"]:
                raise ValueError(f"New anatomy mesh is not watertight: {name}")
            total_mesh_vertices += mesh["stats"]["vertices"]
            planned.append((part, name, mesh, material))
        parent = part["attachment"]
        if parent["mode"] == "bone":
            arm = bpy.data.objects.get(parent["armature"])
            if arm is None or arm.type != "ARMATURE" or parent["bone"] not in arm.data.bones:
                raise ValueError(f"Missing reviewed bone for anatomy part {part['id']}")
    if total_mesh_vertices > 2_000_000:
        raise ValueError("Generated anatomical geometry exceeds 2 million vertices")
    baseline = {obj.name: _baseline_mesh(obj) for obj in bpy.data.objects if obj.type == "MESH"}
    skeletons = {rig.name: _rig_pose(rig) for rig in bpy.data.objects if rig.type == "ARMATURE"}
    collection = bpy.data.collections.new(f"Kumiki Anatomy {plan['character_id']}")
    bpy.context.scene.collection.children.link(collection)
    created = {}
    measurements = []
    for part, name, mesh, material in planned:
        obj = _create_mesh(collection, name, mesh, material)
        obj["kumiki_part_id"] = part["id"]
        obj["kumiki_kind"] = part["kind"]
        obj["kumiki_source_status"] = part["evidence"]["status"]
        obj["kumiki_source_views"] = ",".join(part["evidence"]["source_views"])
        obj["kumiki_reference_review_required"] = True
        _attach(obj, part["attachment"])
        created.setdefault(part["id"], []).append(obj)
        measurements.append({
            "object": name,
            "part_id": part["id"],
            "kind": part["kind"],
            "source_status": part["evidence"]["status"],
            "source_views": part["evidence"]["source_views"],
            "attachment": part["attachment"],
            **mesh["stats"],
        })
    after = {name: _baseline_mesh(bpy.data.objects[name]) for name in baseline}
    after_skeletons = {name: _rig_pose(bpy.data.objects[name]) for name in skeletons}
    if baseline != after or skeletons != after_skeletons:
        raise RuntimeError("Non-target original model, rig, UV, shape keys, or material assignments changed")
    folder.parent.mkdir(parents=True, exist_ok=True)
    folder.mkdir()
    bpy.ops.wm.save_as_mainfile(filepath=str(folder / "anatomy.blend"))
    parts_folder = folder / "parts"
    parts_folder.mkdir()
    for pid, objects in created.items():
        bpy.ops.object.select_all(action="DESELECT")
        for obj in objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = objects[0]
        bpy.ops.wm.obj_export(
            filepath=str(parts_folder / f"{pid}.obj"),
            export_selected_objects=True, export_materials=False)
    write_report(folder, {
        "schema_version": 1,
        "status": "anatomy-proposal-artistic-review-required",
        "source_blend": str(source),
        "source_sha256": plan["source_sha256"],
        "source_spec_sha256": plan["spec_sha256"],
        "character_id": plan["character_id"],
        "part_ids": sorted(created),
        "component_count": len(planned),
        "total_new_vertices": total_mesh_vertices,
        "original_mesh_hashes": baseline,
        "original_meshes_unchanged": True,
        "original_armatures_unchanged": True,
        "components": measurements,
        "files": ["anatomy.blend"] + [f"parts/{part}.obj" for part in sorted(created)],
        "limitations": [
            "Anatomy is built from supplied anchors/stations and cannot establish original illustration fidelity by itself.",
            "The registered-verified evidence field is an input assertion, not independently certified by this worker.",
            "Separate overlapping eye layers are intentional; all new part components have distinct 3D topology.",
            "No voxel unions or remeshing of the source body; the new parts may still require seam and animation QA.",
            "Run compare-region with calibrated source masks, then review the actual Blender rig and final GLB.",
        ],
    })


main()
