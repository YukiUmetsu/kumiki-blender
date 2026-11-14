"""Blender worker: bind reviewed rest-space corrective sculpts to joint bend.

The sculpt is geometry on a separate, identically indexed, hidden mesh
object. Only the named source's *relative shape keys* and their safe Blender
rotation-channel drivers are created. No sculpting or geometry inference is
performed, and no original file is overwritten.

Driver curves improve Blender-native deformation. Raw glTF export does not
serialize Blender drivers; exported animations need separately baked morph
channels before their corrections can be considered portable.
"""
from __future__ import annotations

import hashlib
import math
import struct
import sys
from pathlib import Path

import bpy
from mathutils import Matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kumiki_blender.character_build.blender_common import arguments, write_report  # noqa: E402
from kumiki_blender.character_build.joint_corrective_plan import (  # noqa: E402
    activation_expression, validate_corrective_plan,
)


def file_sha256(path: Path) -> str:
    hash_ = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hash_.update(chunk)
    return hash_.hexdigest()


def source_mesh_hash(obj) -> str:
    """Verify underlying topology, neutral coords and world-space placement."""
    hash_ = hashlib.sha256(b"kumiki-joint-corrective-geometry-v1")
    hash_.update(struct.pack("<3Q", len(obj.data.vertices),
                             len(obj.data.edges), len(obj.data.polygons)))
    for row in obj.matrix_world:
        hash_.update(struct.pack("<4d", *row))
    for vertex in obj.data.vertices:
        hash_.update(struct.pack("<3d", *vertex.co))
    for edge in obj.data.edges:
        hash_.update(struct.pack("<2I", *edge.vertices))
    for polygon in obj.data.polygons:
        hash_.update(struct.pack("<I", len(polygon.vertices)))
        for index in polygon.vertices:
            hash_.update(struct.pack("<I", index))
    return hash_.hexdigest()


def _match_topology(mesh, target) -> None:
    source_data, target_data = mesh.data, target.data
    if (len(source_data.vertices) != len(target_data.vertices) or
            len(source_data.edges) != len(target_data.edges) or
            len(source_data.polygons) != len(target_data.polygons)):
        raise ValueError(f"{mesh.name}/{target.name}: different source/target vertex or face counts")
    for a, b in zip(source_data.edges, target_data.edges):
        if tuple(a.vertices) != tuple(b.vertices):
            raise ValueError(f"{mesh.name}/{target.name}: source/target edge indices differ")
    for a, b in zip(source_data.polygons, target_data.polygons):
        if tuple(a.vertices) != tuple(b.vertices):
            raise ValueError(f"{mesh.name}/{target.name}: source/target face order differs")
    if any(abs(a - b) > 1e-6
           for source_row, target_row in zip(mesh.matrix_world, target.matrix_world)
           for a, b in zip(source_row, target_row)):
        raise ValueError(f"{mesh.name}/{target.name}: object world transforms are not aligned")


def _preflight_row(row, rig):
    mesh = bpy.data.objects.get(row["mesh"])
    target = bpy.data.objects.get(row["target_object"])
    if mesh is None or mesh.type != "MESH":
        raise ValueError(f"Corrective source mesh not found: {row['mesh']}")
    if target is None or target.type != "MESH":
        raise ValueError(f"Author-sculpted target mesh not found: {row['target_object']}")
    if (mesh.library or mesh.data.library or target.library or target.data.library or
            mesh.data.users != 1 or mesh.data == target.data):
        raise ValueError("Corrective source must have unique local mesh data, separate from its sculpt")
    if not target.hide_render or target.modifiers or target.data.shape_keys:
        raise ValueError(
            f"{target.name}: rest-space corrective sculpt must be hidden from render, "
            "have no modifiers and have no shape keys")
    if not any(mod.type == "ARMATURE" and mod.object == rig for mod in mesh.modifiers):
        raise ValueError(f"{mesh.name}: source mesh is not skinned to {rig.name}")
    if not rig.data.bones.get(row["bone"]):
        raise ValueError(f"Corrective joint {row['bone']} is not on {rig.name}")
    keys = mesh.data.shape_keys
    if keys is not None:
        if not keys.use_relative:
            raise ValueError("Correctives require relative, not absolute, shape keys")
        if row["shape_key"] in keys.key_blocks:
            raise ValueError(f"Existing corrective name must not be overwritten: {row['shape_key']}")
        if any(key != keys.key_blocks[0] and abs(key.value) > 1e-6 for key in keys.key_blocks):
            raise ValueError("Save all existing facial/corrective shape keys at zero before binding")
    if len(mesh.data.vertices) > 300_000:
        raise ValueError("Corrective mesh exceeds 300000 source vertices; split into parts")
    _match_topology(mesh, target)
    local_coords = []
    changed = 0
    largest = 0.0
    for vertex, authored in zip(mesh.data.vertices, target.data.vertices):
        if (any(not math.isfinite(x) for x in vertex.co) or
                any(not math.isfinite(x) for x in authored.co)):
            raise ValueError("Corrective has nonfinite source or target geometry")
        displacement = (mesh.matrix_world.to_3x3() @ (authored.co - vertex.co)).length
        largest = max(largest, displacement)
        changed += displacement > 1e-7
        local_coords.append(authored.co.copy())
    if not row["min_changed_vertices"] <= changed <= row["max_changed_vertices"]:
        raise ValueError(
            f"{mesh.name}/{target.name}: edited {changed} vertices outside approved "
            f"[{row['min_changed_vertices']}, {row['max_changed_vertices']}] budget")
    if largest > row["max_displacement_m"] + 1e-8:
        raise ValueError(
            f"{mesh.name}/{target.name}: max sculpt delta {largest:.6f}m "
            f"exceeds {row['max_displacement_m']}m")
    mesh.data.update()
    target.data.update()
    for a, b in zip(mesh.data.polygons, target.data.polygons):
        if a.normal.length > 1e-7 and b.normal.length > 1e-7 and a.normal.dot(b.normal) <= 0:
            raise ValueError(f"{mesh.name}/{target.name}: authored corrective flips a face")
    return {
        "mesh": mesh, "target": target, "coords": local_coords,
        "changed_vertices": changed, "max_displacement_m": largest,
        "source_mesh_sha256": source_mesh_hash(mesh),
    }


def _create_key(row, ready, rig):
    mesh = ready["mesh"]
    if mesh.data.shape_keys is None:
        mesh.shape_key_add(name="Basis", from_mix=False)
    key = mesh.shape_key_add(name=row["shape_key"], from_mix=False)
    key.value = 0.0
    key.slider_min = 0.0
    key.slider_max = 1.0
    for index, coord in enumerate(ready["coords"]):
        key.data[index].co = coord
    curve = key.driver_add("value")
    driver = curve.driver
    driver.type = "SCRIPTED"
    variable = driver.variables.new()
    variable.name = "bend"
    variable.type = "TRANSFORMS"
    variable.targets[0].id = rig
    variable.targets[0].bone_target = row["bone"]
    variable.targets[0].transform_type = f"ROT_{row['axis']}"
    variable.targets[0].transform_space = "LOCAL_SPACE"
    driver.expression = activation_expression(
        row["direction"], row["start_degrees"], row["full_degrees"])
    # Blender ShapeKey blocks do not support ID custom properties in 4.5 LTS.
    # The immutable report and the driver's named bone target are the
    # authoritative provenance and binding, without unsupported metadata.
    return driver.expression


def main() -> None:
    args = arguments()
    plan = args["plan"]
    validate_corrective_plan(plan)
    output = Path(args["output"]).resolve()
    source = Path(plan["source_blend"]).resolve()
    if output.exists():
        raise FileExistsError(f"Corrective revision already exists: {output}")
    if not bpy.data.filepath or Path(bpy.data.filepath).resolve() != source:
        raise ValueError("Load the precise source Blender .blend bound in the plan")
    if not source.is_file() or file_sha256(source) != plan["source_sha256"]:
        raise ValueError("Source .blend SHA-256 differs from the reviewed plan")
    if source.is_relative_to(output):
        raise ValueError("Original .blend may not be inside the new revision")
    rig = bpy.data.objects.get(plan["armature"])
    if rig is None or rig.type != "ARMATURE" or rig.library or rig.data.library:
        raise ValueError(f"Reviewed local armature is missing: {plan['armature']}")
    if any(max(abs(bone.matrix_basis[i][j] - Matrix.Identity(4)[i][j])
               for i in range(4) for j in range(4)) > 1e-5
           for bone in rig.pose.bones):
        raise ValueError("Bind joint correctives only from a saved neutral rig pose")
    before = {obj.name: source_mesh_hash(obj) for obj in bpy.data.objects
              if obj.type == "MESH"}
    # Reject malformed regions and overlapping auto-generated names before
    # creating ANY shape keys. The original remains unchanged on failures.
    prepared = [_preflight_row(row, rig) for row in plan["correctives"]]
    expressions = []
    for row, ready in zip(plan["correctives"], prepared):
        expressions.append(_create_key(row, ready, rig))
    after = {obj.name: source_mesh_hash(obj) for obj in bpy.data.objects
             if obj.type == "MESH"}
    if before != after:
        raise ValueError("Corrective binding unexpectedly changed neutral mesh geometry")
    if file_sha256(source) != plan["source_sha256"]:
        raise ValueError("Source file changed during joint-corrective binding")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "corrected.blend"))
    write_report(output, {
        "schema_version": 1, "status": "bound-pose-correctives-animation-review-required",
        "source_blend": str(source), "source_blend_sha256": plan["source_sha256"],
        "armature": rig.name,
        "correctives": [
            {**{key: row[key] for key in (
                "mesh", "target_object", "shape_key", "bone",
                "axis", "direction", "start_degrees", "full_degrees")},
             "changed_vertices": ready["changed_vertices"],
             "max_displacement_m": round(ready["max_displacement_m"], 8),
             "source_mesh_sha256": ready["source_mesh_sha256"],
             "driver_expression": expression}
            for row, ready, expression in zip(plan["correctives"], prepared, expressions)
        ],
        "geometry_source_hashes_unchanged": before == after,
        "files": ["corrected.blend"],
        "limitations": [
            "This binds artist-authored rest-space shape changes; it does not infer character anatomy.",
            "Correctives use named LOCAL_SPACE bone rotation channels and must be inspected with actual controller poses.",
            "Blender drivers are not glTF animation channels; raw GLB does not preserve these effects.",
            "Before portable export, corrective weights must be sampled and morph-animated alongside the skeleton.",
            "Facial shape keys and unrelated mesh topology are preserved but must be independently QA'd.",
            "Run validate-rig and visual checks against source artwork before animation approval.",
        ],
    })


main()
