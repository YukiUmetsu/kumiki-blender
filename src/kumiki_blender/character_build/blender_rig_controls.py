"""Blender worker: add optional, non-deforming IK/FK animation controls.

Reads a reviewed deform skeleton from a saved .blend. Does not rewrite that
source and deliberately does not export a glTF until controller animation is
baked to the deform armature.
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
from kumiki_blender.character_build.blender_common import arguments, write_report  # noqa: E402


def geometry_hash(obj):
    sha = hashlib.sha256()
    for vertex in obj.data.vertices:
        sha.update(struct.pack("<3d", *vertex.co))
    for face in obj.data.polygons:
        sha.update(struct.pack("<I", len(face.vertices)))
        for i in face.vertices:
            sha.update(struct.pack("<I", i))
    return sha.hexdigest()


def sample_rest(obj):
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        size = len(mesh.vertices)
        stride = max(1, size // 2000)
        coords = {i: tuple(evaluated.matrix_world @ mesh.vertices[i].co)
                  for i in range(0, size, stride)}
        return (size, coords)
    finally:
        evaluated.to_mesh_clear()


def main():
    args = arguments()
    plan = args["plan"]
    target = Path(args["output"]).resolve()
    source = Path(args["source_blend"]).resolve()
    if target.exists():
        raise FileExistsError(f"Revision exists: {target}")
    if not bpy.data.filepath or Path(bpy.data.filepath).resolve() != source:
        raise ValueError("Open the reviewed source .blend specified in the plan")
    rig = bpy.data.objects.get(plan["armature"])
    if rig is None or rig.type != "ARMATURE":
        raise ValueError(f"Armature not found: {plan['armature']}")
    if rig.library or rig.data.library:
        raise ValueError("Linked armatures cannot be edited in-place")
    controls = plan["controls"]
    constraints = plan["constraints"]
    deform_names = {row["owner"] for row in constraints}
    if any(c["name"] in rig.data.bones for c in controls):
        raise ValueError("Control bones already exist; use an unmodified source revision")
    for owner in deform_names:
        if owner not in rig.pose.bones:
            raise ValueError(f"Deforming rig missing required bone {owner}")
        if len(rig.pose.bones[owner].constraints):
            raise ValueError(f"Bone {owner} already has constraints; refusing to overwrite existing rig")
    for c in controls:
        parent = c["parent"]
        if parent not in rig.data.bones and not any(row["name"] == parent for row in controls):
            raise ValueError(f"Control {c['name']} references missing parent {parent}")
        if c["source_bone"] not in rig.data.bones:
            raise ValueError(f"Control {c['name']} uses missing deform bone {c['source_bone']}")
    # Verify the source bone positions against the plan. Otherwise the rig
    # may have been edited after joints.json was generated.
    inverse = rig.matrix_world.inverted_safe()
    expected_fk = {c["source_bone"]: c for c in controls if c["kind"] == "fk"}
    for name, control in expected_fk.items():
        bone = rig.data.bones[name]
        if ((rig.matrix_world @ bone.head_local - Vector(control["head"])).length > .001
                or (rig.matrix_world @ bone.tail_local - Vector(control["tail"])).length > .001):
            raise ValueError(f"{name}: saved rig rest positions differ from reviewed control plan")
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH" and any(
        modifier.type == "ARMATURE" and modifier.object == rig for modifier in obj.modifiers)]
    if not meshes:
        raise ValueError("No skinned meshes reference the reviewed rig")
    geometry_before = {obj.name: geometry_hash(obj) for obj in meshes}
    neutral_before = {obj.name: sample_rest(obj) for obj in meshes}
    bpy.ops.object.mode_set(mode="OBJECT") if bpy.context.object and bpy.context.object.mode != "OBJECT" else None
    bpy.ops.object.select_all(action="DESELECT")
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    for control in controls:
        edit = rig.data.edit_bones.new(control["name"])
        edit.head = inverse @ Vector(control["head"])
        edit.tail = inverse @ Vector(control["tail"])
        edit.use_deform = False
        edit.use_connect = False
        if control["kind"] == "fk":
            edit.roll = rig.data.edit_bones[control["source_bone"]].roll
    for control in controls:
        bone = rig.data.edit_bones[control["name"]]
        bone.parent = rig.data.edit_bones[control["parent"]]
    bpy.ops.object.mode_set(mode="OBJECT")
    for row in constraints:
        deform = rig.pose.bones[row["owner"]]
        if row["type"] == "IK":
            c = deform.constraints.new("IK")
            c.name = "Kumiki Native Two-Bone IK"
            c.target = rig
            c.subtarget = row["target"]
            c.pole_target = rig
            c.pole_subtarget = row["pole"]
            c.chain_count = 2
            c.pole_angle = math.radians(row["pole_angle_deg"])
            c.use_stretch = False
        else:
            c = deform.constraints.new("COPY_ROTATION")
            c.name = "Kumiki Native FK"
            c.target = rig
            c.subtarget = row["target"]
            c.owner_space = "LOCAL"
            c.target_space = "LOCAL"
            c.mix_mode = "REPLACE"
    bpy.context.view_layer.update()
    after = {obj.name: geometry_hash(obj) for obj in meshes}
    if geometry_before != after:
        raise RuntimeError("Mesh topology or rest geometry changed when adding rig controls")
    neutral_after = {obj.name: sample_rest(obj) for obj in meshes}
    neutrality = {}
    for name, before in neutral_before.items():
        current = neutral_after[name]
        if current[0] != before[0]:
            neutrality[name] = {"changed_topology": True, "max_sampled_rest_shift_m": None}
            continue
        delta = max((math.dist(x, current[1][i]) for i, x in before[1].items()), default=0.0)
        neutrality[name] = {"changed_topology": False,
                            "max_sampled_rest_shift_m": round(delta, 7),
                            "sampled_vertices": len(before[1])}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    bpy.ops.wm.save_as_mainfile(filepath=str(target / "controlled.blend"))
    warnings = [
        "Native IK/FK controls are for Blender authoring. Bake actions onto the deform skeleton for GLB.",
        "Review knee pole direction and bird-wing/tail poses before animation approval.",
    ]
    if any(row["changed_topology"] or (row["max_sampled_rest_shift_m"] or 0) > .001
           for row in neutrality.values()):
        warnings.append("Adding controls changed the neutral evaluated pose by >1 mm or topology; review rig.")
    write_report(target, {
        "schema_version": 1,
        "status": "animator-review-required",
        "source_blend": str(source),
        "armature": rig.name,
        "profile": plan["profile"],
        "controls": [c["name"] for c in controls],
        "constraints": [{"type": c["type"], "owner": c["owner"]} for c in constraints],
        "skinned_mesh_geometry_sha256": after,
        "rest_pose_change": neutrality,
        "warnings": warnings,
        "files": ["controlled.blend"],
    })


main()
