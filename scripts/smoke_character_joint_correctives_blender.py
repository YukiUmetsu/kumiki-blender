"""Native Blender smoke: authored wing joint geometry activates only at its pose.

No private source artwork is needed; this validates real Blender dependency
evaluation, not final silhouette or artistic approval.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kumiki_blender.character_build.blender import run_worker  # noqa: E402
from kumiki_blender.character_build.joint_corrective_plan import make_corrective_plan  # noqa: E402


def fixture():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    armature = bpy.data.armatures.new("CharacterRig")
    rig = bpy.data.objects.new("CharacterRig", armature)
    bpy.context.scene.collection.objects.link(rig)
    bpy.ops.object.select_all(action="DESELECT")
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    root = armature.edit_bones.new("root")
    root.head = (0, 0, 0)
    root.tail = (0, 0, .1)
    wing = armature.edit_bones.new("upper_wing.L")
    wing.head = (0, 0, .1)
    wing.tail = (0, 0, .24)
    wing.parent = root
    bpy.ops.object.mode_set(mode="OBJECT")
    rig.select_set(False)

    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=20, ring_count=12, radius=.042, location=(.055, 0, .17))
    mesh = bpy.context.object
    mesh.name = "Wing.L"
    group = mesh.vertex_groups.new(name="upper_wing.L")
    group.add(list(range(len(mesh.data.vertices))), 1, "REPLACE")
    skin = mesh.modifiers.new("WingSkin", "ARMATURE")
    skin.object = rig
    sculpt = bpy.data.objects.new("WingRaiseSculpt.L", mesh.data.copy())
    bpy.context.scene.collection.objects.link(sculpt)
    sculpt.matrix_world = mesh.matrix_world.copy()
    sculpt.hide_render = True
    for i in range(12):
        sculpt.data.vertices[i].co.x += .004
    sculpt.data.update()
    bpy.context.view_layer.update()


def evaluated_positions(degrees: float):
    rig = bpy.data.objects["CharacterRig"]
    bone = rig.pose.bones["upper_wing.L"]
    bone.rotation_mode = "XYZ"
    bone.rotation_euler = (0, 0, math.radians(degrees))
    bpy.context.view_layer.update()
    graph = bpy.context.evaluated_depsgraph_get()
    obj = bpy.data.objects["Wing.L"].evaluated_get(graph)
    surface = obj.to_mesh()
    try:
        return [tuple(obj.matrix_world @ v.co) for v in surface.vertices]
    finally:
        obj.to_mesh_clear()


def maximum_error(left, right):
    return max(math.dist(x, y) for x, y in zip(left, right))


def main():
    with tempfile.TemporaryDirectory(prefix="kumiki-pose-corrective-") as directory:
        out = Path(directory)
        fixture()
        source = out / "rig-source.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(source))
        checksum = hashlib.sha256(source.read_bytes()).hexdigest()
        config = {
            "armature": "CharacterRig",
            "correctives": [{
                "mesh": "Wing.L", "target_object": "WingRaiseSculpt.L",
                "shape_key": "CORR_WingRaise.L", "bone": "upper_wing.L",
                "axis": "Z", "direction": "positive",
                "start_degrees": 10, "full_degrees": 40,
                "max_displacement_m": .005, "min_changed_vertices": 10,
                "max_changed_vertices": 20,
            }],
        }
        plan = make_corrective_plan(source, config)
        candidate = out / "corrective-r01"
        receipt = run_worker(
            "joint-correctives", {"plan": plan, "output": str(candidate)},
            blender=bpy.app.binary_path, blend_file=source, timeout_s=120)
        assert receipt["status"] == "bound-pose-correctives-animation-review-required"
        assert receipt["correctives"][0]["changed_vertices"] == 12
        assert receipt["geometry_source_hashes_unchanged"]
        assert (candidate / "corrected.blend").is_file()
        assert hashlib.sha256(source.read_bytes()).hexdigest() == checksum

        bpy.ops.wm.open_mainfile(filepath=str(source))
        original_rest = evaluated_positions(0)
        original_plus = evaluated_positions(40)
        original_minus = evaluated_positions(-40)

        bpy.ops.wm.open_mainfile(filepath=str(candidate / "corrected.blend"))
        block = bpy.data.objects["Wing.L"].data.shape_keys.key_blocks["CORR_WingRaise.L"]
        assert block.value == 0.0
        corrected_rest = evaluated_positions(0)
        corrected_plus = evaluated_positions(40)
        corrected_minus = evaluated_positions(-40)

        assert maximum_error(original_rest, corrected_rest) < 1e-6
        assert maximum_error(original_minus, corrected_minus) < 1e-6, (
            "One-sided positive correction erroneously activates on a negative wing bend")
        changed = maximum_error(original_plus, corrected_plus)
        assert changed > .0025, (
            f"Actual posed geometry unchanged ({changed:.7f} m); a driver label alone is insufficient")
        assert changed < .005, "Corrective vertex displacement exceeded approved authoring amount"

        driver_curves = bpy.data.objects["Wing.L"].data.shape_keys.animation_data.drivers
        assert len(driver_curves) == 1
        drv = driver_curves[0].driver
        assert not drv.is_valid is False, "Blender corrective driver is invalid"
        var = drv.variables[0]
        assert var.targets[0].bone_target == "upper_wing.L"
        assert var.targets[0].transform_type == "ROT_Z"
        assert var.targets[0].transform_space == "LOCAL_SPACE"
        assert bpy.data.objects["WingRaiseSculpt.L"].hide_render

        print(json.dumps({
            "status": "corrective-joint-motion-verified",
            "blender": bpy.app.version_string,
            "source_unchanged": True,
            "rest_max_error_m": maximum_error(original_rest, corrected_rest),
            "wrong_direction_max_error_m": maximum_error(original_minus, corrected_minus),
            "corrected_bend_delta_m": round(changed, 7),
        }, indent=2))


main()
