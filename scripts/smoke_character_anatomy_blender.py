"""Blender 4.5 smoke: source-preserving, source-anchored facial anatomy geometry.

Synthetic, permissively authored fixture only. This validates mesh topology,
UVs, bone attachments, part-only OBJ exports and untouched source objects,
not artistic fidelity of any specific real character.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from tests.test_character_anatomy import anatomy_fixture  # noqa: E402
from kumiki_blender.character_build.anatomy_plan import plan_anatomy  # noqa: E402


def setup_source():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.object.armature_add(enter_editmode=False, location=(0, 0, 0))
    armature = bpy.context.object
    armature.name = "CharacterRig"
    armature.data.bones[0].name = "head"
    bpy.ops.object.mode_set(mode="EDIT")
    bone = armature.data.edit_bones["head"]
    bone.head = (0, 0, .19)
    bone.tail = (0, 0, .33)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.mesh.primitive_uv_sphere_add(segments=28, ring_count=18,
                                         radius=.083, location=(0, 0, .26))
    source = bpy.context.object
    source.name = "OriginalHead"
    source.shape_key_add(name="Basis", from_mix=False)
    bump = source.shape_key_add(name="Expression", from_mix=False)
    for i, vertex in enumerate(source.data.vertices):
        if vertex.co.z > 0:
            bump.data[i].co.z += .001
    mat = bpy.data.materials.new("OriginalHeadMaterial")
    mat.diffuse_color = (.5, .3, .2, 1)
    source.data.materials.append(mat)
    bpy.context.view_layer.update()


def run_worker(source, plan, output):
    args_path = output.parent / "worker-arguments.json"
    args_path.write_text(json.dumps({"plan": plan, "output": str(output)}))
    worker = ROOT / "src/kumiki_blender/character_build/blender_anatomy.py"
    process = subprocess.run([
        bpy.app.binary_path, "--background", "--python-exit-code", "1",
        str(source), "--python", str(worker), "--", str(args_path),
    ], capture_output=True, text=True, timeout=125)
    if process.returncode or not (output / "report.json").is_file():
        print(process.stdout[-12000:], file=sys.stderr)
        print(process.stderr[-12000:], file=sys.stderr)
        raise RuntimeError(f"Anatomy Blender worker exited {process.returncode}")
    return json.loads((output / "report.json").read_text(encoding="utf-8"))


def main():
    with tempfile.TemporaryDirectory(prefix="kumiki-source-anchored-anatomy-") as temp:
        folder = Path(temp)
        setup_source()
        source = folder / "original.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(source))
        spec = folder / "anatomy.json"
        spec.write_text(json.dumps(anatomy_fixture()), encoding="utf-8")
        plan = plan_anatomy(source, spec, folder / "plan")
        target = folder / "modeled"
        report = run_worker(source, plan, target)
        assert report["status"] == "anatomy-proposal-artistic-review-required"
        assert report["original_meshes_unchanged"] is True
        assert report["original_armatures_unchanged"] is True
        assert report["component_count"] == 5
        assert report["part_ids"] == ["beak", "eye.L", "upper_lid.L"]
        assert all(row["boundary_edges"] == 0 and row["nonmanifold_edges"] == 0
                   for row in report["components"])
        assert all(row["signed_volume_m3"] > 0 for row in report["components"])
        assert all((target / "parts" / (pid + ".obj")).is_file() for pid in report["part_ids"])
        # A source eye at z=0.27m must stay z=0.27m in an exported
        # OBJ. Blender's implicit Y-up export would rotate it toward z≈0.07.
        eye_obj = target / "parts" / "eye.L.source.obj"
        with eye_obj.open("r", encoding="utf-8") as stream:
            world_vertices = [
                tuple(float(x) for x in line.split()[1:4])
                for line in stream if line.startswith("v ")]
        assert world_vertices
        assert (target / "parts" / "eye.L.obj").is_file()
        assert any(.25 < z < .29 and -.09 < y < -.04
                   for _x, y, z in world_vertices), (
                   "Eye OBJ was rotated out of canonical source-camera frame")
        assert (target / "anatomy.blend").is_file()
        bpy.ops.wm.open_mainfile(filepath=str(target / "anatomy.blend"))
        eye = bpy.data.objects["ANATOMY.eye.L.sclera"]
        iris = bpy.data.objects["ANATOMY.eye.L.iris"]
        pupil = bpy.data.objects["ANATOMY.eye.L.pupil"]
        rim = bpy.data.objects["ANATOMY.upper_lid.L"]
        beak = bpy.data.objects["ANATOMY.beak"]
        for obj in (eye, iris, pupil, rim, beak):
            assert obj.type == "MESH"
            assert len(obj.data.uv_layers) >= 1
            assert len(obj.data.materials) == 1
            assert obj["kumiki_reference_review_required"] is True
        assert eye.parent == bpy.data.objects["CharacterRig"]
        assert eye.parent_type == "BONE" and eye.parent_bone == "head"
        assert beak.parent is None
        assert "OriginalHead" in bpy.data.objects
        assert len(bpy.data.objects["OriginalHead"].data.shape_keys.key_blocks) == 2
        bpy.context.view_layer.update()
        graph = bpy.context.evaluated_depsgraph_get()
        evaluated = eye.evaluated_get(graph)
        baseline = evaluated.matrix_world @ evaluated.data.vertices[0].co
        rig = bpy.data.objects["CharacterRig"]
        rig.pose.bones["head"].rotation_mode = "XYZ"
        rig.pose.bones["head"].rotation_euler = (0, .20, 0)
        bpy.context.view_layer.update()
        graph = bpy.context.evaluated_depsgraph_get()
        evaluated = eye.evaluated_get(graph)
        posed = evaluated.matrix_world @ evaluated.data.vertices[0].co
        assert (posed - baseline).length > .00001, (
            f"Bone-attached eye geometry did not move: before {tuple(baseline)}, "
            f"after {tuple(posed)}, bone {tuple(rig.pose.bones['head'].rotation_euler)}")
        print(json.dumps({
            "status": "native-anatomy-geometry-verified",
            "blender_version": bpy.app.version_string,
            "parts": report["part_ids"],
            "components": report["component_count"],
            "new_vertices": report["total_new_vertices"],
            "all_components_closed": True,
            "original_meshes_and_armature_unchanged": True,
            "eye_moves_with_bone": True,
        }, indent=2))


main()
