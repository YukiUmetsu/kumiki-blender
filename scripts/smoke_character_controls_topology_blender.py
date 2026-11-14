"""Fast synthetic native-Blender character-worker smoke without private artwork.

Run with:
  PYTHONPATH=src:tests blender -b --factory-startup --threads 2 \
    --python-exit-code 1 --python scripts/smoke_character_controls_topology_blender.py

Requires the existing reviewed official Blender binary. Child workers are
factory-started in isolated processes with explicit .blend source revisions.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from tests.test_character_rig_controls import rig_fixture  # noqa: E402
from kumiki_blender.character_build.rig_control_plan import save_control_plan  # noqa: E402
from kumiki_blender.character_build.animation_bake_plan import save_bake_plan  # noqa: E402
from kumiki_blender.character_build.roundtrip_plan import make_roundtrip_plan  # noqa: E402
from kumiki_blender.character_build.topology_zones import save_topology_plan  # noqa: E402

WORKER_DIR = ROOT / "src/kumiki_blender/character_build"


def fixture_scene(data: dict):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    armature = bpy.data.armatures.new("CharacterRig")
    rig = bpy.data.objects.new("CharacterRig", armature)
    bpy.context.scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    for bone in data["bones"]:
        edit = armature.edit_bones.new(bone["name"])
        edit.head = data["joints"][bone["head"]]
        edit.tail = data["joints"][bone["tail"]]
        if bone["parent"]:
            edit.parent = armature.edit_bones[bone["parent"]]
            edit.use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")
    rig.select_set(False)

    # A small deforming mesh on a single wing bone and a root-bound body are
    # sufficient to exercise actual scene depsgraph evaluation, not sculpt quality.
    parts = [
        ("Skin", (0, 0, .17), .055, "root"),
        ("Wing.L", (.105, -.006, .16), .016, "upper_wing.L"),
        ("Eye.L", (.035, -.04, .275), .013, "head"),
    ]
    for name, position, radius, group in parts:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=12,
                                             radius=radius, location=position)
        mesh_obj = bpy.context.object
        mesh_obj.name = name
        weights = mesh_obj.vertex_groups.new(name=group)
        weights.add(list(range(len(mesh_obj.data.vertices))), 1.0, "REPLACE")
        if name == "Wing.L":
            zone = mesh_obj.vertex_groups.new(name="REGION_WingLeft")
            zone.add(list(range(len(mesh_obj.data.vertices))), 1.0, "REPLACE")
        skin = mesh_obj.modifiers.new("Skin", "ARMATURE")
        skin.object = rig
    bpy.context.view_layer.update()


def run_worker(source: Path, name: str, args: dict, target: Path):
    arguments = target.parent / f"{name}-arguments.json"
    arguments.write_text(json.dumps(args), encoding="utf-8")
    command = [
        bpy.app.binary_path, "--background", "--python-exit-code", "1",
        str(source), "--python", str(WORKER_DIR / name), "--", str(arguments),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    report = target / "report.json"
    if result.returncode or not report.is_file():
        print(result.stdout[-6000:], file=sys.stderr)
        print(result.stderr[-6000:], file=sys.stderr)
        raise RuntimeError(f"{name} failed (returncode {result.returncode}): "
                           f"{(result.stdout + result.stderr)[-2500:]}")
    return json.loads(report.read_text(encoding="utf-8"))


def main():
    with tempfile.TemporaryDirectory(prefix="kumiki-character-native-") as temp:
        folder = Path(temp)
        joints = rig_fixture(limbs="wings")
        joints_file = folder / "joints.json"
        joints_file.write_text(json.dumps(joints), encoding="utf-8")
        fixture_scene(joints)
        source = folder / "source.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(source))

        control_spec = save_control_plan(joints_file, "avian-biped-wings", folder / "control-plan")
        output_controls = folder / "controls-output"
        controls = run_worker(source, "blender_rig_controls.py", {
            "source_blend": str(source), "plan": control_spec,
            "output": str(output_controls),
        }, output_controls)
        assert (output_controls / "controlled.blend").is_file()
        assert "CTRL_upper_wing.L" in controls["controls"]
        assert "Wing.L" in controls["rest_pose_change"]

        # Drive one native FK control with a real two-key pose action. This
        # proves the exporter can bake evaluated IK/FK motion, not just export
        # a static skinned model containing a misleading animation name.
        bpy.ops.wm.open_mainfile(filepath=str(output_controls / "controlled.blend"))
        rig = bpy.data.objects["CharacterRig"]
        wing = rig.pose.bones["CTRL_upper_wing.L"]
        wing.rotation_mode = "XYZ"
        bpy.context.scene.frame_set(1)
        wing.rotation_euler = (0, 0, 0)
        wing.keyframe_insert(data_path="rotation_euler", frame=1)
        bpy.context.scene.frame_set(12)
        wing.rotation_euler = (0, 0, .45)
        wing.keyframe_insert(data_path="rotation_euler", frame=12)
        animated = folder / "animated-controls.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(animated))
        bake_plan = save_bake_plan(animated, "wing_fold", 1, 12, folder / "bake-plan")
        bake_out = folder / "bake-output"
        baked = run_worker(animated, "blender_animation_bake.py", {
            "plan": bake_plan, "output": str(bake_out),
        }, bake_out)
        assert (bake_out / "baked.blend").is_file()
        assert (bake_out / "animated.glb").is_file()
        assert baked["glb"]["skins"] >= 1 and baked["glb"]["channels"] >= 1
        assert baked["pose_parity"]["max_position_error_m"] <= .002
        assert baked["pose_parity_after_controller_removal"]["max_position_error_m"] <= .002
        assert baked["glb"]["deform_joint_channels"] > 0
        assert baked["glb"]["animations"] == 1
        assert baked["glb"]["clip_names"] == ["wing_fold"]

        # M8: a reviewed wing corrective must survive skeletal control baking
        # as genuine GLB "weights" animation in the SAME action. A morph name
        # in Blender or a static glTF target alone does not prove portability.
        bpy.ops.wm.open_mainfile(filepath=str(animated))
        rig_morph = bpy.data.objects["CharacterRig"]
        wing_mesh = bpy.data.objects["Wing.L"]
        wing_mesh.shape_key_add(name="Basis", from_mix=False)
        corrective = wing_mesh.shape_key_add(name="CORR_WingRaise.L", from_mix=False)
        for i in range(12):
            corrective.data[i].co.x += .004
        driver = corrective.driver_add("value").driver
        driver.type = "SCRIPTED"
        variable = driver.variables.new()
        variable.name = "bend"
        variable.type = "TRANSFORMS"
        target = variable.targets[0]
        target.id = rig_morph
        target.bone_target = "upper_wing.L"
        target.transform_type = "ROT_Z"
        target.transform_space = "LOCAL_SPACE"
        driver.expression = "min(1.0,max(0.0,bend/0.3))"
        corrected_animated = folder / "animated-with-pose-corrective.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(corrected_animated))
        corrected_plan = save_bake_plan(
            corrected_animated, "wing_fold_corrected", 1, 12,
            folder / "corrective-bake-plan", bake_correctives=True)
        corrected_out = folder / "baked-with-morph-animation"
        corrected_receipt = run_worker(corrected_animated, "blender_animation_bake.py", {
            "plan": corrected_plan, "output": str(corrected_out),
        }, corrected_out)
        assert corrected_receipt["glb"]["animations"] == 1
        assert corrected_receipt["glb"]["deform_joint_channels"] > 0
        assert corrected_receipt["morph_parity"]["status"] == "exported-morph-parity-verified"
        assert corrected_receipt["morph_parity"]["checked_correctives"] == 1
        assert corrected_receipt["morph_parity"]["max_weight_error"] <= .025
        assert corrected_receipt["morph_parity"]["meshes"][0]["object"] == "Wing.L"
        assert corrected_receipt["morph_driver_to_baked_pose_max_error"] <= .02
        assert (corrected_out / "animated.glb").is_file()
        assert (corrected_out / "baked.blend").is_file()
        # M9: authored mouth viseme Action + pose-driven wing correction +
        # FK control rig must land in ONE GLB animation, preserving actual
        # per-frame mouth and wing weights, not merely morph target names.
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=20, ring_count=12, radius=.03, location=(0, -.05, .21))
        mouth = bpy.context.object
        mouth.name = "Mouth"
        mouth_weights = mouth.vertex_groups.new(name="root")
        mouth_weights.add(list(range(len(mouth.data.vertices))), 1.0, "REPLACE")
        mouth_skin = mouth.modifiers.new("MouthSkin", "ARMATURE")
        mouth_skin.object = rig_morph
        mouth.shape_key_add(name="Basis", from_mix=False)
        speech_a = mouth.shape_key_add(name="viseme_A", from_mix=False)
        speech_c = mouth.shape_key_add(name="viseme_C", from_mix=False)
        for i, vertex in enumerate(mouth.data.vertices):
            if vertex.co.z > 0:
                speech_a.data[i].co.z += .008
            if vertex.co.x > 0:
                speech_c.data[i].co.x += .008
        for frame, a_value, c_value in [
            (1, 0, 0), (6, 1, 0), (12, 0, 1)]:
            bpy.context.scene.frame_set(frame)
            for block, value in ((speech_a, a_value), (speech_c, c_value)):
                block.value = value
                assert block.keyframe_insert(
                    data_path="value", frame=frame, group="KumikiVisemes")
        assert mouth.data.shape_keys.animation_data.action is not None
        mouth.data.shape_keys.animation_data.action.name = "authored_speech"
        bpy.context.scene.frame_set(1)
        unified_source = folder / "animated-with-corrective-and-visemes.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(unified_source))
        unified_sha = hashlib.sha256(unified_source.read_bytes()).hexdigest()
        unified_plan = save_bake_plan(
            unified_source, "wing_speech", 1, 12, folder / "unified-bake-plan",
            bake_correctives=True, bake_visemes=True)
        unified_out = folder / "unified-glb-clip"
        unified = run_worker(unified_source, "blender_animation_bake.py", {
            "plan": unified_plan, "output": str(unified_out),
        }, unified_out)
        assert unified["glb"]["animations"] == 1
        assert unified["glb"]["clip_names"] == ["wing_speech"]
        assert unified["glb"]["deform_joint_channels"] > 0
        assert unified["morph_visemes_requested"]
        assert unified["morph_correctives_requested"]
        assert unified["morph_viseme_objects"] == ["Mouth"]
        assert unified["morph_parity"]["checked_correctives"] == 3
        assert unified["morph_parity"]["target_meshes"] == 2
        assert {row["object"] for row in unified["morph_parity"]["meshes"]} == {"Mouth", "Wing.L"}
        assert unified["morph_parity"]["max_weight_error"] <= .025
        assert (unified_out / "animated.glb").is_file()
        assert (unified_out / "baked.blend").is_file()
        assert hashlib.sha256(unified_source.read_bytes()).hexdigest() == unified_sha

        # Independently re-import the actual combined GLB into a brand-new
        # Blender scene. Comparing only exported channel names cannot prove
        # that the facial and wing deformations survived transport.
        portable = unified_out / "animated.glb"
        baked_source = unified_out / "baked.blend"
        portable_sha = hashlib.sha256(portable.read_bytes()).hexdigest()
        baked_sha = hashlib.sha256(baked_source.read_bytes()).hexdigest()
        roundtrip = make_roundtrip_plan(
            baked_source, portable, first_frame=1, last_frame=12, fps=24,
            samples=5, max_p95_error_m=.002, max_max_error_m=.01)
        roundtrip_out = folder / "roundtrip-surface-qa"
        roundtrip_report = run_worker(
            baked_source, "blender_glb_surface_roundtrip.py", {
                "plan": roundtrip, "output": str(roundtrip_out),
            }, roundtrip_out)
        assert roundtrip_report["status"] == "evaluated-glb-roundtrip-review-required", roundtrip_report["problems"]
        assert roundtrip_report["source_meshes"] == [
            "Eye.L", "Mouth", "Skin", "Wing.L",
        ], roundtrip_report["source_meshes"]
        assert roundtrip_report["sampled_frames"] == [1, 3, 6, 9, 12]
        assert roundtrip_report["checked_mesh_frames"] == 5 * len(roundtrip_report["source_meshes"])
        assert roundtrip_report["global_max_nearest_error_m"] <= .01
        assert roundtrip_report["global_p95_worst_frame_m"] <= .002
        assert hashlib.sha256(portable.read_bytes()).hexdigest() == portable_sha
        assert hashlib.sha256(baked_source.read_bytes()).hexdigest() == baked_sha

        # A structurally valid *different* GLB (without the added animated
        # Mouth) must fail even though both assets have a skin and one clip.
        wrong_export = bake_out / "animated.glb"
        wrong_plan = make_roundtrip_plan(
            baked_source, wrong_export, first_frame=1, last_frame=12,
            fps=24, samples=3)
        wrong_out = folder / "must-reject-incomplete-glb"
        wrong_receipt = run_worker(
            baked_source, "blender_glb_surface_roundtrip.py", {
                "plan": wrong_plan, "output": str(wrong_out),
            }, wrong_out)
        assert wrong_receipt["status"] == "structural-fail"
        assert any("missing=" in p for p in wrong_receipt["problems"])

        # Fail closed when either modality is not explicitly approved.
        no_face_plan = save_bake_plan(
            unified_source, "must_reject_face_drop", 1, 12,
            folder / "missing-face-opt-in", bake_correctives=True)
        no_face_out = folder / "must-refuse-face-drop"
        try:
            run_worker(unified_source, "blender_animation_bake.py", {
                "plan": no_face_plan, "output": str(no_face_out),
            }, no_face_out)
        except RuntimeError as exc:
            assert "requires --bake-visemes" in str(exc), str(exc)
            assert not no_face_out.exists()
        else:
            raise AssertionError("Exporter silently discarded authored mouth animation")

        no_corr_plan = save_bake_plan(
            unified_source, "must_reject_corrective_drop", 1, 12,
            folder / "missing-corrective-opt-in", bake_visemes=True)
        no_corr_out = folder / "must-refuse-corrective-drop"
        try:
            run_worker(unified_source, "blender_animation_bake.py", {
                "plan": no_corr_plan, "output": str(no_corr_out),
            }, no_corr_out)
        except RuntimeError as exc:
            assert "require --bake-correctives" in str(exc), str(exc)
            assert not no_corr_out.exists()
        else:
            raise AssertionError("Exporter silently discarded sculpted wing correction")

        # Explicit opt-in contract: a skeletal-only export must still reject
        # a driven corrective rather than silently render an incorrect wing.
        old_plan = save_bake_plan(
            corrected_animated, "must_reject_morph_drop", 1, 12,
            folder / "no-morph-opt-in-plan")
        refused_output = folder / "must-refuse-shape-drop"
        try:
            run_worker(corrected_animated, "blender_animation_bake.py", {
                "plan": old_plan, "output": str(refused_output),
            }, refused_output)
        except RuntimeError as exc:
            assert "require --bake-correctives" in str(exc), str(exc)
            assert not refused_output.exists()
        else:
            raise AssertionError("Skeletal-only bake silently dropped authored morph correction")

        topology_spec = save_topology_plan(joints_file, folder / "topology-plan",
                                           mesh="Skin", joints=["neck", "shoulder.L", "elbow.L"])
        topology_out = folder / "topology-output"
        topology = run_worker(source, "blender_topology_qa.py", {
            "source_blend": str(source), "plan": topology_spec,
            "output": str(topology_out), "annotate": True,
        }, topology_out)
        assert len(topology["zones"]) == 3
        assert (topology_out / "topology-reviewed.blend").is_file()

        sculpt_out = folder / "sculpt-output"
        sculpt = run_worker(source, "blender_region_sculpt.py", {
            "source_blend": str(source), "output": str(sculpt_out),
            "plan": {
                "schema_version": 1, "object": "Eye.L",
                "max_displacement_m": .005,
                "expected_vertex_count": len(bpy.data.objects["Eye.L"].data.vertices),
                "operations": [{
                    "center_m": [.035, -.053, .275],
                    "radius_m": .014,
                    "move_m": [0, -.002, 0],
                    "falloff": "smoothstep",
                }]
            }
        }, sculpt_out)
        assert sculpt["edited_vertices"] > 0
        assert (sculpt_out / "region.obj").is_file()
        assert sculpt["flipped_faces"] == 0

        # Inspect source revision without altering its .blend: obtain stable
        # vertex-order SHA plus a candidate anchor index for surface editing.
        probe_out = folder / "region-source-probe"
        probe = run_worker(source, "blender_region_sculpt.py", {
            "source_blend": str(source),
            "output": str(probe_out),
            "inspect_only": True, "object": "Eye.L",
            "anchors": [[.035, -.053, .275]],
        }, probe_out)
        assert probe["status"] == "source-probe-only"
        seed = probe["anchor_candidates"][0]["nearest_vertex"]
        assert probe["anchor_candidates"][0]["distance_m"] < .004
        pinned = (seed + 1) % probe["expected_vertex_count"]
        geo_out = folder / "geodesic-sculpt-output"
        geodesic_plan = {
                "schema_version": 1, "object": "Eye.L",
                "expected_vertex_count": probe["expected_vertex_count"],
                "expected_mesh_sha256": probe["expected_mesh_sha256"],
                "protected_vertex_indices": [pinned],
                "max_displacement_m": .002,
                "operations": [{
                    "center_m": [.035, -.053, .275], "radius_m": .014,
                    "move_m": [0, -.001, 0], "falloff": "smoothstep",
                    "distance_mode": "geodesic", "seed_vertex": seed,
                    "max_seed_distance_m": .004,
                }]
        }
        geodesic = run_worker(source, "blender_region_sculpt.py", {
            "source_blend": str(source),
            "output": str(geo_out), "plan": geodesic_plan,
        }, geo_out)
        assert geodesic["expected_mesh_sha256_verified"]
        assert geodesic["protected_vertex_count"] == 1
        assert geodesic["operations"][0]["distance_mode"] == "geodesic"
        assert geodesic["operations"][0]["edited_vertices"] > 0
        assert geodesic["flipped_faces"] == 0
        assert (geo_out / "refined.blend").is_file()

        # Identical vertex counts are insufficient: a moved object transform
        # must invalidate the saved-mesh plan before any derived edit is written.
        bpy.ops.wm.open_mainfile(filepath=str(source))
        bpy.data.objects["Eye.L"].location.x += .001
        altered = folder / "repositioned-source.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(altered))
        stale = folder / "must-reject-stale-region"
        try:
            run_worker(altered, "blender_region_sculpt.py", {
                "source_blend": str(altered),
                "output": str(stale), "plan": geodesic_plan,
            }, stale)
        except RuntimeError as exc:
            assert "Source mesh has changed" in str(exc)
            assert not stale.exists()
        else:
            raise AssertionError("Geodesic sculpt accepted stale object transform")

        poses_out = folder / "poses-output"
        poses = run_worker(output_controls / "controlled.blend", "blender_pose_qa.py", {
            "source_blend": str(output_controls / "controlled.blend"),
            "output": str(poses_out),
            "plan": {
                "schema_version": 1, "armature": "CharacterRig",
                "poses": [{"name": "wing-open",
                           "bones": {"CTRL_upper_wing.L": {"axis": "Z", "degrees": 25}}}],
                "collision_pairs": [], "render": False,
            }
        }, poses_out)
        assert len(poses["poses"]) == 1
        assert poses["status"] in ("deformation-review-required", "structural-fail")
        # New M6 reliability gate: a real deform bone moves evaluated vertices,
        # while a no-op pose is explicitly rejected instead of marked reviewed.
        real_out = folder / "verified-moving-pose"
        real = run_worker(source, "blender_pose_qa.py", {
            "source_blend": str(source), "output": str(real_out),
            "plan": {
                "schema_version": 1, "armature": "CharacterRig",
                "poses": [{"name": "wing-moves",
                           "bones": {"upper_wing.L": {"axis": "Z", "degrees": 45}}}],
                "min_pose_displacement_m": .0005,
                "collision_pairs": [], "render": False,
            },
        }, real_out)
        assert real["status"] == "deformation-review-required", real["problems"]
        assert real["poses"][0]["max_vertex_displacement_m"] > .0005

        idle_out = folder / "rejected-noop-pose"
        idle = run_worker(source, "blender_pose_qa.py", {
            "source_blend": str(source), "output": str(idle_out),
            "plan": {
                "schema_version": 1, "armature": "CharacterRig",
                "poses": [{"name": "no-op",
                           "bones": {"root": {"axis": "Z", "degrees": 0}}}],
                "min_pose_displacement_m": .0005,
                "collision_pairs": [], "render": False,
            },
        }, idle_out)
        assert idle["status"] == "structural-fail"
        assert any("moved less than" in message for message in idle["problems"])

        zone_plan = {
            "schema_version": 1, "armature": "CharacterRig",
            "poses": [{"name": "wing-pose",
                       "bones": {"upper_wing.L": {"axis": "Z", "degrees": 25}}}],
            "collision_pairs": [], "render": False,
            "weight_zones": [{
                "mesh": "Wing.L", "region_group": "REGION_WingLeft",
                "allowed_bones": ["upper_wing.L", "fore_wing.L"],
                "max_outside_weight": .02, "min_vertices": 20,
            }],
        }
        approved_out = folder / "approved-skin-zone"
        approved = run_worker(source, "blender_pose_qa.py", {
            "source_blend": str(source), "output": str(approved_out),
            "plan": zone_plan,
        }, approved_out)
        assert approved["status"] == "deformation-review-required", approved["problems"]
        assert approved["rest_weight_zones"][0]["violating_vertices"] == 0
        assert approved["rest_weight_zones"][0]["selected_vertices"] >= 20

        # Introduce a normalized but anatomically wrong spine influence.
        # The generic sum-to-one skin audit must still pass; semantic QA fails.
        bpy.ops.wm.open_mainfile(filepath=str(source))
        wing_obj = bpy.data.objects["Wing.L"]
        wing_obj.vertex_groups["upper_wing.L"].add([0], .9, "REPLACE")
        wing_obj.vertex_groups.new(name="spine").add([0], .1, "REPLACE")
        leaking = folder / "wing-weights-bleed.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(leaking))
        leaking_out = folder / "rejected-weight-bleed"
        bad = run_worker(leaking, "blender_pose_qa.py", {
            "source_blend": str(leaking), "output": str(leaking_out),
            "plan": zone_plan,
        }, leaking_out)
        assert bad["status"] == "structural-fail"
        assert bad["rest_weight_zones"][0]["violating_vertices"] == 1
        assert bad["rest_weight_zones"][0]["worst_examples"][0]["outside_bones"] == ["spine"]
        assert any("outside-weight limit" in problem for problem in bad["problems"])

        # M7: fix the actual deform weights without introducing new bone
        # influences, editing the source asset or altering the mesh shape.
        source_sha_before = hashlib.sha256(leaking.read_bytes()).hexdigest()
        repair_args = {
            "source_blend": str(leaking),
            "source_blend_sha256": source_sha_before,
            "armature": "CharacterRig",
            "weight_zones": zone_plan["weight_zones"],
            "max_removed_weight": .15, "max_changed_vertices": 20,
        }
        preview_out = folder / "weight-repair-preview"
        preview = run_worker(leaking, "blender_weight_repair.py", {
            **repair_args, "output": str(preview_out), "dry_run": True,
        }, preview_out)
        assert preview["status"] == "weight-repair-proposal-only"
        assert preview["changed_vertices"] == 1
        assert preview["zones"][0]["before"]["violating_vertices"] == 1
        assert not (preview_out / "repaired.blend").exists()

        repair_out = folder / "repaired-wing-weights"
        repaired = run_worker(leaking, "blender_weight_repair.py", {
            **repair_args, "output": str(repair_out), "dry_run": False,
        }, repair_out)
        assert repaired["status"] == "weight-repaired-animation-review-required"
        assert repaired["changed_vertices"] == 1
        assert repaired["zones"][0]["after"]["violating_vertices"] == 0
        assert (repair_out / "repaired.blend").is_file()
        assert hashlib.sha256(leaking.read_bytes()).hexdigest() == source_sha_before

        repaired_qa_out = folder / "repaired-wing-pose-qa"
        validated = run_worker(repair_out / "repaired.blend", "blender_pose_qa.py", {
            "source_blend": str(repair_out / "repaired.blend"),
            "output": str(repaired_qa_out), "plan": zone_plan,
        }, repaired_qa_out)
        assert validated["status"] == "deformation-review-required", validated["problems"]
        assert validated["rest_weight_zones"][0]["violating_vertices"] == 0
        assert validated["rest_skin_weights"]["Wing.L"]["non_normalized"] == 0

        print(json.dumps({
            "status": "native-workers-executed",
            "blender_version": bpy.app.version_string,
            "controls": len(controls["controls"]),
            "topology_zones": len(topology["zones"]),
            "sculpt_vertices": sculpt["edited_vertices"],
            "pose_qa_status": poses["status"],
            "exported_animation_channels": baked["glb"]["channels"],
            "deform_joint_channels": baked["glb"]["deform_joint_channels"],
            "animation_names": baked["glb"]["clip_names"],
        }, indent=2))


main()
