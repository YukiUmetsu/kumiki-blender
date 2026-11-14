"""Fast synthetic native-Blender character-worker smoke without private artwork.

Run with:
  PYTHONPATH=src:tests blender -b --factory-startup --threads 2 \
    --python-exit-code 1 --python scripts/smoke_character_controls_topology_blender.py

Requires the existing reviewed official Blender binary. Child workers are
factory-started in isolated processes with explicit .blend source revisions.
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
from tests.test_character_rig_controls import rig_fixture  # noqa: E402
from kumiki_blender.character_build.rig_control_plan import save_control_plan  # noqa: E402
from kumiki_blender.character_build.animation_bake_plan import save_bake_plan  # noqa: E402
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
