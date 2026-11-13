"""Native Blender smoke for pre-rig, hinge-local quad densification.

Creates a synthetic low-poly neck with UVs and an unrelated head to protect.
Requires no private Pip/Quill artwork or internet-dependent code execution.
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
from kumiki_blender.character_build.topology_zones import save_topology_plan  # noqa: E402
from kumiki_blender.character_build.joint_refine_plan import save_joint_refine_plan  # noqa: E402


def main():
    with tempfile.TemporaryDirectory(prefix="kumiki-native-hinge-") as tmp:
        folder = Path(tmp)
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        bpy.ops.mesh.primitive_cylinder_add(vertices=20, radius=.035,
                                            depth=.15, location=(0, 0, .245))
        neck = bpy.context.object
        neck.name = "Neck"
        bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=12,
                                             radius=.06, location=(.25, 0, .25))
        unrelated = bpy.context.object
        unrelated.name = "UnrelatedHead"
        original_verts = {tuple(v.co) for v in neck.data.vertices}
        before_other = [tuple(v.co) for v in unrelated.data.vertices]
        origUV = [uv.name for uv in neck.data.uv_layers]
        source = folder / "original.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(source))
        source_digest = hashlib.sha256(source.read_bytes()).hexdigest()

        joints_file = folder/"joints.json"
        joints_file.write_text(json.dumps(rig_fixture()), encoding="utf-8")
        topo = save_topology_plan(joints_file, folder/"topology-plan",
                                  mesh="Neck", joints=["neck"])
        plan = save_joint_refine_plan(source, folder/"topology-plan/topology-plan.json",
                                      ["neck"], folder/"refine-plan", cuts=2,
                                      max_added_vertices=1500)
        target = folder/"refined"
        args = folder/"args.json"
        args.write_text(json.dumps({
            "plan": plan, "topology": topo, "output": str(target),
        }), encoding="utf-8")
        script = ROOT/"src/kumiki_blender/character_build/blender_joint_refine.py"
        process = subprocess.run([
            bpy.app.binary_path, "--background", "--python-exit-code", "1",
            str(source), "--python", str(script), "--", str(args),
        ], capture_output=True, text=True, timeout=100)
        if process.returncode or not (target/"report.json").is_file():
            print(process.stdout[-10000:], file=sys.stderr)
            print(process.stderr[-10000:], file=sys.stderr)
            raise RuntimeError(f"Native hinge worker failed: {process.returncode}")
        report = json.loads((target/"report.json").read_text())
        assert report["status"] == "local-hinge-topology-candidate-review-required"
        assert report["new_vertices"] >= 40
        assert report["reviewed_joint_bands"][0]["crossing_axial_edges"] == 20
        assert report["before"]["boundary_edges"] == report["after"]["boundary_edges"] == 0
        assert report["after"]["quads"] > report["before"]["quads"]
        assert report["original_source_vertices_preserved"] is True
        assert report["unrelated_character_meshes_preserved"] is True
        assert (target/"hinge-retopo.obj").is_file()
        with (target/"hinge-retopo.obj").open("r", encoding="utf-8") as stream:
            exported = [
                tuple(float(v) for v in line.split()[1:4])
                for line in stream if line.startswith("v ")]
        assert exported
        assert any(.17 <= vertex[2] <= .32 and abs(vertex[1]) < .05
                   for vertex in exported), (
                   "Joint OBJ was rotated out of Z-up registered source frame")
        assert hashlib.sha256(source.read_bytes()).hexdigest() == source_digest
        bpy.ops.wm.open_mainfile(filepath=str(target/"hinge-retopo.blend"))
        updated = bpy.data.objects["Neck"]
        assert len(updated.data.vertices) > len(original_verts)
        after_verts = {tuple(v.co) for v in updated.data.vertices}
        assert original_verts <= after_verts
        assert [uv.name for uv in updated.data.uv_layers] == origUV
        assert [tuple(v.co) for v in bpy.data.objects["UnrelatedHead"].data.vertices] == before_other
        print(json.dumps({
            "status": "native-hinge-loop-densification-verified",
            "blender_version": bpy.app.version_string,
            "added_vertices": report["new_vertices"],
            "added_quads": report["after"]["quads"]-report["before"]["quads"],
            "volume_ratio": report["volume_ratio"],
            "unrelated_objects_preserved": True,
        }, indent=2))


main()
