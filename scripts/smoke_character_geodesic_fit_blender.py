"""Native Blender 4.5 smoke for source-pinned geodesic fitting.

A synthetic input receipt is used so Blender embedded Python does not need
Pillow or SciPy. Full source-image SHA and 2D triangulation checks are tested
separately by the pure-Python focused suite.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix="kumiki-geodesic-landmark-") as tmp:
        folder = Path(tmp)
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=13, y_subdivisions=13,
                                        size=.12, location=(0, 0, .14))
        patch = bpy.context.object
        patch.name = "FacePatch"
        index = min((v.index for v in patch.data.vertices),
                    key=lambda i: patch.data.vertices[i].co.length_squared)
        anchor = patch.matrix_world @ patch.data.vertices[index].co
        target = anchor + Vector((0, 0, .0024))
        source_count = len(patch.data.vertices)
        source_uv = [uv.name for uv in patch.data.uv_layers]
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=10,
                                             radius=.02, location=(.28, 0, .15))
        other = bpy.context.object
        other.name = "UnrelatedNose"
        nose_vertices = [tuple(v.co) for v in other.data.vertices]
        source = folder / "original.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(source))
        source_sha = hashlib.sha256(source.read_bytes()).hexdigest()

        # The complete plan/reprojection receipt is verified by the host CLI.
        # The Blender worker receives its frozen geometry parameters only.
        plan = {
            "schema_version": 1, "source_blend": str(source),
            "source_sha256": source_sha,
            "landmark_report_sha256": "synthetic-fixture-only",
            "fit_spec_sha256": "synthetic-fixture-only",
            "character_id": "synthetic", "mesh": "FacePatch",
            "expected_vertex_count": source_count,
            "max_displacement_m": .005,
            "max_relative_edge_change": .25,
            "protected_vertices": [0, 12, 156, 168],
            "pins": [{
                "id": "lid.center", "source_vertex": index,
                "radius_m": .044, "target_world_m": list(target),
            }],
        }
        output = folder / "fitted"
        args = folder / "worker-args.json"
        args.write_text(json.dumps({"plan": plan, "output": str(output)}))
        script = ROOT / "src/kumiki_blender/character_build/blender_landmark_fit.py"
        proc = subprocess.run([
            bpy.app.binary_path, "--background", "--python-exit-code", "1",
            str(source), "--python", str(script), "--", str(args),
        ], capture_output=True, text=True, timeout=100)
        if proc.returncode or not (output / "report.json").is_file():
            print(proc.stdout[-12000:], file=sys.stderr)
            print(proc.stderr[-12000:], file=sys.stderr)
            raise RuntimeError(f"Native geodesic fit worker failed: {proc.returncode}")
        report = json.loads((output / "report.json").read_text())
        assert report["status"] == "geodesic-landmark-candidate-visual-qa-required"
        assert report["changed_vertices"] > 1
        assert report["changed_vertices"] < source_count
        assert report["max_displacement_m"] <= .005
        assert report["flipped_or_degenerate_faces"] == 0
        assert report["unrelated_meshes_unchanged"] is True
        assert report["original_materials_and_uvs_unchanged"] is True
        assert (output / "landmark-fitted.obj").is_file()
        # Explicitly audit OBJ world axes against the immutable registered
        # Blender frame. Default Blender export Y-up would fail this check.
        obj_vertices = []
        with (output / "landmark-fitted.obj").open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("v "):
                    obj_vertices.append(Vector(tuple(float(x) for x in line.split()[1:4])))
        assert len(obj_vertices) == source_count
        assert min((candidate - target).length for candidate in obj_vertices) < 1e-6, (
            "OBJ export changed Blender Z-up registered source coordinates")
        assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha
        bpy.ops.wm.open_mainfile(filepath=str(output / "landmark-fitted.blend"))
        obj = bpy.data.objects["FacePatch"]
        placed = obj.matrix_world @ obj.data.vertices[index].co
        assert (placed - target).length < 1e-6, (
            (placed - target).length, tuple(placed), tuple(target))
        assert [uv.name for uv in obj.data.uv_layers] == source_uv
        assert [tuple(v.co) for v in bpy.data.objects["UnrelatedNose"].data.vertices] == nose_vertices
        print(json.dumps({
            "status": "native-source-geodesic-fit-verified",
            "blender": bpy.app.version_string,
            "landmark_error_m": round((placed - target).length, 10),
            "changed_vertices": report["changed_vertices"],
            "source_blend_unchanged": True,
            "unrelated_meshes_unchanged": True,
        }, indent=2))


main()
