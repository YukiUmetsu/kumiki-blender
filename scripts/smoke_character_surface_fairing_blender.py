"""Native Blender smoke: pre-rig feature-aware fairing without touching source.

Run: blender --background --factory-startup --python-exit-code 1 \
        --python scripts/smoke_character_surface_fairing_blender.py
"""
from __future__ import annotations

import hashlib
import json
import math
import runpy
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kumiki_blender.character_build.surface_fairing import load_surface_policy  # noqa: E402


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    with tempfile.TemporaryDirectory(prefix="kumiki-surface-fairing-smoke-") as name:
        root = Path(name)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=12, radius=1.0)
        mesh = bpy.context.object
        mesh.name = "Body"
        for vertex in mesh.data.vertices:
            polar = math.acos(max(-1, min(1, vertex.co.z)))
            theta = math.atan2(vertex.co.y, vertex.co.x)
            r = 1.0 + 0.035 * math.cos(7 * theta) * math.sin(3 * polar)
            vertex.co *= r
        mesh.data.update()
        anchor = mesh.vertex_groups.new(name="Eyes")
        ids = [v.index for v in mesh.data.vertices if v.co.z > .90]
        assert ids
        anchor.add(ids, 1.0, "REPLACE")
        original_positions = {i: tuple(mesh.data.vertices[i].co) for i in ids}
        bpy.ops.mesh.primitive_cube_add(size=.12, location=(3, 0, 0))
        accessory = bpy.context.object
        accessory.name = "Unrelated"
        original_accessory = [tuple(v.co) for v in accessory.data.vertices]
        source = root / "source.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(source))
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        config = root / "policy.json"
        config.write_text(json.dumps({
            "schema_version": 1, "mesh": "Body",
            "iterations": 8, "lambda_step": .45, "mu_step": -.32,
            "feature_angle_deg": 80, "curvature_stop": 8,
            "max_displacement_m": .08, "max_edge_strain": .50,
            "max_volume_change": .10, "min_roughness_improvement": .0001,
            "protected_vertex_groups": ["Eyes"],
        }))
        policy = load_surface_policy(config)
        dest = root / "candidate"
        args = root / "arguments.json"
        args.write_text(json.dumps({
            "plan": asdict(policy),
            "source_blend": str(source), "source_sha256": source_hash,
            "output": str(dest),
        }))
        old_args = sys.argv
        try:
            sys.argv = ["blender", "--", str(args)]
            runpy.run_path(str(ROOT / "src" / "kumiki_blender" / "character_build"
                               / "blender_surface_fairing.py"), run_name="__main__")
        finally:
            sys.argv = old_args
        report = json.loads((dest / "report.json").read_text())
        assert report["status"] == "surface-candidate-art-review-required", report
        assert report["final_roughness"] < report["baseline_roughness"]
        assert report["maximum_displacement_m"] <= .08 + 1e-7
        assert report["other_meshes_preserved"]
        assert report["source_blend_preserved"]
        assert (dest / "surface-candidate.blend").is_file()
        assert (dest / "baseline-source.obj").is_file()
        assert (dest / "surface-candidate-source.obj").is_file()
        assert {i: tuple(mesh.data.vertices[i].co) for i in ids} == original_positions
        assert [tuple(v.co) for v in accessory.data.vertices] == original_accessory
        assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
        print("Native character surface-fairing smoke: PASS")


main()
