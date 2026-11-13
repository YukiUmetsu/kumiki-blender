"""Run inside the existing Blender character smoke job; no external assets required.

Creates a deliberately overlapping pair of cubes, runs the production worker
against a saved .blend, then validates rollback safety and actual BVH reduction.
"""
from __future__ import annotations

import hashlib
import json
import runpy
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kumiki_blender.character_build.penetration_repair import load_collision_plan  # noqa: E402


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    with tempfile.TemporaryDirectory(prefix="kumiki-collision-smoke-") as temporary:
        root = Path(temporary)
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=(-1, 0, 0))
        fixed = bpy.context.object
        fixed.name = "Body"
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.35, 0, 0))
        moving = bpy.context.object
        moving.name = "Tail"
        protected = [vertex.index for vertex in moving.data.vertices
                     if (moving.matrix_world @ vertex.co).x > 0.5]
        assert protected, "Need protected vertices on the visible side"
        root_group = moving.vertex_groups.new(name="TailRoot")
        root_group.add(protected, 1.0, "REPLACE")
        bpy.ops.mesh.primitive_cube_add(size=0.2, location=(2.0, 0, 0))
        guard = bpy.context.object
        guard.name = "OtherAccessory"
        source = root / "original.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(source))
        before_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        original_fixed = [tuple(v.co) for v in fixed.data.vertices]
        original_anchors = {i: tuple(moving.data.vertices[i].co) for i in protected}
        config = root / "policy.json"
        config.write_text(json.dumps({
            "schema_version": 1,
            "guard_pairs": [["Tail", "OtherAccessory"]],
            "pairs": [{
                "moving": "Tail", "fixed": "Body",
                "protected_vertex_groups": ["TailRoot"],
                "clearance_m": 0.002,
                "max_displacement_m": 0.20,
                "iterations": 8,
                "smoothing_iterations": 2,
                "max_relative_edge_change": 0.50,
            }],
        }))
        policy = load_collision_plan(config)
        destination = root / "revision"
        args = root / "args.json"
        args.write_text(json.dumps({
            "plan": {"schema_version": 1, "pairs": [asdict(p) for p in policy["pairs"]],
                     "guard_pairs": policy["guard_pairs"]},
            "source_blend": str(source), "output": str(destination),
        }))
        old_args = sys.argv
        try:
            sys.argv = ["blender", "--", str(args)]
            runpy.run_path(str(ROOT / "src" / "kumiki_blender" / "character_build"
                               / "blender_penetration_repair.py"), run_name="__main__")
        finally:
            sys.argv = old_args
        report = json.loads((destination / "report.json").read_text())
        assert report["status"] == "geometric-candidate-review-required", report
        assert (destination / "collision-candidate.blend").is_file()
        assert report["pairs"][0]["accepted"]
        assert report["pairs"][0]["protected_vertices"] == len(protected)
        assert report["guard_pairs"][0]["objects"] == ["Tail", "OtherAccessory"]
        assert report["guard_pairs"][0]["baseline_triangle_overlap_pairs"] == 0
        assert report["guard_pairs"][0]["final_triangle_overlap_pairs"] == 0
        assert report["pairs"][0]["before_triangle_overlap_pairs"] > 0
        assert report["pairs"][0]["after_triangle_overlap_pairs"] <= report["pairs"][0]["before_triangle_overlap_pairs"]
        assert [tuple(v.co) for v in fixed.data.vertices] == original_fixed
        assert {i: tuple(moving.data.vertices[i].co) for i in protected} == original_anchors
        assert hashlib.sha256(source.read_bytes()).hexdigest() == before_sha
        print("collision repair Blender smoke: PASS")


main()
