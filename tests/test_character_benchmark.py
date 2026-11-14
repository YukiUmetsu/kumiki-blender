"""M0 reproducible benchmark tests using actual registered CharacterSpec evidence.

Private Pip/Quill assets are not required: synthetic reference masks, pose
receipts and a minimal skinned+animated GLB document exercise the same contract.
"""
from __future__ import annotations

import hashlib
import io
import json
import struct
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from PIL import Image, ImageDraw

from kumiki_blender.character_build.__main__ import main as character_cli
from kumiki_blender.character_build.benchmark import (
    character_benchmark, inspect_glb,
)
from kumiki_blender.character_build.character_spec_v2 import plan_character
from kumiki_blender.character_build.region_revision_gate import gate_region_revision


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def minimal_glb(path: Path, *, clips=("talk",), skinned=True, morph=True):
    doc = {
        "asset": {"version": "2.0"},
        "nodes": [{"name": "CharacterRoot"}, {"name": "spine"},
                  {"name": "face", "mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0},
                                   "indices": 1,
                                   "targets": [{"POSITION": 2}] if morph else []}]}],
        "accessors": [
            {"count": 3, "type": "VEC3", "componentType": 5126},
            {"count": 3, "type": "SCALAR", "componentType": 5123},
            {"count": 3, "type": "VEC3", "componentType": 5126},
        ],
        "skins": [{"joints": [1]}] if skinned else [],
        "animations": [
            {"name": name, "samplers": [{"input": 0, "output": 0}],
             "channels": [{"sampler": 0, "target": {"node": 1, "path": "rotation"}},
                          {"sampler": 0, "target": {"node": 2, "path": "weights"}}]
             } for name in clips
        ],
    }
    body = json.dumps(doc).encode("utf-8")
    body += b" " * (-len(body) % 4)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 20+len(body))
                     + struct.pack("<II", len(body), 0x4E4F534A) + body)


class CharacterBenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/"masks").mkdir()
        (self.root/"mesh").mkdir()
        img = Image.new("L", (1600, 1200))
        ImageDraw.Draw(img).rectangle((771, 528, 829, 586), fill=255)
        img.save(self.root/"masks/front.png")
        ply = self.root/"mesh/eye.ply"
        ply.write_text(
            "ply\nformat ascii 1.0\n"
            "element vertex 4\nproperty float x\nproperty float y\nproperty float z\n"
            "element face 2\nproperty list uchar int vertex_indices\nend_header\n"
            "-0.01 0 0.18\n0.01 0 0.18\n0.01 0 0.20\n-0.01 0 0.20\n"
            "3 0 1 2\n3 0 2 3\n", encoding="utf-8")
        spec = {
            "schema_version": 2, "character_id": "test-owl",
            "height_m": .35, "profile": "avian-biped-wings",
            "reference_frame": {"width": 1600, "height": 1200, "figure_px": 1000},
            "regions": [
                {"id": "head", "type": "head", "source_confidence": "estimated"},
                {"id": "eye.L", "parent": "head", "type": "eye",
                 "source_confidence": "verified",
                 "reference_masks": {"front": "masks/front.png"},
                 "verified_views": ["front"], "source_mesh": "mesh/eye.ply",
                 "detail_critical": True},
            ],
        }
        (self.root/"character.json").write_text(json.dumps(spec))
        self.plan = plan_character(self.root/"character.json", self.root/"plan")
        self.region_report = gate_region_revision(
            self.root/"plan", "eye.L", ply, self.root/"eye-qa")
        self.blend = self.root/"rig.blend"
        self.blend.write_bytes(b"BLENDER-original-unmodified-rig")
        pose = {
            "schema_version": 1,
            "status": "deformation-review-required",
            "source_blend": str(self.blend.resolve()),
            "source_blend_sha256": sha(self.blend),
            "armature": "CharacterRig",
            "rest_skin_weights": {
                "Body": {"vertices": 24, "unweighted": 0,
                         "over_four_influences": 0, "non_normalized": 0},
            },
            "poses": [{"name": "wing_fold", "meshes": {"Body": {
                        "topology_changed": False,
                        "p95_relative_edge_stretch": .06,
                        "closed_manifold": True}},
                       "collision_candidates": []}],
            "problems": [],
        }
        (self.root/"pose").mkdir()
        (self.root/"pose/report.json").write_text(json.dumps(pose))
        self.glb = self.root/"character.glb"
        minimal_glb(self.glb)
        inputs = {}
        for name in ("handoff_spec","mesh_obj","baseline_texture","candidate_texture","aligned_front"):
            file = self.root / f"{name}.data"
            file.write_bytes(name.encode("utf-8"))
            inputs[name] = file
        (self.root/"texture").mkdir()
        self.texture_report = {
            "schema_version": 1, "numeric_gate_pass": True,
            "status": "no-change-artistic-review-required",
            "source_view_names": ["front"],
            "failed_views": [],
            "views": [{"view": "front", "baseline": {"mean_delta_e": 0},
                       "candidate": {"mean_delta_e": 0},
                       "rejection_reasons": []}],
            "input_sha256": {name: sha(file) for name,file in inputs.items()},
            "max_projection_samples": 2_000_000,
        }
        (self.root/"texture/report.json").write_text(json.dumps(self.texture_report))
        self.manifest = {
            "schema_version": 1,
            "character_plan": "plan",
            "required_checks": ["regions", "textures", "pose", "glb"],
            "required_regions": ["eye.L"],
            "required_clips": ["talk"],
            "required_poses": ["wing_fold"],
            "region_reports": [{"region": "eye.L", "report": "eye-qa/report.json"}],
            "texture_reports": [{
                "id": "basecolor", "report": "texture/report.json",
                "inputs": {name: file.name for name, file in inputs.items()},
            }],
            "pose_report": {"report": "pose/report.json", "blend": "rig.blend"},
            "glb": "character.glb",
            "runtime_observations": [{
                "stage": "sample-head-sculpt", "wall_time_s": 1.8,
                "cpu_time_s": .7, "peak_rss_mb": 100.0, "machine": "synthetic",
            }],
        }
        self.file = self.root/"benchmark.json"
        self.write_manifest()

    def write_manifest(self):
        self.file.write_text(json.dumps(self.manifest))

    def test_complete_benchmark_is_numeric_review_not_artistic_approval(self):
        result = character_benchmark(self.file, self.root/"result")
        self.assertTrue(result["numeric_gates_pass"])
        self.assertTrue(result["artistic_review_required"])
        self.assertEqual(result["status"], "numeric-gates-passed-artistic-review-required")
        self.assertEqual(result["required_regions"], ["eye.L"])
        self.assertEqual(result["missing_evidence"], [])
        self.assertEqual(result["numeric_failures"], [])
        self.assertEqual(result["checks"]["regions"]["items"][0]["verified_views"], ["front"])
        self.assertEqual(result["checks"]["pose"]["items"][0]["pose_names"], ["wing_fold"])
        glb = result["checks"]["glb"]["items"][0]
        self.assertEqual(glb["clip_names"], ["talk"])
        self.assertEqual(glb["joint_count"], 1)
        self.assertEqual(glb["morph_target_count"], 1)
        self.assertEqual(glb["animation_channels"][0]["morph_weight_channels"], 1)
        self.assertEqual(result["runtime_observations_source"],
                         "externally_supplied_not_measured_by_benchmark")
        self.assertIn("artistic review", (self.root/"result/SUMMARY.md").read_text())
        with self.assertRaises(FileExistsError):
            character_benchmark(self.file, self.root/"result")

    def test_semantic_weight_zone_failure_blocks_numerical_character_readiness(self):
        path = self.root / "pose/report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        report["rest_weight_zones"] = [{
            "mesh": "Wing.L",
            "region_group": "REGION_WingLeft",
            "selected_vertices": 100,
            "minimum_required_vertices": 25,
            "violating_vertices": 1,
            "max_outside_weight_observed": .1,
            "status": "fail",
        }]
        # An external receipt that misleadingly says "review required"
        # must never hide a real semantic skin-weight failure.
        report["status"] = "deformation-review-required"
        path.write_text(json.dumps(report), encoding="utf-8")
        result = character_benchmark(self.file, self.root / "bad-zone")
        self.assertFalse(result["numeric_gates_pass"])
        self.assertEqual(result["checks"]["pose"]["items"][0]["weight_zone_count"], 1)
        self.assertEqual(result["checks"]["pose"]["items"][0]["failed_weight_zones"],
                         ["Wing.L/REGION_WingLeft"])
        self.assertIn("pose:structural-or-skinning", result["numeric_failures"])

    def test_healthy_semantic_weight_zone_is_included_in_evidence(self):
        path = self.root / "pose/report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        report["rest_weight_zones"] = [{
            "mesh": "Wing.L",
            "region_group": "REGION_WingLeft",
            "selected_vertices": 100,
            "minimum_required_vertices": 25,
            "violating_vertices": 0,
            "status": "pass",
        }]
        path.write_text(json.dumps(report), encoding="utf-8")
        result = character_benchmark(self.file, self.root / "valid-zone")
        self.assertTrue(result["numeric_gates_pass"])
        self.assertEqual(result["checks"]["pose"]["items"][0]["weight_zone_count"], 1)
        self.assertEqual(result["checks"]["pose"]["items"][0]["failed_weight_zones"], [])

    def test_missing_report_does_not_fake_release_readiness(self):
        self.manifest["region_reports"] = []
        self.manifest["texture_reports"] = []
        self.manifest["pose_report"] = None
        self.manifest["glb"] = None
        self.write_manifest()
        result = character_benchmark(self.file, self.root/"missing")
        self.assertFalse(result["numeric_gates_pass"])
        self.assertEqual(result["status"], "missing-required-evidence")
        self.assertTrue(any("eye.L" in name for name in result["missing_evidence"]))
        self.assertIn("pose:no-source-locked-pose-report", result["missing_evidence"])
        self.assertFalse(result["checks"]["glb"]["checked"])

    def test_failed_region_gate_cannot_be_mislabeled_as_success(self):
        path = self.root/"eye-qa/report.json"
        report = json.loads(path.read_text())
        report["numeric_gate_pass"] = False
        report["failed_views"] = ["front"]
        report["views"][0]["rejection_reasons"] = ["registered_contour_regression"]
        path.write_text(json.dumps(report))
        result = character_benchmark(self.file, self.root/"failed")
        self.assertEqual(result["status"], "numeric-gate-failed")
        self.assertIn("regions:eye.L", result["numeric_failures"])
        self.assertTrue((self.root/"failed/SUMMARY.md").is_file())

    def test_stale_reference_image_is_rejected_before_output(self):
        img = Image.open(self.root/"masks/front.png")
        data = img.copy()
        img.close()
        data.putpixel((0, 0), 255)
        data.save(self.root/"masks/front.png")
        with self.assertRaisesRegex(ValueError, "changed"):
            character_benchmark(self.file, self.root/"invalid")
        self.assertFalse((self.root/"invalid").exists())

    def test_stale_region_geometry_and_texture_inputs_are_rejected(self):
        ply = self.root/"mesh/eye.ply"
        ply.write_text(ply.read_text()+"\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            character_benchmark(self.file, self.root/"invalid")
        self.assertFalse((self.root/"invalid").exists())

    def test_stale_pose_source_or_missing_origin_hash_rejected(self):
        self.blend.write_bytes(b"BLENDER-updated-rig")
        with self.assertRaisesRegex(ValueError, "stale|source-locked"):
            character_benchmark(self.file, self.root/"invalid")
        self.assertFalse((self.root/"invalid").exists())

    def test_unreferenced_glb_clip_blocks_numeric_gate(self):
        self.manifest["required_clips"] = ["talk", "blink"]
        self.write_manifest()
        result = character_benchmark(self.file, self.root/"missing-clip")
        self.assertEqual(result["status"], "numeric-gate-failed")
        self.assertIn("glb:missing-skinned-animation-or-clips", result["numeric_failures"])
        self.assertEqual(result["checks"]["glb"]["items"][0]["missing_required_clips"], ["blink"])

    def test_invalid_glb_chunk_rejected_without_report(self):
        self.glb.write_bytes(b"broken")
        with self.assertRaisesRegex(ValueError, "truncated"):
            character_benchmark(self.file, self.root/"invalid")
        self.assertFalse((self.root/"invalid").exists())

    def test_unverified_region_cannot_be_declared_required(self):
        self.manifest["required_regions"] = ["head"]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "verified"):
            character_benchmark(self.file, self.root/"invalid")

    def test_cli_failure_and_evidence_are_stable(self):
        self.manifest["pose_report"] = None
        self.write_manifest()
        out = io.StringIO()
        with redirect_stdout(out):
            code = character_cli([
                "character-benchmark", "--manifest", str(self.file),
                "--output", str(self.root/"cli")])
        self.assertEqual(code, 1)
        self.assertFalse(json.loads(out.getvalue())["numeric_gates_pass"])
        self.assertTrue((self.root/"cli/report.json").is_file())

    def test_texture_source_mismatch_detected(self):
        (self.root/"aligned_front.data").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed since texture QA"):
            character_benchmark(self.file, self.root/"invalid")

    def test_invalid_runtime_measurement_rejected(self):
        self.manifest["runtime_observations"][0]["peak_rss_mb"] = float("nan")
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "runtime observation"):
            character_benchmark(self.file, self.root/"invalid")


if __name__ == "__main__":
    unittest.main()
