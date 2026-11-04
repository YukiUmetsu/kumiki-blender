"""Evidence-bound, non-destructive generated-part admission and fitting."""

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.reconstruction.completion import (
    CompletionError, evaluate_completion, load_completion_plan, verify_completion_sources,
)
from tests.test_asset_import import fixture_glb


SOURCE = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base = self.root / "base.glb"
        self.candidate = self.root / "ear.glb"
        fixture_glb(self.base)
        fixture_glb(self.candidate)
        self.manifest = self.root / "completion.json"
        self.data = {
            "schema_version": 1,
            "asset_id": "great-sphinx",
            "mode": "cinematic",
            "base_origin": "observed",
            "base_glb": "base.glb",
            "base_sha256": hashlib.sha256(self.base.read_bytes()).hexdigest(),
            "regions": [{
                "id": "left-ear", "candidate_glb": "ear.glb",
                "candidate_sha256": hashlib.sha256(self.candidate.read_bytes()).hexdigest(),
                "origin": "generated", "generator": {
                    "adapter": "supplied-glb", "model_id": "local-image-to-3d",
                },
                "max_error_m": 0.02,
                "anchors": [
                    {"id": f"point-{index}", "source_xyz_m": list(point),
                     "target_xyz_m": list(point)}
                    for index, point in enumerate(SOURCE)
                ],
            }],
        }

    def write(self):
        self.manifest.write_text(json.dumps(self.data), encoding="utf-8")

    def test_cinematic_includes_generated_geometry_without_writing_files(self):
        self.write()
        result = evaluate_completion(load_completion_plan(self.manifest))
        self.assertEqual(result["mode"], "cinematic")
        self.assertEqual(result["regions"][0]["geometry_origin"], "inferred")
        self.assertTrue(result["regions"][0]["included"])
        self.assertFalse(result["historical_fidelity_verified"])
        self.assertEqual({path.name for path in self.root.iterdir()},
                         {"base.glb", "ear.glb", "completion.json"})

    def test_historical_excludes_generated_but_includes_observed_scan(self):
        self.data["mode"] = "historical"
        self.write()
        result = evaluate_completion(load_completion_plan(self.manifest))
        self.assertFalse(result["regions"][0]["included"])
        self.assertIn("historical", result["regions"][0]["reason"])
        self.data["regions"][0]["origin"] = "observed-scan"
        self.data["regions"][0]["source_url"] = "https://example.org/survey"
        self.data["regions"][0]["license"] = "CC0-1.0"
        self.data["regions"][0].pop("generator")
        self.write()
        result = evaluate_completion(load_completion_plan(self.manifest))
        self.assertTrue(result["regions"][0]["included"])
        self.assertEqual(result["regions"][0]["geometry_origin"], "observed")

    def test_rejects_mismatched_hash_and_symlink_escape(self):
        self.data["base_sha256"] = "0" * 64
        self.write()
        with self.assertRaisesRegex(CompletionError, "SHA-256"):
            load_completion_plan(self.manifest)
        self.data["base_sha256"] = hashlib.sha256(self.base.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as other:
            outer = Path(other) / "external.glb"
            fixture_glb(outer)
            (self.root / "external.glb").symlink_to(outer)
            self.data["regions"][0]["candidate_glb"] = "external.glb"
            self.data["regions"][0]["candidate_sha256"] = hashlib.sha256(outer.read_bytes()).hexdigest()
            self.write()
            with self.assertRaisesRegex(CompletionError, "escapes"):
                load_completion_plan(self.manifest)

    def test_rejects_unknown_fields_and_invalid_mode(self):
        self.data["mode"] = "made-up"
        self.write()
        with self.assertRaisesRegex(CompletionError, "mode"):
            load_completion_plan(self.manifest)
        self.data["mode"] = "historical"
        self.data["surprise"] = True
        self.write()
        with self.assertRaisesRegex(CompletionError, "schema"):
            load_completion_plan(self.manifest)

    def test_rejects_unattributed_observational_source(self):
        self.data["regions"][0]["origin"] = "observed-survey"
        self.data["regions"][0].pop("generator")
        self.write()
        with self.assertRaisesRegex(CompletionError, "source_url"):
            load_completion_plan(self.manifest)

    def test_fits_proper_rotation_scale_translation_to_measured_anchors(self):
        # Rotation about Z by 90 degrees, scale 2, translation (3, -1, 4).
        for anchor in self.data["regions"][0]["anchors"]:
            x, y, z = anchor["source_xyz_m"]
            anchor["target_xyz_m"] = [3 - 2*y, -1 + 2*x, 4 + 2*z]
        self.write()
        result = evaluate_completion(load_completion_plan(self.manifest))["regions"][0]
        self.assertLess(result["max_error_m"], 1e-6)
        self.assertAlmostEqual(result["transform"]["scale"], 2.0, places=6)
        for actual, expected in zip(result["transform"]["translation_m"], (3, -1, 4)):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_rejects_mirrored_or_coplanar_fit(self):
        for anchor in self.data["regions"][0]["anchors"]:
            x, y, z = anchor["source_xyz_m"]
            anchor["target_xyz_m"] = [-x, y, z]
        self.write()
        with self.assertRaisesRegex(CompletionError, "mirror"):
            evaluate_completion(load_completion_plan(self.manifest))
        for anchor in self.data["regions"][0]["anchors"]:
            anchor["target_xyz_m"][2] = 0
        self.write()
        with self.assertRaisesRegex(CompletionError, "coplanar"):
            evaluate_completion(load_completion_plan(self.manifest))

    def test_rejects_excessive_anchor_error_even_in_cinematic_mode(self):
        self.data["regions"][0]["anchors"][-1]["target_xyz_m"] = [0, 0, 2]
        self.write()
        with self.assertRaisesRegex(CompletionError, "residual"):
            evaluate_completion(load_completion_plan(self.manifest))

    def test_malformed_nested_types_fail_with_actionable_completion_errors(self):
        self.data["regions"][0]["origin"] = []
        self.write()
        with self.assertRaises(CompletionError):
            load_completion_plan(self.manifest)
        self.data["regions"][0]["origin"] = "generated"
        self.data["regions"][0]["generator"] = []
        self.write()
        with self.assertRaises(CompletionError):
            load_completion_plan(self.manifest)
        self.data["regions"][0]["generator"] = {"adapter": "supplied-glb", "model_id": "example"}
        self.data["regions"][0]["origin"] = "observed-scan"
        self.data["regions"][0]["license"] = "CC0-1.0"
        self.data["regions"][0]["source_url"] = "https://[invalid"
        self.write()
        with self.assertRaises(CompletionError):
            load_completion_plan(self.manifest)

    def test_records_realistic_model_repository_identifiers_as_metadata(self):
        self.data["regions"][0]["generator"]["model_id"] = "microsoft/TRELLIS.2-4B"
        self.write()
        record = evaluate_completion(load_completion_plan(self.manifest))["regions"][0]
        self.assertEqual(record["generator"]["model_id"], "microsoft/TRELLIS.2-4B")

    def test_requires_explicit_base_origin_and_rejects_inferred_historical_base(self):
        self.data.pop("base_origin")
        self.write()
        with self.assertRaisesRegex(CompletionError, "base_origin"):
            load_completion_plan(self.manifest)
        self.data["base_origin"] = "inferred"
        self.data["mode"] = "historical"
        self.write()
        with self.assertRaisesRegex(CompletionError, "historical"):
            load_completion_plan(self.manifest)

    def test_revalidates_source_bytes_after_loading_before_export(self):
        self.data["base_origin"] = "observed"
        self.write()
        plan = load_completion_plan(self.manifest)
        fixture_glb(self.candidate, materials=["Changed-after-planning"])
        with self.assertRaisesRegex(CompletionError, "SHA-256"):
            verify_completion_sources(plan)

    def test_thin_but_nonplanar_mirror_never_passes_on_small_residual(self):
        self.data["base_origin"] = "observed"
        for anchor in self.data["regions"][0]["anchors"]:
            x, y, z = anchor["source_xyz_m"]
            anchor["source_xyz_m"] = [x, y, z / 100_000]
            anchor["target_xyz_m"] = [-x, y, z / 100_000]
        self.write()
        with self.assertRaisesRegex(CompletionError, "mirror"):
            evaluate_completion(load_completion_plan(self.manifest))

    def test_huge_json_integers_reject_with_completion_error(self):
        self.data["base_origin"] = "observed"
        self.data["regions"][0]["anchors"][0]["source_xyz_m"][0] = 10**400
        self.write()
        with self.assertRaises(CompletionError):
            load_completion_plan(self.manifest)

    def test_historical_preflight_rejects_inferred_tags_inside_base_glb(self):
        data = {"asset": {"version": "2.0"}, "meshes": [{"primitives": [{}]}],
                "nodes": [{"mesh": 0, "extras": {"kumiki_geometry_origin": "inferred",
                                                 "kumiki_part": "completion-ear"}}]}
        payload = json.dumps(data).encode("utf-8")
        payload += b" " * (-len(payload) % 4)
        self.base.write_bytes(struct.pack("<4sII", b"glTF", 2, len(payload)+20)
                              + struct.pack("<I4s", len(payload), b"JSON") + payload)
        self.data["base_sha256"] = hashlib.sha256(self.base.read_bytes()).hexdigest()
        self.data["mode"] = "historical"
        self.write()
        with self.assertRaisesRegex(CompletionError, "inferred"):
            load_completion_plan(self.manifest)

    def test_observed_candidate_cannot_launder_inferred_mesh_tags(self):
        data = {"asset": {"version": "2.0"}, "meshes": [{"primitives": [{}]}],
                "nodes": [{"mesh": 0, "extras": {"kumiki_geometry_origin": "inferred"}}]}
        payload = json.dumps(data).encode("utf-8")
        payload += b" " * (-len(payload) % 4)
        self.candidate.write_bytes(struct.pack("<4sII", b"glTF", 2, len(payload)+20)
                                   + struct.pack("<I4s", len(payload), b"JSON") + payload)
        region = self.data["regions"][0]
        region["candidate_sha256"] = hashlib.sha256(self.candidate.read_bytes()).hexdigest()
        region["origin"] = "observed-scan"
        region["source_url"] = "https://example.org/real-scan"
        region["license"] = "CC0-1.0"
        region.pop("generator")
        self.data["mode"] = "historical"
        self.write()
        with self.assertRaisesRegex(CompletionError, "inferred"):
            load_completion_plan(self.manifest)
        self.data["regions"][0]["anchors"][0]["source_xyz_m"][0] = 0
        self.data["regions"][0]["max_error_m"] = 10**400
        self.write()
        with self.assertRaises(CompletionError):
            load_completion_plan(self.manifest)


if __name__ == "__main__":
    unittest.main()
