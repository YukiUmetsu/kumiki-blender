"""AI bridge previews remain separate from camera evidence and source photos."""

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from kumiki_blender.reconstruction.ai_bridge_previews import build_ai_bridge_previews


class AIBridgePreviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.sparse = self.root / "cameras"
        (self.sparse / "images").mkdir(parents=True)
        for n in (1, 2):
            Image.new("RGB", (160, 100), (80 + n, 150, 210)).save(
                self.sparse / f"images/photo-000{n}.png")
        records = []
        for number, name in ((1, "sphinx-left.jpg"), (2, "sphinx-front.jpg")):
            path = self.sparse / f"images/photo-000{number}.png"
            records.append({"filename": path.name, "original_filename": name,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        (self.sparse / "staged-reference-index.json").write_text(json.dumps({"images": records}))
        self.coverage = {"existing_photo_priorities": [{
            "filename": "sphinx-left.jpg", "kind": "bridge-and-retry",
            "best_registered_bridge_photo": "sphinx-front.jpg",
            "existing_2d3d_candidates": 15}]}

    def test_without_workflow_generates_safe_plan_only(self):
        report = build_ai_bridge_previews(
            self.sparse, self.coverage, self.root / "candidate", auto_workflow=False)
        self.assertEqual(report["status"], "plan-only / no-workflow")
        self.assertEqual(len(report["requests"]), 1)
        self.assertFalse(report["usable_as_sfm_input"])
        self.assertEqual(report["generated"], [])
        self.assertIn("registered_anchor_sha256", report["requests"][0])
        self.assertFalse((self.sparse / "images/synthetic-01.png").exists())

    def test_local_workflow_generates_separately_labeled_candidate(self):
        workflow = self.root / "workflow.json"
        workflow.write_text(json.dumps({"1": {"class_type": "LoadImage", "inputs": {}},
                                        "9": {"class_type": "SaveImage", "inputs": {}}}))
        calls = []

        def fake_generate(image, graph, output, **options):
            calls.append((image, graph, options))
            output.mkdir()
            Image.new("RGB", (16, 16), "white").save(output / "candidate.png")
            return {"origin": "generated", "output_file": "candidate.png"}

        report = build_ai_bridge_previews(
            self.sparse, self.coverage, self.root / "generated",
            workflow=workflow, generator=fake_generate)
        self.assertEqual(report["status"], "synthetic-previews-created / artist-review-required")
        self.assertFalse(report["usable_as_sfm_input"])
        self.assertTrue((self.root / "generated/synthetic-01/candidate.png").is_file())
        self.assertEqual(calls[0][2]["image_node"], "1")
        self.assertEqual(calls[0][2]["output_node"], "9")
        self.assertEqual(calls[0][2]["require_mps"], True)

    def test_automatically_generated_workflow_runs_same_local_adapter(self):
        calls = []

        def mock_factory(destination, *, asset_id, server):
            self.assertEqual(asset_id, "great-sphinx")
            self.assertEqual(server, "http://127.0.0.1:8188")
            destination.write_text(json.dumps({
                "1": {"class_type": "LoadImage", "inputs": {}},
                "8": {"class_type": "SaveImage", "inputs": {}}}))
            return {"checkpoint": "local-model.safetensors"}

        def mock_generator(image, graph, destination, **kwargs):
            calls.append(graph)
            destination.mkdir()
            Image.new("RGB", (16, 16), "white").save(destination / "candidate.png")
            return {"origin": "generated"}

        report = build_ai_bridge_previews(
            self.sparse, self.coverage, self.root / "auto", asset_id="great-sphinx",
            workflow_factory=mock_factory, generator=mock_generator)
        self.assertEqual(report["status"], "synthetic-previews-created / artist-review-required")
        self.assertEqual(report["auto_workflow"]["checkpoint"], "local-model.safetensors")
        self.assertEqual(calls[0].name, "auto-comfyui-workflow.json")
        self.assertFalse(report["usable_as_sfm_input"])

    def test_missing_pairs_produce_no_fake_bridges(self):
        report = build_ai_bridge_previews(
            self.sparse, {"existing_photo_priorities": []}, self.root / "empty")
        self.assertEqual(report["status"], "no-unregistered-photo-pairs")
        self.assertFalse(report["usable_as_sfm_input"])

    def test_source_tampering_is_detected_before_synthetic_generation(self):
        from kumiki_blender.reconstruction.references import ReferenceError
        Image.new("RGB", (160, 100), "white").save(
            self.sparse / "images/photo-0001.png")
        with self.assertRaisesRegex(ReferenceError, "changed after camera registration"):
            build_ai_bridge_previews(self.sparse, self.coverage, self.root / "candidate")
        self.assertFalse((self.root / "candidate").exists())


if __name__ == "__main__":
    unittest.main()
