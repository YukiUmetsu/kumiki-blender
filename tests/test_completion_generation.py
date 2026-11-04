"""Ungated, accelerator-aware completion candidate generation contracts."""

import hashlib
import contextlib
import io
import json
import tempfile
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from kumiki_blender.cli import main
from kumiki_blender.reconstruction.completion_generation import (
    GenerationError, _enable_mps_marching_cubes_compat,
    _resolve_public_model, _load_weights_only,
    choose_torch_device, generate_triposr_candidate,
    suggest_completion_regions,
)
from tests.test_asset_import import fixture_glb


class CompletionGenerationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.image = self.root / "input.png"
        self.image.write_bytes(b"private-image")

    def test_selects_mps_over_cuda_on_apple_silicon_and_never_silently_falls_back(self):
        torch = SimpleNamespace(
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
            cuda=SimpleNamespace(is_available=lambda: True),
        )
        self.assertEqual(choose_torch_device(torch, "auto"), "mps")
        self.assertEqual(choose_torch_device(torch, "mps"), "mps")
        torch.backends.mps.is_available = lambda: False
        self.assertEqual(choose_torch_device(torch, "auto"), "cuda:0")
        torch.cuda.is_available = lambda: False
        with self.assertRaisesRegex(GenerationError, "No accelerated"):
            choose_torch_device(torch, "auto")
        with self.assertRaisesRegex(GenerationError, "not available"):
            choose_torch_device(torch, "mps")
        self.assertEqual(choose_torch_device(torch, "cpu"), "cpu")

    def test_candidate_records_hashes_accelerator_and_ungated_model(self):
        def fake_inference(image, output, **kwargs):
            fixture_glb(output)
            self.assertEqual(image, self.image.resolve())
            self.assertEqual(kwargs["device"], "mps")
            return {"device": "mps", "checkpoint_sha256": "a" * 64,
                    "config_sha256": "b" * 64, "revision": "c" * 40}

        out = self.root / "candidate-r01"
        result = generate_triposr_candidate(self.image, out, requested_device="mps",
                                            inference=fake_inference)
        self.assertEqual(result["generator"]["model_id"], "stabilityai/TripoSR")
        self.assertEqual(result["generator"]["license"], "MIT")
        self.assertEqual(result["device"], "mps")
        self.assertEqual(result["origin"], "generated")
        self.assertEqual(result["input_sha256"], hashlib.sha256(b"private-image").hexdigest())
        self.assertEqual(result["candidate_sha256"],
                         hashlib.sha256((out / "candidate.glb").read_bytes()).hexdigest())
        self.assertEqual(json.loads((out / "provenance.json").read_text()), result)
        self.assertFalse((out / "input.png").exists())  # private source never copied

    def test_output_is_exclusive_and_unusable_mesh_cleans_up(self):
        out = self.root / "candidate-r01"
        out.mkdir()
        with self.assertRaisesRegex(GenerationError, "already exists"):
            generate_triposr_candidate(self.image, out, inference=Mock())
        self.assertEqual(list(out.iterdir()), [])
        out.rmdir()

        def bad_inference(_image, output, **_kwargs):
            output.write_bytes(b"not a glb")
            return {"device": "mps", "checkpoint_sha256": "a" * 64,
                    "config_sha256": "b" * 64, "revision": "c" * 40}

        with self.assertRaises(GenerationError):
            generate_triposr_candidate(self.image, out, inference=bad_inference)
        self.assertFalse(out.exists())

    def test_rejects_unknown_model_without_download(self):
        with self.assertRaisesRegex(GenerationError, "ungated"):
            generate_triposr_candidate(self.image, self.root / "out",
                                        model_id="stabilityai/stable-fast-3d", inference=Mock())
        self.assertFalse((self.root / "out").exists())

    def test_mps_shim_avoids_cuda_only_marching_cubes_extension(self):
        with patch.dict(sys.modules):
            _enable_mps_marching_cubes_compat()
            self.assertTrue(callable(sys.modules["torchmcubes"].marching_cubes))

    def test_ungated_download_does_not_forward_token_and_loader_rejects_pickle(self):
        download = Mock(return_value=str(self.root))
        self.assertEqual(_resolve_public_model(download, "main", False), self.root)
        download.assert_called_once_with(
            repo_id="stabilityai/TripoSR", revision="main",
            allow_patterns=["config.yaml", "model.ckpt"], token=False,
            local_files_only=False,
        )
        fake_model = Mock()
        tsr = Mock(return_value=fake_model)
        yaml = SimpleNamespace(load=Mock(return_value={"config": "test"}), resolve=Mock())
        torch = SimpleNamespace(load=Mock(return_value={"a": "tensor"}))
        self.assertIs(_load_weights_only(torch, tsr, yaml,
                                         self.root / "config.yaml", self.root / "model.ckpt"),
                      fake_model)
        self.assertTrue(torch.load.call_args.kwargs["weights_only"])

    def test_mutated_reference_during_inference_rejected(self):
        def mutate(image, output, **kwargs):
            fixture_glb(output)
            image.write_bytes(b"changed-source")
            return {"device": "mps", "checkpoint_sha256": "a" * 64,
                    "config_sha256": "b" * 64, "revision": "c" * 40}
        out = self.root / "candidate-r01"
        with self.assertRaisesRegex(GenerationError, "changed"):
            generate_triposr_candidate(self.image, out, inference=mutate)
        self.assertFalse(out.exists())

    def test_gaps_are_suggestions_not_claims_of_missing_geometry(self):
        audit = {
            "schema_version": 1, "asset_id": "great-sphinx", "missing_regions": ["tail"],
            "region_counts": {"face": 1, "torso": 3}, "missing_views": ["rear"],
        }
        path = self.root / "reference-audit.json"
        path.write_text(json.dumps(audit))
        report = suggest_completion_regions(path, min_views=2)
        self.assertEqual([x["region"] for x in report["candidates"]], ["face", "tail"])
        self.assertEqual(report["missing_views"], ["rear"])
        self.assertFalse(report["geometry_missing_verified"])

    def test_malformed_audit_rejected(self):
        path = self.root / "bad.json"
        path.write_text(json.dumps({"schema_version": 1, "asset_id": "demo",
                                   "missing_regions": ["../../../evil"], "region_counts": {},
                                   "missing_views": []}))
        with self.assertRaises(GenerationError):
            suggest_completion_regions(path)

    def test_cli_gap_suggestion_and_gated_model_rejection(self):
        audit = self.root / "audit.json"
        audit.write_text(json.dumps({"schema_version": 1, "asset_id": "sphinx",
                                     "region_counts": {"head": 1}, "missing_regions": ["tail"],
                                     "missing_views": ["rear"]}))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(main(["suggest-completion", "--audit", str(audit)]), 0)
        self.assertEqual(len(json.loads(stdout.getvalue())["candidates"]), 2)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(main(["generate-completion-part", "--image", str(self.image),
                                   "--output", str(self.root / "generated"),
                                   "--model", "stabilityai/stable-fast-3d"]), 2)
        self.assertIn("ungated", stderr.getvalue())
        self.assertFalse((self.root / "generated").exists())

    def test_completion_doctor_does_not_download_weights(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(main(["completion-doctor"]), 0)
        report = json.loads(stdout.getvalue())
        self.assertFalse(report["model_gated"])
        self.assertEqual(report["model_license"], "MIT")
        self.assertIn("torch", report["dependency_available"])


if __name__ == "__main__":
    unittest.main()
