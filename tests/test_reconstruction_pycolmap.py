"""Exercise optional CPU-PyCOLMAP entry points without requiring a wheel in CI."""

import contextlib
import io
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from kumiki_blender.reconstruction.colmap import build_sparse_reconstruction
from kumiki_blender.reconstruction.pycolmap_worker import main
from kumiki_blender.reconstruction.references import ReferenceError

from tests import test_reconstruction_colmap as fixture_colmap


class PycolmapBackendTests(unittest.TestCase):
    def test_help_exposes_expected_cpu_flags_without_importing_wheel(self):
        for stage, expected in (("feature_extractor", "--FeatureExtraction.use_gpu"),
                                ("exhaustive_matcher", "--FeatureMatching.use_gpu")):
            with io.StringIO() as stream, contextlib.redirect_stdout(stream):
                self.assertEqual(main([stage, "--help"]), 0)
                self.assertIn(expected, stream.getvalue())

    def test_missing_optional_wheel_is_a_preflight_error(self):
        fixture = fixture_colmap.ColmapWorkflowTests()
        fixture.setUp()
        try:
            with patch("kumiki_blender.reconstruction.colmap.importlib.util.find_spec",
                       return_value=None):
                with self.assertRaisesRegex(ReferenceError, "PyCOLMAP is not installed"):
                    build_sparse_reconstruction(fixture.references(), fixture.root / "out",
                                                engine="pycolmap")
            self.assertFalse((fixture.root / "out").exists())
        finally:
            fixture.temp.cleanup()

    def test_worker_passes_cpu_and_image_size_to_real_entry_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fake = types.SimpleNamespace(
                Device=types.SimpleNamespace(cpu="cpu"),
                ImageReaderOptions=lambda: types.SimpleNamespace(mask_path=""),
                FeatureExtractionOptions=lambda: types.SimpleNamespace(max_image_size=-1),
                extract_features=Mock(),
            )
            with patch.dict("sys.modules", {"pycolmap": fake}):
                self.assertEqual(main(["feature_extractor", "--database_path", str(root / "db"),
                                       "--image_path", str(root / "images"),
                                       "--ImageReader.mask_path", str(root / "masks"),
                                       "--FeatureExtraction.use_gpu", "0", "--max_image_size", "1280"]), 0)
                kwargs = fake.extract_features.call_args.kwargs
                self.assertEqual(kwargs["device"], "cpu")
                self.assertEqual(kwargs["extraction_options"].max_image_size, 1280)
                self.assertEqual(kwargs["reader_options"].mask_path, str(root / "masks"))

    def test_worker_rejects_accidental_gpu_or_wrong_output_format(self):
        fake = types.SimpleNamespace(Device=types.SimpleNamespace(cpu="cpu"))
        with patch.dict("sys.modules", {"pycolmap": fake}):
            with self.assertRaisesRegex(ValueError, "CPU-only"):
                main(["exhaustive_matcher", "--database_path", "unused",
                      "--FeatureMatching.use_gpu", "1"])
            with self.assertRaisesRegex(ValueError, "Only TXT"):
                main(["model_converter", "--input_path", "unused",
                      "--output_path", "unused", "--output_type", "PLY"])


if __name__ == "__main__":
    unittest.main()
