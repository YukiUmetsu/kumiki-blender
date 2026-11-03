"""Read-only camera coverage comparison including legacy sparse reports."""

import json
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.cli import main
from kumiki_blender.reconstruction.camera_compare import compare_camera_registrations
from kumiki_blender.reconstruction.references import ReferenceError


class CameraComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def revision(self, name, *, registered, names=("front.png", "left.png", "back.png"),
                 modern=True, hashes=None):
        directory = self.root / name
        directory.mkdir()
        hashes = hashes or tuple(f"{idx:064x}" for idx in range(1, len(names) + 1))
        (directory / "staged-reference-index.json").write_text(json.dumps({"images": [
            {"filename": filename, "original_sha256": digest}
            for filename, digest in zip(names, hashes)]}))
        report = {"models": [{"model": "0", "registered_images": len(registered)}],
                  "matching_preset": "loma" if modern else "sift"}
        if modern:
            report["registered_filenames"] = list(registered)
        else:
            text_model = directory / "text-models" / "0"
            text_model.mkdir(parents=True)
            (text_model / "images.txt").write_text("".join(
                f"{idx} 1 0 0 0 0 0 0 1 {filename}\n\n"
                for idx, filename in enumerate(registered, 1)))
        (directory / "reconstruction-report.json").write_text(json.dumps(report))
        return directory

    def test_reports_newly_registered_and_remaining_missing_views(self):
        old = self.revision("old", registered=["front.png"], modern=False)
        new = self.revision("new", registered=["front.png", "left.png"])
        result = compare_camera_registrations(old, new)
        self.assertEqual(result["status"], "coverage-improved")
        self.assertEqual(result["newly_registered"], ["left.png"])
        self.assertEqual(result["still_unregistered"], ["back.png"])
        self.assertEqual(result["lost_registrations"], [])
        self.assertEqual(result["baseline_registered_shared"], 1)
        self.assertEqual(result["candidate_registered_shared"], 2)
        self.assertFalse(result["historical_fidelity_verified"])

    def test_uses_original_hashes_across_photo_renaming(self):
        old = self.revision("old", registered=["front.png", "left.png"])
        new = self.revision("new", registered=["front-new.png", "back-new.png"],
                            names=("front-new.png", "left-new.png", "back-new.png"))
        result = compare_camera_registrations(old, new)
        self.assertEqual(result["newly_registered"], ["back-new.png"])
        self.assertEqual(result["lost_registrations"], ["left.png"])
        self.assertEqual(result["status"], "mixed-coverage")

    def test_refuses_missing_or_invalid_evidence(self):
        old = self.revision("old", registered=["front.png"])
        invalid = self.revision("invalid", registered=["not-staged.png"])
        with self.assertRaisesRegex(ReferenceError, "Registered images"):
            compare_camera_registrations(old, invalid)
        other = self.revision("other", registered=["front.png"],
                              hashes=("a" * 64, "b" * 64, "c" * 64))
        with self.assertRaisesRegex(ReferenceError, "no common"):
            compare_camera_registrations(old, other)

    def test_compare_cli_produces_machine_readable_report(self):
        old = self.revision("old", registered=["front.png"])
        new = self.revision("new", registered=["front.png", "left.png"])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["compare-camera-registrations", "--baseline", str(old),
                                   "--candidate", str(new)]), 0)
        self.assertEqual(json.loads(output.getvalue())["newly_registered"], ["left.png"])


if __name__ == "__main__":
    unittest.main()
