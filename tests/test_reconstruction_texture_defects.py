"""Photo defect review must preserve genuine surface damage and original bytes."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from kumiki_blender.cli import main
from kumiki_blender.reconstruction.references import ReferenceError
from kumiki_blender.reconstruction.texture_defects import (
    DefectDetectionOptions, detect_texture_defects,
)


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PhotoDefectCandidateTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        folder = self.root / "dense" / "workspace"
        folder.mkdir(parents=True)
        self.dense = folder.parent
        self.obj, self.mtl = folder / "mesh.obj", folder / "mesh.mtl"
        self.atlas = folder / "stone.png"
        self.obj.write_text(
            "mtllib mesh.mtl\n"
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
            "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
            "usemtl rock\nf 1/1 2/2 3/3 4/4\n")
        self.mtl.write_text("newmtl rock\nmap_Kd stone.png\n")
        source = Image.new("RGBA", (128, 96), (120, 109, 95, 255))
        ImageDraw.Draw(source).rectangle((35, 35, 40, 40), fill=(236, 10, 20, 255))
        # Authentic local patina is visible in every observation as well.
        ImageDraw.Draw(source).rectangle((83, 59, 87, 63), fill=(79, 65, 55, 255))
        source.save(self.atlas)
        (self.dense / "dense-report.json").write_text(json.dumps({
            "status": "dense-mesh-candidate / artistic-review-required",
            "textured_obj": {"path": "workspace/mesh.obj", "sha256": checksum(self.obj)},
            "texture_atlases": [{"path": "workspace/stone.png", "sha256": checksum(self.atlas)}],
        }))
        self.evidence = self.root / "evidence"
        self.evidence.mkdir()
        self.recipe = self.evidence / "surface-recipe.json"
        Image.new("L", (128, 96), 255).save(self.evidence / "region.png")
        self.observations = []
        for n in range(3):
            photo = Image.new("RGBA", (128, 96), (120, 109, 95, 255))
            ImageDraw.Draw(photo).rectangle((83, 59, 87, 63), fill=(79, 65, 55, 255))
            photo.save(self.evidence / f"view-{n}.png")
            Image.new("L", (128, 96), 255).save(self.evidence / f"confidence-{n}.png")
            self.observations.append({"id": f"view-{n}", "photo": f"view-{n}.png",
                                      "confidence": f"confidence-{n}.png"})
        self._recipe()

    def _recipe(self, observations=None):
        self.recipe.write_text(json.dumps({"schema_version": 1,
            "atlases": {self.atlas.name: {
                "regions": [{"id": "stone", "mask": "region.png", "roughness": .86}],
                "observations": self.observations if observations is None else observations,
            }}}))

    def test_detects_local_transient_color_without_erasing_real_patina(self):
        before = self.atlas.read_bytes()
        output = self.root / "candidates"
        report = detect_texture_defects(self.dense, self.recipe, output)
        self.assertFalse(report["automatic_cleanup_performed"])
        item = report["atlases"][self.atlas.name]
        self.assertGreater(item["candidate_texels"], 0)
        self.assertTrue(item["cleanup_area_within_five_percent"])
        with Image.open(output / item["candidate_mask"]["path"]) as image:
            self.assertEqual(image.getpixel((37, 37)), 255)
            self.assertEqual(image.getpixel((85, 61)), 0)
            self.assertEqual(image.getpixel((0, 0)), 0)
        self.assertEqual(before, self.atlas.read_bytes())
        for key in ("candidate_mask", "severity", "overlay"):
            entry = item[key]
            self.assertEqual(checksum(output / entry["path"]), entry["sha256"])
        self.assertEqual(report["source_surface_recipe_sha256"], checksum(self.recipe))

    def test_requires_three_independent_views_and_refuses_output_overwrite(self):
        output = self.root / "candidates"
        self._recipe(observations=self.observations[:2])
        with self.assertRaises(ReferenceError):
            detect_texture_defects(self.dense, self.recipe, output)
        self.assertFalse(output.exists())
        self._recipe()
        detect_texture_defects(self.dense, self.recipe, output)
        with self.assertRaisesRegex(ReferenceError, "unused output"):
            detect_texture_defects(self.dense, self.recipe, output)

    def test_without_three_usable_cameras_outlier_is_not_labeled(self):
        path = self.evidence / "confidence-2.png"
        confidence = Image.new("L", (128, 96), 255)
        ImageDraw.Draw(confidence).rectangle((30, 30, 45, 45), fill=0)
        confidence.save(path)
        output = self.root / "incomplete"
        report = detect_texture_defects(self.dense, self.recipe, output)
        mask = Image.open(output / report["atlases"][self.atlas.name]["candidate_mask"]["path"])
        self.assertEqual(mask.getpixel((37, 37)), 0)

    def test_pinned_source_tampering_fails_without_partial_output(self):
        output = self.root / "bad-source"
        with Image.open(self.atlas) as original:
            changed = original.copy()
        changed.putpixel((1, 1), (0, 0, 0, 255))
        changed.save(self.atlas)
        with self.assertRaisesRegex(ReferenceError, "changed"):
            detect_texture_defects(self.dense, self.recipe, output)
        self.assertFalse(output.exists())

    def test_cli_and_invalid_thresholds(self):
        output = self.root / "cli-candidates"
        with self.assertRaisesRegex(ReferenceError, "color_difference"):
            detect_texture_defects(self.dense, self.recipe, output,
                                   options=DefectDetectionOptions(color_difference=3))
        with contextlib.redirect_stdout(io.StringIO()) as capture:
            self.assertEqual(main(["detect-photo-texture-defects",
                                   "--dense-revision", str(self.dense),
                                   "--surface-recipe", str(self.recipe),
                                   "--output", str(output)]), 0)
        self.assertIn("manual-review-required", capture.getvalue())


if __name__ == "__main__":
    unittest.main()
