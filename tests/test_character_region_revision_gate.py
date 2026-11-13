"""Registered source-camera no-regression gate for individual character revisions.

The synthetic closed box is positioned in the 1600x1200 canonical frame,
with front and right reference masks. Translating only its X coordinate
degrades the front silhouette but leaves the right projection unchanged.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from kumiki_blender.character_build.character_spec_v2 import plan_character
from kumiki_blender.character_build.region_revision_gate import gate_region_revision
from kumiki_blender.character_build.__main__ import main as character_cli


def _box(path: Path, *, dx: float = 0.0):
    vertices = [
        (-.01, -.01, .18), (.01, -.01, .18),
        (.01, -.01, .20), (-.01, -.01, .20),
        (-.01, .01, .18), (.01, .01, .18),
        (.01, .01, .20), (-.01, .01, .20),
    ]
    triangles = [
        (0, 1, 2), (0, 2, 3),  # front
        (4, 7, 6), (4, 6, 5),  # rear
        (4, 0, 3), (4, 3, 7),  # right view (-X)
        (1, 5, 6), (1, 6, 2),  # left view
        (0, 4, 5), (0, 5, 1),  # underside
        (3, 2, 6), (3, 6, 7),  # top
    ]
    with path.open("w", encoding="utf-8") as out:
        out.write("ply\nformat ascii 1.0\n")
        out.write("element vertex 8\nproperty float x\nproperty float y\nproperty float z\n")
        out.write("element face 12\nproperty list uchar int vertex_indices\nend_header\n")
        for x, y, z in vertices:
            out.write(f"{x+dx:.8f} {y:.8f} {z:.8f}\n")
        for a, b, c in triangles:
            out.write(f"3 {a} {b} {c}\n")


class RegionRevisionGateTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = self.root = Path(tmp.name)
        (root / "masks").mkdir()
        (root / "geometry").mkdir()
        for view in ("front", "right"):
            image = Image.new("L", (1600, 1200))
            ImageDraw.Draw(image).rectangle((771, 529, 829, 585), fill=255)
            image.save(root / "masks" / f"{view}.png")
        self.source_mesh = root / "geometry" / "original.ply"
        _box(self.source_mesh)
        spec = {
            "schema_version": 2, "character_id": "synthetic",
            "height_m": .35, "profile": "generic-biped",
            "reference_frame": {"width": 1600, "height": 1200, "figure_px": 1000},
            "regions": [
                {"id": "head", "type": "head", "source_confidence": "estimated"},
                {
                    "id": "eye.L", "type": "eye", "parent": "head",
                    "source_confidence": "verified",
                    "reference_masks": {
                        "front": "masks/front.png", "right": "masks/right.png"
                    }, "verified_views": ["front", "right"],
                    "source_mesh": "geometry/original.ply",
                    "detail_critical": True,
                },
                {"id": "eye.R", "type": "eye", "parent": "head",
                 "source_confidence": "estimated"},
            ],
        }
        self.spec_path = root / "character.json"
        self.spec_path.write_text(json.dumps(spec), encoding="utf-8")
        self.plan = root / "plan"
        plan_character(self.spec_path, self.plan)
        self.candidate = root / "geometry" / "candidate.ply"

    def test_identical_source_remains_art_review_not_false_artistic_pass(self):
        self.candidate.write_bytes(self.source_mesh.read_bytes())
        result = gate_region_revision(self.plan, "eye.L", self.candidate, self.root / "checked")
        self.assertTrue(result["numeric_gate_pass"])
        self.assertEqual(result["status"], "no-change-review-required")
        self.assertFalse(result["any_geometry_change"])
        self.assertEqual(result["failed_views"], [])
        self.assertEqual({x["view"] for x in result["views"]}, {"front", "right"})
        self.assertEqual(result["worst_iou_delta"], 0)
        for view in ("front", "right"):
            with Image.open(self.root / "checked" / "panels" / f"{view}.png") as image:
                self.assertEqual(image.size, (800, 300))
            self.assertEqual(result["views"][0]["iou_delta"], 0)

    def test_rejects_single_view_regression_even_when_other_view_stable(self):
        _box(self.candidate, dx=.031)
        result = gate_region_revision(
            self.plan, "eye.L", self.candidate, self.root / "rejected",
            max_regression=.01)
        self.assertFalse(result["numeric_gate_pass"])
        self.assertEqual(result["status"], "registered-silhouette-regression-rejected")
        self.assertIn("front", result["failed_views"])
        self.assertNotIn("right", result["failed_views"])
        rows = {r["view"]: r for r in result["views"]}
        self.assertLess(rows["front"]["iou_delta"], -.05)
        self.assertEqual(rows["right"]["iou_delta"], 0)
        self.assertIn("registered_silhouette_regression", rows["front"]["rejection_reasons"])
        self.assertTrue((self.root / "rejected" / "baseline" / "report.json").is_file())
        self.assertTrue((self.root / "rejected" / "candidate" / "report.json").is_file())
        self.assertTrue((self.root / "rejected" / "panels" / "front.png").is_file())

    def test_accepts_revised_candidate_within_allowed_regression(self):
        _box(self.candidate, dx=.001)
        result = gate_region_revision(self.plan, "eye.L", self.candidate, self.root/"lenient",
                                      max_regression=1)
        self.assertTrue(result["numeric_gate_pass"])
        self.assertEqual(result["status"], "no-registered-regression-artistic-review-required")
        self.assertTrue(result["any_geometry_change"])
        self.assertGreaterEqual(result["worst_iou_delta"], -1)

    def test_absolute_iou_floor_is_independent_from_regression(self):
        _box(self.candidate, dx=.031)
        result = gate_region_revision(self.plan, "eye.L", self.candidate,
                                      self.root/"strict", max_regression=1, min_iou=.9)
        self.assertFalse(result["numeric_gate_pass"])
        failed = {r["view"] for r in result["views"]
                  if "below_absolute_iou_floor" in r["rejection_reasons"]}
        self.assertIn("front", failed)

    def test_legacy_obj_rejected_before_output_not_created(self):
        old_style = self.root / "geometry" / "region.obj"
        old_style.write_text("v 0 0 0\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "ambiguous Y-up OBJ"):
            gate_region_revision(self.plan, "eye.L", old_style, self.root/"invalid")
        self.assertFalse((self.root/"invalid").exists())

    def test_missing_source_baseline_requires_explicit_before(self):
        self.candidate.write_bytes(self.source_mesh.read_bytes())
        with self.assertRaisesRegex(ValueError, "no original source_mesh"):
            gate_region_revision(self.plan, "eye.R", self.candidate, self.root/"invalid")
        self.assertFalse((self.root/"invalid").exists())

    def test_unchanged_source_art_is_enforced_before_writing(self):
        self.candidate.write_bytes(self.source_mesh.read_bytes())
        with Image.open(self.root / "masks" / "front.png") as im:
            modified = im.copy()
        modified.putpixel((0, 0), 255)
        modified.save(self.root / "masks" / "front.png")
        with self.assertRaisesRegex(ValueError, "changed"):
            gate_region_revision(self.plan, "eye.L", self.candidate, self.root/"invalid")
        self.assertFalse((self.root/"invalid").exists())

    def test_existing_output_refused_without_mutation(self):
        self.candidate.write_bytes(self.source_mesh.read_bytes())
        output = self.root / "already"
        output.mkdir()
        (output / "keep.txt").write_text("original")
        with self.assertRaises(FileExistsError):
            gate_region_revision(self.plan, "eye.L", self.candidate, output)
        self.assertEqual((output/"keep.txt").read_text(), "original")

    def test_cli_returns_nonzero_for_visual_regression(self):
        _box(self.candidate, dx=.04)
        code = character_cli([
            "audit-region-revision", "--character", str(self.plan), "--region",
            "eye.L", "--after", str(self.candidate), "--max-regression", ".01",
            "--output", str(self.root/"cli-report")])
        self.assertEqual(code, 1)
        self.assertTrue((self.root/"cli-report"/"report.json").is_file())


if __name__ == "__main__":
    unittest.main()
