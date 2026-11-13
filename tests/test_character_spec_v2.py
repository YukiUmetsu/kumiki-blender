"""Local, license-free regression fixtures for CharacterSpec v2 and region QA."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from kumiki_blender.character_build.__main__ import main as cli
from kumiki_blender.character_build.character_spec_v2 import (
    affected_regions, load_spec, open_plan, plan_character, plan_region,
)
from kumiki_blender.character_build.region_compare import compare_region


class CharacterPlanV2Test(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "masks").mkdir()
        (self.root / "mesh").mkdir()
        image = Image.new("L", (1600, 1200))
        ImageDraw.Draw(image).rectangle((771, 528, 829, 586), fill=255)
        image.save(self.root / "masks" / "front.png")
        # A 2D square in the canonical Blender Z-up frame written as PLY.
        (self.root / "mesh" / "eye.ply").write_text(
            "ply\nformat ascii 1.0\n"
            "element vertex 4\nproperty float x\nproperty float y\nproperty float z\n"
            "element face 2\nproperty list uchar int vertex_indices\nend_header\n"
            "-0.01 0 0.18\n0.01 0 0.18\n0.01 0 0.20\n-0.01 0 0.20\n"
            "3 0 1 2\n3 0 2 3\n", encoding="utf-8")
        self.spec = {
            "schema_version": 2,
            "character_id": "sample",
            "height_m": .35,
            "profile": "stylized-biped",
            "reference_frame": {"width": 1600, "height": 1200, "figure_px": 1000},
            "regions": [
                {"id": "head", "type": "head", "source_confidence": "estimated"},
                {"id": "eye.L", "parent": "head", "type": "eye",
                 "source_confidence": "verified", "reference_masks": {"front": "masks/front.png"},
                 "verified_views": ["front"], "source_mesh": "mesh/eye.ply",
                 "attachment": {"mode": "bone-parent", "bone": "head"},
                 "geometry_policy": "separate", "detail_critical": True,
                 "landmarks": {"center": [0, 0, 0.19]}},
                {"id": "eye.R", "parent": "head", "type": "eye", "source_confidence": "estimated"},
                {"id": "lid.L", "parent": "eye.L", "type": "eyelid", "source_confidence": "estimated"},
            ],
        }
        self.spec_path = self.root / "character.json"
        self.save()

    def save(self):
        self.spec_path.write_text(json.dumps(self.spec), encoding="utf-8")

    def test_validate_and_plan_idempotent_source_immutable_output(self):
        self.assertEqual(len(load_spec(self.spec_path)["regions"]), 4)
        plan = plan_character(self.spec_path, self.root / "plan")
        self.assertEqual(plan["region_order"], ["head", "eye.L", "eye.R", "lid.L"])
        self.assertEqual(plan["regions"]["eye.L"]["dependencies"], ["head"])
        with self.assertRaises(FileExistsError):
            plan_character(self.spec_path, self.root / "plan")
        self.assertEqual(open_plan(self.root / "plan")[0]["character_id"], "sample")

    def test_region_isolation_and_affected_dependencies(self):
        plan_character(self.spec_path, self.root / "plan")
        task = plan_region(self.root / "plan", "eye.L", self.root / "eye-task")
        self.assertEqual(task["invalidate_on_change"], ["eye.L", "lid.L"])
        self.assertEqual(task["preserve_unchanged"], ["head", "eye.R"])
        self.assertTrue((self.root / "eye-task" / "region-task.json").is_file())
        with self.assertRaises(FileExistsError):
            plan_region(self.root / "plan", "eye.L", self.root / "eye-task")
        self.assertEqual(affected_regions(self.spec, "head"), ["head", "eye.L", "eye.R", "lid.L"])

    def test_source_hash_detects_modification(self):
        plan_character(self.spec_path, self.root / "plan")
        with Image.open(self.root / "masks" / "front.png") as source:
            image = source.copy()
        image.putpixel((0, 0), 255)
        image.save(self.root / "masks" / "front.png")
        with self.assertRaisesRegex(ValueError, "changed"):
            plan_region(self.root / "plan", "eye.L", self.root / "task")
        self.assertFalse((self.root / "task").exists())

    def test_invalid_paths_symlinks_and_cycles(self):
        self.spec["regions"][1]["reference_masks"]["front"] = "../private.png"
        self.save()
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            load_spec(self.spec_path)
        self.spec["regions"][1]["reference_masks"]["front"] = "masks/front.png"
        self.spec["regions"][0]["parent"] = "lid.L"
        self.save()
        with self.assertRaisesRegex(ValueError, "Cycle"):
            load_spec(self.spec_path)
        del self.spec["regions"][0]["parent"]
        self.save()
        (self.root / "masks" / "linked.png").symlink_to(self.root / "masks" / "front.png")
        self.spec["regions"][1]["reference_masks"]["front"] = "masks/linked.png"
        self.save()
        with self.assertRaisesRegex(ValueError, "Linked"):
            load_spec(self.spec_path)

    def test_reject_inconsistent_verified_evidence_and_symmetry(self):
        self.spec["regions"][1]["source_confidence"] = "estimated"
        self.save()
        with self.assertRaisesRegex(ValueError, "verified"):
            load_spec(self.spec_path)
        self.spec["regions"][1]["source_confidence"] = "verified"
        self.spec["regions"][1]["symmetry"] = {"partner": "eye.R", "verified": True}
        self.save()
        with self.assertRaisesRegex(ValueError, "reciprocal"):
            load_spec(self.spec_path)

    def test_compare_registered_region_without_recentring(self):
        plan_character(self.spec_path, self.root / "plan")
        first = compare_region(self.root / "plan", "eye.L", self.root / "mesh" / "eye.ply",
                               self.root / "comparison", min_iou=.25)
        self.assertEqual(first["status"], "artistic-review-required")
        self.assertGreater(first["views"][0]["iou"], .25)
        self.assertTrue((self.root / "comparison" / "overlays" / "front.png").is_file())
        second = compare_region(self.root / "plan", "eye.L", self.root / "mesh" / "eye.ply",
                                self.root / "comparison2",
                                baseline=self.root / "comparison" / "report.json", min_iou=.25)
        self.assertEqual(second["views"][0]["iou_delta"], 0)
        # An offset eye must not be automatically translated to appear aligned.
        mesh = (self.root / "mesh" / "eye.ply").read_text()
        moved = mesh.replace("-0.01 0 0.18", "0.19 0 0.18").replace("0.01 0 0.18", "0.21 0 0.18")
        moved = moved.replace("0.01 0 0.20", "0.21 0 0.20").replace("-0.01 0 0.20", "0.19 0 0.20")
        (self.root / "mesh" / "moved.ply").write_text(moved)
        third = compare_region(self.root / "plan", "eye.L", self.root / "mesh" / "moved.ply",
                               self.root / "comparison3", baseline=self.root / "comparison" / "report.json")
        self.assertEqual(third["status"], "geometric-review-failed")
        self.assertIn("front", third["failed_views"])

    def test_native_source_obj_and_legacy_interchange_project_identically(self):
        """The Blender source OBJ must not be silently rotated during QA."""
        plan_character(self.spec_path, self.root / "plan")
        ply = self.root / "mesh" / "eye.ply"
        expected = compare_region(self.root / "plan", "eye.L", ply,
                                  self.root / "from-ply", min_iou=.25)
        # Coordinates are already in Blender Z-up, character facing -Y.
        sources = [
            (-.01, 0, .18), (.01, 0, .18),
            (.01, 0, .20), (-.01, 0, .20),
        ]
        native = self.root / "mesh" / "eye.L.source.obj"
        legacy = self.root / "mesh" / "eye.L.obj"
        for path, rows in (
            (native, sources),
            # Existing OBJ uses Y-up interchange with the inverse transform.
            (legacy, [(x, z, -y) for x, y, z in sources]),
        ):
            path.write_text(
                "\n".join("v " + " ".join(str(v) for v in vertex)
                          for vertex in rows)
                + "\nf 1 2 3\nf 1 3 4\n",
                encoding="utf-8")
        a = compare_region(self.root / "plan", "eye.L", native,
                           self.root / "from-native-source", min_iou=.25)
        b = compare_region(self.root / "plan", "eye.L", legacy,
                           self.root / "from-legacy", min_iou=.25)
        self.assertEqual(a["views"][0]["iou"], expected["views"][0]["iou"])
        self.assertEqual(b["views"][0]["iou"], expected["views"][0]["iou"])
        self.assertGreater(a["views"][0]["boundary_f1_2px"], 0)
        self.assertEqual(a["views"][0]["boundary_f1_2px"],
                         b["views"][0]["boundary_f1_2px"])

    def test_cli_and_unverified_source_rejection(self):
        self.assertEqual(cli(["plan-character", "--spec", str(self.spec_path), "--output",
                              str(self.root / "cli-plan")]), 0)
        self.assertEqual(cli(["plan-region", "--character", str(self.root / "cli-plan"),
                              "--region", "eye.L", "--output", str(self.root / "cli-region")]), 0)
        self.assertEqual(cli(["compare-region", "--character", str(self.root / "cli-plan"),
                              "--region", "eye.L", "--mesh", str(self.root / "mesh" / "eye.ply"),
                              "--min-iou", ".25", "--output", str(self.root / "cli-compare")]), 0)
        with self.assertRaisesRegex(ValueError, "no verified"):
            compare_region(self.root / "cli-plan", "head", self.root / "mesh" / "eye.ply",
                           self.root / "invalid")


if __name__ == "__main__":
    unittest.main()
