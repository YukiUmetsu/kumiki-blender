"""Source-locked multiview visual-review plan and contact-sheet contracts."""
from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from kumiki_blender.character_build.__main__ import main
from kumiki_blender.character_build.character_review_plan import (
    confirm_review_sources, digest, make_character_review_plan,
    validate_character_review_plan,
)
from kumiki_blender.character_build.character_review_sheet import (
    make_character_review_sheets,
)


class ReviewPlanTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / "pip.blend"
        self.source.write_bytes(b"saved rig fixture")
        self.front = self.root / "approved-front.png"
        self.right = self.root / "approved-right.png"
        Image.new("RGBA", (96, 128), (255, 255, 255, 255)).save(self.front)
        Image.new("RGBA", (96, 128), (240, 245, 250, 255)).save(self.right)
        self.manifest = self.root / "references.json"
        self.info = {
            "schema_version": 1,
            "views": {"front": str(self.front)},
            "regions": [{
                "id": "tail", "object": "Tail", "vertex_group": "REGION_Tail",
                "views": {"right": str(self.right)},
            }],
        }
        self.manifest.write_text(json.dumps(self.info), encoding="utf-8")

    def plan(self, **kwargs):
        return make_character_review_plan(
            self.source, frames=[1, 12], views=["front", "right"],
            references=self.manifest, **kwargs)

    def test_source_and_reference_hashes_are_locked(self):
        plan = self.plan()
        self.assertEqual(plan["source_sha256"], digest(self.source))
        self.assertEqual(plan["reference_views"]["front"]["sha256"], digest(self.front))
        self.assertEqual(plan["regions"][0]["references"]["right"]["sha256"], digest(self.right))
        self.assertEqual(plan["reference_alignment"], "unregistered-visual-review-only")
        validate_character_review_plan(plan)
        confirm_review_sources(plan)
        self.right.write_bytes(b"edited reference image")
        with self.assertRaisesRegex(ValueError, "changed"):
            confirm_review_sources(plan)

    def test_requires_approved_front_and_safe_exact_region_names(self):
        cases = [
            {"views": {"right": str(self.right)}},
            {"regions": [{"id": "bad path", "object": "Skin", "vertex_group": None, "views": {}}]},
            {"regions": [{"id": "tail", "object": "../Tail", "vertex_group": None, "views": {}}]},
            {"regions": [{"id": "tail", "object": "Tail", "vertex_group": "../../Escape", "views": {}}]},
            {"regions": [
                {"id": "tail", "object": "Tail", "vertex_group": None, "views": {}},
                {"id": "tail", "object": "Tail", "vertex_group": None, "views": {}},
            ]},
            {"unknown_option": True},
        ]
        for change in cases:
            with self.subTest(change=change):
                self.manifest.write_text(json.dumps({**self.info, **change}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.plan()

    def test_frame_view_and_render_limits(self):
        for kwargs in (
            {"frames": []}, {"frames": [5, 1]}, {"frames": [1, 1]},
            {"frames": [1, 2, 3, 4, 5, 6]}, {"frames": [True]},
            {"views": []}, {"views": ["front", "front"]},
            {"views": ["north"]}, {"resolution": 128}, {"resolution": 2049},
            {"margin": float("nan")}, {"margin": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_character_review_plan(
                    self.source, frames=kwargs.get("frames", [1]),
                    views=kwargs.get("views", ["front"]),
                    resolution=kwargs.get("resolution", 720),
                    margin=kwargs.get("margin", 1.18))
        self.info["regions"] = [
            {"id": f"region{i}", "object": "Skin", "vertex_group": None, "views": {}}
            for i in range(8)
        ]
        self.manifest.write_text(json.dumps(self.info), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "48"):
            make_character_review_plan(
                self.source, frames=[1, 6], views=[
                    "front", "left", "right", "back", "front_left", "front_right"
                ], references=self.manifest)

    def test_sheet_includes_original_art_only_as_unregistered_reference(self):
        plan = self.plan()
        output = self.root / "review-r01"
        (output / "images").mkdir(parents=True)
        entries = []
        for scope in ("full", "tail"):
            for view in ("front", "right"):
                for frame in (1, 12):
                    filename = f"images/{scope}-{view}-f{frame:05d}.png"
                    Image.new("RGB", (128, 128), (180, 180, 190)).save(output / filename)
                    ref = (
                        plan["reference_views"].get(view)
                        if scope == "full" else plan["regions"][0]["references"].get(view))
                    entries.append({
                        "scope": scope, "view": view, "frame": frame,
                        "render": filename, "reference": ref, "camera": {},
                    })
        (output / "report.json").write_text(json.dumps({
            "status": "source-visual-review-required",
            "source_sha256": plan["source_sha256"],
            "images": entries, "limitations": [],
        }), encoding="utf-8")
        report = make_character_review_sheets(output, plan)
        self.assertEqual(len(report["contact_sheets"]), 4)
        for filename in report["contact_sheets"]:
            with Image.open(output / filename) as im:
                self.assertEqual(im.width, 844)
                self.assertGreater(im.height, 300)
        self.assertEqual(report["status"], "source-visual-review-required")
        with self.assertRaises(FileExistsError):
            make_character_review_sheets(output, plan)
        # Source illustrations and character source bytes remain untouched.
        self.assertEqual(digest(self.front), plan["reference_views"]["front"]["sha256"])
        self.assertEqual(digest(self.source), plan["source_sha256"])

    def test_cli_routes_the_exact_locked_plan_and_source(self):
        with patch("kumiki_blender.character_build.blender.run_worker") as worker, patch(
            "kumiki_blender.character_build.character_review_sheet.make_character_review_sheets"
        ) as compose:
            worker.return_value = {"status": "source-visual-review-required"}
            compose.return_value = {"status": "source-visual-review-required"}
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main([
                    "render-character-review", "--blend", str(self.source),
                    "--references", str(self.manifest),
                    "--frames", "1,12", "--views", "front,right",
                    "--output", str(self.root / "native-review"),
                ]), 0)
            self.assertEqual(worker.call_args.args[0], "character-review")
            self.assertEqual(worker.call_args.kwargs["blend_file"], self.source.resolve())
            self.assertEqual(worker.call_args.args[1]["plan"]["reference_views"]["front"]["sha256"],
                             digest(self.front))
            self.assertEqual(compose.call_count, 1)


if __name__ == "__main__":
    unittest.main()
