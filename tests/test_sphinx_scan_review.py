"""Regression coverage for the scan-first, non-distributing review workflow."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kumiki_blender.asset_import import AssetImportError
from kumiki_blender.sphinx_scan_review import (
    prepare_sphinx_review, review_camera_layout,
)


class SphinxScanReviewTests(unittest.TestCase):
    def test_accepts_locally_supplied_obj_fbx_and_glb_without_copying_mesh(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for extension in (".obj", ".fbx", ".glb"):
                source = root / ("source"+extension)
                source.write_bytes(b"source-owned-test")
                destination = root / ("review-"+extension[1:])
                planned = prepare_sphinx_review(source, destination,
                                                front_axis="+Y")
                self.assertEqual(planned.source, source.resolve())
                self.assertEqual(planned.front_axis, "+Y")
                self.assertEqual(planned.output, destination.resolve())
                self.assertFalse(destination.exists())
                self.assertEqual(source.read_bytes(), b"source-owned-test")

    def test_rejects_missing_wrong_format_empty_and_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "previews"
            with self.assertRaises(OSError):
                prepare_sphinx_review(root / "missing.obj", output)
            for name, data in (("archive.zip", b"zip"), ("empty.obj", b"")):
                source = root / name
                source.write_bytes(data)
                with self.assertRaises(AssetImportError):
                    prepare_sphinx_review(source, output)
            real = root / "scan.obj"
            real.write_bytes(b"v 0 0 0\n")
            output.mkdir()
            with self.assertRaisesRegex(AssetImportError, "already exists"):
                prepare_sphinx_review(real, output)
            with self.assertRaisesRegex(AssetImportError, "front axis"):
                prepare_sphinx_review(real, root / "fresh", front_axis="diagonal")

    def test_cameras_target_actual_head_end_with_profile_separate(self):
        layout = review_camera_layout((-10., -1.2, 0.), (0., 1.2, 3.6),
                                      front_axis="-X")
        self.assertEqual(set(layout), {
            "frontal", "face-three-quarter", "side-profile", "whole-frontal",
        })
        self.assertLess(layout["frontal"]["target"][0], -8.)
        self.assertLess(layout["frontal"]["camera"][0],
                        layout["frontal"]["target"][0])
        self.assertNotEqual(layout["side-profile"]["camera"][1],
                            layout["side-profile"]["target"][1])
        self.assertGreater(layout["side-profile"]["ortho_scale"],
                           layout["frontal"]["ortho_scale"])
        self.assertGreater(layout["frontal"]["target"][2], 2.5)
        other = review_camera_layout((-1., -10., 0.), (1., 0., 3.6),
                                     front_axis="+Y")
        self.assertGreater(other["frontal"]["target"][1], -2.)
        self.assertGreater(other["frontal"]["camera"][1], 0.)

    def test_refuses_invalid_bounds(self):
        for bounds in (
            ((0., 0., 0.), (0., 1., 1.)),
            ((0., 0., 0.), (float("nan"), 1., 1.)),
            ((0., 0., 0.), (1., 1., float("inf"))),
        ):
            with self.assertRaises(AssetImportError):
                review_camera_layout(*bounds, front_axis="-X")


if __name__ == "__main__":
    unittest.main()
