"""Provenance and deterministic inspection contracts for the scan-assisted face."""

import hashlib
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from kumiki_blender.cli import main
from kumiki_blender.blender_liberty_scan import plan_liberty_scan_face_shots
from kumiki_blender.liberty_face_sculpt import (face_relief_offset,
                                                hair_relief_offset,
                                                sculpt_relief_offset)
from kumiki_blender.liberty_scan import (ScanSourceError, load_scan_source,
                                        verify_scan_source)
from kumiki_blender.procedural_pbr import make_pbr_maps


class LibertyScanTests(unittest.TestCase):
    def test_candidate_is_item_level_attributed_and_hash_locked(self):
        source = load_scan_source()
        self.assertEqual(source["source_id"], "jerryfisher-liberty-bronze-photogrammetry")
        self.assertEqual(source["license"], "CC-BY-4.0")
        self.assertEqual(source["author"], "jerryfisher")
        self.assertIn("sketchfab.com/3d-models/", source["original_url"])
        self.assertIn("creativecommons.org/licenses/by/4.0/", source["license_url"])
        self.assertEqual(len(source["sha256"]), 64)
        self.assertGreater(source["size_bytes"], 100000)

    def test_source_must_match_recorded_binary_not_merely_have_glb_extension(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "scan.glb"
            target.write_bytes(b"glTF" + bytes(48))
            metadata = {**load_scan_source(), "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                        "size_bytes": target.stat().st_size}
            self.assertEqual(verify_scan_source(target, metadata), metadata)
            target.write_bytes(b"glTF" + bytes(47) + b"!")
            with self.assertRaisesRegex(ScanSourceError, "SHA-256"):
                verify_scan_source(target, metadata)

    def test_cli_rejects_missing_scan_without_creating_revision(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "revision"
            stream = io.StringIO()
            with redirect_stderr(stream):
                status = main(["build-liberty-scan-face", "--source", str(Path(temp) / "missing.glb"),
                               "--output", str(destination)])
            self.assertEqual(status, 2)
            self.assertIn("scan", stream.getvalue().lower())
            self.assertFalse(destination.exists())

    def test_scan_sculpt_relieves_eye_and_lips_without_touching_crown_or_back(self):
        # The same eyelid has a genuinely recessed crease AND a protruding
        # supporting ledge: subdivision can resolve the surface contrast.
        self.assertGreater(face_relief_offset(-.53, -.5, 8.788),
                           face_relief_offset(-.53, -.5, 8.811) + .01)
        self.assertLess(face_relief_offset(-.53, -.5, 8.811), -.015)
        self.assertLess(face_relief_offset(-.53, -.5, 8.861), 0)
        self.assertLess(face_relief_offset(-.353, -.5, 8.405), -.015)
        self.assertGreater(face_relief_offset(-.353, -.5, 8.244), 0)
        self.assertLess(face_relief_offset(-.353, -.5, 8.203), 0)
        for point in ((-.53, .2, 8.755), (-.53, -.5, 9.25),
                      (-1.45, -.5, 8.755), (-.53, -.5, 7.4)):
            self.assertEqual(face_relief_offset(*point), 0.0)

    def test_hair_is_sculpted_with_localized_strands_not_free_tubes(self):
        self.assertNotEqual(hair_relief_offset(-.5, -.5, 9.15), 0)
        self.assertEqual(hair_relief_offset(-.5, -.5, 8.7), 0)
        self.assertEqual(hair_relief_offset(-.5, .5, 9.15), 0)
        self.assertEqual(hair_relief_offset(-.5, -.5, 9.6), 0)
        self.assertNotEqual(sculpt_relief_offset(-.5, -.5, 9.15), 0)

    def test_custom_patina_has_packed_color_roughness_and_normal(self):
        maps = make_pbr_maps("verdigris-copper", "#658F82", .79,
                             resolution=64, weathering_strength=.72)
        self.assertEqual(maps.resolution, 64)
        for image in (maps.base_color_png, maps.roughness_png, maps.normal_png):
            self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertGreater(len(image), 128)
        self.assertNotEqual(maps.base_color_png, maps.normal_png)

    def test_face_review_has_tight_multiviews_and_full_reference(self):
        shots = plan_liberty_scan_face_shots(12.43596)
        self.assertEqual(shots[0].name, "hero")
        self.assertEqual(len(shots), 11)
        self.assertEqual(len({shot.name for shot in shots}), len(shots))
        self.assertTrue({"face-front", "eyes-macro", "nose-mouth-macro",
                         "face-left-profile", "face-right-profile",
                         "face-low-angle", "hair-rear", "crown-top"} <=
                        {shot.name for shot in shots})
        face = next(s for s in shots if s.name == "face-front")
        self.assertLess(face.ortho_scale, 2.2)
        self.assertLess(face.position[1], 0)


if __name__ == "__main__":
    unittest.main()
