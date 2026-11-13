"""Fixed-support original-art color QA and source-frame UV OBJ regression tests."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from kumiki_blender.character_build.mesh import Mesh
from kumiki_blender.character_build.texture import read_obj_uv, render_textured
from kumiki_blender.character_build.texture_revision_gate import (
    _scores, audit_texture_revision,
)
from kumiki_blender.character_build.__main__ import main as cli


def write_uv_obj(path: Path, *, source_frame: bool):
    """Two orthogonal, separate surfaces with distinct UV material islands."""
    verts = [
        (-.04, -.03, .14), (.04, -.03, .14),
        (.04, -.03, .22), (-.04, -.03, .22),
        (.03, -.04, .14), (.03, .04, .14),
        (.03, .04, .22), (.03, -.04, .22),
    ]
    uv = [(0, 0), (.48, 0), (.48, 1), (0, 1),
          (.52, 0), (1, 0), (1, 1), (.52, 1)]
    def xyz(v):
        x, y, z = v
        return (x, y, z) if source_frame else (x, z, -y)
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in map(xyz, verts)]
    lines += [f"vt {u} {v}" for u, v in uv]
    lines += [
        "f 1/1 2/2 3/3", "f 1/1 3/3 4/4",
        "f 5/5 6/6 7/7", "f 5/5 7/7 8/8",
    ]
    path.write_text("\n".join(lines) + "\n")


class TextureRevisionTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.handoff = self.root / "handoff"
        (self.handoff / "aligned").mkdir(parents=True)
        self.model = self.root / "rig-ready.source.obj"
        self.legacy_model = self.root / "rig-ready.obj"
        write_uv_obj(self.model, source_frame=True)
        write_uv_obj(self.legacy_model, source_frame=False)
        self.old = self.root / "old.png"
        self.new = self.root / "new.png"
        initial = np.full((128, 128, 3), (120, 80, 40), np.uint8)
        initial[:, 67:, :] = (40, 140, 160)
        Image.fromarray(initial).save(self.old)
        Image.fromarray(initial).save(self.new)
        self.initial = initial
        self.handoff_spec = {
            "views": ["front", "right"],
            "metres_per_aligned_px": .35 / 1000,
        }
        (self.handoff / "spec.json").write_text(json.dumps(self.handoff_spec))
        mesh, uv = read_obj_uv(self.model)
        for view in self.handoff_spec["views"]:
            frame, visible = render_textured(mesh, uv, initial, view,
                                             self.handoff_spec["metres_per_aligned_px"], .25)
            rgba = np.empty(visible.shape + (4,), dtype=np.uint8)
            rgba[..., :3] = np.clip(frame, 0, 255).astype(np.uint8)
            rgba[..., 3] = np.where(visible, 255, 0).astype(np.uint8)
            Image.fromarray(rgba).resize((1600, 1200), Image.Resampling.NEAREST).save(
                self.handoff / "aligned" / f"{view}.png")

    def audit(self, folder="report", **kwargs):
        return audit_texture_revision(
            self.handoff, self.model, self.old, self.new,
            self.root / folder, **kwargs)

    def test_source_frame_uv_reader_preserves_xyz_and_uvs(self):
        one, uv1 = read_obj_uv(self.model)
        two, uv2 = read_obj_uv(self.legacy_model)
        np.testing.assert_array_equal(one.vertices, two.vertices)
        np.testing.assert_array_equal(uv1, uv2)
        np.testing.assert_allclose(one.vertices[0], [-.04, -.03, .14], atol=1e-12)

    def test_identical_revision_is_not_claimed_as_artistic_improvement(self):
        result = self.audit()
        self.assertTrue(result["numeric_gate_pass"])
        self.assertEqual(result["status"], "no-change-artistic-review-required")
        self.assertTrue(result["mesh_and_uv_fixed"])
        self.assertEqual(result["failed_views"], [])
        self.assertEqual(len(result["views"]), 2)
        for row in result["views"]:
            self.assertGreater(row["comparable_source_pixels"], 100)
            self.assertEqual(row["mean_delta_e_increase"], 0)
            self.assertTrue((self.root / "report" / row["panel"]).is_file())

    def test_wrong_color_on_one_uv_island_is_rejected_without_hiding_errors(self):
        changed = self.initial.copy()
        changed[:, :62, :] = (0, 255, 0)
        Image.fromarray(changed).save(self.new)
        result = self.audit()
        self.assertFalse(result["numeric_gate_pass"])
        self.assertIn("front", result["failed_views"])
        frontal = next(r for r in result["views"] if r["view"] == "front")
        self.assertGreater(frontal["mean_delta_e_increase"], 5)
        self.assertIn("mean_delta_e_regression", frontal["rejection_reasons"])
        self.assertTrue((self.root / "report" / "front.png").is_file())
        self.assertTrue((self.root / "report" / "report.json").is_file())
        with Image.open(self.root / "report" / "front.png") as image:
            self.assertEqual(image.size, (1600, 300))

    def test_candidate_that_corrects_broken_texture_passes(self):
        bad = np.full_like(self.initial, (255, 30, 230))
        Image.fromarray(bad).save(self.old)
        result = self.audit("improved")
        self.assertTrue(result["numeric_gate_pass"])
        self.assertEqual(result["status"],
                         "no-registered-texture-regression-artistic-review-required")
        self.assertLess(next(r for r in result["views"]
                            if r["view"] == "front")["mean_delta_e_increase"], -5)

    def test_modified_reference_fails_with_no_partial_final_output(self):
        with Image.open(self.handoff/"aligned"/"front.png") as im:
            changed = im.copy()
        changed.putpixel((0, 0), (0, 0, 0, 255))
        changed.save(self.handoff/"aligned"/"front.png")
        # This is a *new* immutable reference revision, so it is evaluated
        # normally; missing sources, by contrast, must fail closed.
        (self.handoff/"aligned"/"right.png").unlink()
        with self.assertRaisesRegex(ValueError, "selected source-camera"):
            self.audit()
        self.assertFalse((self.root / "report").exists())

    def test_unverified_view_and_bad_thresholds_fail_before_writing(self):
        for kwargs in (
            {"views": ["front", "fabricated"]},
            {"views": ["front", "front"]},
            {"max_mean_increase": float("nan")},
            {"max_p90_increase": -1},
            {"max_within_10_drop": float("inf")},
            {"min_pixels_per_view": 100000},
            {"scale": 0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    self.audit(**kwargs)
                self.assertFalse((self.root/"report").exists())

    def test_existing_report_is_immutable(self):
        folder = self.root/"report"
        folder.mkdir()
        (folder/"reviewed.txt").write_text("approved")
        with self.assertRaises(FileExistsError):
            self.audit()
        self.assertEqual((folder/"reviewed.txt").read_text(), "approved")

    def test_cli_returns_one_with_candidate_evidence(self):
        changed = np.full_like(self.initial, (255, 0, 255))
        Image.fromarray(changed).save(self.new)
        code = cli([
            "audit-texture-revision", str(self.handoff), str(self.model),
            str(self.old), str(self.new), "--output", str(self.root/"cli")])
        self.assertEqual(code, 1)
        self.assertTrue((self.root/"cli"/"report.json").is_file())


class PureMetricTests(unittest.TestCase):
    def test_fixed_pixel_support_and_metric_direction(self):
        target = np.full((6, 5, 3), 120, dtype=np.uint8)
        mask = np.zeros((6, 5), dtype=bool)
        mask[1:5, 1:4] = True
        unchanged, delta = _scores(target, target, mask)
        self.assertEqual(unchanged["mean_delta_e"], 0)
        self.assertEqual(unchanged["p90_delta_e"], 0)
        self.assertEqual(unchanged["within_10"], 1)
        modified = target.copy()
        modified[mask] = 255
        degraded, _ = _scores(target, modified, mask)
        self.assertGreater(degraded["mean_delta_e"], unchanged["mean_delta_e"])
        self.assertLess(degraded["within_10"], unchanged["within_10"])
        self.assertEqual(len(delta), int(mask.sum()))
        with self.assertRaisesRegex(ValueError, "same registered frame"):
            _scores(target, modified, mask[:4])


if __name__ == "__main__":
    unittest.main()
