"""character_build parts.py and polish.py: recognise parts by shape and colour, bake by part, close the seams."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from character_build_fixtures import BAG, HEIGHT_M, SKIN, ellipsoid_mesh, write_views
from kumiki_blender.character_build import parts, polish
from kumiki_blender.character_build.handoff import build_handoff
from kumiki_blender.character_build.mesh import Mesh, read_mesh
from kumiki_blender.character_build.texture import texel_samples


def plane(width: int, height: int, spacing: float = 0.001):
    """Texels of a flat sheet that faces the front camera: points, normals, column and row of each."""
    col, row = np.meshgrid(np.arange(width), np.arange(height))
    points = np.column_stack([col.ravel() * spacing, np.zeros(width * height), row.ravel() * spacing])
    return points, np.tile([0.0, -1.0, 0.0], (width * height, 1)), col.ravel(), row.ravel()


class ShapeTests(unittest.TestCase):
    def test_a_valley_separates_two_shapes(self):
        """Two balls joined by a narrow neck: the crease between them splits the mesh in two."""
        ball = ellipsoid_mesh(centre=(0, 0, 0), radii=(.2, .2, .5), subdivisions=5)
        waist = 1 - 0.6 * np.exp(-(ball.vertices[:, 2] / 0.08) ** 2)
        mesh = Mesh(ball.vertices * np.column_stack([waist, waist, np.ones(len(waist))]), ball.faces)
        shape, concavity = parts.shape_parts(mesh, crease_radius=0.3, smooth_rounds=4)
        top, bottom = mesh.vertices[:, 2] > 0.15, mesh.vertices[:, 2] < -0.15
        self.assertEqual(shape.max() + 1, 2)
        self.assertEqual(len(np.unique(shape[top])), 1)
        self.assertEqual(len(np.unique(shape[bottom])), 1)
        self.assertNotEqual(shape[top][0], shape[bottom][0])
        self.assertGreater(concavity[np.abs(mesh.vertices[:, 2]) < 0.02].mean(), 3.0, "the neck bends inward")
        self.assertLess(concavity[top].mean(), 0, "a ball is convex")
        whole, _ = parts.shape_parts(ball, crease_radius=0.3, smooth_rounds=4)
        self.assertEqual(whole.max() + 1, 1, "a convex mesh is one shape")


class RecogniseTests(unittest.TestCase):
    def sheet(self, gradient: float = 40.0):
        """Green for 108 columns, then skin; a soft gradient inside the skin; fine noise as fur has."""
        points, normals, col, _ = plane(200, 60)
        rng = np.random.default_rng(0)
        colour = np.where((col < 108)[:, None], (70, 120, 60), (175, 125, 75)).astype(float)
        colour[col >= 108] += ((col[col >= 108] - 108) / 92)[:, None] * gradient   # no edge inside the skin
        colour = np.clip(colour + rng.normal(0, 4, colour.shape), 0, 255).astype(np.uint8)
        return points, normals, col, colour

    def test_a_sharp_border_makes_two_parts_and_a_gradient_does_not(self):
        points, normals, col, colour = self.sheet()
        label, found, notes = parts.recognise_parts(points, normals, colour, np.ones(len(points), bool), 0.001, classes=4)
        self.assertEqual(len(found), 2, found)
        self.assertEqual(notes, [])
        self.assertEqual(len(np.unique(label[col < 100])), 1)
        self.assertEqual(len(np.unique(label[col > 116])), 1, "the gradient does not split the skin")
        self.assertNotEqual(label[col < 100][0], label[col > 116][0])
        self.assertNotEqual(found[0]["material"], found[1]["material"])
        green = found[int(label[col < 100][0])]
        self.assertLess(np.abs(np.array(green["rgb"]) - (70, 120, 60)).max(), 8)
        border = [np.nonzero(np.diff(label.reshape(60, 200)[r]))[0] for r in range(10, 50)]
        self.assertTrue(all(len(b) == 1 and abs(int(b[0]) - 107) <= 2 for b in border), "the border follows the paint")

    def test_unpainted_texels_take_the_part_beside_them(self):
        points, normals, col, colour = self.sheet(gradient=0.0)
        seen = ~((col > 140) & (col < 170))
        colour[~seen] = (255, 0, 255)          # a fill colour that must not become a part
        label, found, _ = parts.recognise_parts(points, normals, colour, seen, 0.001, classes=4)
        # The band cuts the skin in two; the texels in it join one of the two sides, and the
        # fill colour is no part.
        self.assertTrue(set(np.unique(label[~seen])) <= {label[col == 125][0], label[col == 190][0]})
        self.assertTrue(all(np.abs(np.array(part["rgb"]) - (255, 0, 255)).sum() > 200 for part in found), found)
        self.assertEqual(len({part["material"] for part in found}), 2, "green and skin")

    def test_a_material_on_the_edge_of_another_shape_is_bleed(self):
        """The shapes meet at column 100; the green reaches column 108. Geometry wins."""
        points, normals, col, colour = self.sheet()
        shape = (col >= 100).astype(np.int64)
        seen = np.ones(len(points), bool)
        plain, _, _ = parts.recognise_parts(points, normals, colour, seen, 0.001, classes=4, shape=shape, bleed_share=0)
        label, found, notes = parts.recognise_parts(points, normals, colour, seen, 0.001, classes=4, shape=shape,
                                                    bleed_share=0.1)
        sliver = (col >= 101) & (col < 107)
        self.assertEqual(plain[sliver][0], plain[col == 50][0], "without the rule the sliver is green")
        self.assertTrue((label[sliver] == label[col == 150][0]).all(), "with it, the sliver belongs to the skin")
        self.assertTrue((label[col < 99] == label[col == 50][0]).all(), "the green keeps its own shape")
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["shape"], 1)
        self.assertLess(notes[0]["share_of_shape"], 0.1)
        self.assertLess(np.abs(np.array(notes[0]["material_rgb"]) - (70, 120, 60)).max(), 10)
        # A material that lies mostly on this shape is not bleed, however small it is.
        small = colour.copy()
        small[(col > 150) & (col < 158)] = (40, 40, 160)
        _, found, notes = parts.recognise_parts(points, normals, small, seen, 0.001, classes=5, shape=shape,
                                                bleed_share=0.1, min_share=0.001)
        self.assertEqual(len(found), 3)
        self.assertEqual(len(notes), 1)

    def test_materials_and_label_smoothing(self):
        strap_cuts_vest = [{"rgb": (90, 70, 45)}, {"rgb": (200, 160, 130)}, {"rgb": (88, 73, 48)}, {"rgb": (130, 48, 54)}]
        self.assertEqual(parts.group_materials(strap_cuts_vest), [0, 1, 0, 2])
        labels = np.ones((20, 20), np.uint8)
        labels[:, 10:] = 2
        labels[5, 4] = 2        # a speck
        labels[0, 0] = 0        # no part: stays
        smooth = parts.smooth_labels(labels)
        self.assertEqual(smooth[5, 4], 1)
        self.assertEqual(smooth[0, 0], 0)
        self.assertTrue((smooth[:, 12:] == 2).all() and (smooth[1:, 1:8] == 1).all())


class PolishTests(unittest.TestCase):
    def two_sources(self, size=48, step=30.0):
        rng = np.random.default_rng(1)
        image = np.full((size, size, 3), 100.0) + rng.normal(0, 2, (size, size, 3))
        image[:, size // 2:] += step
        owner = np.zeros((size, size), np.int64)
        owner[:, size // 2:] = 1
        return image, owner, np.zeros((size, size), np.int64), np.ones((size, size), bool)

    def test_a_seam_inside_a_part_is_closed_and_the_rest_is_kept(self):
        image, owner, part, covered = self.two_sources()
        before = polish.seam_contrast(image, owner, part, covered)
        result, report = polish.polish_seams(image, owner, part, covered, reach=8)
        after = polish.seam_contrast(result, owner, part, covered)
        self.assertGreater(before["ratio"], 3)
        self.assertLess(after["ratio"], 1.5)
        self.assertGreater(report["polished_texels"], 0)
        np.testing.assert_array_equal(result[:, :12], image[:, :12])       # beyond the reach: untouched
        np.testing.assert_array_equal(result[:, -12:], image[:, -12:])
        grain = np.abs(np.diff(result[:, 18:22], axis=0)).mean()
        self.assertGreater(grain, 0.7 * np.abs(np.diff(image[:, 18:22], axis=0)).mean(), "the grain stays")

    def test_the_border_of_two_parts_is_not_blended(self):
        image, owner, _, covered = self.two_sources()
        result, report = polish.polish_seams(image, owner, owner.copy(), covered, reach=8)
        np.testing.assert_array_equal(result, image)
        self.assertEqual(report["polished_texels"], 0)
        self.assertEqual(polish.seam_contrast(image, owner, owner.copy(), covered)["seam_pairs"], 0)

    def test_unseen_texels_are_filled_with_a_smooth_tone(self):
        image, owner, part, covered = self.two_sources(step=60.0)
        owner[:, 16:32] = -1
        image[:, 16:32] = image[:, 15:16]          # the first fill: the nearest painted texel
        result, report = polish.smooth_fill(image, owner, part, covered, detail=0.0)
        self.assertEqual(report["filled_texels"], 48 * 16)
        profile = result[:, :, 0].mean(axis=0)
        self.assertTrue((np.diff(profile[15:33]) > 0).all(), "the tone rises steadily across the gap")
        self.assertAlmostEqual(profile[24], 130, delta=6)
        np.testing.assert_array_equal(result[:, :16], image[:, :16])
        np.testing.assert_array_equal(result[:, 32:], image[:, 32:])
        other_part = part.copy()
        other_part[:, 24:] = 1                      # the gap spans two parts: each half keeps to its own side
        split, _ = polish.smooth_fill(image, owner, other_part, covered, detail=0.0)
        self.assertLess(split[:, 20, 0].mean(), 106)
        self.assertGreater(split[:, 28, 0].mean(), 154)

    def test_a_seam_across_two_uv_islands_is_closed_through_the_stitches(self):
        """One sheet, cut into two islands that lie apart in the layout; the cut is also a seam."""
        size = 48
        covered = np.zeros((size, size), bool)
        covered[8:40, 2:18] = covered[8:40, 30:46] = True
        rows, cols = np.nonzero(covered)
        x = np.where(cols < 24, cols - 2, cols - 30 + 16) * 0.001     # the second island continues the first
        points = np.column_stack([x, np.zeros(len(x)), rows * 0.001])
        normals = np.tile([0.0, -1.0, 0.0], (len(x), 1))
        first, second = polish.stitch_islands(rows, cols, points, normals, covered)
        self.assertGreaterEqual(len(first), 32)
        self.assertTrue(set(np.unique(first % size)) | set(np.unique(second % size)) <= {17, 30})
        image = np.full((size, size, 3), 100.0)
        image[:, 24:] = 130.0
        owner = np.where(np.arange(size)[None, :] < 24, 0, 1) * np.ones((size, 1), np.int64)
        part = np.zeros((size, size), np.int64)
        alone, _ = polish.polish_seams(image, owner, part, covered, reach=6)
        np.testing.assert_array_equal(alone, image)                    # in the layout the two never touch
        joined, _ = polish.polish_seams(image, owner, part, covered, reach=6, stitches=(first, second))
        self.assertLess(abs(joined[20, 30, 0] - joined[20, 17, 0]), 6)
        self.assertEqual(polish.seam_contrast(image, owner, part, covered, stitches=(first, second))["seam_pairs"] > 0, True)


class BakeByPartTests(unittest.TestCase):
    """The synthetic figure: skin, with a red bag on its right side."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        write_views(root / "refs")
        build_handoff(root / "refs", HEIGHT_M)
        cls.handoff = root / "refs" / "handoff"
        cls.mesh = read_mesh(cls.handoff / "blockout.obj")
        low, high = cls.mesh.bounds()
        uv = (cls.mesh.vertices[:, [1, 2]] - low[[1, 2]]) / (high[[1, 2]] - low[[1, 2]])   # seen from the side
        corners = cls.mesh.vertices[cls.mesh.faces]
        left = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])[:, 0] > 0
        cls.corner_uv = uv[cls.mesh.faces] * [0.49, 0.98] + [0.005, 0.01]
        cls.corner_uv[left, :, 0] += 0.5
        cls.out = root / "parts"
        cls.report = parts.find_parts(cls.handoff, cls.mesh, cls.corner_uv, cls.out, size=256)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def bag(self) -> dict:
        return min(self.report["parts"], key=lambda part: np.abs(np.array(part["rgb"]) - BAG).sum())

    def test_the_bag_is_a_part_with_masks_in_every_view_that_shows_it(self):
        report = self.report
        self.assertEqual((report["source_confidence"], report["review"]), ("estimated", "required"))
        self.assertGreaterEqual(len(report["parts"]), 2)
        bag, body = self.bag(), report["parts"][0]
        self.assertLess(np.abs(np.array(bag["rgb"]) - BAG).max(), 30)
        self.assertLess(np.abs(np.array(body["rgb"]) - SKIN).max(), 30)
        self.assertNotEqual(bag["material"], body["material"])
        self.assertIn("right", bag["views"])
        self.assertNotIn("left", bag["views"], "the bag hangs on the right side only")
        for name in ("parts.json", "semantic.json", "uv_labels.png", "uv_shapes.png", "turnaround.png"):
            self.assertTrue((self.out / name).is_file(), name)
        policy = json.loads((self.out / "semantic.json").read_text())
        self.assertEqual(set(policy), {"parts"})
        masks = {}
        for entry in policy["parts"]:
            self.assertEqual(set(entry), {"id", "uv_mask", "view_masks"})
            self.assertTrue(entry["view_masks"])
            for view, file in entry["view_masks"].items():
                mask = np.asarray(Image.open(self.out / file)) > 127
                with Image.open(self.handoff / "raw" / f"{view}.png") as raw:
                    self.assertEqual(mask.shape, (raw.height, raw.width), "masks are in the frame of the raw drawing")
                self.assertTrue(mask.any())
                self.assertFalse((masks.get(view, np.zeros_like(mask)) & mask).any(), "one part for each pixel")
                masks[view] = masks.get(view, np.zeros_like(mask)) | mask
        right = np.asarray(Image.open(self.handoff / "raw" / "right.png").convert("RGB")).astype(int)
        shown = np.asarray(Image.open(self.out / "views" / "right" / f"{bag['id']}.png")) > 127
        red = np.abs(right - BAG).sum(axis=2) < 60
        self.assertGreater((shown & red).sum() / max(1, shown.sum()), 0.85)
        self.assertGreater((shown & red).sum() / max(1, red.sum()), 0.85)

    def test_each_part_is_baked_from_its_own_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            image, report = parts.bake_parts(self.handoff, self.mesh, self.corner_uv, self.out, size=256,
                                             diagnostics=Path(tmp) / "d")
            self.assertEqual({p.name for p in (Path(tmp) / "d").iterdir()}, {"owners.png", "parts.png", "turnaround.png"})
        self.assertEqual(report["texels_from_parts"], 1.0)
        self.assertTrue(all(row["baked"] for row in report["materials"]))
        self.assertLessEqual(report["seam_contrast_after"]["across_delta_e"],
                             report["seam_contrast_before"]["across_delta_e"] + 0.5)
        rows, cols, _, _ = texel_samples(self.mesh, self.corner_uv, 256)
        label = np.asarray(Image.open(self.out / "uv_labels.png"))[rows, cols].astype(int) - 1
        index = self.report["parts"].index(self.bag())
        colour = image[rows, cols].astype(int)
        red = np.abs(colour - BAG).sum(axis=1) < 90
        self.assertGreater(red[label == index].mean(), 0.9, "the bag part is red all over")
        self.assertLess(red[label == 0].mean(), 0.01, "no red on the body")

    def test_the_parts_must_come_from_the_same_sources_as_the_bake(self):
        with self.assertRaises(ValueError):
            parts.bake_parts(self.handoff, self.mesh, self.corner_uv, self.out, size=128, mirror_missing=True)


if __name__ == "__main__":
    unittest.main()
