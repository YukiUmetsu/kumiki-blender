"""character_build split_views.py and part_place.py: model a part alone, place it, join the parts."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from character_build_fixtures import BAG, HEIGHT_M, SKIN, ellipsoid_mesh, write_views
from kumiki_blender.character_build import split_views
from kumiki_blender.character_build.handoff import build_handoff, reference_masks
from kumiki_blender.character_build.mesh import Mesh, rasterize
from kumiki_blender.character_build.part_place import join_parts, place_part, write_obj_uv
from kumiki_blender.character_build.repaint import BACKGROUND
from kumiki_blender.character_build.texture import read_obj_uv
from kumiki_blender.character_build.views import camera


class RemovedRegionTests(unittest.TestCase):
    def drawing(self):
        """A body with a tail beside it and a patch on its chest, as RGBA; and the two masks."""
        rgba = np.zeros((300, 300, 4), np.uint8)
        rgba[60:260, 100:180] = (*SKIN, 255)
        tail = np.zeros((300, 300), bool)
        tail[80:200, 180:250] = True
        rgba[tail] = (120, 90, 60, 255)
        patch = np.zeros((300, 300), bool)
        patch[120:170, 115:165] = True
        rgba[patch] = (*BAG, 255)
        return rgba, tail, patch

    def edited(self, rgba, remove, fill):
        picture = np.where(rgba[..., 3:] > 0, rgba[..., :3], BACKGROUND).astype(np.uint8)
        picture[remove] = fill
        return picture

    def test_an_outline_that_went_away_is_the_part(self):
        rgba, tail, _ = self.drawing()
        region, figure = split_views.removed_region(rgba, self.edited(rgba, tail, BACKGROUND))
        self.assertGreater((region & tail).sum() / (region | tail).sum(), 0.9)
        self.assertFalse(figure[100:180, 200:240].any(), "the edit shows no figure where the tail was")

    def test_a_colour_change_inside_the_figure_is_the_part(self):
        rgba, _, patch = self.drawing()
        region, _ = split_views.removed_region(rgba, self.edited(rgba, patch, SKIN))
        self.assertGreater((region & patch).sum() / (region | patch).sum(), 0.8)

    def test_a_small_shift_of_the_whole_picture_is_no_part(self):
        """The editor never returns the rest pixel for pixel."""
        rgba, tail, _ = self.drawing()
        shifted = np.roll(self.edited(rgba, np.zeros_like(tail), BACKGROUND), 2, axis=1)
        region, _ = split_views.removed_region(rgba, shifted)
        self.assertEqual(int(region.sum()), 0)

    def test_square_canvas_round_trip(self):
        rgba, _, _ = self.drawing()
        tall = Image.fromarray(rgba[:, 60:260])
        square, back = split_views.to_square(tall, 256)
        self.assertEqual(square.size, (256, 256))
        again = np.asarray(split_views.from_square(square, back, tall.size)).astype(int)
        flat = np.where(rgba[:, 60:260, 3:] > 0, rgba[:, 60:260, :3], BACKGROUND).astype(int)
        self.assertLess(np.abs(again - flat).mean(), 4)


class SplitViewsTests(unittest.TestCase):
    """The synthetic figure; the red bag on its right side is the part to take out."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        write_views(root / "refs", names=("front", "back", "left", "right"))
        build_handoff(root / "refs", HEIGHT_M)
        cls.handoff, cls.out, cls.calls = root / "refs" / "handoff", root / "split", []

        class Client:
            def upload_image(self, path):
                return str(path)

        def paint_out(client, workflow):
            cls.calls.append(Path(workflow["4"]["inputs"]["image"]).name)
            with Image.open(workflow["4"]["inputs"]["image"]) as picture:
                pixels = np.asarray(picture.convert("RGB")).copy()
            pixels[np.abs(pixels.astype(int) - BAG).sum(axis=2) < 90] = SKIN     # the bag turns into body
            data = io.BytesIO()
            Image.fromarray(pixels).save(data, format="PNG")
            return data.getvalue()

        cls.original, split_views.run_comfy = split_views.run_comfy, paint_out
        cls.client = Client()
        cls.report = split_views.split_views(cls.handoff, cls.out, cls.client, part="bag", size=512,
                                             views=["front", "right", "left"])

    @classmethod
    def tearDownClass(cls):
        split_views.run_comfy = cls.original
        cls.tmp.cleanup()

    def red(self, path):
        with Image.open(path) as picture:
            pixels = np.asarray(picture.convert("RGBA")).astype(int)
        return (np.abs(pixels[..., :3] - BAG).sum(axis=2) < 90) & (pixels[..., 3] > 127), pixels[..., 3] > 127

    def test_the_part_is_found_only_where_a_drawing_shows_it(self):
        found = {row["view"]: row["part_found"] for row in self.report["views"]}
        self.assertEqual(found, {"front": True, "right": True, "left": False})
        self.assertEqual((self.report["source_confidence"], self.report["review"]), ("estimated", "required"))
        self.assertEqual(sorted(p.name for p in (self.out / "bag" / "raw").iterdir()), ["front.png", "right.png"])
        self.assertEqual(sorted(p.name for p in (self.out / "rest" / "raw").iterdir()),
                         ["back.png", "front.png", "left.png", "right.png"])

    def test_the_part_keeps_the_drawings_own_pixels_and_the_rest_has_none_of_it(self):
        bag_in_drawing, figure = self.red(self.handoff / "raw" / "right.png")
        _, part = self.red(self.out / "bag" / "raw" / "right.png")
        self.assertGreater((part & bag_in_drawing).sum() / (part | bag_in_drawing).sum(), 0.8)
        with Image.open(self.out / "bag" / "raw" / "right.png") as cut, Image.open(self.handoff / "raw" / "right.png") as raw:
            np.testing.assert_array_equal(np.asarray(cut)[..., :3][part], np.asarray(raw.convert("RGBA"))[..., :3][part])
        bag_left, rest = self.red(self.out / "rest" / "raw" / "right.png")
        self.assertLess(bag_left.sum(), 0.02 * bag_in_drawing.sum())
        self.assertGreater((rest & figure).sum() / (rest | figure).sum(), 0.97, "the body keeps its outline")
        for name in ("back", "left"):       # not edited, or nothing found: the drawing itself
            with Image.open(self.out / "rest" / "raw" / f"{name}.png") as kept, Image.open(self.handoff / "raw" / f"{name}.png") as raw:
                np.testing.assert_array_equal(np.asarray(kept), np.asarray(raw.convert("RGBA")))

    def test_both_folders_are_handoffs_in_the_frame_of_the_character(self):
        whole = reference_masks(self.handoff, 0.25)
        bag, rest = reference_masks(self.out / "bag", 0.25), reference_masks(self.out / "rest", 0.25)
        self.assertEqual(set(bag), {"front", "right"})
        self.assertEqual(set(rest), set(whole))
        for name in bag:
            with Image.open(self.out / "bag" / "masks" / f"{name}.png") as mask:
                self.assertEqual(mask.size, (1600, 1200))
            self.assertLess((bag[name] & ~whole[name]).sum(), 0.03 * bag[name].sum(), "the part lies inside the figure")
            self.assertGreater(bag[name].sum(), 20)
        spec = json.loads((self.out / "bag" / "spec.json").read_text())
        self.assertEqual(spec["views"], ["front", "right"])
        self.assertEqual(spec["raw"]["right"], json.loads((self.handoff / "spec.json").read_text())["raw"]["right"])
        self.assertNotIn("blockout", spec)

    def test_finished_edits_are_reused(self):
        before = len(self.calls)
        self.assertEqual(before, 3)
        again = split_views.split_views(self.handoff, self.out, self.client, part="bag", size=512,
                                        views=["front", "right", "left"])
        self.assertEqual(len(self.calls), before)
        self.assertTrue(all(row["reused"] for row in again["views"]))
        with self.assertRaises(ValueError):
            split_views.split_views(self.handoff, self.out, self.client, part="bag", views=["top"])

    def test_the_part_folder_has_a_safe_name(self):
        self.assertEqual(self.report["name"], "bag")
        for bad in ("rest", "edits", "../x", "a b"):
            with self.assertRaises(ValueError):
                split_views.split_views(self.handoff, self.out, self.client, part="bag", name=bad)
        before = len(self.calls)
        named = split_views.split_views(self.handoff, self.out, self.client, part="bag", name="satchel", size=512,
                                        views=["front", "right", "left"])
        self.assertEqual(len(self.calls), before, "the same edits serve a different folder name")
        self.assertEqual(set(named["folders"]), {"satchel", "rest"})
        self.assertTrue((self.out / "satchel" / "raw" / "right.png").is_file())


class PlaceAndJoinTests(unittest.TestCase):
    PX = 0.32 / 1000

    def tail(self):
        base = ellipsoid_mesh(centre=(0, 0, 0), radii=(.04, .03, .09), subdivisions=3)
        bent = base.vertices.copy()
        bent[:, 1] += 0.6 * bent[:, 2] ** 2 / 0.09          # not symmetric front to back
        return base, bent

    def test_a_turned_scaled_moved_part_returns_to_its_place(self):
        base, bent = self.tail()
        truth = Mesh(bent + [0.01, 0.07, 0.15], base.faces)
        masks = {name: rasterize(truth, camera(name), self.PX, 0.25)[0] for name in ("left", "back", "back_left")}
        a = np.radians(200)
        turn = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        placed, info = place_part(Mesh((bent @ turn.T) * 3.7 + [5, -2, 1], base.faces), masks, self.PX)
        self.assertLess(np.abs(placed.vertices - truth.vertices).max(), 0.004)
        self.assertGreater(info["mean_iou"], 0.95)
        self.assertEqual(info["position_measured"], {"x": True, "y": True, "z": True})
        self.assertAlmostEqual(info["scale"], 1 / 3.7, delta=0.01)
        self.assertLess(min(abs(info["yaw_deg"] - 160), abs(info["yaw_deg"] - 160 - 360)), 4)

    def test_one_view_leaves_the_depth_unmeasured(self):
        base, bent = self.tail()
        truth = Mesh(bent + [0.01, 0.07, 0.15], base.faces)
        masks = {"left": rasterize(truth, camera("left"), self.PX, 0.25)[0]}
        placed, info = place_part(Mesh(bent * 2 + [1, 1, 1], base.faces), masks, self.PX, yaw_deg=0.0)
        self.assertEqual(info["position_measured"], {"x": False, "y": True, "z": True})
        self.assertLess(np.abs(placed.vertices[:, 1:] - truth.vertices[:, 1:]).max(), 0.004)
        self.assertAlmostEqual(float(placed.vertices[:, 0].min() + placed.vertices[:, 0].max()), 0.0, delta=1e-6)
        with self.assertRaises(ValueError):
            place_part(truth, {"back_left": masks["left"]}, self.PX)

    def test_joined_parts_keep_their_own_region_of_the_texture(self):
        big = ellipsoid_mesh(centre=(0, 0, .2), radii=(.1, .1, .2), subdivisions=2)
        small = ellipsoid_mesh(centre=(0, .15, .2), radii=(.03, .03, .08), subdivisions=2)

        def planar(mesh):
            low, high = mesh.bounds()
            return ((mesh.vertices[:, [0, 2]] - low[[0, 2]]) / (high[[0, 2]] - low[[0, 2]]))[mesh.faces]

        red, blue = np.full((64, 64, 3), (200, 30, 30), np.uint8), np.full((64, 64, 3), (30, 30, 200), np.uint8)
        mesh, corner_uv, texture, layout = join_parts([(big, planar(big), red), (small, planar(small), blue)], size=256)
        self.assertEqual(len(mesh.vertices), len(big.vertices) + len(small.vertices))
        self.assertEqual(layout[0]["u_range"][0], 0.0)
        self.assertEqual(layout[1]["u_range"][1], 1.0)
        self.assertGreaterEqual(layout[1]["u_range"][1] - layout[1]["u_range"][0], 0.14, "a small part keeps a usable share")
        for row, colour in zip(layout, ((200, 30, 30), (30, 30, 200))):
            uv = corner_uv[row["faces"][0]:row["faces"][1]].reshape(-1, 2)
            self.assertTrue((uv[:, 0] >= row["u_range"][0] - 1e-9).all() and (uv[:, 0] <= row["u_range"][1] + 1e-9).all())
            inner = uv[(uv[:, 0] > row["u_range"][0] + 0.02) & (uv[:, 0] < row["u_range"][1] - 0.02)]
            sampled = texture[np.clip(((1 - inner[:, 1]) * 256).astype(int), 0, 255), (inner[:, 0] * 256).astype(int)]
            self.assertLess(np.abs(sampled.astype(int) - colour).max(), 12)
        with tempfile.TemporaryDirectory() as tmp:
            write_obj_uv(mesh, corner_uv, Path(tmp) / "joined.obj")
            again, uv_again = read_obj_uv(Path(tmp) / "joined.obj")
        np.testing.assert_allclose(again.vertices, mesh.vertices, atol=1e-6)
        np.testing.assert_array_equal(again.faces, mesh.faces)
        np.testing.assert_allclose(uv_again, corner_uv, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
