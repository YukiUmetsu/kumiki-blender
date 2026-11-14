"""character_build split_views.py and part_place.py: model a part alone, place it, join the parts."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from character_build_fixtures import BAG, HEIGHT_M, SKIN, ellipsoid_mesh, signed_volume, write_views
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

    def test_within_and_no_fill(self):
        """A mask limits where the part can be; without fill, the rest has a hole and no edited pixels."""
        with Image.open(self.handoff / "raw" / "right.png") as picture:
            raw = np.asarray(picture.convert("RGBA"))
        bag = (np.abs(raw[..., :3].astype(int) - BAG).sum(axis=2) < 90) & (raw[..., 3] > 127)
        ys, xs = np.nonzero(bag)
        with tempfile.TemporaryDirectory() as tmp:
            allowed = np.zeros(bag.shape, np.uint8)
            allowed[:, :int(xs.mean())] = 255                      # only the half of the bag on the image left
            Image.fromarray(allowed).save(Path(tmp) / "right.png")
            before = len(self.calls)
            report = split_views.split_views(self.handoff, self.out, self.client, part="bag", name="half", size=512,
                                             views=["right"], within=Path(tmp), fill=False)
        self.assertEqual(len(self.calls), before, "the edits are reused")
        self.assertTrue(report["views"][0]["changes_outside_dropped"])
        with Image.open(self.out / "half" / "raw" / "right.png") as picture:
            part = np.asarray(picture)[..., 3] > 127
        self.assertFalse(part[:, int(xs.mean()) + 2:].any())
        self.assertGreater(part.sum(), 0.3 * bag.sum())
        with Image.open(self.out / "rest" / "raw" / "right.png") as picture:
            rest = np.asarray(picture)
        self.assertFalse((rest[..., 3] > 127)[part].any(), "a hole where the part was")
        kept = rest[..., 3] > 127
        np.testing.assert_array_equal(rest[kept], raw[kept])       # every pixel that is left is the drawing's own
        # Put the shared folders back for the other tests.
        split_views.split_views(self.handoff, self.out, self.client, part="bag", size=512, views=["front", "right", "left"])

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
        np.testing.assert_allclose(info["stretch"], [1, 1, 1], atol=0.03)
        self.assertLess(min(abs(info["yaw_deg"] - 160), abs(info["yaw_deg"] - 160 - 360)), 4)

    def test_the_placed_part_has_the_height_of_its_mask(self):
        """A flat base holds many surface points and few pixels: the size must come from the silhouette."""
        from kumiki_blender.character_build.mesh import surface_samples
        ball = ellipsoid_mesh(centre=(0, 0, 0), radii=(.05, .05, .15), subdivisions=4)
        v = ball.vertices.copy()
        v[:, 2] = np.maximum(v[:, 2], -0.05)           # cut flat at the bottom, like soles
        truth = Mesh(v + [0, 0, 0.05], ball.faces)
        self.assertGreater((surface_samples(truth, 0.002)[0][:, 2] < 0.001).mean(), 0.02)
        masks = {name: rasterize(truth, camera(name), self.PX, 0.25)[0] for name in ("front", "left")}
        placed, info = place_part(Mesh(v * 5, ball.faces), masks, self.PX, yaw_deg=0.0, refine=False)
        self.assertAlmostEqual(float(np.ptp(placed.vertices[:, 2])), 0.2, delta=0.004)
        self.assertGreater(info["mean_iou"], 0.97)
        np.testing.assert_allclose(info["stretch"], [1, 1, 1], atol=0.02)
        # A generator made the part 10% too squat: each direction gets its own scale, within the limit.
        squat = Mesh(v * [5, 5, 4.5], ball.faces)
        placed, info = place_part(squat, masks, self.PX, yaw_deg=0.0)
        self.assertAlmostEqual(float(np.ptp(placed.vertices[:, 2])), 0.2, delta=0.004)
        self.assertAlmostEqual(float(np.ptp(placed.vertices[:, 0])), 0.1, delta=0.004)
        self.assertGreater(info["stretch"][2] / info["stretch"][0], 1.07)
        rigid, info = place_part(squat, masks, self.PX, yaw_deg=0.0, stretch=0.0)
        self.assertEqual(info["stretch"], [1.0, 1.0, 1.0])
        self.assertLess(float(np.ptp(rigid.vertices[:, 2])), 0.198, "one scale cannot give both the height and the width")

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

    def test_a_part_inside_the_body_is_seated_on_its_surface(self):
        from kumiki_blender.character_build.part_place import seat_part
        body = ellipsoid_mesh(centre=(0, 0, .16), radii=(.06, .05, .16), subdivisions=3)
        bag = ellipsoid_mesh(centre=(.045, 0, .12), radii=(.015, .02, .02), subdivisions=2)   # on the left hip, mostly inside
        seated, info = seat_part(bag, body, self.PX)
        self.assertEqual(info["view"], "left")
        self.assertGreater(info["buried_before"], 0.5)
        self.assertAlmostEqual(info["buried_after"], 0.15, delta=0.05)
        moved = seated.vertices - bag.vertices
        self.assertGreater(moved[0, 0], 0.005)                      # out, along +X
        np.testing.assert_allclose(moved, np.tile(moved[0], (len(moved), 1)), atol=1e-12)   # as one rigid piece
        np.testing.assert_allclose(moved[:, 1:], 0, atol=1e-12)
        again, info = seat_part(seated, body, self.PX, view="left")
        self.assertLess(info["moved_m"], 0.0005, "a seated part stays")
        outside = Mesh(bag.vertices + [0.1, 0, 0], bag.faces)
        np.testing.assert_array_equal(seat_part(outside, body, self.PX)[0].vertices, outside.vertices)

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


class SmallPartTests(unittest.TestCase):
    """A hand at the end of an arm: more pixels for it, its place, the body without its own hand, the other side."""

    PX = 0.32 / 1000

    def body_and_hand(self):
        body = ellipsoid_mesh(centre=(0, 0.01, .16), radii=(.12, .03, .03), subdivisions=4)     # an arm along X, at y = 1 cm
        hand = ellipsoid_mesh(centre=(.12, 0.01, .16), radii=(.02, .012, .025), subdivisions=3)
        return body, hand

    def test_a_part_seen_from_front_and_back_takes_its_depth_from_the_body(self):
        body, hand = self.body_and_hand()
        masks = {name: rasterize(hand, camera(name), self.PX, 0.25)[0] for name in ("front", "back")}
        source = Mesh((hand.vertices - [.12, 0.01, .16]) * 4, hand.faces)
        free, info = place_part(source, masks, self.PX, yaw_deg=0.0)
        self.assertEqual((info["position_measured"]["y"], info["unmeasured_from_anchor"]), (False, False))
        self.assertAlmostEqual(float(free.vertices[:, 1].mean()), 0.0, delta=0.002)
        placed, info = place_part(source, masks, self.PX, yaw_deg=0.0, anchor=body)
        self.assertTrue(info["unmeasured_from_anchor"])
        self.assertLess(np.abs(placed.vertices - hand.vertices).max(), 0.004)

    def test_the_body_loses_what_the_part_replaces_and_nothing_else(self):
        from kumiki_blender.character_build.part_place import cut_under
        body, hand = self.body_and_hand()
        masks = {name: rasterize(hand, camera(name), self.PX, 0.25)[0] for name in ("front", "back")}
        uv = np.zeros((len(body.faces), 3, 2))
        uv[:, :, 0] = np.arange(len(body.faces))[:, None]           # to follow the faces through the cut
        cut, cut_uv, removed = cut_under(body, uv, masks, self.PX)
        self.assertGreater(removed, 20)
        self.assertEqual(len(cut.faces), len(body.faces) - removed)
        self.assertEqual(len(cut_uv), len(cut.faces))
        centres = cut.vertices[cut.faces].mean(axis=1)
        self.assertLess(centres[:, 0].max(), 0.105, "the end of the arm under the hand is gone")
        self.assertLess(centres[:, 0].min(), -0.115, "the other end of the arm stays")
        kept = cut_uv[:, 0, 0].astype(int)
        np.testing.assert_allclose(cut.vertices[cut.faces], body.vertices[body.faces[kept]])   # faces and UVs stay paired
        band = masks["front"].copy()                                 # the outline misses the end of the hand
        columns = np.nonzero(band.any(axis=0))[0]
        band[:, columns[len(columns) // 4]:] = False                 # x from 0.10 to 0.11; the arm ends at 0.12
        with_tip = cut_under(body, uv, {"front": band}, self.PX, grow_px=1, loose_faces=0)[0]
        without = cut_under(body, uv, {"front": band}, self.PX, grow_px=1, loose_faces=400)[0]
        self.assertGreater(with_tip.vertices[:, 0].max(), 0.115, "the end of the arm is beyond the band")
        self.assertLess(without.vertices[:, 0].max(), 0.105, "the small piece that the cut separated goes too")
        self.assertLess(without.vertices[:, 0].min(), -0.115)
        half = masks["back"].copy()                                  # the back drawing shows less of the hand
        half[:, :int(np.nonzero(half.any(axis=0))[0].mean())] = False
        self.assertEqual(cut_under(body, uv, {"front": masks["front"], "back": half}, self.PX)[2], removed,
                         "opposite views draw one outline: what one of them shows is the part")
        side = rasterize(hand, camera("left"), self.PX, 0.25)[0]
        self.assertEqual(cut_under(body, uv, {"front": masks["front"], "left": np.zeros_like(side)}, self.PX)[2], removed,
                         "a view that does not show the part does not limit the cut")
        shifted = np.roll(side, 40, axis=1)                          # a side view that puts the hand elsewhere in depth
        self.assertEqual(cut_under(body, uv, {"front": masks["front"], "left": shifted}, self.PX)[2], 0,
                         "views along different directions must both show the face")
        self.assertEqual(cut_under(body, None, {"front": np.zeros_like(masks["front"])}, self.PX)[2], 0,
                         "no view shows the part: nothing is cut")
        self.assertEqual(cut_under(body, None, {}, self.PX)[2], 0)

    def test_mirror_keeps_the_outside_out(self):
        from kumiki_blender.character_build.part_place import mirror_part
        _, hand = self.body_and_hand()
        uv = np.random.default_rng(0).random((len(hand.faces), 3, 2))
        other, other_uv = mirror_part(hand, uv, mid_x=0.01)
        np.testing.assert_allclose(other.vertices[:, 0].mean(), 0.02 - hand.vertices[:, 0].mean(), atol=1e-9)
        self.assertGreater(signed_volume(other), 0)
        self.assertAlmostEqual(signed_volume(other), signed_volume(hand), delta=1e-12)
        face = 7                                                     # each corner keeps its UV
        for corner in range(3):
            original = [0, 2, 1][corner]
            np.testing.assert_allclose(other_uv[face, corner], uv[face, original])
            np.testing.assert_allclose(other.vertices[other.faces[face, corner]] * [-1, 1, 1] + [0.02, 0, 0],
                                       hand.vertices[hand.faces[face, original]])

    def test_detail_views_enlarge_a_part_in_the_same_frame(self):
        from kumiki_blender.character_build import detail, split_views as S
        from kumiki_blender.character_build.handoff import reference_masks as masks_of
        _, hand = self.body_and_hand()
        spec = {"schema_version": 1, "height_m": 0.32, "metres_per_aligned_px": self.PX, "views": ["front", "back"],
                "raw": {n: {"left": 0, "top": 0, "scale": 1.0, "x0": 0, "y0": 0, "shift_px": 0} for n in ("front", "back")}}
        images = {}
        for name in spec["views"]:
            mask, _ = rasterize(hand, camera(name), self.PX, 1.0)
            images[name] = np.dstack([np.where(mask[..., None], (230, 160, 150), 0).astype(np.uint8),
                                      np.where(mask, 255, 0).astype(np.uint8)])
        calls = []

        class Client:
            def upload_image(self, path):
                return str(path)

        def sharpen(client, workflow):                               # the "editor" returns the picture with a line on it
            calls.append(1)
            with Image.open(workflow["4"]["inputs"]["image"]) as picture:
                pixels = np.asarray(picture.convert("RGB")).copy()
            figure = np.abs(pixels.astype(int) - (217, 217, 217)).max(axis=2) > 24
            line = figure & (np.abs(np.arange(pixels.shape[1])[None, :] - pixels.shape[1] // 2) < 3)
            pixels[line] = (120, 60, 60)
            data = io.BytesIO()
            Image.fromarray(pixels).save(data, format="PNG")
            return data.getvalue()

        original, detail.run_comfy = detail.run_comfy, sharpen
        try:
            with tempfile.TemporaryDirectory() as tmp:
                small, large = Path(tmp) / "hand", Path(tmp) / "hand_detail"
                S.write_handoff(small, spec, images, "hand")
                report = detail.detail_views(small, large, Client(), part="the hand", size=512)
                again = detail.detail_views(small, large, Client(), part="the hand", size=512)
                before, after = masks_of(small, 0.5), masks_of(large, 0.5)
                new_spec = json.loads((large / "spec.json").read_text())
                with Image.open(large / "raw" / "front.png") as picture:
                    front = np.asarray(picture)
        finally:
            detail.run_comfy = original
        self.assertEqual((len(calls), [row["reused"] for row in again["views"]]), (2, [True, True]))
        self.assertTrue(all(row["redrawn"] and row["enlarged"] > 2 for row in report["views"]), report["views"])
        self.assertEqual(front.shape[:2], (512, 512))
        self.assertLess(new_spec["raw"]["front"]["scale"], 0.5, "more drawing pixels for each aligned pixel")
        for name in ("front", "back"):                               # the part is at the same place in the character's frame
            overlap = (before[name] & after[name]).sum() / (before[name] | after[name]).sum()
            self.assertGreater(overlap, 0.9, name)
        figure = front[..., 3] > 127
        row = front[256, :, :3].astype(int)                           # across the hand: the drawn line is in the middle
        self.assertGreater(row[230:245].sum(axis=1).max() - row[254:258].sum(axis=1).min(), 60, "the redrawn detail is kept")
        mean = front[..., :3][figure].astype(float).mean(axis=0)
        self.assertLess(np.abs(mean - (230, 160, 150)).max(), 14, "with the colours of the drawing")

    def test_boxes_and_polygons_split_a_drawing_without_an_editor(self):
        from kumiki_blender.character_build import split_views as S
        spec = {"schema_version": 1, "height_m": 0.32, "metres_per_aligned_px": self.PX, "views": ["front", "back"],
                "raw": {n: {"left": 0, "top": 0, "scale": 1.0, "x0": 0, "y0": 0, "shift_px": 0} for n in ("front", "back")}}
        figure = np.zeros((120, 200, 4), np.uint8)
        figure[20:100, 40:160] = (200, 150, 120, 255)
        with tempfile.TemporaryDirectory() as tmp:
            whole = Path(tmp) / "whole"
            S.write_handoff(whole, spec, {"front": figure, "back": figure}, "whole")
            report = S.box_part(whole, Path(tmp) / "out", name="hand", boxes={"front": [(0, 0, 60, 120)]},
                                polygons={"front": [[(160, 20), (160, 100), (120, 100)]]})
            with Image.open(Path(tmp) / "out" / "hand" / "raw" / "front.png") as picture:
                part = np.asarray(picture)[..., 3] > 127
            with Image.open(Path(tmp) / "out" / "rest" / "raw" / "front.png") as picture:
                rest = np.asarray(picture)[..., 3] > 127
            self.assertFalse((Path(tmp) / "out" / "hand" / "raw" / "back.png").exists(), "no box: the view has no part")
            with Image.open(Path(tmp) / "out" / "rest" / "raw" / "back.png") as picture:
                self.assertEqual(int((np.asarray(picture)[..., 3] > 127).sum()), 80 * 120)
            with self.assertRaises(ValueError):
                S.box_part(whole, Path(tmp) / "bad", name="hand", boxes={"front": [(0, 0, 30, 15)]})    # background only
            with self.assertRaises(ValueError):
                S.box_part(whole, Path(tmp) / "bad", name="hand", polygons={"front": [[(50, 30), (60, 40)]]})
            with self.assertRaises(ValueError):
                S.box_part(whole, Path(tmp) / "bad", name="hand", boxes={"left": [(0, 0, 60, 120)]})
        self.assertTrue(part[60, 50] and not part[60, 70], "the box takes the figure inside it")
        self.assertTrue(part[90, 150] and not part[30, 130], "the polygon takes its triangle")
        self.assertFalse(part[10, 10], "the background is no part")
        np.testing.assert_array_equal(part | rest, figure[..., 3] > 127)
        self.assertFalse((part & rest).any())
        self.assertEqual(report["views"][0]["part_px"], int(part.sum()))


class RedrawTests(unittest.TestCase):
    def test_the_change_is_taken_from_the_edit_and_the_rest_from_the_drawing(self):
        from kumiki_blender.character_build import redraw
        rng = np.random.default_rng(0)
        rgba = np.zeros((400, 400, 4), np.uint8)
        rgba[60:340, 150:250] = (*SKIN, 255)                       # body
        rgba[60:340, 150:250, :3] = np.clip(SKIN + rng.normal(0, 6, (280, 100, 3)), 0, 255)   # fine detail
        rgba[150:180, 80:150] = (*SKIN, 255)                       # a bare arm
        calls = []

        class Client:
            def upload_image(self, path):
                return str(path)

        def add_sleeve(client, workflow):
            calls.append(1)
            with Image.open(workflow["4"]["inputs"]["image"]) as picture:
                pixels = np.asarray(picture.convert("RGB")).copy()
            size = pixels.shape[0]
            small = np.asarray(Image.fromarray(pixels).resize((size // 2, size // 2)).resize((size, size)))   # softer
            scale = size / 400
            a, b, c, d = (int(v * scale) for v in (140, 190, 80, 150))
            out = small.copy()
            out[a:b, c:d] = (245, 228, 175)                        # a sleeve, thicker than the arm
            data = io.BytesIO()
            Image.fromarray(out).save(data, format="PNG")
            return data.getvalue()

        original, redraw.run_comfy = redraw.run_comfy, add_sleeve
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                Image.fromarray(rgba).save(root / "front.png")
                Image.fromarray(rgba).save(root / "back.png")
                report = redraw.redraw_view(root / "front.png", root / "back.png", root / "new" / "front.png", Client(),
                                            change="the arm wears a sleeve", size=400)
                with Image.open(root / "new" / "front.png") as picture:
                    result = np.asarray(picture)
                again = redraw.redraw_view(root / "front.png", root / "back.png", root / "new" / "front.png", Client(),
                                           change="the arm wears a sleeve", size=400)
        finally:
            redraw.run_comfy = original
        self.assertEqual((len(calls), report["reused"], again["reused"]), (1, False, True))
        self.assertLess(np.abs(result[165, 110, :3].astype(int) - (245, 228, 175)).max(), 12, "the sleeve is there")
        self.assertEqual(result[145, 110, 3], 255, "also where it is thicker than the arm was")
        np.testing.assert_array_equal(result[250:330, 160:240], rgba[250:330, 160:240])   # far from the change: the drawing
        self.assertGreater(report["changed_share_of_figure"], 0.03)
        self.assertLess(report["changed_share_of_figure"], 0.5)
        self.assertIn("the arm wears a sleeve", report["prompt"])


    def test_a_region_of_the_front_comes_from_the_mirrored_back(self):
        from kumiki_blender.character_build import redraw, split_views as S
        maps = {"front": {"left": 0, "top": 0, "scale": 1.0, "x0": 600, "y0": 100, "shift_px": 0},
                "back": {"left": 10, "top": 5, "scale": 2.0, "x0": 500, "y0": 100, "shift_px": 0},   # drawn at half the size
                "left": {"left": 0, "top": 0, "scale": 1.0, "x0": 600, "y0": 100, "shift_px": 0}}
        spec = {"schema_version": 1, "height_m": 0.32, "metres_per_aligned_px": 0.00032, "views": list(maps), "raw": maps}
        front = np.zeros((600, 400, 4), np.uint8)
        front[100:500, 100:300] = (*SKIN, 255)                       # body, in the middle of the character's frame
        front[150:190, 300:400] = (200, 40, 40, 255)                 # the character's left arm, drawn high
        back = np.zeros((300, 320, 4), np.uint8)
        back[55:255, 110:210] = (90, 70, 50, 255)                    # the same body from behind
        back[155:175, 60:110] = (40, 40, 200, 255)                   # the same arm, drawn lower; it is on the left here
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            S.write_handoff(root / "in", spec, {"front": front, "back": back, "left": front}, "whole")
            report = redraw.borrow_view(root / "in", root / "one", view="front", source="back",
                                        boxes=[(300, 100, 400, 400)], feather_px=0)
            both = redraw.borrow_view(root / "one", root / "two", view="front", source="front",
                                      polygons=[[(0, 100), (100, 100), (100, 400), (0, 400)]], feather_px=0)
            with Image.open(root / "two" / "raw" / "front.png") as picture:
                result = np.asarray(picture)
            with Image.open(root / "two" / "raw" / "back.png") as picture:
                np.testing.assert_array_equal(np.asarray(picture), back)
            self.assertTrue((root / "two" / "masks" / "front.png").is_file())
            with self.assertRaises(ValueError):
                redraw.borrow_view(root / "in", root / "bad", view="front", source="left", boxes=[(300, 100, 400, 400)])
            with self.assertRaises(ValueError):
                redraw.borrow_view(root / "in", root / "bad", view="left", source="left", boxes=[(300, 100, 400, 400)])
            with self.assertRaises(ValueError):
                redraw.borrow_view(root / "in", root / "bad", view="front", source="back")
        self.assertEqual(report["outline_iou_in_region_before"], 0.0, "the two drawings put the arm at different heights")
        self.assertEqual(both["view"], "front")
        self.assertEqual(result[170, 350, 3], 0, "the high arm is gone")
        np.testing.assert_array_equal(result[320, 350], (40, 40, 200, 255))      # the arm of the back drawing, at its height
        self.assertEqual((result[296, 350, 3], result[304, 350, 3], result[336, 350, 3], result[344, 350, 3]),
                         (0, 255, 255, 0), "at the size of the front drawing")
        np.testing.assert_array_equal(result[320, 50], (40, 40, 200, 255))       # and the other arm from this one
        self.assertEqual(result[170, 50, 3], 0)
        np.testing.assert_array_equal(result[100:500, 100:300], front[100:500, 100:300])   # outside the regions: no change


class EyeTests(unittest.TestCase):
    """A ball head with two drawn eyes: front and left drawings, each with its own idea of where the eye is."""

    PX = 0.32 / 1000

    @classmethod
    def setUpClass(cls):
        from PIL import ImageDraw
        from kumiki_blender.character_build import eyes, split_views as S
        cls.eyes = eyes
        cls.head = ellipsoid_mesh(centre=(0, 0, .16), radii=(.1, .1, .16), subdivisions=4)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.handoff = Path(cls.tmp.name) / "handoff"
        images = {}
        for name in ("front", "left", "back"):
            mask, _ = rasterize(cls.head, camera(name), cls.PX, 1.0)
            picture = Image.fromarray(np.where(mask[..., None], (200, 170, 140), 0).astype(np.uint8))
            pen = ImageDraw.Draw(picture)
            spots = {"front": [(800 - 95, 500), (800 + 95, 500)], "left": [(800 - 235, 505)], "back": []}[name]   # the true place is 800 - 290
            for x, y in spots:                      # white of the eye, lid line, iris, pupil, highlight
                pen.ellipse((x - 70, y - 60, x + 70, y + 60), fill=(240, 232, 220), outline=(30, 20, 15), width=7)
                pen.ellipse((x - 42, y - 42, x + 42, y + 42), fill=(170, 90, 25), outline=(60, 30, 10), width=4)
                pen.ellipse((x - 26, y - 26, x + 26, y + 26), fill=(8, 6, 6))
                pen.ellipse((x + 6, y - 22, x + 18, y - 10), fill=(250, 250, 250))
            images[name] = np.dstack([np.asarray(picture), np.where(mask, 255, 0).astype(np.uint8)])
        spec = {"schema_version": 1, "height_m": 0.32, "metres_per_aligned_px": cls.PX, "views": list(images),
                "raw": {n: {"left": 0, "top": 0, "scale": 1.0, "x0": 0, "y0": 0, "shift_px": 0} for n in images}}
        S.write_handoff(cls.handoff, spec, images, "test")
        cls.front = images["front"]
        cls.out = Path(cls.tmp.name) / "eyes"
        cls.report = eyes.split_eyes(cls.handoff, cls.head, cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_pupil_and_iris_are_found_and_the_white_is_left_alone(self):
        found, mask = self.eyes.find_eyes(self.front)
        self.assertEqual(len(found), 2)
        for eye, x in zip(found, (705, 895)):
            self.assertAlmostEqual(eye["centre_px"][0], x, delta=3)
            self.assertAlmostEqual(eye["centre_px"][1], 500, delta=3)
            self.assertAlmostEqual(eye["pupil_radius_px"], 26, delta=3)
            self.assertAlmostEqual(eye["iris_radius_px"], 42, delta=4)
        self.assertTrue(mask[500, 705] and mask[500, 740])
        self.assertFalse(mask[500, 765], "the white of the eye is not part of the eyeball")
        plain = self.front.copy()
        plain[..., :3] = (200, 170, 140)
        self.assertEqual(self.eyes.find_eyes(plain)[0], [])
        bead = self.front.copy()                    # no coloured iris: the pupil counts
        bead[np.abs(bead[..., :3].astype(int) - (170, 90, 25)).sum(axis=2) < 60, :3] = (240, 232, 220)
        self.assertAlmostEqual(self.eyes.find_eyes(bead)[0][0]["iris_radius_px"], 26, delta=4)

    def test_eyeballs_sit_on_the_head_at_the_drawn_place(self):
        ids = [eye["id"] for eye in self.report["eyes"]]
        self.assertEqual(sorted(ids), ["eye.L", "eye.R"])
        for eye in self.report["eyes"]:
            centre, radii = np.array(eye["center_m"]), np.array(eye["radii_m"])
            self.assertAlmostEqual(abs(centre[0]), 95 * self.PX, delta=2 * self.PX)
            self.assertEqual(centre[0] > 0, eye["id"] == "eye.L", "the character's left is +X")
            self.assertAlmostEqual(centre[2], (1100 - 500) * self.PX, delta=2 * self.PX)
            self.assertAlmostEqual(radii[0], 1.06 * 42 * 1.03 * self.PX, delta=3 * self.PX)
            self.assertLess(radii[1], radii[0], "a lens, flatter along the view")
            surface = -0.1 * np.sqrt(1 - (centre[0] / 0.1) ** 2 - ((centre[2] - 0.16) / 0.16) ** 2)   # the head at the eye
            pole = centre[1] - radii[1]                # the front camera looks along +Y: the pole is at the lowest y
            self.assertLess(pole, surface, "the eyeball stands out of the head")
            self.assertGreater(centre[1], surface, "and its centre is inside")
            mesh, uv = read_obj_uv(self.out / eye["mesh"])
            np.testing.assert_allclose(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0), 2 * radii, rtol=0.02)
            self.assertTrue((uv >= 0).all() and (uv <= 1).all())
        anatomy = json.loads((self.out / "anatomy.json").read_text())
        self.assertEqual([part["kind"] for part in anatomy["parts"]], ["ocular_globe", "ocular_globe"])
        self.assertTrue(all(part["pupil_fraction"] < part["iris_fraction"] for part in anatomy["parts"]))

    def test_eyeballs_from_a_part_include_the_white_of_the_eye(self):
        """The part is what `split-views` took out: the whole eye, 140 x 120 px, not only the iris."""
        from kumiki_blender.character_build import split_views as S
        yy, xx = np.mgrid[:1200, :1600]
        whole = np.zeros((1200, 1600), bool)
        for x in (705, 895):
            whole |= ((xx - x) / 70) ** 2 + ((yy - 500) / 60) ** 2 <= 1
        whole[300:306, 300:306] = True                  # a speck that the edit also changed: no eye
        part = self.front.copy()
        part[..., 3] = np.where(whole, 255, 0)
        spec = json.loads((self.handoff / "spec.json").read_text())
        root = Path(self.tmp.name) / "part"
        S.write_handoff(root, spec, {"front": part}, "eyes")
        report = self.eyes.eyeballs_from_part(root, self.head, Path(self.tmp.name) / "balls", shape="ball")
        self.assertTrue(report["white_included"])
        self.assertEqual(sorted(eye["id"] for eye in report["eyes"]), ["eye.L", "eye.R"])
        for eye in report["eyes"]:
            radii = np.array(eye["radii_m"])
            self.assertAlmostEqual(radii[0], 70 * 1.02 * self.PX, delta=2 * self.PX)
            self.assertAlmostEqual(radii[2], 60 * 1.02 * self.PX, delta=2 * self.PX)
            self.assertAlmostEqual(eye["iris_radius_px"], 42, delta=4)
            self.assertAlmostEqual(abs(eye["center_m"][0]), 95 * self.PX, delta=2 * self.PX)
            mesh, uv = read_obj_uv(Path(self.tmp.name) / "balls" / eye["mesh"])
            np.testing.assert_allclose(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0), 2 * radii, rtol=0.02)
            # The texture: the drawn iris in front, and eye-white on the side that the drawing cannot see.
            with Image.open(Path(self.tmp.name) / "balls" / eye["texture"]) as picture:
                texture = np.asarray(picture.convert("RGB")).astype(int)
            self.assertLess(np.abs(np.array(eye["white_rgb"]) - (240, 232, 220)).max(), 12)
            back = texture[texture.shape[0] // 2, 5]            # u = 0: the seam, at the back of the eyeball
            self.assertLess(np.abs(back - (240, 232, 220)).max(), 12, "not the lid line, not fur")
            front = texture[texture.shape[0] // 2, texture.shape[1] // 2]
            self.assertLess(front.max(), 60, "the pupil is at the front")
        anatomy = json.loads((Path(self.tmp.name) / "balls" / "anatomy.json").read_text())
        self.assertEqual(len(anatomy["parts"]), 2)
        self.assertLess(anatomy["parts"][0]["iris_fraction"], 0.75, "the iris is a part of the eyeball, not all of it")

    def test_an_eye_shell_follows_the_head_and_shows_the_drawing(self):
        """The default: no ball that stands out, but the patch of the head under the eye, with the drawn pixels."""
        from kumiki_blender.character_build import split_views as S
        from kumiki_blender.character_build.mesh import surface_samples
        from kumiki_blender.character_build.texture import render_textured
        yy, xx = np.mgrid[:1200, :1600]
        whole = ((xx - 895) / 70) ** 2 + ((yy - 500) / 60) ** 2 <= 1          # the character's left eye
        part = self.front.copy()
        part[..., 3] = np.where(whole, 255, 0)
        spec = json.loads((self.handoff / "spec.json").read_text())
        root, out = Path(self.tmp.name) / "part_shell", Path(self.tmp.name) / "shells"
        S.write_handoff(root, spec, {"front": part}, "eyes")
        report = self.eyes.eyeballs_from_part(root, self.head, out)
        self.assertEqual((report["shape"], [eye["id"] for eye in report["eyes"]]), ("shell", ["eye.L"]))
        mesh, uv = read_obj_uv(out / "eye.L.obj")
        # On the head: every vertex within the dome height above the ball's surface, none inside it.
        level = np.sqrt((mesh.vertices[:, 0] / .1) ** 2 + (mesh.vertices[:, 1] / .1) ** 2 + ((mesh.vertices[:, 2] - .16) / .16) ** 2)
        self.assertGreater(level.min(), 0.995)
        self.assertLess(level.max(), 1.06, "a low dome, not a ball that stands out")
        self.assertGreater(level.max(), 1.01)
        # Seen from the front it covers the drawn eye and nothing else, with the drawn pixels.
        samples = surface_samples(mesh, self.PX)[0]
        col, row, _ = camera("front").project(samples, self.PX, 1.0)
        covered = np.zeros((1200, 1600), bool)
        covered[row.astype(int), col.astype(int)] = True
        self.assertGreater((covered & whole).sum() / whole.sum(), 0.97)
        self.assertLess((covered & ~whole).sum() / whole.sum(), 0.2)
        with Image.open(out / "eye.L.png") as picture:
            texture = np.asarray(picture.convert("RGB"))
        rendered, mask = render_textured(mesh, uv, texture, "front", self.PX, 1.0)
        inner = whole & mask & (((xx - 895) / 60) ** 2 + ((yy - 500) / 50) ** 2 <= 1)
        difference = np.abs(rendered[inner] - self.front[..., :3][inner].astype(float)).mean()
        self.assertLess(difference, 12, "the eye from the front is the drawing")
        self.assertLess(rendered[500, 895].max(), 60, "the pupil")
        with_masks = self.eyes.eyeballs_from_part(root, self.head, Path(self.tmp.name) / "shells_masks",
                                                  source_handoff=self.handoff)
        self.assertEqual(with_masks["shape"], "shell", "also when it writes the masks for the other drawings")
        self.assertTrue(any((Path(self.tmp.name) / "shells_masks" / "within").iterdir()))
        with self.assertRaises(ValueError):
            self.eyes.eyeballs_from_part(root, self.head, Path(self.tmp.name) / "bad", shape="cube")

    def test_only_the_face_drawing_paints_around_the_eyes(self):
        def alpha(folder, name):
            with Image.open(self.out / folder / "raw" / f"{name}.png") as picture:
                return np.asarray(picture)[..., 3] > 127

        self.assertEqual(sorted(p.name for p in (self.out / "eyes" / "raw").iterdir()), ["front.png"])
        eyes_only = alpha("eyes", "front")
        self.assertTrue(eyes_only[500, 705] and not eyes_only[500, 800] and not eyes_only[500, 765])
        front, left, back = alpha("rest", "front"), alpha("rest", "left"), alpha("rest", "back")
        self.assertFalse(front[500, 705], "the iris disc is off the head texture")
        self.assertTrue(front[500, 768] and front[445, 705], "the white and the lid stay in the face drawing")
        self.assertFalse(left[505, 565], "the side drawing loses its own eye, drawn 55 px from the true place")
        self.assertFalse(left[505, 520] or left[505, 590])
        self.assertTrue(left[505, 880] and left[800, 800], "and keeps the rest of the head")
        self.assertTrue(back.all() == alpha("rest", "back").all() and back[500, 800], "the back drawing sees no eye")
        removed = {row["view"]: row["removed_px"] for row in self.report["removed"]}
        self.assertEqual(removed["back"], 0)
        self.assertGreater(removed["left"], removed["front"])


if __name__ == "__main__":
    unittest.main()
