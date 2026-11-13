"""character_build: views, checks, hull, mesh IO, handoff, fit, projection, scatter, joints, assembly, CLI."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from character_build_fixtures import (HEIGHT_M, MARGINS, SKIN, ellipsoid_mesh, figure_points, glb_bytes, render_view,
                                      signed_volume, write_views)
from kumiki_blender.character_build import checks, views
from kumiki_blender.character_build.__main__ import main as cli
from kumiki_blender.character_build.assembly import load_assembly
from kumiki_blender.character_build.fit import fit_silhouettes, fold_faces, normalise
from kumiki_blender.character_build.handoff import build_handoff, compare, reference_masks
from kumiki_blender.character_build.hull import solidify
from kumiki_blender.character_build.mesh import Mesh, orient_faces, rasterize, read_mesh, write_mesh
from kumiki_blender.character_build.swap import align, parse_region, region_box
from kumiki_blender.character_build.texture import appearance_report, bake_texture, read_obj_uv
from kumiki_blender.character_build.project import mirror_missing, project_images, relief, subdivide
from kumiki_blender.character_build.rig import estimate_joints, validate_joints
from kumiki_blender.character_build.scatter import ScatterSpec, fur_mask, scatter_cards
from kumiki_blender.character_fidelity.schema import load_manifest


def statuses(report: dict) -> dict[str, str]:
    return {row["check"]: row["status"] for row in report["checks"]}


class SharedSet(unittest.TestCase):
    """One consistent synthetic set (8 views + clay) and its handoff, built once."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.good = cls.root / "good"
        write_views(cls.good)
        points, labels = figure_points()
        for name in ("front", "right"):
            render_view(points, labels, name, clay=True, margin=MARGINS[name]).save(cls.good / f"{name}_clay.png")
        cls.report = checks.check_folder(cls.good)
        cls.spec = build_handoff(cls.good, HEIGHT_M)
        cls.handoff = cls.good / "handoff"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()


class ViewTests(unittest.TestCase):
    def test_camera_conventions(self):
        front, right = views.camera("front"), views.camera("right")
        np.testing.assert_allclose(front.toward, [0, -1, 0], atol=1e-12)
        np.testing.assert_allclose(front.right, [1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(right.toward, [-1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(right.right, [0, -1, 0], atol=1e-12)  # the figure faces screen-right
        np.testing.assert_allclose(views.camera("three_quarter").toward, views.camera("front_right").toward)
        self.assertEqual({n: views.mirror_of(n) for n in ("front", "left", "front_left", "back_left")},
                         {"front": "back", "left": "right", "front_left": "back_right", "back_left": "front_right"})

    def test_aligned_frame_ignores_framing(self):
        points, labels = figure_points()
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.png", Path(tmp) / "b.png"
            render_view(points, labels, "front", margin=(40, 30)).save(a)
            render_view(points, labels, "front", margin=(400, 200), size=(1400, 2000)).save(b)
            va, vb = views.load(a, "front"), views.load(b, "front")
        self.assertGreater(checks.iou(va.mask, vb.mask), 0.98)
        rows = np.nonzero(va.mask.any(axis=1))[0]
        self.assertEqual((rows.min(), rows.max() + 1), (views.BASELINE - views.FIGURE_PX, views.BASELINE))

    def test_extra_object_hole_fill_and_ingest(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Image.new("RGB", (400, 600), (217, 217, 217))
            draw = ImageDraw.Draw(image)
            draw.ellipse((100, 100, 300, 500), fill=(120, 80, 40))
            draw.ellipse((180, 250, 220, 290), fill=(217, 217, 217))  # light eye: an enclosed hole
            draw.rectangle((10, 10, 60, 30), fill=(0, 0, 0))  # a label
            path = Path(tmp) / "front.png"
            image.save(path)
            view = views.load(path)
            self.assertTrue(any("extra object" in note for note in view.notes))
            self.assertTrue(view.raw_mask[270, 200], "enclosed holes are filled")
            (Path(tmp) / "src").mkdir()
            image.save(Path(tmp) / "src" / "x-1.png")
            image.save(Path(tmp) / "src" / "x-2.png")
            mapping = Path(tmp) / "map.json"
            mapping.write_text(json.dumps({"x-1.png": "front", "x-2.png": "three_quarter"}))
            views.ingest(Path(tmp) / "src", mapping, Path(tmp) / "out")
            self.assertEqual(sorted(p.name for p in (Path(tmp) / "out").iterdir()), ["front.png", "front_right.png"])
            mapping.write_text(json.dumps({"x-1.png": "front", "x-2.png": "front"}))
            with self.assertRaises(views.ViewError):
                views.ingest(Path(tmp) / "src", mapping, Path(tmp) / "out2")


class CheckTests(SharedSet):
    def test_consistent_set_has_no_failure(self):
        failed = [r for r in self.report["checks"] if r["status"] == "FAIL"]
        self.assertEqual(failed, [])
        for name in views.AZIMUTH:
            self.assertLess(self.report["hull"]["missing"][name], 0.03, name)

    def test_each_break_is_caught(self):
        bad = self.root / "bad"
        bad.mkdir()
        for item in self.good.glob("*.png"):
            (bad / item.name).write_bytes(item.read_bytes())
        render_view(*figure_points(widen=1.45), "back", margin=MARGINS["back"]).save(bad / "back.png")
        render_view(*figure_points(bag=3.2), "front_right", margin=MARGINS["front_right"]).save(bad / "front_right.png")
        Image.open(self.good / "front.png").save(bad / "front_clay.png")
        shadow = Image.open(self.good / "right.png")
        ImageDraw.Draw(shadow).rectangle((40, 1450, 300, 1500), fill=(40, 40, 40))
        shadow.save(bad / "right.png")
        result = statuses(checks.check_folder(bad))
        self.assertEqual(result["back is the mirror image of front"], "FAIL")
        self.assertEqual(result["front_clay is grey clay (no colour)"], "FAIL")
        self.assertEqual(result["right: one clean figure"], "FAIL")
        self.assertEqual(result["front_right agrees with the 3D shape of all views"], "FAIL")


class HullAndHandoffTests(SharedSet):
    def test_blockout_matches_views_and_points_outward(self):
        mesh = read_mesh(self.handoff / "blockout.obj")
        self.assertGreater(signed_volume(mesh), 0)
        report = compare(self.handoff, self.handoff / "blockout.obj")
        self.assertGreater(report["mean_iou"], 0.95)
        for row in report["views"]:  # the outline blur shortens thin tips (ears) by about 2%
            self.assertLess(abs(row["height_error"]), 0.025, row)

    def test_blockout_surface_is_smooth(self):
        """No voxel terraces: a vertex normal stays close to the mean normal of its neighbours."""
        mesh = read_mesh(self.handoff / "blockout.obj")
        normals = mesh.vertex_normals()
        mean = mesh.adjacency() @ normals
        mean /= np.maximum(np.linalg.norm(mean, axis=1), 1e-12)[:, None]
        rough = float((np.einsum("ij,ij->i", normals, mean) < np.cos(np.radians(20))).mean())
        self.assertLess(rough, 0.01)

    def test_spec_and_fidelity_manifest(self):
        self.assertEqual(sorted(self.spec["views"]), sorted(views.AZIMUTH))
        self.assertEqual(len(self.spec["bands"]), 21)
        manifest = load_manifest(self.handoff / "fidelity_manifest.json")
        self.assertFalse(manifest["views"][0]["alignment"]["verified"])
        self.assertTrue((self.handoff / "aligned" / "front_clay.png").is_file())
        with self.assertRaises(FileExistsError):
            build_handoff(self.good, HEIGHT_M)


class MeshTests(unittest.TestCase):
    def test_obj_ply_round_trip_and_glb(self):
        mesh = ellipsoid_mesh(subdivisions=2)
        mesh.colours = np.tile(np.array([[10, 200, 30]], np.uint8), (len(mesh.vertices), 1))
        with tempfile.TemporaryDirectory() as tmp:
            for suffix in (".obj", ".ply"):
                path = Path(tmp) / f"m{suffix}"
                write_mesh(mesh, path)
                back = read_mesh(path)
                np.testing.assert_allclose(back.vertices, mesh.vertices, atol=1e-5)
                np.testing.assert_array_equal(back.faces, mesh.faces)
                self.assertTrue(np.abs(back.colours.astype(int) - mesh.colours).max() <= 1)
            glb = Path(tmp) / "m.glb"
            for nested in (False, True):
                glb.write_bytes(glb_bytes(mesh, translation=(0.0, 0.5, 0.0), nested=nested))  # glTF Y up
                back = read_mesh(glb)
                np.testing.assert_allclose(back.vertices, mesh.vertices + [0, 0, 0.5], atol=1e-5)

    def test_rasterize_area(self):
        sphere = ellipsoid_mesh((0, 0, 0.5), (0.3, 0.3, 0.3), 4)
        mask, depth = rasterize(sphere, views.camera("front"), 1 / 1000)
        self.assertAlmostEqual(mask.sum() / (np.pi * 300 ** 2), 1.0, delta=0.02)
        self.assertAlmostEqual(np.nanmax(depth), 0.3, delta=0.002)


class FitProjectScatterTests(SharedSet):
    def test_fit_improves_an_ellipsoid(self):
        masks = reference_masks(self.handoff, 0.25)
        result = fit_silhouettes(ellipsoid_mesh(radii=(.3, .25, .5), subdivisions=3), masks, HEIGHT_M, iterations=40)
        start, best = result.history[0]["weighted_iou"], result.history[result.best_iteration]["weighted_iou"]
        self.assertGreater(best - start, 0.2)
        self.assertGreater(best, 0.9)
        self.assertEqual(result.folds, 0, "the flip guard and untangle leave no folded triangles")
        self.assertEqual(len(fold_faces(result.mesh, 0.0)), 0, "no crease sharper than 90 degrees")

    def test_orientation_search_finds_a_turned_generator_mesh(self):
        """A generator mesh comes turned (45 degrees from a 3/4 view), scaled and moved. Front and
        back silhouettes are mirror images, so the side views must settle a half turn."""
        blockout = read_mesh(self.handoff / "blockout.obj")
        masks = reference_masks(self.handoff, 0.25)
        for yaw in (47, 180):
            a = np.radians(-yaw)
            turn = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
            moved = Mesh(blockout.vertices @ turn.T * 3.0 + [1, 2, 3], blockout.faces)
            placed, info = normalise(moved, masks, HEIGHT_M, HEIGHT_M / 1000, 0.25)
            self.assertLess(abs((info["yaw_deg"] - yaw + 180) % 360 - 180), 4, info)
            self.assertAlmostEqual(placed.vertices[:, 2].max(), HEIGHT_M, places=6)

    def test_visibility_and_relief(self):
        image = np.full((views.CANVAS_H, views.CANVAS_W, 3), 100.0)
        sphere = ellipsoid_mesh((0, 0, .5), (.3, .3, .3), 3)
        hidden = ellipsoid_mesh((0, .5, .5), (.1, .1, .1), 2)
        both = Mesh(np.vstack([sphere.vertices, hidden.vertices]),
                    np.vstack([sphere.faces, hidden.faces + len(sphere.vertices)]))
        projection = project_images(both, {"front": image}, 1 / 1000)
        facing = (sphere.vertex_normals() @ [0, -1, 0]) > 0.05
        self.assertTrue((projection.weight[:len(sphere.vertices)][facing] > 0).all())
        self.assertFalse((projection.weight[len(sphere.vertices):] > 0).any())
        flat = np.full(len(sphere.vertices), 0.6)
        moved, report = relief(sphere, flat, np.ones(len(sphere.vertices), bool), 0.002)
        self.assertEqual(report["moved_fraction"], 0.0)
        np.testing.assert_allclose(moved.vertices, sphere.vertices, atol=1e-12)
        images, added = mirror_missing({"left": image})
        self.assertEqual(added, ["right"])

    def test_subdivide_keeps_a_closed_surface(self):
        mesh = subdivide(ellipsoid_mesh(subdivisions=1))
        edges = np.sort(np.concatenate([mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]]), axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        self.assertTrue((counts == 2).all())
        self.assertAlmostEqual(signed_volume(mesh) / signed_volume(ellipsoid_mesh(subdivisions=1)), 1, delta=0.05)

    def test_scatter_density_and_placement(self):
        sphere = ellipsoid_mesh((0, 0, .5), (.2, .2, .2), 3)
        spec = ScatterSpec(density_per_m2=2000, z_min_m=0.5, seed=3)
        cards, report = scatter_cards(sphere, spec)
        expected = 2000 * report["region_area_m2"]
        self.assertLess(abs(report["cards"] - expected), 4 * expected ** 0.5)
        bases = cards.vertices.reshape(report["cards"], -1, 3)[:, 0]
        self.assertTrue(np.all(bases[:, 2] > 0.46))  # face centres >= 0.5; a face spans about 3 cm
        self.assertTrue(np.allclose(np.linalg.norm(bases - [0, 0, .5], axis=1), 0.2, atol=0.01))
        again, _ = scatter_cards(sphere, spec)
        np.testing.assert_array_equal(cards.vertices, again.vertices)
        behind, report = scatter_cards(sphere, ScatterSpec(density_per_m2=2000, y_min_m=0.1, seed=3))
        bases = behind.vertices.reshape(report["cards"], -1, 3)[:, 0]
        self.assertTrue(np.all(bases[:, 1] > 0.07), "a y band selects the back of the figure only")


class GeneratorMeshTests(unittest.TestCase):
    """Repairs for image-to-3D output: winding, hollow shells, and region swaps between generations."""

    def test_orient_faces_repairs_random_winding(self):
        sphere = ellipsoid_mesh((0, 0, .5), (.3, .3, .3), 3)
        faces = sphere.faces.copy()
        flip = np.random.default_rng(0).random(len(faces)) < 0.5
        faces[flip] = faces[flip][:, ::-1]
        fixed, flipped = orient_faces(Mesh(sphere.vertices, faces))
        self.assertAlmostEqual(signed_volume(fixed), signed_volume(sphere), places=9)
        self.assertGreater(flipped, 0)

    def test_solidify_fills_a_hollow_shell(self):
        """TRELLIS.2 exports a closed outer wall with an inner wall just behind it."""
        outer = ellipsoid_mesh((0, 0, .1), (.08, .08, .08), 4)
        inner = ellipsoid_mesh((0, 0, .1), (.078, .078, .078), 4)
        shell = Mesh(np.vstack([outer.vertices, inner.vertices]),
                     np.vstack([outer.faces, inner.faces[:, ::-1] + len(outer.vertices)]))
        self.assertLess(signed_volume(shell), 0.1 * signed_volume(outer))
        solid = solidify(shell, 0.002)
        self.assertAlmostEqual(signed_volume(solid) / signed_volume(outer), 1.0, delta=0.08)

    def test_region_parsing(self):
        self.assertEqual(region_box(parse_region(["y>0.02", "z<0.3"]))[0][1], 0.02)
        with self.assertRaises(ValueError):
            parse_region(["tail>1"])
        with self.assertRaises(ValueError):
            region_box(parse_region(["y>0.2", "y<0.1"]))


class AlignmentTests(SharedSet):
    def test_similarity_icp_recovers_a_known_transform(self):
        """Two generations share the body; matching it outside the swap region lines them up.
        (A smooth ellipsoid is a poor test: ICP slides along it. The figure has ears and feet.)"""
        base = read_mesh(self.handoff / "blockout.obj")
        a = np.radians(10)
        turn = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        donor = Mesh(1.1 * base.vertices @ turn.T + [0.01, -0.02, 0.005], base.faces)
        aligned, info = align(donor, base, parse_region(["y>0.1"]))
        self.assertAlmostEqual(info["scale"], 1 / 1.1, delta=0.005)
        self.assertAlmostEqual(info["rotation_deg"], 10, delta=0.5)
        self.assertLess(float(np.median(np.linalg.norm(aligned.vertices - base.vertices, axis=1))), 0.002)
        # Matching only a shared part. Both meshes come from `place`, so they differ only a
        # little. The part must hold distinctive shapes (feet, a face): a round lower body alone
        # leaves the rotation loose (it settles at 3.2 of 5 degrees).
        b = np.radians(5)
        small = np.array([[np.cos(b), -np.sin(b), 0], [np.sin(b), np.cos(b), 0], [0, 0, 1]])
        donor = Mesh(1.02 * base.vertices @ small.T + [0.005, 0, 0], base.faces)
        aligned, info = align(donor, base, parse_region(["y>0.1"]), match=parse_region(["z<0.8"]))
        self.assertAlmostEqual(info["rotation_deg"], 5, delta=0.5)

    def test_placement_transform_reproduces_the_placed_mesh(self):
        """place writes yaw, scale and offset so the source file (and its texture) can be put in place."""
        blockout = read_mesh(self.handoff / "blockout.obj")
        a = np.radians(-30)
        turn = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        source = Mesh(blockout.vertices @ turn.T * 2.0 + [0.3, -0.1, 0.2], blockout.faces)
        placed, info = normalise(source, reference_masks(self.handoff, 0.25), HEIGHT_M, HEIGHT_M / 1000, 0.25)
        b = np.radians(info["yaw_deg"])
        rz = np.array([[np.cos(b), -np.sin(b), 0], [np.sin(b), np.cos(b), 0], [0, 0, 1]])
        rebuilt = info["scale"] * source.vertices @ rz.T + info["offset_m"]
        np.testing.assert_allclose(rebuilt, placed.vertices, atol=1e-9)


class TextureTests(SharedSet):
    def test_bake_takes_reference_colours(self):
        """A planar front UV on the blockout: texels must carry the figure's skin colour."""
        mesh = read_mesh(self.handoff / "blockout.obj")
        low, high = mesh.bounds()
        uv = (mesh.vertices[:, [0, 2]] - low[[0, 2]]) / (high[[0, 2]] - low[[0, 2]])
        corner_uv = uv[mesh.faces] * 0.98 + 0.01
        image, report = bake_texture(self.handoff, mesh, corner_uv, size=256)
        self.assertGreater(report["seen_fraction"], 0.8)
        self.assertEqual(set(report["per_view"]), set(self.spec["views"]))
        centre = image[96:160, 96:160].reshape(-1, 3).astype(int)
        distance = np.abs(centre - np.array(SKIN)).sum(axis=1)
        self.assertGreater(float((distance < 40).mean()), 0.7, "the belly is skin-coloured")

    def test_appearance_of_a_baked_texture(self):
        """Baked from the views and rendered back from the front camera, the colours agree."""
        mesh = read_mesh(self.handoff / "blockout.obj")
        low, high = mesh.bounds()
        uv = (mesh.vertices[:, [0, 2]] - low[[0, 2]]) / (high[[0, 2]] - low[[0, 2]])
        corner_uv = uv[mesh.faces] * 0.98 + 0.01
        image, _ = bake_texture(self.handoff, mesh, corner_uv, size=512)
        report = appearance_report(self.handoff, mesh, corner_uv, image)
        front = next(r for r in report["views"] if r["view"] == "front")
        self.assertLess(front["median_delta_e"], 5)
        self.assertGreater(front["within_10"], 0.8)

    def test_mirror_missing_paints_the_unseen_side(self):
        """Without the right views, the right side is unseen; mirrored left views paint it."""
        mesh = read_mesh(self.handoff / "blockout.obj")
        low, high = mesh.bounds()
        uv = (mesh.vertices[:, [1, 2]] - low[[1, 2]]) / (high[[1, 2]] - low[[1, 2]])
        corner_uv = uv[mesh.faces] * 0.98 + 0.01
        _, plain = bake_texture(self.handoff, mesh, corner_uv, size=128, views=["left"])
        _, mirrored = bake_texture(self.handoff, mesh, corner_uv, size=128, views=["left"], mirror_missing=True)
        self.assertEqual(mirrored["mirrored_views"], ["right"])
        self.assertGreater(mirrored["seen_fraction"], plain["seen_fraction"] + 0.2)

    def test_obj_uv_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "q.obj"
            path.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nf 1/1 2/2 3/3 4/4\n")
            mesh, corner_uv = read_obj_uv(path)
        self.assertEqual(mesh.faces.shape, (2, 3))
        np.testing.assert_allclose(corner_uv[1], [[0, 0], [1, 1], [0, 1]])
        np.testing.assert_allclose(mesh.vertices[2], [1, 0, 1])  # OBJ Y up -> Blender Z up


class RepaintTests(SharedSet):
    def test_free_cameras(self):
        from kumiki_blender.character_build.views import camera
        level = camera("az-45_el0")
        np.testing.assert_allclose(level.toward, views.camera("front_right").toward, atol=1e-12)
        above = camera("az0_el35")
        points = np.array([[0, 0, HEIGHT_M / 2], [0, -0.1, HEIGHT_M / 2]])
        _, row, _ = above.project(points, HEIGHT_M / 1000)
        self.assertAlmostEqual(row[0], views.BASELINE - views.FIGURE_PX / 2, places=6)  # the pivot stays put
        self.assertGreater(row[1], row[0], "seen from above, a point nearer the camera sits lower")
        with self.assertRaises(views.ViewError):
            camera("az0_el95")

    def test_render_round_trip_through_bake(self):
        """A render baked back as an extra view puts each colour where it came from."""
        from kumiki_blender.character_build.repaint import describe, nearest_drawing, render_condition
        mesh = read_mesh(self.handoff / "blockout.obj")
        low, high = mesh.bounds()
        uv = (mesh.vertices[:, [0, 2]] - low[[0, 2]]) / (high[[0, 2]] - low[[0, 2]])
        corner_uv = uv[mesh.faces] * 0.98 + 0.01
        texture = np.zeros((256, 256, 3), np.uint8)
        texture[:, :128] = (200, 40, 40)   # one half red, one half blue in UV (left/right of the figure)
        texture[:, 128:] = (40, 40, 200)
        picture, mask, mapping = render_condition(mesh, corner_uv, texture, "az-45_el30", HEIGHT_M / 1000, size=256)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paint.png"
            Image.fromarray(np.dstack([np.asarray(picture), np.where(mask, 255, 0).astype(np.uint8)])).save(path)
            baked, report = bake_texture(self.handoff, mesh, corner_uv, size=256, views=[],
                                         extra=[{"name": "az-45_el30", "image": str(path), "map": mapping,
                                                 "weight": 1.0}])
        self.assertGreater(report["per_view"]["az-45_el30"], 0.2)
        seen = baked.reshape(-1, 3)
        reds = (seen[:, 0] > 150) & (seen[:, 2] < 90)
        self.assertGreater(reds.mean(), 0.2)
        np.testing.assert_array_less(np.abs(baked[60:200, 40:100].astype(int) - [200, 40, 40]).max(axis=2).mean(), 25)
        self.assertIn("front-right", describe("az-45_el0"))
        self.assertIn("above", describe("az0_el35"))
        self.assertEqual(nearest_drawing("az170_el20", ["front", "back", "left"]), "back")

    def test_repaints_do_not_outvote_a_drawing(self):
        """Extra images share one source: three that agree must not overrule the drawing they face."""
        from kumiki_blender.character_build.repaint import render_condition
        mesh = read_mesh(self.handoff / "blockout.obj")
        low, high = mesh.bounds()
        uv = (mesh.vertices[:, [0, 2]] - low[[0, 2]]) / (high[[0, 2]] - low[[0, 2]])
        corner_uv = uv[mesh.faces] * 0.98 + 0.01
        texture = np.full((64, 64, 3), 128, np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            extra = []
            for view in ("az20_el0", "az-20_el0", "az0_el20"):
                _, mask, mapping = render_condition(mesh, corner_uv, texture, view, HEIGHT_M / 1000, size=256)
                green = np.zeros((256, 256, 4), np.uint8)
                green[..., 1], green[..., 3] = 200, np.where(mask, 255, 0)
                Image.fromarray(green).save(Path(tmp) / f"{view}.png")
                extra.append({"name": view, "image": str(Path(tmp) / f"{view}.png"), "map": mapping, "weight": 1.0})
            _, report = bake_texture(self.handoff, mesh, corner_uv, size=128, power=4, views=["front"], extra=extra)
        self.assertGreater(report["owned"]["front"], max(report["owned"][v] for v in report["owned"] if v != "front"))

    def test_tone_matching_restores_the_render_brightness(self):
        from kumiki_blender.character_build.checks import lab
        from kumiki_blender.character_build.repaint import match_tone
        rng = np.random.default_rng(0)
        render = np.clip(rng.normal((170, 130, 100), 12, (96, 96, 3)), 0, 255).astype(np.uint8)
        darker = (render * 0.8).astype(np.uint8)            # the editor's own shading
        mask = np.zeros((96, 96), bool)
        mask[10:86, 10:86] = True
        matched = match_tone(darker, render, mask)
        before = abs(lab(darker[mask]).mean(0)[0] - lab(render[mask]).mean(0)[0])
        after = abs(lab(matched[mask]).mean(0)[0] - lab(render[mask]).mean(0)[0])
        self.assertGreater(before, 5)
        self.assertLess(after, 0.5)

    def test_screen_drops_a_repaint_that_contradicts_the_drawings(self):
        from kumiki_blender.character_build.repaint import mirror_twin, render_condition, screen_extras
        mesh = read_mesh(self.handoff / "blockout.obj")
        low, high = mesh.bounds()
        uv = (mesh.vertices[:, [0, 2]] - low[[0, 2]]) / (high[[0, 2]] - low[[0, 2]])
        # Front-facing faces in the left half of the UV square, back-facing ones in the right
        # half: a planar projection alone would give both sides the same texels.
        corners = mesh.vertices[mesh.faces]
        back = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])[:, 1] > 0
        corner_uv = uv[mesh.faces] * [0.49, 0.98] + [0.005, 0.01]
        corner_uv[back, :, 0] += 0.5
        drawn, _ = bake_texture(self.handoff, mesh, corner_uv, size=256, power=4)
        with tempfile.TemporaryDirectory() as tmp:
            extra = []
            for view, tint in (("az30_el20", (1.0, 1.0, 1.0)), ("az-30_el20", (1.0, 0.7, 0.7))):
                picture, mask, mapping = render_condition(mesh, corner_uv, drawn, view, HEIGHT_M / 1000, size=256)
                # The second repaint drew something else: close enough in colour to survive the
                # agreement check, different enough to show in the drawn views.
                rgb = np.clip(np.asarray(picture) * np.array(tint), 0, 255).astype(np.uint8)
                Image.fromarray(np.dstack([rgb, np.where(mask, 255, 0).astype(np.uint8)])).save(Path(tmp) / f"{view}.png")
                extra.append({"name": view, "image": str(Path(tmp) / f"{view}.png"), "map": mapping, "weight": 3.0})
            # Agreement off: test the screen alone (in the fixture the agreement check already catches this one).
            kept, rows = screen_extras(self.handoff, mesh, corner_uv, extra, size=256, power=4, agree_delta_e=0)
        self.assertEqual([e["name"] for e in kept], ["az30_el20"], rows)
        self.assertEqual(mirror_twin("az45_el0"), "az-45_el0")
        self.assertEqual(mirror_twin("az180_el35"), "az180_el35")
        self.assertIsNone(mirror_twin("front"))

    def test_workflow_starts_from_the_render(self):
        from kumiki_blender.character_build.repaint import klein_workflow
        workflow = klein_workflow("r.png", "s.png", "p", size=768, steps=20, seed=1, denoise=0.6)
        sampler = workflow["20"]["inputs"]
        self.assertEqual(sampler["latent_image"], ["6", 0])  # the render's latent
        self.assertEqual(sampler["sigmas"], ["19", 1])       # the low-noise part of the schedule
        self.assertEqual(workflow["19"]["inputs"]["denoise"], 0.6)


    def test_batch_resumes_without_repainting_finished_views(self):
        from kumiki_blender.character_build import repaint
        mesh = read_mesh(self.handoff / "blockout.obj")
        low, high = mesh.bounds()
        uv = (mesh.vertices[:, [0, 2]] - low[[0, 2]]) / (high[[0, 2]] - low[[0, 2]])
        corner_uv = uv[mesh.faces] * 0.98 + 0.01
        texture = np.full((64, 64, 3), 150, np.uint8)

        class Client:
            uploads = []

            def upload_image(self, path):
                self.uploads.append(Path(path).name)
                return Path(path).name

        def fake_comfy(client, workflow):
            render = workflow["4"]["inputs"]["image"]   # echo the render back: a perfect repaint
            return (out / render).read_bytes()

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            original = repaint.run_comfy
            repaint.run_comfy = fake_comfy
            try:
                run = lambda views, image: repaint.repaint_views(self.handoff, mesh, corner_uv, image, views, out,
                                                                 Client(), size=128)
                run(["az45_el0"], texture)
                Client.uploads.clear()
                manifest = run(["az45_el0", "az-45_el0"], texture)
                self.assertEqual([r["reused"] for r in manifest["views"]], [True, False])
                self.assertEqual(Client.uploads, ["az-45_el0_render.png", "az-45_el0_style.png"])
                self.assertTrue(all(r["accepted"] for r in manifest["views"]))
                changed = run(["az45_el0"], np.full_like(texture, 90))   # a new texture: the old repaint is stale
                self.assertEqual([r["reused"] for r in changed["views"]], [False])
            finally:
                repaint.run_comfy = original


def two_sided_uv(mesh):
    """Front-facing faces in the left half of the UV square, back-facing ones in the right half."""
    low, high = mesh.bounds()
    uv = (mesh.vertices[:, [0, 2]] - low[[0, 2]]) / (high[[0, 2]] - low[[0, 2]])
    corners = mesh.vertices[mesh.faces]
    back = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])[:, 1] > 0
    corner_uv = uv[mesh.faces] * [0.49, 0.98] + [0.005, 0.01]
    corner_uv[back, :, 0] += 0.5
    return corner_uv


class SeamTests(SharedSet):
    """seams.py: one source per region, objections, leveling, and the conflict report."""

    def plane(self, quality, colours, side=40, spacing=0.002):
        """A flat square of ``side`` x ``side`` texels that faces the front camera, as regions."""
        from kumiki_blender.character_build import seams
        x, z = np.meshgrid(np.arange(side) * spacing, np.arange(side) * spacing)
        points = np.column_stack([x.ravel(), np.zeros(side * side), z.ravel()])
        normals = np.tile([0.0, -1.0, 0.0], (side * side, 1))
        regions = seams.build_regions(points, normals, sigma=3 * spacing, spacing=spacing)
        regions.add_sources(np.asarray(quality, dtype=np.float32), np.asarray(colours, dtype=np.uint8))
        return regions, x.ravel() / spacing, z.ravel() / spacing

    @staticmethod
    def ball():
        """Dense points on a sphere of radius 10 cm (about 25 per region cell, as texels are)."""
        from kumiki_blender.character_build.mesh import surface_samples
        return surface_samples(ellipsoid_mesh(centre=(0, 0, 0), radii=(.1, .1, .1), subdivisions=3), 0.002)[0]

    def test_bumps_do_not_change_the_facing(self):
        from kumiki_blender.character_build import seams
        points = self.ball()
        true = points / np.linalg.norm(points, axis=1)[:, None]
        rng = np.random.default_rng(0)
        bumpy = true + rng.normal(0, 0.6, true.shape)   # fur: the normal swings by tens of degrees
        bumpy /= np.linalg.norm(bumpy, axis=1)[:, None]
        regions = seams.build_regions(points, bumpy, sigma=0.03, spacing=0.01)
        raw = np.degrees(np.arccos(np.clip((bumpy * true).sum(axis=1), -1, 1))).mean()
        nodes = regions.centre / np.linalg.norm(regions.centre, axis=1)[:, None]
        smooth = np.degrees(np.arccos(np.clip((regions.normal * nodes).sum(axis=1), -1, 1))).mean()
        self.assertGreater(raw, 30)
        self.assertLess(smooth, 12)

    def test_a_part_in_the_wrong_place_is_not_painted(self):
        """Source 0 shows a red patch where the witness shows green and no red anywhere near."""
        from kumiki_blender.character_build import seams
        from kumiki_blender.character_build.checks import lab
        side = 40
        green, red = (70, 120, 60), (170, 40, 50)
        col, row = np.meshgrid(np.arange(side), np.arange(side))
        patch = ((col - 20) ** 2 + (row - 20) ** 2 <= 9).ravel()
        colours = np.tile(np.array(green, dtype=np.uint8), (2, side * side, 1))
        colours[0, patch] = red
        regions, x, z = self.plane([np.full(side * side, 0.9), np.full(side * side, 0.5)], colours)
        node_col, node_row = np.round(regions.centre[:, 0] / 0.002).astype(int), np.round(regions.centre[:, 2] / 0.002).astype(int)

        def picture(image):
            weighted = lab(image.reshape(-1, 3)).reshape(side, side, 3)
            weighted[..., 0] *= seams.LIGHTNESS_WEIGHT
            return weighted

        seen_by = [seams.Testimony(picture(colours[v].reshape(side, side, 3)), np.ones((side, side), bool),
                                   node_col, node_row, np.ones(len(regions.texels))) for v in range(2)]
        here, nearby = seams.slack_match(np.stack([picture(colours[v].reshape(side, side, 3)).reshape(-1, 3)[
            node_row * side + node_col] for v in range(2)]), seen_by[0], 5)
        on_patch = ((node_col - 20) ** 2 + (node_row - 20) ** 2 <= 1)
        self.assertGreater(here[1, on_patch].min(), 20, "green claimed where the witness shows red")
        self.assertLess(nearby[1, on_patch].max(), 5, "but the witness shows green close by: no objection")
        chosen = seams.choose_sources(regions, np.array([True, True]), seen_by, slack_cells=5, min_region_nodes=1)
        self.assertTrue((regions.label[on_patch] == 1).all(), "the witness paints the patch")
        self.assertTrue(chosen["misplaced"][on_patch].all())
        far = (node_col - 20) ** 2 + (node_row - 20) ** 2 > 100
        self.assertTrue((regions.label[far] == 0).all(), "the better source keeps the rest")
        self.assertFalse(chosen["misplaced"][far].any())

    def test_leveling_makes_two_tones_meet(self):
        """Left half painted by a source that is 30 units darker: the offsets close the gap at the seam."""
        from kumiki_blender.character_build import seams
        side = 40
        col = np.tile(np.arange(side), side)
        quality = [np.where(col < 20, 0.9, 0.3), np.where(col < 20, 0.3, 0.9)]
        colours = np.stack([np.full((side * side, 3), 120), np.full((side * side, 3), 150)])
        regions, _, _ = self.plane(quality, colours)
        seams.choose_sources(regions, np.array([True, True]), None, min_region_nodes=1)
        offset, report = seams.level_seams(regions, reach=0.03)
        self.assertGreater(report["seam_delta_e_before"], 8)
        self.assertLess(report["seam_delta_e_after"], 0.3 * report["seam_delta_e_before"])
        node_col = regions.centre[:, 0] / 0.002
        self.assertGreater(offset[node_col < 19].mean(), 0)    # the dark side rises, the light side falls
        self.assertLess(offset[node_col > 21].mean(), 0)
        self.assertLess(np.abs(offset[(node_col < 3) | (node_col > 36)]).mean(),
                        0.6 * np.abs(offset[(node_col > 17) & (node_col < 23)]).mean(), "the offset fades from the seam")

    def test_views_that_face_a_patch(self):
        from kumiki_blender.character_build import seams
        points = self.ball()
        regions = seams.build_regions(points, points / np.linalg.norm(points, axis=1)[:, None], sigma=0.03, spacing=0.01)
        left_side = regions.normal[:, 0] > 0.9     # the character's left is +X: camera azimuth +90
        found = seams.suggest_views(regions, left_side, min_nodes=3)
        self.assertEqual(found[0]["camera"], "az90_el0")
        self.assertEqual(seams.suggest_views(regions, np.zeros(len(left_side), bool)), [])

    def test_bake_reports_conflicts_with_pictures(self):
        import shutil
        from kumiki_blender.character_build.texture import conflict_advice
        mesh = read_mesh(self.handoff / "blockout.obj")
        corner_uv = two_sided_uv(mesh)
        with tempfile.TemporaryDirectory() as tmp:
            darker = Path(tmp) / "handoff"
            shutil.copytree(self.handoff, darker)
            for name in ("left", "right", "front_left", "front_right", "back_left", "back_right"):
                path = darker / "raw" / f"{name}.png"
                pixels = np.asarray(Image.open(path).convert("RGBA")).copy()
                pixels[..., :3] = (pixels[..., :3] * 0.8).astype(np.uint8)   # these drawings are darker in tone
                Image.fromarray(pixels).save(path)
            plain, off = bake_texture(darker, mesh, corner_uv, size=256, level=False)
            leveled, report = bake_texture(darker, mesh, corner_uv, size=256, diagnostics=Path(tmp) / "d")
            self.assertEqual({p.name for p in (Path(tmp) / "d").iterdir()}, {"owners.png", "conflicts.png", "turnaround.png"})
        self.assertNotIn("seam_delta_e_after", off)
        self.assertGreater(report["seam_delta_e_before"], 4)
        self.assertLess(report["seam_delta_e_after"], 0.6 * report["seam_delta_e_before"])
        self.assertFalse(np.array_equal(plain, leveled))
        found = report["conflicts"]
        self.assertEqual(set(found["surface"]), {"misplaced_part", "contested_feature", "tone_seam", "unseen"})
        self.assertLessEqual(sum(found["surface"].values()), 1.0)
        self.assertGreater(found["surface"]["tone_seam"], 0.01)
        self.assertTrue(found["pairs"] and all(pair["overlap"] > 0 for pair in found["pairs"]))
        advice = conflict_advice(report)
        self.assertTrue(any("Tone seams" in line for line in advice), advice)

    def test_a_repaint_counts_only_where_its_outline_agrees(self):
        from kumiki_blender.character_build.repaint import BACKGROUND, alignment
        render_mask = np.zeros((200, 200), bool)
        render_mask[40:160, 40:110] = True     # body
        render_mask[60:120, 110:170] = True    # tail
        picture = np.full((200, 200, 3), BACKGROUND, np.uint8)
        picture[40:160, 40:110] = (150, 100, 80)
        picture[100:160, 110:170] = (150, 100, 80)   # the editor drew the tail 40 px lower
        iou, trusted = alignment(Image.fromarray(picture), render_mask)
        self.assertLess(iou, 0.9)
        self.assertTrue(trusted[60:140, 45:80].all(), "the body far from the tail is kept")
        self.assertFalse(trusted[60:160, 112:170].any(), "nothing of the tail zone is kept")
        self.assertFalse((trusted & ~render_mask).any())


class ReconcileTests(unittest.TestCase):
    def test_a_part_without_room_in_the_opposite_view_is_cut(self):
        """The front shows a lobe beside the body; the back has no room for it. A thin offset stays."""
        from kumiki_blender.character_build.reconcile import parts_without_room, trim_view
        from kumiki_blender.character_build.views import CX, load

        def draw(lobe: bool, shift: int = 0) -> Image.Image:
            image = Image.new("RGB", (400, 600), (217, 217, 217))
            pen = ImageDraw.Draw(image)
            pen.rectangle((140 + shift, 60, 260 + shift, 560), fill=(150, 100, 80))   # body; shift: a small disagreement
            if lobe:
                pen.ellipse((30, 100, 150, 260), fill=(110, 80, 60))                  # a tail lobe on the image left
            return image

        with tempfile.TemporaryDirectory() as tmp:
            draw(True).save(Path(tmp) / "front.png")
            draw(False, shift=4).save(Path(tmp) / "back.png")
            front, back = load(Path(tmp) / "front.png"), load(Path(tmp) / "back.png")
        # Each view is centred on its own outline; the lobe moved the front body off centre.
        self.assertGreater(abs(np.nonzero(front.mask[600])[0].mean() - np.nonzero(back.mask[600])[0].mean()), 30)
        remove = parts_without_room(front.mask, back.mask)
        self.assertGreater(remove.sum(), 0.8 * (front.mask[:, :CX - 150]).sum(), "most of the lobe")
        self.assertFalse(remove[:, CX:].any(), "nothing on the side without a lobe")
        image, share = trim_view(front, remove)
        self.assertEqual(image.mode, "RGBA")
        self.assertGreater(share, 0.1)
        alpha = np.asarray(image)[..., 3]
        self.assertEqual(alpha[180, 60], 0, "the lobe is transparent")
        self.assertEqual(alpha[300, 200], 255, "the body is kept")
        self.assertFalse(parts_without_room(front.mask, front.mask[:, ::-1] | front.mask).any())


class FurMaskTests(unittest.TestCase):
    def test_clothing_and_empty_texels_get_no_fur(self):
        texture = np.zeros((40, 40, 3), np.uint8)
        texture[:, :20] = (160, 115, 85)    # fur
        texture[:, 20:30] = (70, 95, 50)    # vest green
        mask = fur_mask(texture, [(70, 95, 50)], grow_px=0)
        self.assertTrue((mask[:, :20] == 255).all())
        self.assertTrue((mask[:, 20:30] == 0).all())
        self.assertTrue((mask[:, 30:] == 0).all(), "unpainted texels between UV islands")


class JointsAndAssemblyTests(SharedSet):
    def test_joints_follow_the_body_midline(self):
        mesh = read_mesh(self.handoff / "blockout.obj")
        joints = estimate_joints(mesh, HEIGHT_M)
        mid = joints["midline_x_m"]
        self.assertGreater(mid, 0.02, "the bag on the right side moves the frame centre off the body")
        left, right = joints["joints"]["knee.L"][0] - mid, mid - joints["joints"]["knee.R"][0]
        self.assertAlmostEqual(left, right, delta=0.015)
        validate_joints(joints)
        broken = json.loads(json.dumps(joints))
        broken["bones"][1]["parent"] = "nothing"
        with self.assertRaises(ValueError):
            validate_joints(broken)

    def test_assembly_fits_anchors_and_rejects_bad_plans(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_mesh(ellipsoid_mesh(), root / "body.ply")
            write_mesh(ellipsoid_mesh(radii=(.1, .1, .1), subdivisions=1), root / "hat.obj")
            corners = ([0, 0, 0], [.1, 0, 0], [0, .1, 0], [0, 0, .1])
            anchors = [{"part": p, "target": [0.5 * c + t for c, t in zip(p, (0, 0, 1))]} for p in corners]
            plan = {"schema_version": 1, "body": "body.ply", "parts": [{"id": "hat", "file": "hat.obj", "anchors": anchors}]}
            (root / "plan.json").write_text(json.dumps(plan))
            loaded = load_assembly(root / "plan.json")
            self.assertAlmostEqual(loaded["parts"][0]["scale"], 0.5, places=6)
            self.assertAlmostEqual(loaded["parts"][0]["matrix"][2][3], 1.0, places=6)
            anchors[0]["target"] = [0.3, 0.3, 0.3]
            (root / "plan.json").write_text(json.dumps(plan))
            with self.assertRaises(ValueError):
                load_assembly(root / "plan.json")
            plan["parts"][0]["file"] = "../hat.obj"
            (root / "plan.json").write_text(json.dumps(plan))
            with self.assertRaises(ValueError):
                load_assembly(root / "plan.json")


class CliTests(SharedSet):
    def test_face_shapes_pair_with_the_head_of_their_side(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "front.png").write_bytes((self.good / "front.png").read_bytes())
            head = Image.open(self.good / "front.png").crop((0, 0, 1024, 560))
            head.save(folder / "head_front.png")
            head.save(folder / "head_right.png")
            head.save(folder / "expr_smile.png")
            head.crop((150, 60, 1024, 560)).resize((1024, 560)).save(folder / "expr_smile_right.png")
            result = statuses(checks.check_folder(folder))
        self.assertEqual(result["expr_smile keeps the framing of head_front"], "PASS")
        self.assertEqual(result["expr_smile_right keeps the framing of head_right"], "FAIL")
        self.assertNotIn("expr_smile_right keeps the framing of head_front", result)

    def test_compare_refuses_an_existing_report(self):
        out = self.root / "compare-once"
        compare(self.handoff, self.handoff / "blockout.obj", output=out)
        with self.assertRaises(FileExistsError):
            compare(self.handoff, self.handoff / "blockout.obj", output=out)

    def test_place_rejects_duplicate_file_names(self):
        first, second = self.root / "a" / "mesh.obj", self.root / "b" / "mesh.obj"
        for path in (first, second):
            path.parent.mkdir(exist_ok=True)
            path.write_bytes((self.handoff / "blockout.obj").read_bytes())
        code = cli(["place", str(self.handoff), str(first), str(second), "--output", str(self.root / "ranked")])
        self.assertEqual(code, 2)
        self.assertFalse((self.root / "ranked").exists())

    def test_outputs_are_new_revisions(self):
        out = self.root / "fitted.ply"
        out.write_text("existing")
        code = cli(["fit", str(self.handoff), str(self.handoff / "blockout.obj"), "--output", str(out)])
        self.assertEqual(code, 2)
        self.assertEqual(out.read_text(), "existing")


if __name__ == "__main__":
    unittest.main()
