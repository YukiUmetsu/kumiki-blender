"""Character source-anchored anatomy: pure geometry and strict authoring plan tests."""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.character_build.anatomy_geometry import (
    contour_loft, ocular_globe, swept_tube,
)
from kumiki_blender.character_build.anatomy_plan import (
    load_anatomy_plan, plan_anatomy, validate_anatomy_spec,
)


def _material(rgb):
    return {"rgba": [*rgb, 1], "roughness": .45, "metallic": 0}


def anatomy_fixture():
    color = _material([.7, .35, .2])
    return {
        "schema_version": 1, "character_id": "fixture",
        "parts": [
            {
                "id": "eye.L", "kind": "ocular_globe",
                "evidence": {"status": "artist-placed", "source_views": ["front", "front_right"],
                             "notes": "Synthetic registered-frame design anchors"},
                "attachment": {"mode": "bone", "armature": "CharacterRig", "bone": "head"},
                "center_m": [.036, -.063, .27], "forward_m": [0, -1, 0],
                "up_m": [0, 0, 1], "radii_m": [.014, .011, .017],
                "iris_fraction": .47, "pupil_fraction": .21,
                "radial_segments": 32, "latitudes": 20,
                "materials": {
                    "sclera": _material([.95, .93, .87]),
                    "iris": _material([.26, .38, .48]),
                    "pupil": _material([.025, .025, .03]),
                },
            },
            {
                "id": "upper_lid.L", "kind": "swept_tube",
                "evidence": {"status": "artist-placed", "source_views": ["front"],
                             "notes": "Explicit upper eyelid rim anchors"},
                "attachment": {"mode": "bone", "armature": "CharacterRig", "bone": "head"},
                "control_points_m": [
                    [.023, -.071, .275], [.028, -.075, .282], [.036, -.076, .286],
                    [.044, -.075, .282], [.05, -.07, .274],
                ],
                "radii_m": [[.0012, .0009], [.0016, .0012], [.0019, .0012],
                            [.0015, .001], [.001, .0009]],
                "radial_segments": 20, "subdivisions": 6, "reference_up": [0, 0, 1],
                "material": color,
            },
            {
                "id": "beak", "kind": "contour_loft",
                "evidence": {"status": "inferred", "source_views": ["front"],
                             "notes": "Demo only; not a source-verified Quill beak"},
                "attachment": {"mode": "separate"},
                "sections_m": [
                    [[.014, y, .251], [.011, y, .257], [0, y, .261],
                     [-.011, y, .257], [-.014, y, .251], [-.010, y, .244],
                     [0, y, .241], [.010, y, .244]]
                    for y in (-.068, -.079, -.092, -.102)
                ],
                "section_subdivisions": 4, "contour_subdivisions": 3,
                "material": _material([.87, .52, .14]),
            },
        ],
    }


class AnatomyGeometryTest(unittest.TestCase):
    def test_source_anchored_eye_three_separate_closed_3d_solids(self):
        part = anatomy_fixture()["parts"][0]
        meshes = ocular_globe(
            part["center_m"], part["forward_m"], part["up_m"],
            part["radii_m"], iris_fraction=part["iris_fraction"],
            pupil_fraction=part["pupil_fraction"], radial_segments=32, latitudes=20)
        self.assertEqual(set(meshes), {"sclera", "iris", "pupil"})
        for label, mesh in meshes.items():
            with self.subTest(mesh=label):
                self.assertEqual(mesh["stats"]["boundary_edges"], 0)
                self.assertEqual(mesh["stats"]["nonmanifold_edges"], 0)
                self.assertGreater(mesh["stats"]["signed_volume_m3"], 0)
                self.assertEqual(len(mesh["vertices"]), len(mesh["uvs"]))
                self.assertGreater(len(mesh["faces"]), 32)
                self.assertTrue(all(math.isfinite(v) for p in mesh["vertices"] for v in p))
        self.assertNotEqual(meshes["sclera"]["stats"]["vertices"],
                            meshes["iris"]["stats"]["vertices"])

    def test_curved_lid_tube_is_volumetric_with_exact_end_anchors(self):
        p = anatomy_fixture()["parts"][1]
        mesh = swept_tube(p["control_points_m"], p["radii_m"],
                          radial_segments=20, subdivisions=6, reference_up=p["reference_up"])
        self.assertEqual(mesh["stats"]["boundary_edges"], 0)
        self.assertGreater(mesh["stats"]["signed_volume_m3"], 0)
        self.assertEqual(mesh["stats"]["vertices"], 20*(4*6+1))

    def test_beak_loft_interpolates_authored_contours_without_welding_others(self):
        p = anatomy_fixture()["parts"][2]
        mesh = contour_loft(p["sections_m"], section_subdivisions=4,
                            contour_subdivisions=3)
        self.assertEqual(mesh["stats"]["boundary_edges"], 0)
        self.assertEqual(mesh["stats"]["nonmanifold_edges"], 0)
        self.assertGreater(mesh["stats"]["signed_volume_m3"], 0)
        self.assertEqual(mesh["stats"]["vertices"], 13*8*3)
        first_point = p["sections_m"][0][0]
        self.assertTrue(any(math.dist(first_point, v) < 1e-10 for v in mesh["vertices"]))

    def test_reject_bad_geometric_inputs(self):
        p = anatomy_fixture()["parts"][1]
        with self.assertRaises(ValueError):
            swept_tube(p["control_points_m"][:2], p["radii_m"][:2])
        with self.assertRaises(ValueError):
            contour_loft([p["control_points_m"]]*3)
        with self.assertRaisesRegex(ValueError, "fractions"):
            ocular_globe([0,0,0], [0,-1,0], [0,0,1], [.01,.01,.01],
                          iris_fraction=.2, pupil_fraction=.6)


class AnatomyPlanTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / "source.blend"
        self.source.write_bytes(b"BLENDER-anatomy-fixture")
        self.spec = self.root / "anatomy.json"
        self.data = anatomy_fixture()
        self.save()

    def save(self):
        self.spec.write_text(json.dumps(self.data), encoding="utf-8")

    def test_source_hashes_and_immutable_revision(self):
        self.assertGreater(validate_anatomy_spec(self.data), 1000)
        plan = plan_anatomy(self.source, self.spec, self.root / "out")
        self.assertEqual(plan["character_id"], "fixture")
        self.assertEqual(load_anatomy_plan(self.root / "out/anatomy-plan.json"), plan)
        with self.assertRaises(FileExistsError):
            plan_anatomy(self.source, self.spec, self.root / "out")
        self.source.write_bytes(b"BLENDER-changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            load_anatomy_plan(self.root / "out/anatomy-plan.json")

    def test_cannot_claim_unsupported_verified_evidence(self):
        self.data["parts"][1]["evidence"]["status"] = "registered-verified"
        self.data["parts"][1]["evidence"]["source_views"] = []
        with self.assertRaisesRegex(ValueError, "verification"):
            validate_anatomy_spec(self.data)
        self.data["parts"][1]["evidence"]["status"] = "inferred"
        self.data["parts"][1]["evidence"]["notes"] = ""
        with self.assertRaisesRegex(ValueError, "provenance"):
            validate_anatomy_spec(self.data)

    def test_reject_bad_axes_colors_collision_and_missing_stations(self):
        import copy
        for mutate, fragment in [
            (lambda d: d["parts"][0].update(up_m=[0, -1, 0]), "parallel"),
            (lambda d: d["parts"][0]["materials"]["iris"].update(rgba=[1, 1]), "rgba"),
            (lambda d: d["parts"][1].update(id="eye.L"), "Duplicate"),
            (lambda d: d["parts"][2].update(sections_m=[[0,0,0]]), "loft"),
            (lambda d: d["parts"][0]["attachment"].update(bone="../unsafe"), "identifier"),
        ]:
            with self.subTest(fragment=fragment):
                data = copy.deepcopy(self.data)
                mutate(data)
                with self.assertRaisesRegex(ValueError, fragment):
                    validate_anatomy_spec(data)

    def test_detects_mutated_plan_and_spec(self):
        plan_anatomy(self.source, self.spec, self.root / "out")
        plan_path = self.root / "out/anatomy-plan.json"
        plan = json.loads(plan_path.read_text())
        plan["parts"][0]["pupil_fraction"] = .3
        plan_path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "contents changed"):
            load_anatomy_plan(plan_path)
        self.data["parts"][0]["pupil_fraction"] = .22
        self.save()
        with self.assertRaisesRegex(ValueError, "source/spec changed"):
            load_anatomy_plan(plan_path)


if __name__ == "__main__":
    unittest.main()
