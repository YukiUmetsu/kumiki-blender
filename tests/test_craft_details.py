"""Hard-surface craftsmanship primitives and compiled asset integration."""

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from kumiki_blender.architecture_geometry import audit_mesh, world_vertices
from kumiki_blender.asset_spec import AssetSpec, load_asset_spec
from kumiki_blender.composer import plan_assembly
from kumiki_blender.compiled_plan import (CompiledPlanError, _canonical,
                                         load_compiled_plan, write_compiled_plan)
from kumiki_blender.craft_details import (detail_frame, fastener_head_mesh,
                                         mitered_panel_rails, rounded_box_mesh)


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/precision-control-enclosure-v2.json"


class CraftDetailsTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))

    def validated(self):
        return AssetSpec.model_validate_json(json.dumps(self.raw))

    def test_rounded_box_has_real_fillets_and_welded_closed_sides(self):
        dimensions = (0.56, 0.16, 0.30)
        radius = 0.024
        mesh = rounded_box_mesh(dimensions, radius, corner_segments=7)
        report = audit_mesh(mesh)
        self.assertGreater(report.vertices, 1000)
        self.assertEqual(report.boundary_edges, 0)
        self.assertEqual(report.nonmanifold_edges, 0)
        self.assertGreater(report.volume_m3, 0)
        for axis, size in enumerate(dimensions):
            self.assertAlmostEqual(max(vertex[axis] for vertex in mesh.vertices), size / 2)
            self.assertAlmostEqual(min(vertex[axis] for vertex in mesh.vertices), -size / 2)
        # A fillet removes corner volume rather than returning the plain box.
        self.assertLess(report.volume_m3, math.prod(dimensions))
        self.assertGreater(report.volume_m3, math.prod(dimensions) * 0.95)

    def test_real_panel_opening_and_four_manifold_miter_rails(self):
        rails = mitered_panel_rails((0.48, 0.25), 0.015, 0.009)
        self.assertEqual([name for name, _ in rails], ["top", "right", "bottom", "left"])
        volume = sum(audit_mesh(mesh).volume_m3 for _, mesh in rails)
        expected = (0.48 * 0.25 - 0.45 * 0.22) * 0.009
        self.assertAlmostEqual(volume, expected)
        for _, mesh in rails:
            report = audit_mesh(mesh)
            self.assertEqual(report.boundary_edges, 0)
            self.assertEqual(report.nonmanifold_edges, 0)

    def test_hex_and_button_heads_are_different_proper_solids(self):
        hex_head = fastener_head_mesh("hex", 0.004, 0.003)
        button = fastener_head_mesh("button", 0.004, 0.003)
        self.assertEqual(len(hex_head.vertices), 18)
        self.assertGreater(len(button.vertices), 100)
        for head in (hex_head, button):
            self.assertEqual(audit_mesh(head).boundary_edges, 0)
            self.assertGreater(audit_mesh(head).volume_m3, 0)

    def test_surface_frames_are_orthonormal_and_right_handed(self):
        for direction in ("+x", "-x", "+y", "-y", "+z", "-z"):
            with self.subTest(direction=direction):
                x, y, z = detail_frame(direction)
                cross = (x[1]*y[2]-x[2]*y[1],
                         x[2]*y[0]-x[0]*y[2],
                         x[0]*y[1]-x[1]*y[0])
                self.assertEqual(cross, z)

    def test_sample_expands_to_named_audited_product_details(self):
        spec = load_asset_spec(EXAMPLE)
        plan = plan_assembly(spec)
        self.assertEqual(len(plan.instances), 23)
        self.assertEqual(dict(plan.named_parts), {
            "solid.rounded-box": 2, "detail.panel-frame": 4,
            "detail.linear-fasteners": 14, "mechanical.rod": 3,
        })
        self.assertEqual(len({part.name for part in plan.instances}), 23)
        self.assertEqual({part.material for part in plan.instances},
                         {"housing", "panel", "trim", "screws", "controls"})
        for part in plan.instances:
            self.assertEqual(audit_mesh(part.mesh).boundary_edges, 0, part.name)
        # The left/right front-facing fasteners actually protrude toward -Y.
        fastener = next(p for p in plan.instances if p.name == "top-hex-screws-000")
        self.assertLess(min(p[1] for p in world_vertices(fastener)), -0.102)

    def test_bounded_detail_inputs_reject_invalid_dimensions_and_spacing(self):
        mutations = (
            (0, "corner_radius_m", 0.08, "half the shortest"),
            (2, "trim_width_m", 0.2, "half the shortest"),
            (3, "head_radius_m", 0.04, "insufficient clearance"),
            (4, "normal_axis", "diagonal", "normal_axis"),
            (4, "count", 200, "less than or equal to 128"),
            (4, "end_m", [0.205, -0.12, 0.073], "plane perpendicular"),
        )
        for index, field, value, message in mutations:
            with self.subTest(index=index, field=field):
                original = self.raw["components"][index][field]
                self.raw["components"][index][field] = value
                with self.assertRaisesRegex(ValidationError, message):
                    self.validated()
                self.raw["components"][index][field] = original

    def test_compiled_detail_handoff_is_lossless_and_checks_expansion_names(self):
        spec = load_asset_spec(EXAMPLE)
        plan = plan_assembly(spec)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "plan.json"
            write_compiled_plan(spec, plan, target)
            parsed_spec, parsed_plan = load_compiled_plan(target)
            self.assertEqual(parsed_spec.asset_id, "precision-control-enclosure")
            self.assertEqual(parsed_plan.named_parts, plan.named_parts)
            self.assertEqual(len(parsed_plan.instances), 23)
            self.assertEqual(parsed_plan.recipe_sha256, plan.recipe_sha256)

            raw = json.loads(target.read_text(encoding="utf-8"))
            name = next(item for item in raw["plan"]["instances"]
                        if item["name"] == "front-mitered-trim-top")
            name["name"] = "front-mitered-trim-malicious"
            raw["sha256"] = hashlib.sha256(_canonical({
                k: v for k, v in raw.items() if k != "sha256"
            })).hexdigest()
            target.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(CompiledPlanError, "does not match"):
                load_compiled_plan(target)


if __name__ == "__main__":
    unittest.main()
