"""Declarative rigid-body authoring can be validated without bpy or NumPy."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from kumiki_blender.physics_plan import PhysicsPlanError, load_physics_plan, parse_physics_plan


VALID = {
    "schema_version": 1, "asset_id": "midcentury-desk-lamp-physics", "action_id": "fastener-drop",
    "duration_frames": 120, "gravity_mps2": [0, 0, -9.81],
    "substeps": 20, "solver_iterations": 20,
    "bodies": [
        {"target": "role:base-fasteners", "kind": "active", "mass_kg": .015,
         "friction": .45, "restitution": .25, "offset_m": [0, 0, .4]},
        {"target": "role:weighted-base", "kind": "passive", "shape": "MESH"},
    ],
}


class PhysicsPlanTests(unittest.TestCase):
    def test_compact_plan_has_unit_aware_defaults_and_rigid_body_rules(self):
        shot = parse_physics_plan(VALID)
        self.assertEqual(shot.duration_frames, 120)
        self.assertEqual(shot.action_id, "fastener-drop")
        self.assertEqual(shot.gravity_mps2, (0, 0, -9.81))
        self.assertEqual(shot.bodies[0].shape, "CONVEX_HULL")
        self.assertEqual(shot.bodies[1].shape, "MESH")
        self.assertEqual(shot.bodies[0].offset_m, (0, 0, .4))

    def test_rejects_unknown_fields_duplicate_targets_and_nonfinite_values(self):
        variants = [
            {**VALID, "run_python": "import os"},
            {**VALID, "bodies": VALID["bodies"] * 2},
            {**VALID, "gravity_mps2": [0, 0, float("nan")]},
            {**VALID, "substeps": 0},
            {**VALID, "bodies": [{**VALID["bodies"][0], "mass_kg": float("inf")}]},
            {**VALID, "bodies": [{**VALID["bodies"][0], "target": "role:../../escape"}]},
        ]
        for variant in variants:
            with self.subTest(variant=variant), self.assertRaises(PhysicsPlanError):
                parse_physics_plan(variant)

    def test_static_only_plan_and_moving_static_colliders_are_refused(self):
        with self.assertRaisesRegex(PhysicsPlanError, "active"):
            parse_physics_plan({**VALID, "bodies": [VALID["bodies"][1]]})
        with self.assertRaisesRegex(PhysicsPlanError, "passive collider"):
            parse_physics_plan({**VALID, "bodies": [VALID["bodies"][0],
                {**VALID["bodies"][1], "offset_m": [0, 0, 1]}]})

    def test_optional_unexported_ground_has_explicit_bounded_physics(self):
        shot = parse_physics_plan({**VALID, "ground": {"height_m": -.002, "friction": .8}})
        self.assertAlmostEqual(shot.ground.height_m, -.002)
        self.assertEqual(shot.ground.half_extent_m, 8)
        for invalid in ({"half_extent_m": 0}, {"thickness_m": -.1},
                        {"friction": 3}, {"bad_key": 1}):
            with self.subTest(invalid=invalid), self.assertRaises(PhysicsPlanError):
                parse_physics_plan({**VALID, "ground": invalid})

    def test_reusable_fixed_hinge_slider_constraints_are_strictly_validated(self):
        hinge = {"id": "door-pin", "kind": "hinge", "body_a": "name:post", "body_b": "name:panel",
                 "pivot_m": [0, 0, 1], "axis": [0, 0, 1], "limits_deg": [-70, 70]}
        slider = {"id": "rail", "kind": "slider", "body_a": "name:post", "body_b": "name:panel",
                  "pivot_m": [0, 0, .4], "axis": [1, 0, 0], "limits_m": [-.2, .2]}
        fixed = {"id": "bolt", "kind": "fixed", "body_a": "name:post", "body_b": "name:panel",
                 "pivot_m": [0, 0, .4]}
        shot = parse_physics_plan({**VALID, "constraints": [hinge, slider, fixed]})
        self.assertEqual([item.kind for item in shot.constraints], ["hinge", "slider", "fixed"])
        self.assertEqual(shot.constraints[0].limits, (-70, 70))
        self.assertEqual(shot.constraints[1].axis, (1, 0, 0))
        for invalid in ({**hinge, "limits_deg": [70, -70]},
                        {**hinge, "axis": [0, 0, 0]},
                        {**hinge, "limits_m": [-1, 1]},
                        {**hinge, "body_b": "name:post"},
                        {**hinge, "disable_collisions": "false"},
                        {**hinge, "kind": "unknown"}):
            with self.subTest(invalid=invalid), self.assertRaises(PhysicsPlanError):
                parse_physics_plan({**VALID, "constraints": [invalid]})
        with self.assertRaisesRegex(PhysicsPlanError, "unique"):
            parse_physics_plan({**VALID, "constraints": [hinge, hinge]})

    def test_load_is_bounded_and_cli_plan_needs_no_blender(self):
        from kumiki_blender.cli import main
        import io
        from contextlib import redirect_stdout

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            spec = root / "plan.json"
            spec.write_text(json.dumps(VALID), encoding="utf-8")
            self.assertEqual(load_physics_plan(spec).asset_id, VALID["asset_id"])
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["plan-physics", "--spec", str(spec)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output.getvalue())["duration_frames"], 120)
            spec.write_bytes(b" " * (128 * 1024 + 1))
            with self.assertRaisesRegex(PhysicsPlanError, "128 KiB"):
                load_physics_plan(spec)


if __name__ == "__main__":
    unittest.main()
