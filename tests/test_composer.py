"""Fast Python-only contracts for the agent-facing AssetSpec compiler."""

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from kumiki_blender.architecture_geometry import audit_mesh, world_vertices
from kumiki_blender.asset_spec import (AssetSpec, AssetSpecError, asset_json_schema,
                                      load_asset_spec)
from kumiki_blender.composer import COMPONENT_REGISTRY, ComposerError, plan_assembly


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/desk-lamp-assembly-v2.json"


def validated(raw):
    return AssetSpec.model_validate_json(json.dumps(raw))


class ComposerTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))

    def test_example_expands_to_named_reusable_closed_parts(self):
        spec = load_asset_spec(EXAMPLE)
        first = plan_assembly(spec)
        second = plan_assembly(spec)
        self.assertEqual(first, second)
        self.assertEqual(first.asset_id, "midcentury-desk-lamp")
        self.assertEqual(len(first.instances), 17)
        self.assertEqual(len({item.name for item in first.instances}), 17)
        self.assertEqual(len(first.recipe_sha256), 64)
        self.assertLess(first.bounds_min[2], 0.01)
        self.assertGreater(first.bounds_max[2], 0.46)
        for part in first.instances:
            report = audit_mesh(part.mesh)
            self.assertEqual(report.boundary_edges, 0, part.name)
            self.assertEqual(report.nonmanifold_edges, 0, part.name)

    def test_semantically_identical_material_key_order_has_same_digest(self):
        first = plan_assembly(validated(self.raw)).recipe_sha256
        self.raw["materials"] = dict(reversed(tuple(self.raw["materials"].items())))
        self.assertEqual(plan_assembly(validated(self.raw)).recipe_sha256, first)

    def test_schema_has_discriminated_component_types(self):
        schema = asset_json_schema()
        self.assertEqual(schema["properties"]["schema_version"]["const"], 2)
        self.assertEqual(len(COMPONENT_REGISTRY), 9)
        self.assertEqual(set(COMPONENT_REGISTRY), {
            "solid.box", "solid.lathe", "solid.hollow-lathe",
            "solid.rounded-box", "mechanical.rod", "detail.radial-studs",
            "detail.linear-fasteners", "detail.panel-frame",
            "assembly.service-panel",
        })

    def test_hollow_shade_is_a_closed_shell_with_an_open_center(self):
        plan = plan_assembly(validated(self.raw))
        shade = next(part for part in plan.instances if part.name == "enameled-shade")
        self.assertEqual(audit_mesh(shade.mesh).boundary_edges, 0)
        self.assertGreater(len(shade.mesh.vertices), 500)
        # Bottom has inner and outer circles rather than a filled disk.
        bottom_z = min(vertex[2] for vertex in shade.mesh.vertices)
        bottom = [v for v in shade.mesh.vertices if abs(v[2]-bottom_z) < 1e-8]
        self.assertEqual(len(bottom), 192)
        self.assertGreater(min((v[0]**2+v[1]**2)**0.5 for v in bottom), 0.12)

    def test_material_reference_and_duplicate_id_rejected(self):
        self.raw["components"][0]["material"] = "missing"
        with self.assertRaisesRegex(ValidationError, "unknown material role"):
            validated(self.raw)
        self.raw["components"][0]["material"] = "enamel"
        self.raw["components"][1]["id"] = self.raw["components"][0]["id"]
        with self.assertRaisesRegex(ValidationError, "component ids must be unique"):
            validated(self.raw)

    def test_unknown_fields_executable_payload_and_bad_shapes_rejected(self):
        self.raw["components"][0]["python"] = "import os"
        with self.assertRaisesRegex(ValidationError, "Extra inputs are not permitted"):
            validated(self.raw)
        self.raw["components"][0].pop("python")
        self.raw["components"][0]["profile_rz_m"][0][0] = -1
        with self.assertRaises(ValidationError):
            validated(self.raw)

    def test_degenerate_rod_rejected_before_blender(self):
        self.raw["components"][2]["end_m"] = self.raw["components"][2]["start_m"]
        with self.assertRaisesRegex(ValidationError, "rod endpoints must differ"):
            validated(self.raw)

    def test_rod_frame_endpoints_match_spec(self):
        plan = plan_assembly(validated(self.raw))
        rod = next(part for part in plan.instances if part.name == "upper-arm")
        world = world_vertices(rod)
        self.assertAlmostEqual(min(point[0] for point in world), -0.012, delta=0.013)
        self.assertAlmostEqual(max(point[0] for point in world), 0.18, delta=0.013)
        self.assertAlmostEqual(max(point[2] for point in world), 0.45, delta=0.013)

    def test_duplicate_json_keys_and_nan_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.json"
            path.write_text('{"schema_version":2,"schema_version":1}')
            with self.assertRaisesRegex(AssetSpecError, "Duplicate JSON property"):
                load_asset_spec(path)
            path.write_text('{"schema_version":NaN}')
            with self.assertRaisesRegex(AssetSpecError, "Nonfinite number"):
                load_asset_spec(path)

    def test_instance_name_collisions_rejected(self):
        self.raw["components"].append({"type": "solid.box", "id": "base-fasteners-000",
                                       "material": "arm", "size_m": [0.02, 0.02, 0.02]})
        with self.assertRaisesRegex(ComposerError, "names collide"):
            plan_assembly(validated(self.raw))


if __name__ == "__main__":
    unittest.main()
