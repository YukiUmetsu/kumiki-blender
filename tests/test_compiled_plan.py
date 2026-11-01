"""Pure-Python interchange is safe to load in Blender without Pydantic."""

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.asset_spec import load_asset_spec
from kumiki_blender.composer import plan_assembly
from kumiki_blender.compiled_plan import (CompiledPlanError, load_compiled_plan,
                                          write_compiled_plan)


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/desk-lamp-assembly-v2.json"


class CompiledPlanTests(unittest.TestCase):
    def setUp(self):
        self.spec = load_asset_spec(EXAMPLE)
        self.plan = plan_assembly(self.spec)

    def test_compilation_round_trip_preserves_instancing_and_materials(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "plan.json"
            write_compiled_plan(self.spec, self.plan, file)
            loaded_spec, loaded_plan = load_compiled_plan(file)
            self.assertEqual(loaded_spec.asset_id, self.spec.asset_id)
            self.assertEqual(loaded_spec.quality.views, self.spec.quality.views)
            self.assertEqual(loaded_plan.bounds_min, self.plan.bounds_min)
            self.assertEqual(loaded_plan.bounds_max, self.plan.bounds_max)
            self.assertEqual([p.name for p in loaded_plan.instances],
                             [p.name for p in self.plan.instances])
            fasteners = [p for p in loaded_plan.instances if p.name.startswith("base-fasteners-")]
            self.assertEqual(len({id(part.mesh) for part in fasteners}), 1)

    def test_tampering_with_compiled_file_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "plan.json"
            write_compiled_plan(self.spec, self.plan, file)
            payload = json.loads(file.read_text())
            payload["plan"]["instances"][0]["origin"] = [99, 0, 0]
            file.write_text(json.dumps(payload))
            with self.assertRaisesRegex(CompiledPlanError, "digest mismatch"):
                load_compiled_plan(file)

    def test_forged_component_count_is_rejected_even_with_matching_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "plan.json"
            write_compiled_plan(self.spec, self.plan, file)
            payload = json.loads(file.read_text())
            payload["plan"]["named_parts"][0][1] += 1
            unsigned = {k: v for k, v in payload.items() if k != "sha256"}
            payload["sha256"] = hashlib.sha256(json.dumps(
                unsigned, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True, allow_nan=False).encode()).hexdigest()
            file.write_text(json.dumps(payload))
            with self.assertRaisesRegex(CompiledPlanError, "instance counts"):
                load_compiled_plan(file)

    def test_compiler_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "plan.json"
            write_compiled_plan(self.spec, self.plan, file)
            before = file.read_bytes()
            with self.assertRaisesRegex(CompiledPlanError, "already exists"):
                write_compiled_plan(self.spec, self.plan, file)
            self.assertEqual(file.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
