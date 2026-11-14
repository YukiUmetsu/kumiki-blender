"""Safe pose-corrective geometry plan and Blender-driver activation contracts."""
from __future__ import annotations

import copy
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from kumiki_blender.character_build.__main__ import main
from kumiki_blender.character_build.joint_corrective_plan import (
    activation_expression, load_corrective_plan,
    make_corrective_plan, save_corrective_plan,
    validate_corrective_config, validate_corrective_plan,
)


def example_config():
    return {
        "armature": "CharacterRig",
        "correctives": [{
            "mesh": "Wing.L",
            "target_object": "WingRaiseSculpt.L",
            "shape_key": "CORR_WingRaise.L",
            "bone": "upper_wing.L",
            "axis": "Z",
            "direction": "positive",
            "start_degrees": 10.0,
            "full_degrees": 50.0,
            "max_displacement_m": 0.008,
            "min_changed_vertices": 3,
            "max_changed_vertices": 200,
        }],
    }


class CorrectiveContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "quill.blend"
        self.source.write_bytes(b"fictional test fixture; Blender runtime not needed")
        self.config = example_config()
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

    def test_immutable_plan_locks_source_exact_bytes(self):
        plan = make_corrective_plan(self.source, self.config)
        self.assertEqual(len(plan["source_sha256"]), 64)
        self.assertEqual(plan["source_blend"], str(self.source.resolve()))
        directory = self.root / "planned"
        self.assertEqual(save_corrective_plan(self.source, self.config_path, directory), plan)
        path = directory / "joint-corrective-plan.json"
        self.assertEqual(load_corrective_plan(path), plan)
        with self.assertRaises(FileExistsError):
            save_corrective_plan(self.source, self.config_path, directory)
        self.source.write_bytes(b"source edited in Blender")
        with self.assertRaisesRegex(ValueError, "changed"):
            load_corrective_plan(path)

    def test_invalid_or_ambiguous_geometry_controls_rejected(self):
        alterations = [
            ("mesh", "WingRaiseSculpt.L"),
            ("shape_key", "bad shape key!"),
            ("target_object", ""),
            ("axis", "W"),
            ("direction", "both"),
            ("start_degrees", 50),
            ("full_degrees", 180),
            ("start_degrees", True),
            ("max_displacement_m", float("nan")),
            ("min_changed_vertices", 0),
            ("max_changed_vertices", 0),
        ]
        for field, value in alterations:
            cfg = copy.deepcopy(self.config)
            cfg["correctives"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_corrective_config(cfg)

    def test_duplicate_shape_name_or_extra_plan_keys_rejected(self):
        cfg = copy.deepcopy(self.config)
        cfg["correctives"].append(copy.deepcopy(cfg["correctives"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_corrective_config(cfg)
        cfg = copy.deepcopy(self.config)
        cfg["extra"] = {"bypass": True}
        with self.assertRaisesRegex(ValueError, "config"):
            validate_corrective_config(cfg)
        plan = make_corrective_plan(self.source, self.config)
        plan["new_field"] = False
        with self.assertRaisesRegex(ValueError, "locked source"):
            validate_corrective_plan(plan)

    def test_driver_expression_is_finite_smooth_clamped_and_sided(self):
        for direction, sign in (("positive", 1), ("negative", -1)):
            expression = activation_expression(direction, 10.0, 50.0)
            values = [
                eval(expression, {"__builtins__": {}, "min": min, "max": max},
                     {"bend": math.radians(x)}) for x in [-90, -50, -30, -10, 0, 10, 30, 50, 90]
            ]
            self.assertTrue(all(0 <= x <= 1 for x in values))
            def eval_angle(deg):
                return eval(expression, {"__builtins__": {}, "min": min, "max": max},
                            {"bend": math.radians(deg)})
            self.assertAlmostEqual(eval_angle(0), 0, places=6)
            self.assertAlmostEqual(eval_angle(sign * 10), 0, places=6)
            self.assertAlmostEqual(eval_angle(sign * 30), .5, places=6)
            self.assertAlmostEqual(eval_angle(sign * 50), 1, places=6)
            self.assertAlmostEqual(eval_angle(-sign * 50), 0, places=6)
        with self.assertRaises(ValueError):
            activation_expression("invalid", 10, 50)
        with self.assertRaises(ValueError):
            activation_expression("positive", 10, 50, variable="__import__('os')")

    def test_cli_plan_bind_worker_receives_source_locked_plan(self):
        directory = self.root / "plan"
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(main([
                "plan-joint-correctives", "--blend", str(self.source),
                "--config", str(self.config_path), "--output", str(directory),
            ]), 0)
        with patch("kumiki_blender.character_build.blender.run_worker") as worker:
            worker.return_value = {"status": "bound-pose-correctives-animation-review-required"}
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main([
                    "bind-joint-correctives", "--plan", str(directory / "joint-corrective-plan.json"),
                    "--output", str(self.root / "corrected"),
                ]), 0)
            self.assertEqual(worker.call_args.args[0], "joint-correctives")
            self.assertEqual(worker.call_args.args[1]["plan"]["source_sha256"],
                             load_corrective_plan(directory / "joint-corrective-plan.json")["source_sha256"])
            self.assertEqual(worker.call_args.kwargs["blend_file"], self.source.resolve())

    def test_cli_refuses_stale_source_and_output_overwrite(self):
        directory = self.root / "plan"
        save_corrective_plan(self.source, self.config_path, directory)
        self.source.write_bytes(b"modified source")
        with patch("kumiki_blender.character_build.blender.run_worker") as worker:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main([
                    "bind-joint-correctives", "--plan", str(directory / "joint-corrective-plan.json"),
                    "--output", str(self.root / "corrected"),
                ]), 2)
            worker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
