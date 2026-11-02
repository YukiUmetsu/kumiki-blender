"""Universal command contract without importing Blender during planning."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kumiki_blender.cli import main
from kumiki_blender.asset_spec import load_asset_spec
from kumiki_blender.composer import plan_assembly
from kumiki_blender.compiled_plan import load_compiled_plan, write_compiled_plan


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/desk-lamp-assembly-v2.json"


class ComposerCLITests(unittest.TestCase):
    def test_plan_and_schema_commands_work_outside_blender(self):
        for command, arguments, expected in (
            ("asset-schema", [], "properties"),
            ("plan-asset", ["--spec", str(EXAMPLE)], "named_parts"),
        ):
            with self.subTest(command=command):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(main([command, *arguments]), 0)
                self.assertIn(expected, json.loads(output.getvalue()))

    def test_invalid_spec_fails_before_blender_or_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "spec.json"
            destination = Path(directory) / "revision"
            source.write_text('{"schema_version":2,"generator":"unsafe"}')
            error = io.StringIO()
            with contextlib.redirect_stderr(error):
                self.assertEqual(main(["build-asset", "--spec", str(source),
                                       "--output", str(destination)]), 2)
            self.assertIn("Invalid AssetSpec", error.getvalue())
            self.assertFalse(destination.exists())

    def test_valid_build_routes_to_generic_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "revision"
            compiled = Path(directory) / "compiled-plan.json"
            spec = load_asset_spec(EXAMPLE)
            write_compiled_plan(spec, plan_assembly(spec), compiled)
            result = {"asset_id": "midcentury-desk-lamp", "generator": "test"}
            output = io.StringIO()
            with patch.dict("sys.modules", {"bpy": object()}), \
                    patch("kumiki_blender.blender_composer.build_assembly_scene",
                          return_value=result) as adapter, contextlib.redirect_stdout(output):
                self.assertEqual(main(["build-asset", "--plan", str(compiled),
                                       "--output", str(destination)]), 0)
            self.assertEqual(json.loads(output.getvalue()), result)
            self.assertEqual(adapter.call_args.args[0].asset_id, "midcentury-desk-lamp")
            self.assertEqual(len(adapter.call_args.args[1].instances), 17)
            self.assertEqual(adapter.call_args.args[2], destination)

    def test_host_build_compiles_and_launches_blender_without_wheel_in_blender(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "revision"
            seen = []

            def verify_worker_command(command, *, check):
                self.assertEqual(check, False)
                self.assertIn("--plan", command)
                compiled = Path(command[command.index("--plan") + 1])
                spec, plan = load_compiled_plan(compiled)
                seen.append((spec.asset_id, len(plan.instances)))
                return type("Completed", (), {"returncode": 0})()

            with patch("kumiki_blender.cli.subprocess.run", side_effect=verify_worker_command), \
                    patch("kumiki_blender.cli.shutil.which", return_value="/usr/bin/blender"):
                self.assertEqual(main(["build-asset", "--spec", str(EXAMPLE),
                                       "--output", str(destination)]), 0)
            self.assertEqual(seen, [("midcentury-desk-lamp", 17)])

    def test_existing_revision_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "revision"
            destination.mkdir()
            error = io.StringIO()
            with contextlib.redirect_stderr(error):
                self.assertEqual(main(["build-asset", "--spec", str(EXAMPLE),
                                       "--output", str(destination)]), 2)
            self.assertIn("Output already exists", error.getvalue())


if __name__ == "__main__":
    unittest.main()
