import contextlib
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kumiki_blender.cli import main
from kumiki_blender.export_audit import ExportAuditError


FIXTURE = Path(__file__).resolve().parents[1] / "examples/lattice-observation-tower.json"


class CLITests(unittest.TestCase):
    def test_audit_command_reports_structural_glb_information(self):
        with tempfile.TemporaryDirectory() as directory:
            glb_path = Path(directory) / "sample.glb"
            document = {"asset": {"version": "2.0"}, "meshes": [{"primitives": [{}]}], "nodes": []}
            chunk = json.dumps(document).encode("utf-8")
            chunk += b" " * (-len(chunk) % 4)
            glb_path.write_bytes(struct.pack("<4sII", b"glTF", 2, 20 + len(chunk)) + struct.pack("<I4s", len(chunk), b"JSON") + chunk)
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                code = main(["audit", "--model", str(glb_path)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stream.getvalue())["meshes"], 1)

    def test_validate_prints_machine_readable_structure(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = main(["validate", "--recipe", str(FIXTURE)])
        self.assertEqual(code, 0)
        payload = json.loads(stream.getvalue())
        self.assertEqual(payload["asset_id"], "lattice-observation-tower")
        self.assertGreater(payload["beams"], 200)
        self.assertEqual(payload["gussets"], 64)
        self.assertEqual(payload["fasteners"], 256)

    def test_plan_portico_reports_modular_study_without_blender(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = main(["plan-portico", "--bays", "3"])
        self.assertEqual(code, 0)
        report = json.loads(stream.getvalue())
        self.assertEqual(report["generator"], "modular-portico")
        self.assertEqual(report["named_parts"]["arch_stones"], 33)
        self.assertEqual(report["named_parts"]["flutes"], 64)

    def test_build_portico_requires_blender_and_does_not_write_partial_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "new-study"
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors):
                code = main(["build-portico", "--bays", "3", "--output", str(destination)])
            self.assertEqual(code, 2)
            self.assertIn("Blender", errors.getvalue())
            self.assertFalse(destination.exists())

    def test_existing_build_destination_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "already-here"
            output.mkdir()
            sentinel = output / "precious.txt"
            sentinel.write_text("keep me")
            with contextlib.redirect_stderr(io.StringIO()):
                code = main(["build", "--recipe", str(FIXTURE), "--output", str(output)])
            self.assertNotEqual(code, 0)
            self.assertEqual(sentinel.read_text(), "keep me")
            self.assertEqual(list(output.iterdir()), [sentinel])

    def test_build_without_blender_is_actionable_and_creates_no_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fresh"
            error_stream = io.StringIO()
            with contextlib.redirect_stderr(error_stream):
                code = main(["build", "--recipe", str(FIXTURE), "--output", str(output)])
            self.assertNotEqual(code, 0)
            self.assertIn("Blender", error_stream.getvalue())
            self.assertFalse(output.exists())

    def test_build_export_failure_is_explained_without_a_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "revision"
            errors = io.StringIO()
            with patch.dict("sys.modules", {"bpy": object()}), \
                    patch("kumiki_blender.blender_scene.build_scene", side_effect=ExportAuditError("missing PNG")), \
                    contextlib.redirect_stderr(errors):
                code = main(["build", "--recipe", str(FIXTURE), "--output", str(target)])
            self.assertEqual(code, 2)
            self.assertIn("missing PNG", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
