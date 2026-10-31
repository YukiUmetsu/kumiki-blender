import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.cli import main


FIXTURE = Path(__file__).resolve().parents[1] / "examples/lattice-observation-tower.json"


class CLITests(unittest.TestCase):
    def test_validate_prints_machine_readable_structure(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = main(["validate", "--recipe", str(FIXTURE)])
        self.assertEqual(code, 0)
        payload = json.loads(stream.getvalue())
        self.assertEqual(payload["asset_id"], "lattice-observation-tower")
        self.assertGreater(payload["beams"], 200)

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


if __name__ == "__main__":
    unittest.main()
