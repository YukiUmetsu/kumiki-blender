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
from tests.test_asset_import import fixture_glb, provenance_json
from tests.test_material_packs import sample as texture_pack_sample


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

    def test_audit_revision_command_checks_asset_package(self):
        stream = io.StringIO()
        with patch("kumiki_blender.cli.audit_revision", return_value={"verified": True, "output_count": 6}), \
                contextlib.redirect_stdout(stream):
            code = main(["audit-revision", "--root", "example/revision"])
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(stream.getvalue())["verified"])

    def test_inspect_asset_command_is_blender_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.glb"
            fixture_glb(model)
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                code = main(["inspect-asset", "--input", str(model)])
            self.assertEqual(code, 0)
            self.assertTrue(json.loads(stream.getvalue())["static_mesh_only"])

    def test_inspect_texture_packs_reports_hashes_without_blender(self):
        with tempfile.TemporaryDirectory() as directory:
            pack, _ = texture_pack_sample(Path(directory))
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                code = main(["inspect-texture-packs", "--input", str(pack)])
            self.assertEqual(code, 0)
            report = json.loads(stream.getvalue())
            self.assertEqual(report["assignments"][0]["target"], "wall-stone")
            self.assertEqual(len(report["assignments"][0]["pack"]["maps"]["normal"]["sha256"]), 64)

    def test_import_preflights_texture_pack_material_name_without_blender(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, record, destination = root / "model.glb", root / "source.json", root / "revision"
            fixture_glb(model, materials=["Original limestone"])
            record.write_text(json.dumps(provenance_json()))
            pack, _ = texture_pack_sample(root)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = main(["import-asset", "--input", str(model), "--provenance", str(record),
                             "--texture-packs", str(pack), "--output", str(destination)])
            self.assertEqual(code, 2)
            self.assertIn("matches 0", stderr.getvalue())
            self.assertFalse(destination.exists())

    def test_import_asset_refuses_rights_without_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, record, destination = root / "model.glb", root / "source.json", root / "revision"
            fixture_glb(model)
            record.write_text(json.dumps(provenance_json(redistribution_allowed=False)))
            stream = io.StringIO()
            with contextlib.redirect_stderr(stream):
                code = main(["import-asset", "--input", str(model), "--provenance",
                             str(record), "--output", str(destination)])
            self.assertEqual(code, 2)
            self.assertIn("redistribution", stream.getvalue())
            self.assertFalse(destination.exists())

    def test_sphinx_scan_build_preflights_rights_and_pairs_without_blender(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, record, destination = root / "scan.glb", root / "source.json", root / "revision"
            fixture_glb(source)
            record.write_text(json.dumps(provenance_json()))
            for arguments, expected in (
                (["--sphinx-source", str(source)], "together"),
                (["--sphinx-source", str(source), "--sphinx-provenance", str(record)], "commercial_use_allowed"),
                (["--kind", "pyramid", "--sphinx-source", str(source),
                  "--sphinx-provenance", str(record)], "only available"),
            ):
                errors = io.StringIO()
                with contextlib.redirect_stderr(errors):
                    code = main(["build-monument", "--output", str(destination), *arguments])
                self.assertEqual(code, 2)
                self.assertIn(expected, errors.getvalue())
                self.assertFalse(destination.exists())

    def test_sphinx_scan_build_passes_prepared_record_to_blender(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, record, destination = root / "scan.glb", root / "source.json", root / "revision"
            fixture_glb(source)
            record.write_text(json.dumps(provenance_json(commercial_use_allowed=True)))
            manifest = {"generator": "test-scan"}
            output = io.StringIO()
            with patch.dict("sys.modules", {"bpy": object()}), \
                    patch("kumiki_blender.blender_monuments.build_monument_scene", return_value=manifest) as adapter, \
                    contextlib.redirect_stdout(output):
                code = main(["build-monument", "--kind", "sphinx", "--sphinx-front-axis=+Y",
                             "--sphinx-source", str(source), "--sphinx-provenance", str(record),
                             "--output", str(destination)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output.getvalue()), manifest)
            self.assertEqual(adapter.call_args.kwargs["sphinx_scan"].front_axis, "+Y")

    def test_import_asset_hands_verified_source_to_blender_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, record, destination = root / "model.glb", root / "source.json", root / "revision"
            fixture_glb(model)
            record.write_text(json.dumps(provenance_json()))
            manifest = {"asset_id": "sample-portico", "generator": "kumiki-blender/static-glb-import"}
            stream = io.StringIO()
            with patch.dict("sys.modules", {"bpy": object()}), \
                    patch("kumiki_blender.blender_import.build_imported_scene", return_value=manifest) as adapter, \
                    contextlib.redirect_stdout(stream):
                code = main(["import-asset", "--input", str(model), "--provenance",
                             str(record), "--output", str(destination)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stream.getvalue()), manifest)
            self.assertEqual(adapter.call_args.args[0], model)
            self.assertEqual(adapter.call_args.args[1].license, "MIT")

    def test_import_asset_preflights_material_override_name_without_blender(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, record = root / "model.glb", root / "source.json"
            config, destination = root / "override.json", root / "revision"
            fixture_glb(model, materials=["Original limestone"])
            record.write_text(json.dumps(provenance_json()))
            config.write_text(json.dumps({
                "schema_version": 1,
                "overrides": [{"material_name": "Typo limestone", "preset": "natural-stone"}],
            }))
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors):
                code = main(["import-asset", "--input", str(model), "--provenance", str(record),
                             "--material-overrides", str(config), "--output", str(destination)])
            self.assertEqual(code, 2)
            self.assertIn("matches 0", errors.getvalue())
            self.assertFalse(destination.exists())

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
        self.assertEqual(report["named_parts"]["volutes"], 32)
        self.assertEqual(report["named_parts"]["column_capitals"], 16)

    def test_cli_plans_gothic_arcade_with_turning_balusters(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = main(["plan-portico", "--bays", "2", "--order", "tuscan",
                         "--arch-style", "pointed", "--balustrade"])
        self.assertEqual(code, 0)
        report = json.loads(stream.getvalue())
        self.assertEqual(report["order"], "tuscan")
        self.assertEqual(report["arch_style"], "pointed")
        self.assertTrue(report["balustrade"])
        self.assertEqual(report["named_parts"]["balusters"], 14)
        self.assertEqual(report["named_parts"]["arch_stones"], 22)
        self.assertEqual(sum(report["named_parts"].values()), report["instances"])

    def test_plan_facade_reports_individually_editable_architectural_parts(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = main(["plan-facade", "--bays", "3"])
        self.assertEqual(code, 0)
        report = json.loads(stream.getvalue())
        self.assertEqual(report["generator"], "architectural-facade-detail-kit")
        self.assertGreater(report["named_parts"]["masonry"], 600)
        self.assertGreater(report["named_parts"]["roof"], 400)

    def test_classical_column_can_be_planned_without_blender(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = main(["plan-classical-column", "--order", "ionic",
                         "--height", "4.55", "--radius", "0.36"])
        self.assertEqual(code, 0)
        report = json.loads(stream.getvalue())
        self.assertEqual(report["parts"], 15)
        self.assertEqual(report["order"], "ionic")
        self.assertEqual(sum("volute-front" in name for name in report["named_parts"]), 2)
        self.assertAlmostEqual(report["bounds_m"][1][2], 4.55)

    def test_facade_build_requires_blender_without_creating_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "facade"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = main(["build-facade", "--output", str(output)])
            self.assertEqual(code, 2)
            self.assertIn("Blender", stderr.getvalue())
            self.assertFalse(output.exists())

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
