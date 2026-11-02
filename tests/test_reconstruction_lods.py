"""Licensed high-poly statue -> PBR LOD contract and Blender launch boundary."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from kumiki_blender.asset_import import AssetImportError, load_asset_provenance
from kumiki_blender.cli import main
from kumiki_blender.reconstruction.lods import ScanLODSettings, plan_scan_lods

from tests.test_asset_import import fixture_glb, provenance_json


class ScanLODTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "licensed-statue.glb"
        fixture_glb(self.source, materials=["Stone"])
        self.provenance = self.root / "provenance.json"
        self.provenance.write_text(json.dumps(provenance_json(asset_id="statue-scan")))

    def test_plans_non_mutating_licensed_lod_export(self):
        plan = plan_scan_lods(self.source, load_asset_provenance(self.provenance),
                              self.root / "lod-r01", ScanLODSettings(5000, 512, 0.02, True))
        self.assertEqual(plan["asset_id"], "statue-scan")
        self.assertEqual(plan["target_triangles"], 5000)
        self.assertEqual(plan["map_size"], 512)
        self.assertTrue(plan["source_provenance"]["redistribution_allowed"])
        self.assertFalse((self.root / "lod-r01").exists())
        (self.root / "lod-r01").mkdir()
        with self.assertRaisesRegex(AssetImportError, "already exists"):
            plan_scan_lods(self.source, load_asset_provenance(self.provenance),
                           self.root / "lod-r01", ScanLODSettings())

    def test_refuses_scan_redistribution_without_provenance_rights(self):
        self.provenance.write_text(json.dumps(provenance_json(redistribution_allowed=False)))
        with self.assertRaisesRegex(AssetImportError, "redistribution_allowed"):
            load_asset_provenance(self.provenance)
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            result = main(["plan-scan-lods", "--source", str(self.source),
                           "--provenance", str(self.provenance),
                           "--output", str(self.root / "lod-r01")])
        self.assertEqual(result, 2)
        self.assertIn("redistribution_allowed", errors.getvalue())

    def test_bounded_texture_budget_and_validated_tolerances(self):
        for settings in (ScanLODSettings(target_triangles=100),
                         ScanLODSettings(map_size=123),
                         ScanLODSettings(ray_distance_m=float("inf")),
                         ScanLODSettings(ray_distance_m=0),
                         ScanLODSettings(render_previews=1)):
            with self.subTest(settings=settings), self.assertRaises(AssetImportError):
                settings.validate()

    def test_cli_plans_and_launches_blender_without_mutating_sourced_assets(self):
        args = ["--source", str(self.source), "--provenance", str(self.provenance),
                "--output", str(self.root / "lod-r01"),
                "--target-triangles", "2000", "--map-size", "512",
                "--ray-distance-m", "0.04"]
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            self.assertEqual(main(["plan-scan-lods", *args]), 0)
        self.assertEqual(json.loads(captured.getvalue())["target_triangles"], 2000)
        import sys
        with patch.dict(sys.modules, {"bpy": None}), \
             patch("kumiki_blender.cli.shutil.which", return_value="/usr/bin/blender"), \
             patch("kumiki_blender.cli.subprocess.run", return_value=Mock(returncode=0)) as run:
            self.assertEqual(main(["build-scan-lods", *args, "--render-previews"]), 0)
        command = run.call_args.args[0]
        self.assertIn("build-scan-lods", command)
        self.assertIn("--render-previews", command)
        self.assertEqual(command[-6:-3], ["2000", "--map-size", "512"])
        self.assertFalse((self.root / "lod-r01").exists())


if __name__ == "__main__":
    unittest.main()
