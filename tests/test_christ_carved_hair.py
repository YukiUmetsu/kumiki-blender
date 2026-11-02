"""Blender-only smoke gate for the experimental continuous carved hair sculpt."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / 'scripts/landmarks/christ_redeemer.py'


@unittest.skipUnless(os.environ.get('KUMIKI_BLENDER_BINARY'),
                     'Set KUMIKI_BLENDER_BINARY to run the carved hair clay sculpt')
class ChristCarvedHairTests(unittest.TestCase):
    def test_anatomical_scalp_has_real_carved_vertices_and_three_hair_meshes(self):
        with tempfile.TemporaryDirectory() as temp:
            metrics = Path(temp) / 'hair.json'
            process = subprocess.run([
                os.environ['KUMIKI_BLENDER_BINARY'], '--background',
                '--factory-startup', '--python-exit-code', '1',
                '--python', str(GENERATOR), '--',
                '--head-study', '--face-method', 'anatomical', '--carved-hair',
                '--metrics', str(metrics), '--verify'],
                capture_output=True, text=True, timeout=120)
            self.assertEqual(process.returncode, 0,
                             process.stdout[-2000:] + process.stderr[-2000:])
            report = json.loads(metrics.read_text())
            self.assertTrue(report['carved_hair'])
            self.assertEqual(report['mesh_count'], 6)
            self.assertEqual(report['carved_hair_grooves'], 6)
            self.assertGreater(report['rooted_curtain_vertices'], 70)
            self.assertLess(report['max_root_snap_distance_m'], .08)
            self.assertEqual(report['groove_projection'], 'continuous-polyline')
            self.assertGreater(report['carved_hair_vertices'], 150)
            self.assertGreater(report['hair_surface_vertices'], 7000)
            self.assertGreater(report['hair_bounds_width_m'], 2.4)
            self.assertLess(report['hair_bounds_width_m'], 4.2)
            self.assertAlmostEqual(report['hair_bounds_max_z_m'], 38., delta=.15)


if __name__ == '__main__':
    unittest.main()
