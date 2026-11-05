import unittest
from pathlib import Path

from tests.test_reconstruction_surface_coverage import SurfaceCoverageTests
from kumiki_blender.reconstruction.photographic_qa import audit_photographs
from kumiki_blender.reconstruction.references import ReferenceError

BLENDER = Path('/Applications/Blender.app/Contents/MacOS/Blender')


@unittest.skipUnless(BLENDER.is_file(), 'Blender executable unavailable')
class PhotographicQATests(SurfaceCoverageTests):
    def test_registered_comparisons_and_source_protection(self):
        with self.assertRaises(ReferenceError):
            audit_photographs(self.dense, self.root / 'sparse' / 'qa', blender=BLENDER,
                              side=64, max_samples=100)
        output = self.root / 'photo-qa'
        report = audit_photographs(self.dense, output, blender=BLENDER, side=64, max_samples=100)
        self.assertEqual(report['audited_cameras'], 3)
        self.assertEqual(len(report['per_camera']), 3)
        self.assertGreater(report['per_camera'][0]['recall'], 0)
        self.assertTrue((output / 'contact-sheet.jpg').is_file())
        self.assertTrue((output / 'visible-support-points.ply').is_file())
        self.assertTrue(report['regions'])
        self.assertFalse(report['distribution_allowed'])
