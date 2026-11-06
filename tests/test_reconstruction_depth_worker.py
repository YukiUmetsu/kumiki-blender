"""Exercise Blender's BVH with two overlapping surfaces and a known camera."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

BLENDER = Path('/Applications/Blender.app/Contents/MacOS/Blender')
WORKER = Path(__file__).resolve().parents[1] / 'src/kumiki_blender/reconstruction/blender_depth_worker.py'


@unittest.skipUnless(BLENDER.is_file(), 'Blender executable unavailable')
class DepthWorkerTests(unittest.TestCase):
    def test_rear_surface_is_occluded_and_silhouette_is_rendered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / 'planes.ply'
            mesh.write_text('''ply
format ascii 1.0
element vertex 8
property float x
property float y
property float z
element face 4
property list uchar int vertex_indices
end_header
-1 -1 -2
1 -1 -2
1 1 -2
-1 1 -2
-1 -1 -4
1 -1 -4
1 1 -4
-1 1 -4
3 0 1 2
3 0 2 3
3 4 5 6
3 4 6 7
''')
            np.save(root / 'samples.npy', np.array([[0.1, 0.1, -2], [0.1, 0.1, -4]]))
            camera = {'intrinsics': {'width': 64, 'height': 64, 'fx_px': 32, 'fy_px': 32,
                                     'cx_px': 32, 'cy_px': 32},
                      'rotation_3x3': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                      'location_colmap': [0, 0, 0]}
            config = {'mesh': str(mesh), 'samples': str(root / 'samples.npy'),
                      'output': directory, 'cameras': [camera], 'side': 64}
            (root / 'config.json').write_text(json.dumps(config))
            subprocess.run([str(BLENDER), '--background', '--factory-startup', '--python-exit-code', '1',
                            '--python', str(WORKER), '--', str(root / 'config.json')],
                           check=True, capture_output=True, text=True)
            data = np.load(root / '0000.npz')
            np.testing.assert_array_equal(data['visible'], [1, 0])
            self.assertEqual(data['silhouette'][32, 32], 255)
            self.assertEqual(data['silhouette'][0, 0], 0)
            from kumiki_blender.reconstruction.silhouette_qa import compare_silhouettes
            # An expanded reference represents a deliberately missing patch;
            # a reduced reference treats the same rendered patch as excess.
            reference = data['silhouette'].copy()
            reference[30:34, 15:16] = 255
            missing, _ = compare_silhouettes(reference, data['silhouette'], boundary_tolerance=0)
            self.assertEqual(missing['missing_pixels'], 4)
            reference = data['silhouette'].copy()
            reference[46:48, 20:30] = 0
            excess, _ = compare_silhouettes(reference, data['silhouette'], boundary_tolerance=0)
            self.assertEqual(excess['excess_pixels'], 20)

