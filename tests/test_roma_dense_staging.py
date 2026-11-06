"""Synthetic provenance gates for a private RoMa camera-to-dense handoff."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from kumiki_blender.reconstruction.dense import DenseSettings, plan_dense_reconstruction
from kumiki_blender.reconstruction.references import ReferenceError
from kumiki_blender.reconstruction.roma_dense_staging import (
    stage_repaired_roma_dense_revision,
)


class Camera:
    width, height = 128, 128
    params = (140, 140, 64, 64)


class RegisteredImage:
    def __init__(self, name, number):
        self.name, self.number, self.camera_id = name, number, 1

    def projection_center(self):
        return (self.number, 0, 0)

    def viewing_direction(self):
        return (0, 0, 1)


class Model:
    def __init__(self, names):
        self.images = {i: RegisteredImage(name, i)
                       for i, name in enumerate(names, 1)}
        self.cameras = {1: Camera()}

    def reg_image_ids(self):
        return list(self.images)


class DenseStagingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'original-rectified-r01'
        self.repair = self.root / 'repair-50'
        self.out = self.root / 'staged-50'
        self.originals = ['a.png', 'b.png', 'top.png', 'drop.png']
        self.selected = self.originals[:-1]
        self.source_model = Model(self.originals)
        self.repaired_model = Model(self.selected)
        image_root = self.source / 'undistorted/images'
        source_sparse = self.source / 'undistorted/sparse'
        repaired_sparse = self.repair / 'sparse'
        for folder in (image_root, source_sparse, repaired_sparse):
            folder.mkdir(parents=True)
        for name in ('cameras.bin', 'images.bin', 'points3D.bin'):
            (source_sparse / name).write_bytes(('original' + name).encode())
            (repaired_sparse / name).write_bytes(('repaired' + name).encode())
        self.receipt = {
            'status': 'isolated-camera-candidate / independent-qa-required',
            'target': 'top.png', 'registered_images': 3,
            'excluded_unverified_cameras': ['drop.png'],
            'before_camera_gate': {'accepted': True, 'new_filenames': ['drop.png', 'top.png']},
            'after_camera_gate': {'accepted': True, 'new_filenames': ['top.png'], 'reasons': []},
            'source_sha256': {'candidate': {
                name: self._sha(source_sparse / name)
                for name in ('cameras.bin', 'images.bin', 'points3D.bin')}},
        }
        (self.repair / 'roma-sparse-repair-report.json').write_text(
            json.dumps(self.receipt))
        index = []
        for name in self.originals:
            path = image_root / name
            Image.new('RGB', (128, 128), (34, 53, 66)).save(path)
            index.append({'filename': name, 'dimensions': [128, 128],
                          'sha256': self._sha(path), 'mask_requires_rectification': True})
        (self.source / 'undistorted-reference-index.json').write_text(
            json.dumps({'images': index}))
        (self.source / 'reconstruction-report.json').write_text(json.dumps({
            'status': 'candidate / visual-review-required', 'asset_id': 'great-sphinx',
            'staged_independent_images': 5, 'registered_images': 4,
            'undistorted_images': 4, 'unregistered_filenames': ['prior-unregistered.png'],
            'models': [{'model': '0', 'registered_images': 4}],
        }))

    @staticmethod
    def _sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _convert(self, argv):
        self.assertEqual(argv[0], 'model_converter')
        out = Path(argv[argv.index('--output_path') + 1])
        (out / 'cameras.txt').write_text('1 PINHOLE 128 128 140 140 64 64\n')
        (out / 'images.txt').write_text(''.join(
            f'{index} 1 0 0 0 0 0 0 1 {name}\n0 0 -1\n'
            for index, name in enumerate(self.selected, 1)))
        return 0

    def _run(self):
        return stage_repaired_roma_dense_revision(
            self.source, self.repair, self.out,
            model_loader=lambda path: (self.source_model if path ==
                                       self.source / 'undistorted/sparse'
                                       else self.repaired_model),
            converter=self._convert)

    def test_stages_exact_rectified_three_view_experiment_for_existing_dense_pipeline(self):
        original_sha = self._sha(self.source / 'undistorted/sparse/images.bin')
        report = self._run()
        self.assertEqual(report['registered_images'], 3)
        self.assertEqual(report['excluded_unverified_cameras'], ['drop.png'])
        self.assertFalse(report['pose_independently_validated'])
        self.assertFalse(report['distribution_allowed'])
        self.assertEqual(self._sha(self.source / 'undistorted/sparse/images.bin'), original_sha)
        self.assertEqual(sorted(path.name for path in (self.out / 'undistorted/images').iterdir()),
                         sorted(self.selected))
        stage_report = json.loads((self.out / 'reconstruction-report.json').read_text())
        self.assertEqual(stage_report['unregistered_filenames'],
                         ['drop.png', 'prior-unregistered.png'])
        self.assertEqual(stage_report['roma_experimental_camera_validation'],
                         'independent-pose-and-mask-qa-pending')
        plan = plan_dense_reconstruction(
            self.out, self.root / 'dense',
            DenseSettings(min_registered_images=3, min_registered_fraction=.6,
                          refine=False, texture=False),
            executables={'InterfaceCOLMAP': 'InterfaceCOLMAP',
                         'DensifyPointCloud': 'DensifyPointCloud',
                         'ReconstructMesh': 'ReconstructMesh'})
        self.assertEqual(plan['registered_images'], 3)

    def test_refuses_rectified_source_with_other_intrinsics_or_binary_model(self):
        (self.source / 'undistorted/sparse/images.bin').write_bytes(b'new-model')
        with self.assertRaisesRegex(ReferenceError, 'differs from the one actually repaired'):
            self._run()
        self.assertFalse(self.out.exists())

    def test_refuses_changed_rectified_image(self):
        Image.new('RGB', (128, 128), (123, 1, 99)).save(
            self.source / 'undistorted/images/top.png')
        with self.assertRaisesRegex(ReferenceError, 'Rectified image or image index'):
            self._run()
        self.assertFalse(self.out.exists())

    def test_refuses_unreviewed_camera_receipt(self):
        receipt = self.receipt.copy()
        receipt['after_camera_gate'] = {'accepted': False, 'new_filenames': ['top.png']}
        (self.repair / 'roma-sparse-repair-report.json').write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ReferenceError, 'lacks reviewed'):
            self._run()
        self.assertFalse(self.out.exists())

    def test_refuses_unexpected_source_camera(self):
        self.source_model.images[5] = RegisteredImage('unexpected.png', 5)
        with self.assertRaisesRegex(ReferenceError, 'roster or approval'):
            self._run()
        self.assertFalse(self.out.exists())

    def test_refuses_camera_pose_drift(self):
        self.repaired_model.images[3].number = 12
        with self.assertRaisesRegex(ReferenceError, 'calibration/pose changed'):
            self._run()
        self.assertFalse(self.out.exists())

    def test_refuses_output_inside_repair(self):
        self.out = self.repair / 'dense'
        with self.assertRaisesRegex(ReferenceError, 'unused isolated output'):
            self._run()
        self.assertFalse(self.out.exists())

    def test_refuses_source_model_symlink(self):
        sparse = self.source / 'undistorted/sparse'
        (sparse / 'images.bin').unlink()
        (sparse / 'images.bin').symlink_to(self.root / 'not-existing.bin')
        with self.assertRaises(ReferenceError):
            self._run()
        self.assertFalse(self.out.exists())


if __name__ == '__main__':
    unittest.main()
