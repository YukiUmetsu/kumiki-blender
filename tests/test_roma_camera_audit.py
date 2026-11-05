import json
import unittest

from kumiki_blender.reconstruction.roma_camera_audit import (
    audit_roma_candidate_camera, measure_target_observations,
)
from kumiki_blender.reconstruction.roma_predictions import generate_roma_predictions
from kumiki_blender.reconstruction.roma_candidate_database import prepare_roma_candidate_database
from tests.test_roma_predictions import PredictionGenerationTests, StubPredictor


class Observation:
    def __init__(self, xy, point_id=10):
        self.xy = xy
        self.point3D_id = point_id

    def has_point3D(self):
        return True


class Transform:
    def __mul__(self, xyz):
        return xyz


class Camera:
    def img_from_cam(self, xyz):
        return [0, 0]


class Image:
    def __init__(self, name, center, points=()):
        self.name, self.center, self.points2D = name, center, list(points)
        self.camera_id = 1
        self.cam_from_world = Transform()
        self.num_points3D = 10

    def projection_center(self):
        return self.center

    def viewing_direction(self):
        return [0, 0, 1]


class Point:
    xyz = [0, 0, 1]


class FakeModel:
    def __init__(self, *, target=False):
        self.images = {i: Image(n, [i, 0, 0]) for i, n in
                       enumerate(['a.png', 'b.png', 'c.png'], 1)}
        if target:
            self.images[4] = Image('target.png', [4, 0, 0],
                                   [Observation([0.5, 0]), Observation([15, 0], 11),
                                    Observation([20, 0], 12)])
        self.cameras = {1: Camera()}
        self.points3D = {10: Point(), 11: Point(), 12: Point()}
        self.deleted = []

    def reg_image_ids(self):
        return list(self.images)

    def delete_observation(self, image_id, point_index):
        self.deleted.append((image_id, point_index))
        self.images[image_id].points2D[point_index] = Observation([0, 0])

    def write(self, folder):
        (folder / 'test-model-marker').write_text('separate copy')


class CameraAuditTests(unittest.TestCase):
    setUp = PredictionGenerationTests.setUp

    def _candidate(self):
        generate_roma_predictions(self.source, self.root / 'predictions',
                                  targets=['target.png'], anchors=['a.png', 'b.png'],
                                  predictor=StubPredictor())
        candidate = self.root / 'candidate'
        prepare_roma_candidate_database(self.source,
                                        self.root / 'predictions/target.png.roma.json', candidate)
        (candidate / 'registered').mkdir()
        return candidate

    def test_audit_distinguishes_original_and_synthetic_features(self):
        model = FakeModel(target=True)
        rows = measure_target_observations(model, 'target.png', 1)
        self.assertEqual([row['source'] for row in rows],
                         ['existing-feature', 'roma-predicted-keypoint', 'roma-predicted-keypoint'])
        self.assertEqual([row['point2D_idx'] for row in rows if row['error_px'] > 8], [1, 2])

    def test_prunes_only_explicit_outliers_to_new_model(self):
        candidate = self._candidate()
        baseline, modified = FakeModel(), FakeModel(target=True)
        def loader(path):
            return modified if path == candidate / 'registered' else baseline
        report = audit_roma_candidate_camera(self.source, candidate, self.root / 'audit',
                                             remove_indices=[1, 2], model_loader=loader)
        self.assertEqual(report['synthetic_feature_observations'], 2)
        self.assertEqual([row['point2D_idx'] for row in report['high_error_observations']], [1, 2])
        self.assertEqual(modified.deleted, [(4, 2), (4, 1)])
        self.assertTrue((self.root / 'audit/pruned-model/test-model-marker').is_file())
        self.assertEqual(json.loads((self.root / 'audit/roma-camera-audit.json').read_text())['status'],
                         'experimental-camera-needs-independent-validation')
        self.assertTrue((candidate / 'registered').is_dir())

    def test_rejects_good_observation_removal_without_creating_output(self):
        candidate = self._candidate()
        with self.assertRaisesRegex(ValueError, 'Only explicitly'):
            audit_roma_candidate_camera(self.source, candidate, self.root / 'bad',
                                        remove_indices=[0],
                                        model_loader=lambda path: FakeModel(target='registered' in str(path)))
        self.assertFalse((self.root / 'bad').exists())

    def test_refuses_pruning_duplicate_3d_point_in_same_image(self):
        candidate = self._candidate()
        model = FakeModel(target=True)
        model.images[4].points2D[2].point3D_id = 11
        with self.assertRaisesRegex(ValueError, 'shares a 3D point'):
            audit_roma_candidate_camera(self.source, candidate, self.root / 'unsafe',
                                        remove_indices=[1],
                                        model_loader=lambda path: model if path == candidate / 'registered' else FakeModel())
        self.assertFalse((self.root / 'unsafe').exists())

    def test_refuses_pruning_two_view_track_that_would_disappear(self):
        candidate = self._candidate()
        model = FakeModel(target=True)
        class Track:
            def length(self):
                return 2
        model.points3D[11].track = Track()
        with self.assertRaisesRegex(ValueError, 'delete the entire 3D point'):
            audit_roma_candidate_camera(self.source, candidate, self.root / 'unsafe-track',
                                        remove_indices=[1],
                                        model_loader=lambda path: model if path == candidate / 'registered' else FakeModel())
        self.assertFalse((self.root / 'unsafe-track').exists())


if __name__ == '__main__':
    unittest.main()
