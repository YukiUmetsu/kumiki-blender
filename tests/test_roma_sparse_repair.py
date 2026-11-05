"""Synthetic guards for the isolated camera surgery; never download models."""
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.reconstruction.roma_sparse_repair import repair_roma_sparse_model


class Observation:
    def __init__(self, xy=(0, 0), pid=-1):
        self.xy, self.point3D_id = xy, pid

    def has_point3D(self):
        return self.point3D_id >= 0


class TrackElement:
    def __init__(self, image_id, index):
        self.image_id, self.point2D_idx = image_id, index


class Track:
    def __init__(self, entries):
        self.elements = [TrackElement(*item) for item in entries]

    def length(self):
        return len(self.elements)


class Point:
    xyz = [0, 0, 1]

    def __init__(self, entries):
        self.track = Track(entries)


class Transform:
    def __mul__(self, xyz):
        return xyz


class Camera:
    def img_from_cam(self, xyz):
        return [0, 0]


class Image:
    def __init__(self, name, image_id, *, observations=(), fixed_count=None):
        self.name, self.frame_id, self.camera_id = name, image_id, 1
        self._id = image_id
        self.points2D = list(observations)
        self._fixed_count = fixed_count
        self.cam_from_world = Transform()

    @property
    def num_points3D(self):
        return self._fixed_count if self._fixed_count is not None else sum(
            o.has_point3D() for o in self.points2D)

    def projection_center(self):
        return [self._id, 0, 0]

    def viewing_direction(self):
        return [0, 0, 1]


class Model:
    def __init__(self, *, with_new=False, duplicate=True):
        self.images = {i: Image(name, i, fixed_count=100) for i, name in enumerate(
            ['first.png', 'second.png', 'third.png'], 1)}
        self.cameras = {1: Camera()}
        self.points3D = {}
        self.registered = {1, 2, 3}
        self.deleted = []
        self.deregistered = []
        if with_new:
            obs = [Observation() for _ in range(1464)]
            for i in range(42):
                obs[i] = Observation((0.2, 0), 100 + i)
                self.points3D[100 + i] = Point([(4, i), (1, i), (2, i)])
            obs[855] = Observation((7.536, 0), 18295)
            obs[1462] = Observation((8.167, 0), 18295 if duplicate else 18296)
            obs[1463] = Observation((11.960, 0), 18298)
            self.points3D[18295] = Point([(4, 855), (4, 1462), (1, 0), (2, 0), (3, 0)])
            self.points3D[18298] = Point([(4, 1463), (1, 1), (2, 1)])
            self.images[4] = Image('photo-0060.png', 4, observations=obs)
            self.images[5] = Image('photo-0013.png', 5, fixed_count=42)
            self.images[6] = Image('photo-0055.png', 6, fixed_count=134)
            self.registered.update((4, 5, 6))

    def reg_image_ids(self):
        return sorted(self.registered)

    def delete_observation(self, image_id, index):
        point_id = self.images[image_id].points2D[index].point3D_id
        self.points3D[point_id].track.elements = [entry for entry in
            self.points3D[point_id].track.elements if
            (entry.image_id, entry.point2D_idx) != (image_id, index)]
        self.images[image_id].points2D[index].point3D_id = -1
        self.deleted.append((image_id, index))

    def deregister_frame(self, frame_id):
        self.registered.remove(frame_id)
        self.deregistered.append(frame_id)

    def write(self, folder):
        (folder / 'marker.txt').write_text('new model')


class SparseRepairTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.base = root / 'base'
        self.source = root / 'candidate'
        self.out = root / 'result'
        for folder in (self.base, self.source):
            folder.mkdir()
            for name in ('cameras.bin', 'images.bin', 'points3D.bin'):
                (folder / name).write_bytes(b'demo')

    def _run(self, *, model=None, indices=(1462, 1463), allow=True, exclude=None):
        candidate = model or Model(with_new=True)
        def loader(path):
            return candidate if path == self.source else Model()
        return repair_roma_sparse_model(self.base, self.source, self.out,
                                       target='photo-0060.png', remove_indices=list(indices),
                                       exclude_cameras=exclude if exclude is not None else
                                       ['photo-0013.png', 'photo-0055.png'],
                                       allow_duplicate_repair=allow, model_loader=loader), candidate

    def test_2_outliers_and_2_unverified_views_are_isolated(self):
        report, candidate = self._run()
        self.assertEqual(report['registered_images'], 4)
        self.assertEqual(report['remaining_target_3d_observations'], 43)
        self.assertEqual(candidate.deleted, [(4, 1463), (4, 1462)])
        self.assertEqual(candidate.deregistered, [5, 6])
        self.assertEqual([e.point2D_idx for e in candidate.points3D[18295].track.elements
                          if e.image_id == 4], [855])
        self.assertTrue((self.out / 'sparse/marker.txt').is_file())
        self.assertEqual(report['after_camera_gate']['new_filenames'], ['photo-0060.png'])

    def test_duplicate_repair_needs_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, 'explicit opt-in'):
            self._run(allow=False)
        self.assertFalse(self.out.exists())

    def test_refuses_good_observation_removal(self):
        with self.assertRaisesRegex(ValueError, 'exceed'):
            self._run(indices=(855, 1463))
        self.assertFalse(self.out.exists())

    def test_refuses_unexpected_extra_or_missing_camera(self):
        with self.assertRaisesRegex(ValueError, 'New camera names'):
            self._run(exclude=['photo-0013.png'])
        self.assertFalse(self.out.exists())

    def test_refuses_output_inside_existing_model(self):
        self.out = self.source / 'new'
        with self.assertRaisesRegex(ValueError, 'isolated output'):
            self._run()
        self.assertFalse(self.out.exists())

    def test_refuses_track_that_would_collapse_to_one_observation(self):
        candidate = Model(with_new=True)
        candidate.points3D[18298].track.elements.pop()
        with self.assertRaisesRegex(ValueError, 'entire 3D point'):
            self._run(model=candidate)
        self.assertFalse(self.out.exists())

    def test_refuses_shared_multicamera_frame_exclusion(self):
        candidate = Model(with_new=True)
        candidate.images[5].frame_id = 4
        with self.assertRaisesRegex(ValueError, 'shares a registered frame'):
            self._run(model=candidate)
        self.assertFalse(self.out.exists())

    def test_refuses_existing_camera_drift(self):
        candidate = Model(with_new=True)
        candidate.images[1]._id = 30
        with self.assertRaisesRegex(ValueError, 'unchanged'):
            self._run(model=candidate)
        self.assertFalse(self.out.exists())


if __name__ == '__main__':
    unittest.main()
