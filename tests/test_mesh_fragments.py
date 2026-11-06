"""Floating pieces are removed; the body and pieces that touch it stay."""
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from kumiki_blender.reconstruction.mesh_fragments import FragmentLimits, write_mesh
from kumiki_blender.reconstruction.references import ReferenceError


def sheet(origin, width, rows=8):
    """Build a flat connected triangle sheet with `width` as its side length."""
    xs, ys = np.meshgrid(np.linspace(0, width, rows), np.linspace(0, width, rows))
    vertices = np.c_[xs.ravel(), ys.ravel(), np.zeros(rows * rows)] + np.array(origin, float)
    corners = np.arange(rows * rows).reshape(rows, rows)
    a, b, c, d = corners[:-1, :-1].ravel(), corners[:-1, 1:].ravel(), corners[1:, :-1].ravel(), corners[1:, 1:].ravel()
    return vertices, np.r_[np.c_[a, b, c], np.c_[b, d, c]]


def join(*parts):
    vertices, faces, offset = [], [], 0
    for v, f in parts:
        vertices.append(v)
        faces.append(f + offset)
        offset += len(v)
    return np.concatenate(vertices), np.concatenate(faces)


@unittest.skipUnless(find_spec("scipy") is not None, "Fragment analysis needs scipy")
class FragmentChoiceTests(unittest.TestCase):
    def choose(self, *parts):
        from kumiki_blender.reconstruction.mesh_fragments import choose_faces

        vertices, faces = join(*parts)
        return choose_faces(vertices, faces, FragmentLimits()), faces

    def test_a_small_far_piece_is_removed_and_recorded(self):
        (keep, removed), faces = self.choose(sheet((0, 0, 0), 10), sheet((30, 0, 0), .5))
        self.assertEqual(int(keep.sum()), len(sheet((0, 0, 0), 10)[1]))
        self.assertEqual(len(removed), 1)
        self.assertGreater(removed[0]["gap_fraction_of_body_size"], .5)

    def test_a_small_piece_that_touches_the_body_stays(self):
        (keep, removed), faces = self.choose(sheet((0, 0, 0), 10), sheet((10.02, 0, 0), .5))
        self.assertTrue(keep.all())
        self.assertEqual(removed, [])

    def test_a_large_separate_piece_stays(self):
        (keep, removed), faces = self.choose(sheet((0, 0, 0), 10), sheet((30, 0, 0), 5))
        self.assertTrue(keep.all())

    def test_a_single_body_has_nothing_to_remove(self):
        (keep, removed), faces = self.choose(sheet((0, 0, 0), 10))
        self.assertTrue(keep.all())
        self.assertEqual(removed, [])


class MeshWriterTests(unittest.TestCase):
    def test_kept_faces_round_trip_without_unused_vertices(self):
        from kumiki_blender.reconstruction.surface_coverage import _load_triangular_ply

        vertices, faces = sheet((0, 0, 0), 1, rows=3)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mesh.ply"
            write_mesh(path, vertices, faces[:2])
            loaded, loaded_faces = _load_triangular_ply(path, expected_vertices=5, expected_faces=2, np=np)
        self.assertEqual(loaded_faces.max(), 4)
        np.testing.assert_allclose(loaded[loaded_faces].reshape(-1, 3),
                                   vertices[faces[:2]].reshape(-1, 3), atol=1e-6)


class LimitTests(unittest.TestCase):
    def test_invalid_limits_are_refused(self):
        with self.assertRaisesRegex(ReferenceError, "between 0 and 1"):
            FragmentLimits(touching_fraction=0)


if __name__ == "__main__":
    unittest.main()
