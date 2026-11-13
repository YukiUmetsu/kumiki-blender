"""Coordinate-system regression: source-registered OBJ is not Y-up interchange."""
from __future__ import annotations

import numpy as np

from kumiki_blender.character_build.mesh import read_mesh


def _write(path, vertices):
    path.write_text("".join(f"v {x} {y} {z}\n" for x, y, z in vertices) +
                    "f 1 2 3\n")


def test_registered_z_up_source_obj_is_not_rotated(tmp_path):
    canonical = np.asarray([[1, 2, 3], [2, 2, 3], [1, 4, 3]], dtype=float)
    for filename in ("surface-candidate-source.obj", "baseline.source.obj"):
        source = tmp_path / filename
        _write(source, canonical)
        np.testing.assert_array_equal(read_mesh(source).vertices, canonical)


def test_legacy_y_up_interchange_obj_still_rotates(tmp_path):
    canonical = np.asarray([[1, 2, 3], [2, 2, 3], [1, 4, 3]], dtype=float)
    y_up = np.column_stack([canonical[:, 0], canonical[:, 2], -canonical[:, 1]])
    path = tmp_path / "surface-candidate.obj"
    _write(path, y_up)
    np.testing.assert_array_equal(read_mesh(path).vertices, canonical)
