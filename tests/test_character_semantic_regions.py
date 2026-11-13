"""Regression test: physically adjacent body parts cannot blend source owners."""
from __future__ import annotations

import numpy as np
import pytest

from kumiki_blender.character_build.seams import (
    build_regions, choose_sources, level_seams, per_texel,
)


def test_touching_semantic_parts_get_independent_paint_graphs():
    # Exact same sample coordinates deliberately stress the previous bug:
    # tail/scarf texels were merged into one coarse source-owner region.
    points = np.zeros((4, 3), dtype=float)
    normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (4, 1))
    owners = np.array([0, 1, 0, 1])
    regions = build_regions(points, normals, sigma=0.01, spacing=0.005,
                            group_labels=owners)
    assert set(regions.group.tolist()) == {0, 1}
    assert len(regions.texels) == 2
    quality = np.zeros((2, 4), dtype=np.float32)
    quality[0, owners == 0] = 1
    quality[1, owners == 1] = 1
    colors = np.zeros((2, 4, 3), dtype=np.uint8)
    colors[0, :, 0] = 250
    colors[1, :, 2] = 250
    regions.add_sources(quality, colors)
    chosen = choose_sources(regions, np.array([False, False]),
                            contest_delta_e=0, min_region_nodes=1)
    surface_source, offset = per_texel(regions, points, chosen["score"], None)
    np.testing.assert_array_equal(surface_source, owners)
    assert np.all(offset == 0)
    _, report = level_seams(regions, reach=0.05)
    assert report["seam_edges"] == 0


def test_semantic_group_shape_must_match_texels():
    points = np.zeros((3, 3))
    normals = np.tile(np.array([[0, 0, 1]]), (3, 1))
    with pytest.raises(ValueError, match="group labels"):
        build_regions(points, normals, sigma=0.01, spacing=0.01,
                      group_labels=np.array([0, 1]))


def test_legacy_grouping_still_available():
    points = np.zeros((4, 3))
    normals = np.tile(np.array([[0, 0, 1]]), (4, 1))
    regions = build_regions(points, normals, sigma=0.01, spacing=0.01)
    assert regions.group is None
    assert regions.feature.shape[1] == 6
    assert len(regions.texels) == 1
