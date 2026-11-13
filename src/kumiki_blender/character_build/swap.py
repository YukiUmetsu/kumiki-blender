"""Swap one region of a mesh for the same region of another mesh in the same frame.

Two generator meshes placed with ``fit.normalise`` share the reference frame,
so a part can move between them without anchor points: the body from one
generation, the tail from another. A region is a list of conditions on the
box, given as conditions such as ``y>0.02`` and ``z>0.04`` (metres, Blender
axes; the back of the figure is +Y). ``blender_swap.py`` cuts and fuses closed
solids.
"""

from __future__ import annotations

import re

import numpy as np
from scipy.spatial import cKDTree

from .mesh import Mesh

_CONDITION = re.compile(r"^\s*([xyz])\s*(<|>)\s*(-?\d+(?:\.\d+)?)\s*$")


def parse_region(conditions: list[str]) -> list[tuple[int, str, float]]:
    parsed = []
    for text in conditions:
        match = _CONDITION.match(text)
        if not match:
            raise ValueError(f"Region condition {text!r} must look like y>0.02 or z<0.3")
        parsed.append(("xyz".index(match[1]), match[2], float(match[3])))
    if not parsed:
        raise ValueError("A region needs one or more conditions")
    return parsed


def region_box(region: list[tuple[int, str, float]], limit_m: float = 100.0) -> tuple[list[float], list[float]]:
    """The box that the conditions describe. An axis without a condition spans +-limit_m."""
    low, high = [-limit_m] * 3, [limit_m] * 3
    for axis, op, value in region:
        if op == ">":
            low[axis] = max(low[axis], value)
        else:
            high[axis] = min(high[axis], value)
    if any(a >= b for a, b in zip(low, high)):
        raise ValueError("The region conditions leave an empty box")
    return low, high


def in_region(points: np.ndarray, region: list[tuple[int, str, float]]) -> np.ndarray:
    inside = np.ones(len(points), dtype=bool)
    for axis, op, value in region:
        inside &= points[:, axis] > value if op == ">" else points[:, axis] < value
    return inside


def _similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares scale, rotation and translation from source to target (Umeyama)."""
    mu_s, mu_t = source.mean(axis=0), target.mean(axis=0)
    s, t = source - mu_s, target - mu_t
    u, sigma, vt = np.linalg.svd(t.T @ s / len(source))
    d = np.eye(3)
    d[2, 2] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ d @ vt
    scale = float(np.trace(np.diag(sigma) @ d) / (s ** 2).sum(axis=1).mean())
    return scale, rotation, mu_t - scale * rotation @ mu_s


def align(donor: Mesh, base: Mesh, region: list[tuple[int, str, float]], *, iterations: int = 150,
          sample: int = 20000, keep: float = 0.8, seed: int = 0) -> tuple[Mesh, dict]:
    """Move the donor onto the base with a similarity ICP on the parts outside the swap region.

    Two generations of one character share the body, head and legs but not the
    part that differs. Matching only the shared parts puts the donor's part where
    it belongs. The worst ``1 - keep`` of the matches are ignored each round.

    Matches go both ways (donor to base and base to donor). With a free scale,
    one-way nearest neighbours shrink the donor, because a smaller shape lies
    closer to every point of the larger one.
    """
    rng = np.random.default_rng(seed)
    fixed = base.vertices[~in_region(base.vertices, region)]
    moving = donor.vertices[~in_region(donor.vertices, region)]
    fixed = fixed[rng.choice(len(fixed), min(sample, len(fixed)), replace=False)]
    moving = moving[rng.choice(len(moving), min(sample, len(moving)), replace=False)]
    tree = cKDTree(fixed)
    total_scale, total_rotation, total_shift = 1.0, np.eye(3), np.zeros(3)
    start = float(np.median(tree.query(moving)[0]))
    for _ in range(iterations):
        forward, nearest = tree.query(moving)
        backward, back_nearest = cKDTree(moving).query(fixed)
        source = np.vstack([moving, moving[back_nearest]])
        target = np.vstack([fixed[nearest], fixed])
        distance = np.concatenate([forward, backward])
        best = distance <= np.quantile(distance, keep)
        scale, rotation, shift = _similarity(source[best], target[best])
        moving = scale * moving @ rotation.T + shift
        total_scale, total_rotation = scale * total_scale, rotation @ total_rotation
        total_shift = scale * rotation @ total_shift + shift
    end = float(np.median(tree.query(moving)[0]))
    moved = total_scale * donor.vertices @ total_rotation.T + total_shift
    angle = float(np.degrees(np.arccos(np.clip((np.trace(total_rotation) - 1) / 2, -1, 1))))
    return Mesh(moved, donor.faces, donor.colours), {
        "median_gap_before_m": round(start, 5), "median_gap_after_m": round(end, 5),
        "scale": round(total_scale, 4), "rotation_deg": round(angle, 2),
        "shift_m": [round(float(v), 4) for v in total_shift]}
