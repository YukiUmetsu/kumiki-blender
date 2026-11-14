"""Deterministic, Blender-independent motion and contact deltas for rig pose QA.

Metrics are geometric evidence, not an aesthetic certificate or collision
penetration depth. Vertex order and triangle indices must remain stable between
the neutral and evaluated pose snapshots.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def _valid_point(point: object) -> bool:
    return (
        isinstance(point, (list, tuple))
        and len(point) == 3
        and all(type(value) in (int, float) and math.isfinite(value) for value in point)
    )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def pose_motion(
    neutral: Sequence[Sequence[float]],
    posed: Sequence[Sequence[float]],
    *,
    moving_epsilon_m: float = 1e-6,
) -> dict:
    """Compare corresponding WORLD-space evaluated vertices without re-aligning.

    Global/root motion contributes to this metric, so it cannot prove that a
    particular limb has a correct influence map. A requested pose with zero
    evaluated movement is, however, definitely not a working deformation test.
    """
    if len(neutral) != len(posed):
        raise ValueError("Pose vertex count changed; cannot compare motion")
    if not neutral:
        raise ValueError("Cannot audit an empty mesh")
    if (type(moving_epsilon_m) not in (int, float) or
            not math.isfinite(moving_epsilon_m) or moving_epsilon_m < 0):
        raise ValueError("Invalid moving epsilon")
    if not all(_valid_point(p) for p in neutral) or not all(_valid_point(p) for p in posed):
        return {
            "finite": False,
            "max_vertex_displacement_m": None,
            "p95_vertex_displacement_m": None,
            "moving_vertices": None,
        }
    lengths = [math.dist(a, b) for a, b in zip(neutral, posed)]
    return {
        "finite": True,
        "max_vertex_displacement_m": round(max(lengths), 9),
        "p95_vertex_displacement_m": round(_percentile(lengths, .95), 9),
        "moving_vertices": sum(length > moving_epsilon_m for length in lengths),
    }


def collision_delta(
    neutral: Iterable[tuple[int, int]],
    posed: Iterable[tuple[int, int]],
) -> dict:
    """Count new/resolved *triangle ID* overlaps, not interpenetration volumes.

    Indices refer to the same ordered object pair in matching topology.
    Explicit contact exemptions must be represented by omitting a pair from
    the rig validation plan, not by silently subtracting arbitrary collisions.
    """
    before, after = set(neutral), set(posed)
    return {
        "neutral_overlapping_triangle_pairs": len(before),
        "posed_overlapping_triangle_pairs": len(after),
        "new_overlapping_triangle_pairs": len(after - before),
        "resolved_overlapping_triangle_pairs": len(before - after),
        "status": (
            "new-contact-review" if after - before
            else "existing-contact-review" if after
            else "clear"
        ),
    }
