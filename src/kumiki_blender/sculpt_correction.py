"""Bounded, topology-preserving local face displacement from photo-fit anchors.

A provisional sculpt experiment only. Does not claim photogrammetric accuracy.
Works in world metres and preserves the original mesh's vertex/face indices.
"""
import math


def _wendland_c2(radius_fraction):
    """Compact C2-continuous radial basis, zero outside its influence radius."""
    r = float(radius_fraction)
    if not math.isfinite(r) or r < 0:
        raise ValueError("Radius fraction must be finite and nonnegative")
    return ((1-r)**4 * (4*r+1)) if r < 1 else 0.


def bounded_sculpt_offsets(vertices, candidates, *, influence_radius_m=.28,
                           max_displacement_m=.25):
    """Return per-vertex displacement vectors in world metres.

    candidate = {source_world_xyz, target_world_xyz}. Corrections use a
    Wendland C2 radial kernel; overlapping influences are normalized to
    avoid an unintended sum of sculpt displacements. This deliberately
    does not alter mesh topology or infer missing anatomy.
    """
    if not (math.isfinite(influence_radius_m) and influence_radius_m > 0 and
            math.isfinite(max_displacement_m) and max_displacement_m > 0):
        raise ValueError("Influence radius and displacement cap must be positive")
    if not candidates:
        raise ValueError("At least one independently triangulated anchor is required")
    prepared = []
    for entry in candidates:
        a = tuple(float(v) for v in entry["source_world_xyz"])
        b = tuple(float(v) for v in entry["target_world_xyz"])
        if len(a) != 3 or len(b) != 3 or not all(math.isfinite(v) for v in (*a, *b)):
            raise ValueError("Sculpt anchor coordinates must be finite 3D vectors")
        delta = tuple(y-x for x, y in zip(a, b))
        if math.dist(a,b) > max_displacement_m:
            raise ValueError("Sculpt anchor displacement exceeds safety cap")
        prepared.append((a, delta))
    result = []
    for point in vertices:
        p = tuple(float(v) for v in point)
        if len(p) != 3 or not all(math.isfinite(v) for v in p):
            raise ValueError("Mesh vertex coordinates must be finite 3D vectors")
        sums = [0.,0.,0.]
        weight_sum = 0.
        for source, delta in prepared:
            distance = math.dist(p,source)
            if distance >= influence_radius_m:
                continue
            weight = _wendland_c2(distance/influence_radius_m)
            weight_sum += weight
            for i in range(3):
                sums[i] += weight*delta[i]
        result.append(tuple(v/max(1.,weight_sum) for v in sums))
    return result
