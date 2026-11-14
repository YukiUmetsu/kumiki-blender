"""Conservative surface-geodesic region influences without Blender imports.

The seed is a reviewed vertex on the intended surface. A short Euclidean
distance across a thin shell does NOT imply connected surface proximity.
Protected vertices form traversal barriers and receive zero influence.
"""
from __future__ import annotations

import heapq
import math
from collections.abc import Sequence


def surface_distances(
    positions: Sequence[Sequence[float]],
    edges: Sequence[Sequence[int]],
    seed: int,
    radius_m: float,
    *,
    protected: frozenset[int] = frozenset(),
) -> dict[int, float]:
    """Dijkstra on native mesh edges, limited by the geodesic radius.

    Edge lengths use WORLD-space coordinates (not Blender object scale).
    A necessary Euclidean-radius filter bounds adjacency memory for local
    edits. Disconnected shells and protected boundary loops are never crossed.
    """
    count = len(positions)
    if type(seed) is not int or not 0 <= seed < count:
        raise ValueError("Seed vertex index is outside the mesh")
    if type(radius_m) not in (float, int) or not math.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("Surface radius must be finite and positive")
    if seed in protected:
        raise ValueError("Seed cannot be a protected vertex")
    if any(type(v) is not int or not 0 <= v < count for v in protected):
        raise ValueError("Protected vertex index is outside the mesh")
    for point in positions:
        if (len(point) != 3 or any(type(x) not in (float, int) or not math.isfinite(x)
                                  for x in point)):
            raise ValueError("Surface vertices must have finite 3D coordinates")

    center = positions[seed]
    # All valid paths of length < radius stay inside this Euclidean ball.
    eligible = {i for i, p in enumerate(positions)
                if i not in protected and math.dist(center, p) < radius_m}
    adjacency: dict[int, list[tuple[int, float]]] = {i: [] for i in eligible}
    for edge in edges:
        if len(edge) != 2:
            raise ValueError("Surface edge must contain two indices")
        a, b = edge
        if type(a) is not int or type(b) is not int or not 0 <= a < count or not 0 <= b < count:
            raise ValueError("Surface edge references an invalid vertex")
        if a == b or a not in eligible or b not in eligible:
            continue
        length = math.dist(positions[a], positions[b])
        if not math.isfinite(length):
            raise ValueError("Nonfinite surface edge length")
        adjacency[a].append((b, length))
        adjacency[b].append((a, length))

    distances = {seed: 0.0}
    pending = [(0.0, seed)]
    while pending:
        distance, current = heapq.heappop(pending)
        if distance > distances[current]:
            continue
        for neighbor, length in adjacency[current]:
            candidate = distance + length
            if candidate < radius_m and candidate < distances.get(neighbor, math.inf):
                distances[neighbor] = candidate
                heapq.heappush(pending, (candidate, neighbor))
    return distances
