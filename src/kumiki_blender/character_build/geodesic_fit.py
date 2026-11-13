"""Constrained mesh-surface landmark fitting, without Blender or ML dependencies.

Vertex distances are measured *along mesh edges* using bounded Dijkstra
search, not ambient Euclidean radii, so a cheek adjustment cannot leak across
two nearby but disconnected surfaces. The caller chooses explicit vertex IDs;
there is no ambiguous automatic correspondence or invented landmark.
"""
from __future__ import annotations

import heapq
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LandmarkPin:
    id: str
    vertex: int
    target_world_m: tuple[float, float, float]
    radius_m: float


def _triangles(faces, count):
    if not isinstance(faces, list) or not faces:
        raise ValueError("Need real triangle or quad face topology")
    triangles = []
    adjacency_edges = set()
    for face in faces:
        if not isinstance(face, (tuple, list)) or not 3 <= len(face) <= 4:
            raise ValueError("Landmark fitting requires triangles or quads")
        if any(type(index) is not int or index < 0 or index >= count for index in face):
            raise ValueError("Mesh has invalid face index")
        if len(set(face)) != len(face):
            raise ValueError("Mesh has repeated face vertices")
        for i, a in enumerate(face):
            b = face[(i+1) % len(face)]
            adjacency_edges.add((min(a, b), max(a, b)))
        for i in range(1, len(face)-1):
            triangles.append((face[0], face[i], face[i+1]))
    if not adjacency_edges:
        raise ValueError("Fitting mesh has no connected edges")
    return np.asarray(triangles, dtype=np.int64), sorted(adjacency_edges)


def _dijkstra(origin, radius, neighbors, obstacles):
    """Radius-bounded shortest paths; protected vertices block propagation."""
    distances = {origin: 0.}
    heap = [(0., origin)]
    while heap:
        d, index = heapq.heappop(heap)
        if d > distances[index] + 1e-12:
            continue
        if d >= radius or (index in obstacles and index != origin):
            continue
        for nbr, edge_length in neighbors[index]:
            if nbr in obstacles:
                continue
            nd = d + edge_length
            if nd < radius and nd < distances.get(nbr, math.inf) - 1e-12:
                distances[nbr] = nd
                heapq.heappush(heap, (nd, nbr))
    return distances


def _triangle_normals(points, faces):
    a, b, c = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    return np.cross(b-a, c-a)


def fit_landmark_surface(vertices, faces, pins: list[LandmarkPin],
                         protected_vertices: tuple[int, ...] = (),
                         *, max_displacement_m: float,
                         max_relative_edge_change: float) -> tuple[np.ndarray, dict]:
    """Produce one guarded candidate, or raise. Never mutates input arrays.

    Expected vertices are *world-space metres*, not Blender local coordinates.
    Shapes/UVs/materials are untouched; never use this on an already skinned
    or shape-key animated mesh without a separate corrective-shape workflow.
    """
    original = np.asarray(vertices, dtype=np.float64)
    if original.ndim != 2 or original.shape[1] != 3 or len(original) < 4:
        raise ValueError("Need a 3D mesh with at least four vertices")
    if len(original) > 300_000 or not np.isfinite(original).all():
        raise ValueError("Mesh is too large or contains nonfinite coordinates")
    if (not isinstance(pins, list) or not 1 <= len(pins) <= 24
            or len({p.id for p in pins}) != len(pins)
            or len({p.vertex for p in pins}) != len(pins)):
        raise ValueError("Need 1–24 unique landmark IDs and distinct source vertices")
    if not math.isfinite(max_displacement_m) or not 1e-6 <= max_displacement_m <= .25:
        raise ValueError("Maximum permitted movement must be in [1e-6, .25] metres")
    if not math.isfinite(max_relative_edge_change) or not 0 <= max_relative_edge_change <= .8:
        raise ValueError("Edge strain threshold must be in [0, .8]")
    obstacles = set(protected_vertices)
    if len(obstacles) != len(protected_vertices) or any(type(i) is not int or i < 0 or i >= len(original) for i in obstacles):
        raise ValueError("Invalid protected vertex indices")
    triangles, edges = _triangles(faces, len(original))
    edge_indices = np.asarray(edges, dtype=np.int64)
    edge_lengths = np.linalg.norm(original[edge_indices[:, 0]] - original[edge_indices[:, 1]], axis=1)
    before_normals = _triangle_normals(original, triangles)
    before_area = np.linalg.norm(before_normals, axis=1)
    if np.any(edge_lengths < 1e-8) or np.any(before_area < 1e-12):
        raise ValueError("Degenerate original geometry cannot be safely fitted")
    neighbors = defaultdict(list)
    for (a, b), d in zip(edges, edge_lengths):
        neighbors[a].append((b, float(d)))
        neighbors[b].append((a, float(d)))
    changes = np.zeros_like(original)
    total_weight = np.zeros(len(original), dtype=float)
    used = []
    for pin in pins:
        if type(pin.vertex) is not int or pin.vertex < 0 or pin.vertex >= len(original) or pin.vertex in obstacles:
            raise ValueError(f"{pin.id}: invalid or protected vertex anchor")
        if (len(pin.target_world_m) != 3 or
                any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in pin.target_world_m)):
            raise ValueError(f"{pin.id}: nonfinite target position")
        if not isinstance(pin.radius_m, (int, float)) or not math.isfinite(pin.radius_m) or not 0.0001 <= pin.radius_m <= 2:
            raise ValueError(f"{pin.id}: radius outside [0.1mm, 2m]")
        delta = np.asarray(pin.target_world_m, dtype=float) - original[pin.vertex]
        if np.linalg.norm(delta) > max_displacement_m + 1e-9:
            raise ValueError(f"{pin.id}: source-landmark displacement exceeds safe plan budget")
        within = _dijkstra(pin.vertex, pin.radius_m, neighbors, obstacles)
        for vertex_index, d in within.items():
            t = min(1., d / pin.radius_m)
            # C1-continuous smooth falloff: zero derivative at both ends.
            falloff = 1 - t*t*(3-2*t)
            changes[vertex_index] += falloff * delta
            total_weight[vertex_index] += falloff
        used.append({"id": pin.id, "vertex": pin.vertex, "radius_m": pin.radius_m,
                     "requested_displacement_m": round(float(np.linalg.norm(delta)), 8),
                     "surface_reachable_vertices": len(within)})
    alpha = np.maximum(1., total_weight)[:, None]
    candidate = original + changes / alpha
    candidate[list(obstacles)] = original[list(obstacles)]
    # Exact anchor correspondence takes precedence over overlapped smooth
    # falloff; face/edge quality gates reject incompatible pin constraints.
    for pin in pins:
        candidate[pin.vertex] = pin.target_world_m
    displacement = np.linalg.norm(candidate - original, axis=1)
    if np.max(displacement) > max_displacement_m + 1e-8:
        raise ValueError("Combined anatomical pins exceeded max displacement")
    current_edges = np.linalg.norm(candidate[edge_indices[:, 0]]-candidate[edge_indices[:, 1]], axis=1)
    strain = abs(current_edges / edge_lengths - 1)
    max_strain = float(np.max(strain))
    if max_strain > max_relative_edge_change + 1e-9:
        raise ValueError(f"Proposed anatomical fit overstretched an edge ({max_strain:.4f})")
    after_normals = _triangle_normals(candidate, triangles)
    after_area = np.linalg.norm(after_normals, axis=1)
    if np.any(after_area < 1e-12) or np.any(np.einsum("ij,ij->i", before_normals, after_normals) <= 0):
        raise ValueError("Proposed anatomical fit flipped or collapsed a source face")
    changed_mask = displacement > 1e-8
    if not np.any(changed_mask):
        raise ValueError("Requested source landmarks do not move any geometry")
    return candidate, {
        "status": "geometry-candidate-source-comparison-required",
        "landmark_count": len(pins),
        "pins": used,
        "changed_vertices": int(np.count_nonzero(changed_mask)),
        "unchanged_vertices": int(len(original)-np.count_nonzero(changed_mask)),
        "max_displacement_m": round(float(np.max(displacement)), 9),
        "max_relative_edge_change": round(max_strain, 8),
        "flipped_or_degenerate_faces": 0,
        "protected_vertices_untouched": len(obstacles),
        "mesh_vertices": len(original),
        "source_vertices_preserved": True,
    }
