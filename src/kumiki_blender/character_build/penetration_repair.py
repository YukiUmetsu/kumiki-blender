"""Bounded, character-agnostic geometric penetration repair.

This module has no Blender dependency. The caller supplies a nearest-surface
oracle for a *verified closed and outward-oriented fixed mesh*. It adjusts the
movable mesh only and refuses edits violating protected vertices, stretch or
normal-orientation constraints. A successful numeric result is NOT a guarantee
of source-faithful character anatomy or a collision-free continuous surface.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

_NAME = re.compile(r"^[A-Za-z0-9_.:-]{1,96}$")


@dataclass(frozen=True)
class CollisionPolicy:
    moving: str
    fixed: str
    max_displacement_m: float
    clearance_m: float
    iterations: int
    smoothing_iterations: int
    max_relative_edge_change: float
    protected_vertices: tuple[int, ...]
    min_improvement_fraction: float


def _number(value, name: str, minimum: float, maximum: float) -> float:
    if type(value) not in (float, int) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be finite within [{minimum}, {maximum}]")
    return float(value)


def load_collision_plan(path: str | Path) -> dict:
    """Validate an explicit collision-pair policy without reading a .blend file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "pairs"} or raw["schema_version"] != 1:
        raise ValueError("Collision plan needs schema_version 1 and pairs")
    pairs = raw["pairs"]
    if not isinstance(pairs, list) or not 1 <= len(pairs) <= 32:
        raise ValueError("Collision plan needs 1 to 32 pairs")
    cleaned = []
    seen = set()
    allowed = {"moving", "fixed", "max_displacement_m", "clearance_m", "iterations",
               "smoothing_iterations", "max_relative_edge_change", "protected_vertices",
               "min_improvement_fraction"}
    for pair in pairs:
        if not isinstance(pair, dict) or set(pair) - allowed:
            raise ValueError("Unrecognized collision-pair setting")
        moving, fixed = pair.get("moving"), pair.get("fixed")
        if (not isinstance(moving, str) or not _NAME.fullmatch(moving)
                or not isinstance(fixed, str) or not _NAME.fullmatch(fixed) or moving == fixed):
            raise ValueError("moving/fixed must be distinct named Blender objects")
        if (moving, fixed) in seen:
            raise ValueError("Duplicate directed collision pair")
        seen.add((moving, fixed))
        protected = pair.get("protected_vertices", [])
        if (not isinstance(protected, list) or len(protected) > 100000
                or any(type(x) is not int or x < 0 for x in protected)
                or len(set(protected)) != len(protected)):
            raise ValueError("protected_vertices must be unique nonnegative vertex indices")
        iterations = pair.get("iterations", 8)
        smoothing = pair.get("smoothing_iterations", 2)
        if type(iterations) is not int or not 1 <= iterations <= 24:
            raise ValueError("iterations must be between 1 and 24")
        if type(smoothing) is not int or not 0 <= smoothing <= 8:
            raise ValueError("smoothing_iterations must be between 0 and 8")
        cleaned.append(CollisionPolicy(
            moving=moving, fixed=fixed,
            max_displacement_m=_number(pair.get("max_displacement_m", 0.008),
                                       "max_displacement_m", 1e-6, 0.25),
            clearance_m=_number(pair.get("clearance_m", 0.0005), "clearance_m", 0, 0.02),
            iterations=iterations, smoothing_iterations=smoothing,
            max_relative_edge_change=_number(pair.get("max_relative_edge_change", 0.25),
                                             "max_relative_edge_change", 0, 1),
            protected_vertices=tuple(protected),
            min_improvement_fraction=_number(pair.get("min_improvement_fraction", 0.02),
                                             "min_improvement_fraction", 0, 1),
        ))
    return {"schema_version": 1, "pairs": cleaned}


def _edges(faces: np.ndarray) -> np.ndarray:
    all_edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    return np.unique(np.sort(all_edges, axis=1), axis=0)


def _adjacency(vertex_count: int, edges: np.ndarray):
    """Return a NumPy-only neighbor-averaging callback (Blender embeds NumPy, not SciPy)."""
    if not len(edges):
        raise ValueError("Moving mesh has no valid triangle edges")
    both = np.concatenate((edges, edges[:, ::-1]), axis=0)
    rows, cols = both[:, 0], both[:, 1]
    degree = np.bincount(rows, minlength=vertex_count)

    def average(values: np.ndarray) -> np.ndarray:
        totals = np.zeros_like(values, dtype=float)
        np.add.at(totals, rows, values[cols])
        return totals / np.maximum(degree, 1)[:, None]
    return average


def _face_vectors(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    a, b, c = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    return np.cross(b - a, c - a)


def _quality(points: np.ndarray, baseline: np.ndarray, faces: np.ndarray,
             edges: np.ndarray, original_edge_length: np.ndarray,
             original_faces: np.ndarray) -> dict:
    current_length = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    edge_strain = abs(current_length / np.maximum(original_edge_length, 1e-10) - 1)
    face_cross = _face_vectors(points, faces)
    face_dot = np.einsum("ij,ij->i", original_faces, face_cross)
    flips = int(((face_dot <= 0) | (np.linalg.norm(face_cross, axis=1) < 1e-12)).sum())
    return {"max_relative_edge_change": float(edge_strain.max(initial=0)),
            "flipped_or_degenerate_faces": flips}


def _surface_score(points: np.ndarray, surface: Callable, clearance: float) -> tuple[float, int, np.ndarray, np.ndarray]:
    """Surface callback returns signed distance (outside positive) and unit outward normals."""
    signed, outward = surface(points)
    signed = np.asarray(signed, dtype=float)
    outward = np.asarray(outward, dtype=float)
    if signed.shape != (len(points),) or outward.shape != points.shape:
        raise ValueError("Surface callback returned incompatible shapes")
    if not np.isfinite(signed).all() or not np.isfinite(outward).all():
        raise ValueError("Surface callback returned NaN or inf")
    norms = np.linalg.norm(outward, axis=1)
    if np.any(norms < 1e-8):
        raise ValueError("Surface callback produced invalid normals")
    outward /= norms[:, None]
    deficit = np.maximum(clearance - signed, 0)
    return float(deficit.sum()), int((signed < 0).sum()), deficit, outward


def repair_penetration(vertices: np.ndarray, faces: np.ndarray, policy: CollisionPolicy,
                       surface: Callable) -> tuple[np.ndarray, dict]:
    """Attempt local displacement, with step-wise rejection and a full rollback path.

    IMPORTANT: the caller must validate actual triangle-triangle BVH overlaps
    after optimization. Vertex clearance by itself can miss crossing faces.
    """
    initial = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    if (initial.ndim != 2 or initial.shape[1] != 3 or not len(initial)
            or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or not np.isfinite(initial).all()
            or np.any(faces < 0) or np.any(faces >= len(initial))):
        raise ValueError("Expected finite vertex coordinates and triangle faces")
    if any(index >= len(initial) for index in policy.protected_vertices):
        raise ValueError("Protected vertex index outside moving mesh")
    coords = initial.copy()
    edges = _edges(faces)
    original_edges = np.linalg.norm(initial[edges[:, 0]] - initial[edges[:, 1]], axis=1)
    if np.any(original_edges <= 1e-10):
        raise ValueError("Degenerate input edges are not safe for automatic repair")
    original_faces = _face_vectors(initial, faces)
    if np.any(np.linalg.norm(original_faces, axis=1) <= 1e-12):
        raise ValueError("Degenerate input faces are not safe for automatic repair")
    adjacency = _adjacency(len(initial), edges)
    anchored = np.zeros(len(initial), dtype=bool)
    anchored[list(policy.protected_vertices)] = True
    baseline_score, baseline_inside, _, _ = _surface_score(coords, surface, policy.clearance_m)
    best_score = baseline_score
    best_inside = baseline_inside
    accepted = 0
    for _ in range(policy.iterations):
        score, inside, deficit, normals = _surface_score(coords, surface, policy.clearance_m)
        if score <= 1e-12:
            break
        desired = np.minimum(deficit, policy.max_displacement_m)[:, None] * normals
        desired[anchored] = 0
        # Blend, rather than replace, the actual projection with neighborhood
        # displacements; avoid moving protected attachments even one micron.
        for _ in range(policy.smoothing_iterations):
            desired = 0.75 * desired + 0.25 * adjacency(desired)
            desired[anchored] = 0
        # Backtracking prevents a large first step from distorting thin geometry.
        was_accepted = False
        for fraction in (1.0, 0.5, 0.25, 0.125):
            candidate = coords + desired * fraction
            travelled = candidate - initial
            travel = np.linalg.norm(travelled, axis=1)
            ratio = np.minimum(1.0, policy.max_displacement_m / np.maximum(travel, 1e-12))
            candidate = initial + travelled * ratio[:, None]
            candidate[anchored] = initial[anchored]
            quality = _quality(candidate, initial, faces, edges, original_edges, original_faces)
            if (quality["flipped_or_degenerate_faces"] or
                    quality["max_relative_edge_change"] > policy.max_relative_edge_change):
                continue
            new_score, new_inside, _, _ = _surface_score(candidate, surface, policy.clearance_m)
            if new_score + 1e-12 < best_score and new_inside <= best_inside:
                coords, best_score, best_inside = candidate, new_score, new_inside
                accepted += 1
                was_accepted = True
                break
        if not was_accepted:
            break
    improvement = ((baseline_score - best_score) / baseline_score) if baseline_score > 1e-12 else 0.0
    approved = accepted > 0 and improvement >= policy.min_improvement_fraction
    result = coords if approved else initial.copy()
    final_quality = _quality(result, initial, faces, edges, original_edges, original_faces)
    final_move = np.linalg.norm(result - initial, axis=1)
    report = {
        "status": "geometric-candidate-review-required" if approved else "no-safe-improvement",
        "baseline_penetrating_vertices": baseline_inside,
        "candidate_penetrating_vertices": best_inside if approved else baseline_inside,
        "baseline_penetration_penalty_m": round(baseline_score, 8),
        "candidate_penetration_penalty_m": round(best_score if approved else baseline_score, 8),
        "improvement_fraction": round(improvement if approved else 0.0, 6),
        "accepted_iterations": accepted if approved else 0,
        "protected_vertices": int(anchored.sum()),
        "changed_vertices": int((final_move > 1e-8).sum()),
        "max_displacement_m": float(final_move.max(initial=0)),
        **final_quality,
    }
    return result, report
