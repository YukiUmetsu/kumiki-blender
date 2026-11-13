"""Bounded, feature-aware pre-rig surface fairing. No Blender/SciPy dependency.

This operates on a stable, closed, outward-oriented triangle surface and keeps
the exact vertex/face indices. It never claims artistic source fidelity. The
optional independent registered-view gate runs *after* Blender exports meshes.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.:-]{1,96}$")


@dataclass(frozen=True)
class SurfacePolicy:
    mesh: str
    iterations: int
    lambda_step: float
    mu_step: float
    feature_angle_deg: float
    curvature_stop: float
    max_displacement_m: float
    max_edge_strain: float
    max_volume_change: float
    min_roughness_improvement: float
    protected_vertex_groups: tuple[str, ...]
    editable_vertex_group: str | None
    reference_gate: dict | None


def _finite(value, name, lower, upper):
    if type(value) not in (float, int) or not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name} must be finite in [{lower}, {upper}]")
    return float(value)


def load_surface_policy(path: str | Path) -> SurfacePolicy:
    file = Path(path).resolve()
    raw = json.loads(file.read_text(encoding="utf-8"))
    keys = {"schema_version", "mesh", "iterations", "lambda_step", "mu_step",
            "feature_angle_deg", "curvature_stop", "max_displacement_m",
            "max_edge_strain", "max_volume_change", "min_roughness_improvement",
            "protected_vertex_groups", "editable_vertex_group", "reference_gate"}
    if (not isinstance(raw, dict) or set(raw) - keys
            or type(raw.get("schema_version")) is not int or raw["schema_version"] != 1):
        raise ValueError("Surface policy requires schema_version 1 and recognized properties")
    mesh = raw.get("mesh")
    if not isinstance(mesh, str) or not _SAFE_NAME.fullmatch(mesh):
        raise ValueError("mesh must be an exact Blender mesh object name")
    iterations = raw.get("iterations", 6)
    if type(iterations) is not int or not 1 <= iterations <= 24:
        raise ValueError("iterations must be 1 to 24")
    groups = raw.get("protected_vertex_groups", [])
    if (not isinstance(groups, list) or len(groups) > 32
            or any(not isinstance(x, str) or not _SAFE_NAME.fullmatch(x) for x in groups)
            or len(groups) != len(set(groups))):
        raise ValueError("protected_vertex_groups must be unique Blender group names")
    editable = raw.get("editable_vertex_group")
    if editable is not None and (not isinstance(editable, str) or not _SAFE_NAME.fullmatch(editable)):
        raise ValueError("editable_vertex_group must be a named Blender group or null")
    gate = raw.get("reference_gate")
    if gate is not None:
        if not isinstance(gate, dict) or set(gate) - {"character_plan", "region", "max_iou_regression", "min_iou"}:
            raise ValueError("reference_gate has unsupported settings")
        character, region = gate.get("character_plan"), gate.get("region")
        if (not isinstance(character, str) or not character
                or not isinstance(region, str) or not _SAFE_NAME.fullmatch(region)):
            raise ValueError("reference_gate needs character_plan folder and semantic region")
        character_dir = (file.parent / character).resolve()
        if not character_dir.is_dir() or not (character_dir / "character-plan.json").is_file():
            raise ValueError("reference_gate.character_plan must point to a frozen CharacterSpec plan")
        gate = {"character_plan": str(character_dir), "region": region,
                "max_iou_regression": _finite(gate.get("max_iou_regression", 0.005),
                                              "max_iou_regression", 0, .1),
                "min_iou": None if gate.get("min_iou") is None else
                           _finite(gate["min_iou"], "min_iou", 0, 1)}
    return SurfacePolicy(
        mesh=mesh, iterations=iterations,
        lambda_step=_finite(raw.get("lambda_step", 0.42), "lambda_step", .01, .65),
        mu_step=_finite(raw.get("mu_step", -0.36), "mu_step", -.65, -.01),
        feature_angle_deg=_finite(raw.get("feature_angle_deg", 48), "feature_angle_deg", 5, 175),
        curvature_stop=_finite(raw.get("curvature_stop", 1.0), "curvature_stop", .02, 10),
        max_displacement_m=_finite(raw.get("max_displacement_m", .004),
                                    "max_displacement_m", 1e-7, .1),
        max_edge_strain=_finite(raw.get("max_edge_strain", .16), "max_edge_strain", .001, 1),
        max_volume_change=_finite(raw.get("max_volume_change", .015), "max_volume_change", 0, .2),
        min_roughness_improvement=_finite(raw.get("min_roughness_improvement", .005),
                                          "min_roughness_improvement", 0, .5),
        protected_vertex_groups=tuple(groups), editable_vertex_group=editable,
        reference_gate=gate,
    )


def _face_vectors(points, faces):
    a, b, c = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    return np.cross(b - a, c - a)


def _volume(points, faces):
    a, b, c = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    return float(np.einsum("ij,ij->", a, np.cross(b, c)) / 6)


def _topology(points, faces, angle_deg):
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError("Expected at least four 3D vertices")
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or (faces < 0).any() or (faces >= len(points)).any()
            or not np.isfinite(points).all()):
        raise ValueError("Expected finite vertices and valid triangle faces")
    original_cross = _face_vectors(points, faces)
    magnitude = np.linalg.norm(original_cross, axis=1)
    if np.any(magnitude < 1e-12):
        raise ValueError("Degenerate faces cannot be safely smoothed")
    raw_edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges, inverse, counts = np.unique(np.sort(raw_edges, axis=1),
                                       axis=0, return_inverse=True, return_counts=True)
    if np.any(counts != 2):
        raise ValueError("Surface must be closed/manifold; open seams are protected by refusing edit")
    original_normals = original_cross / magnitude[:, None]
    which_face = np.repeat(np.arange(len(faces)), 3)
    first = np.full(len(edges), -1, dtype=np.int64)
    second = np.full(len(edges), -1, dtype=np.int64)
    for edge_id, face_id in zip(inverse, which_face):
        if first[edge_id] == -1:
            first[edge_id] = face_id
        else:
            second[edge_id] = face_id
    cos = np.sum(original_normals[first] * original_normals[second], axis=1)
    sharp = cos < math.cos(math.radians(angle_deg))
    sharp_vertices = np.zeros(len(points), dtype=bool)
    sharp_vertices[edges[sharp].reshape(-1)] = True
    both = np.vstack([edges, edges[:, ::-1]])
    rows, cols = both[:, 0], both[:, 1]
    degree = np.bincount(rows, minlength=len(points))

    def laplace(coords):
        sums = np.zeros_like(coords)
        np.add.at(sums, rows, coords[cols])
        return sums / np.maximum(degree, 1)[:, None] - coords

    edge_lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    if np.min(edge_lengths) < 1e-10:
        raise ValueError("Degenerate edges cannot be safely smoothed")
    lengths = np.zeros(len(points))
    np.add.at(lengths, rows, edge_lengths[np.tile(np.arange(len(edges)), 2)])
    local_size = lengths / np.maximum(degree, 1)
    return edges, edge_lengths, original_cross, sharp_vertices, local_size, laplace


def fair_surface(vertices: np.ndarray, faces: np.ndarray, policy: SurfacePolicy,
                 *, protected: np.ndarray | None = None,
                 editable: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Taubin-style weighted fairing with an immutable baseline and hard regression guards.

    The *same* topology and UVs can be retained by a Blender caller. This stage
    performs no subdivision; adaptive topology is an independent, later step.
    """
    original = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(faces, dtype=np.int64)
    edges, edge_lengths, cross, sharp, local_size, laplace = _topology(
        original, tris, policy.feature_angle_deg)
    volume_before = _volume(original, tris)
    if volume_before <= 1e-12:
        raise ValueError("Fairing requires a consistently outward-oriented closed surface")
    if protected is None:
        protected = np.zeros(len(original), dtype=bool)
    protected = np.asarray(protected, dtype=bool)
    if protected.shape != (len(original),):
        raise ValueError("protected mask must match vertex count")
    if editable is None:
        editable = np.ones(len(original))
    editable = np.asarray(editable, dtype=float)
    if editable.shape != (len(original),) or not np.isfinite(editable).all() or (editable < 0).any() or (editable > 1).any():
        raise ValueError("editable weights must be finite and in [0,1]")
    frozen = sharp | protected
    curvature = np.linalg.norm(laplace(original), axis=1) / np.maximum(local_size, 1e-12)
    feature_weight = np.clip(1.0 - curvature / policy.curvature_stop, 0.0, 1.0) ** 2
    strength = np.where(frozen, 0.0, feature_weight * editable)
    active = strength > 1e-8
    if not active.any():
        return original.copy(), {"status": "no-safe-improvement",
                                 "reason": "no_unprotected_smoothable_vertices",
                                 "vertices": int(len(original)), "changed_vertices": 0,
                                 "protected_vertices": int(protected.sum()),
                                 "feature_locked_vertices": int(sharp.sum()),
                                 "accepted_iterations": 0}
    def roughness(coords):
        displacement = np.linalg.norm(laplace(coords), axis=1) / np.maximum(local_size, 1e-12)
        # Evaluate the same fixed set of vertices across all candidates.
        return float(np.sqrt(np.mean(np.square(displacement[active]))))

    baseline_roughness = roughness(original)
    best, best_roughness = original.copy(), baseline_roughness
    history = []
    current = original.copy()
    for step in range(policy.iterations):
        advanced = current + policy.lambda_step * strength[:, None] * laplace(current)
        advanced += policy.mu_step * strength[:, None] * laplace(advanced)
        delta = advanced - original
        length = np.linalg.norm(delta, axis=1)
        advanced = original + delta * np.minimum(
            1.0, policy.max_displacement_m / np.maximum(length, 1e-20))[:, None]
        advanced[frozen] = original[frozen]
        new_cross = _face_vectors(advanced, tris)
        dot = np.einsum("ij,ij->i", cross, new_cross)
        flipped = int(np.sum((dot <= 0) | (np.linalg.norm(new_cross, axis=1) < 1e-12)))
        new_edges = np.linalg.norm(advanced[edges[:, 0]] - advanced[edges[:, 1]], axis=1)
        strain = float(np.max(np.abs(new_edges / edge_lengths - 1)))
        volume = _volume(advanced, tris)
        volume_change = abs(volume / volume_before - 1)
        score = roughness(advanced)
        allowed = (not flipped and strain <= policy.max_edge_strain and
                   volume > 0 and volume_change <= policy.max_volume_change)
        improved = allowed and score < best_roughness - 1e-9
        history.append({"iteration": step + 1, "roughness": round(score, 7),
                        "max_edge_strain": round(strain, 7),
                        "volume_change": round(volume_change, 7),
                        "flipped_faces": flipped, "accepted": bool(improved)})
        if not allowed:
            break
        if improved:
            current, best, best_roughness = advanced, advanced.copy(), score
        else:
            break
    improvement = ((baseline_roughness - best_roughness) / baseline_roughness
                   if baseline_roughness > 1e-12 else 0)
    approved = bool(history) and improvement >= policy.min_roughness_improvement and best_roughness < baseline_roughness
    result = best if approved else original.copy()
    movement = np.linalg.norm(result - original, axis=1)
    report = {
        "status": "surface-candidate-art-review-required" if approved else "no-safe-improvement",
        "vertices": int(len(original)), "triangles": int(len(tris)),
        "protected_vertices": int(protected.sum()), "feature_locked_vertices": int(sharp.sum()),
        "editable_vertices": int(active.sum()),
        "changed_vertices": int((movement > 1e-9).sum()),
        "maximum_displacement_m": float(np.max(movement)),
        "baseline_roughness": round(baseline_roughness, 7),
        "final_roughness": round(best_roughness if approved else baseline_roughness, 7),
        "improvement_fraction": round(improvement if approved else 0.0, 7),
        "volume_change_fraction": round(abs(_volume(result, tris) / volume_before - 1), 7),
        "accepted_iterations": sum(1 for row in history if row["accepted"]) if approved else 0,
        "attempts": history,
        "limitations": [
            "Smoothness score is geometric regularity, not character source fidelity.",
            "Original high-frequency authored details may need protection with vertex groups.",
            "No topology edits, subdivision, rigging, or artistic approval are performed.",
        ],
    }
    return result, report
