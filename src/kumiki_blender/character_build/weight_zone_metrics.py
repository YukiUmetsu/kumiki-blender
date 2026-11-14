"""Pure-Python semantic skin-weight isolation metrics for a Blender rig.

A painted nondeforming region vertex group selects vertices. The corresponding
deform-bone influence must remain inside an explicitly approved bone set.
No attempt is made to infer anatomy or silently redistribute weights.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping


def audit_weight_zone(
    samples: Iterable[tuple[int, float, Mapping[str, float]]],
    allowed_bones: set[str] | frozenset[str],
    *,
    max_outside_weight: float = 0.02,
    min_vertices: int = 1,
    membership_threshold: float = 0.5,
) -> dict:
    """Summarize incorrect deform weights on selected vertices, bounded examples.

    `samples` yields ascending (source vertex index, semantic region membership,
    deform-bone weight mapping). The mapping must contain deform bones only.
    If a vertex belongs to a semantic region, every outside influence counts,
    even when each individual outside bone has a tiny weight.
    """
    if not allowed_bones or not all(isinstance(name, str) and name for name in allowed_bones):
        raise ValueError("A weight zone needs named approved deform bones")
    if (type(max_outside_weight) not in (float, int) or
            not math.isfinite(max_outside_weight) or not 0 <= max_outside_weight <= 1):
        raise ValueError("max_outside_weight must be a finite fraction from 0 to 1")
    if type(min_vertices) is not int or not 1 <= min_vertices <= 10_000_000:
        raise ValueError("min_vertices must be an integer in [1, 10000000]")
    if (type(membership_threshold) not in (float, int) or
            not math.isfinite(membership_threshold) or not 0 < membership_threshold <= 1):
        raise ValueError("membership_threshold must be a finite fraction in (0, 1]")

    included, violating, max_foreign, examples = 0, 0, 0.0, []
    previous_index = -1
    for index, member_weight, weights in samples:
        if type(index) is not int or index <= previous_index:
            raise ValueError("Source vertices must be unique and sorted by ascending index")
        previous_index = index
        if (type(member_weight) not in (float, int) or
                not math.isfinite(member_weight) or not 0 <= member_weight <= 1):
            raise ValueError("Semantic group membership must be a finite weight in [0,1]")
        if not isinstance(weights, Mapping) or any(
                not isinstance(bone, str) or not bone or
                type(weight) not in (float, int) or
                not math.isfinite(weight) or not 0 <= weight <= 1
                for bone, weight in weights.items()):
            raise ValueError("Deform-bone weights must be named finite fractions")
        if member_weight < membership_threshold:
            continue
        included += 1
        foreign = sum(weight for bone, weight in weights.items()
                      if bone not in allowed_bones)
        max_foreign = max(max_foreign, foreign)
        if foreign > max_outside_weight + 1e-9:
            violating += 1
            if len(examples) < 12:
                examples.append({
                    "vertex": index,
                    "outside_weight": round(foreign, 6),
                    "outside_bones": sorted(
                        bone for bone, weight in weights.items()
                        if bone not in allowed_bones and weight > 0),
                })
    return {
        "selected_vertices": included,
        "minimum_required_vertices": min_vertices,
        "violating_vertices": violating,
        "max_outside_weight_observed": round(max_foreign, 6),
        "max_outside_weight_allowed": float(max_outside_weight),
        "membership_threshold": float(membership_threshold),
        "worst_examples": examples,
        "status": ("fail" if included < min_vertices or violating else "pass"),
    }
