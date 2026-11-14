"""Conservative, deterministic redistribution of misassigned skin weights.

Never infers character anatomy. A reviewed painted semantic region supplies the
set of deform bones allowed to influence its vertices. Only existing allowed
influences may receive weight removed from disallowed deform bones.

Pure standard-library implementation, shared by unit tests and Blender worker.
"""
from __future__ import annotations

import math
from collections.abc import Mapping


def _fraction(value: object, name: str) -> float:
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not 0 <= value <= 1):
        raise ValueError(f"{name} must be a finite fraction in [0, 1]")
    return float(value)


def propose_weight_repair(
    weights: Mapping[str, float],
    allowed_bones: set[str] | frozenset[str],
    *,
    outside_tolerance: float,
    max_removed_weight: float = 0.15,
    total_tolerance: float = 0.02,
) -> dict | None:
    """Create one repair without changing the input.

    Returns None if already within tolerance. Raises for uncertain cases:
    unnormalized source weights, no existing allowed bone influence, excessive
    foreign weight, or an invalid configuration. Never invents a new influence.
    """
    if (not allowed_bones or not isinstance(allowed_bones, (set, frozenset))
            or any(not isinstance(b, str) or not b for b in allowed_bones)):
        raise ValueError("allowed_bones must be a nonempty set of names")
    tolerance = _fraction(outside_tolerance, "outside_tolerance")
    maximum = _fraction(max_removed_weight, "max_removed_weight")
    if not 0 < maximum <= 0.5:
        raise ValueError("max_removed_weight must be within (0, 0.5]")
    if tolerance >= maximum:
        raise ValueError("outside_tolerance must be smaller than max_removed_weight")
    if (type(total_tolerance) not in (int, float) or
            not math.isfinite(total_tolerance) or
            not 0 <= total_tolerance <= 0.02):
        raise ValueError("total_tolerance must be a fraction from 0 to 0.02")
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError("The source vertex has no deform-bone influences")
    for name, weight in weights.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Invalid deform-bone name")
        _fraction(weight, name)
    total = math.fsum(weights.values())
    if abs(total - 1) > total_tolerance + 1e-9:
        raise ValueError(f"Source deform weights are not normalized: {total:.6f}")
    forbidden = {bone: float(weight) for bone, weight in weights.items()
                 if bone not in allowed_bones and weight > 0}
    outside = math.fsum(forbidden.values())
    if outside <= tolerance + 1e-9:
        return None
    if outside > maximum + 1e-9:
        raise ValueError(f"Disallowed influence {outside:.6f} exceeds maximum repair {maximum:.6f}")
    allowed = {bone: float(weight) for bone, weight in weights.items()
               if bone in allowed_bones and weight > 0}
    kept_sum = math.fsum(allowed.values())
    if kept_sum <= 1e-8:
        raise ValueError("No existing allowed bone influence; cannot infer anatomy")
    new_weights = {bone: weight / kept_sum for bone, weight in sorted(allowed.items())}
    if len(new_weights) > 4:
        raise ValueError("Repair would retain more than four deform influences")
    return {
        "before": {bone: float(weight) for bone, weight in sorted(weights.items()) if weight > 0},
        "after": new_weights,
        "removed_outside_weight": round(outside, 9),
        "removed_bones": sorted(forbidden),
        "max_absolute_weight_change": round(max(
            abs(new_weights.get(bone, 0.0) - float(weight))
            for bone, weight in weights.items()
        ), 9),
    }
