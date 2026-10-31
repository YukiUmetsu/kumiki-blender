"""Reusable architectural reflection cards for Blender-only studio lighting.

This rig adds physically sized rectangular area lights; it never changes a
model's glTF materials or exports lights into the portable GLB. A Three.js
viewer should supply an environment map for equivalent glazing reflections.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .blender_output import aim_at


@dataclass(frozen=True)
class ReflectionCard:
    name: str
    location: tuple[float, float, float]
    target: tuple[float, float, float]
    width: float
    height: float
    energy: float
    color: tuple[float, float, float]


def plan_architectural_reflections(*, center_x: float, facade_y: float,
                                  center_z: float, span: float) -> tuple[ReflectionCard, ...]:
    """Plan two offset softboxes to give otherwise dark glazing light to reflect."""
    if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in (center_x, facade_y, center_z, span)) or not 2 <= span <= 120):
        raise ValueError("Reflection rig requires finite coordinates and a 2–120 m span")
    return (
        ReflectionCard("Sky / tall glass reflection",
                       (center_x + span * 0.61, facade_y - span * 0.70, center_z + span * 0.40),
                       (center_x, facade_y, center_z),
                       max(0.8, span * 0.12), max(3.0, span * 0.52),
                       span * 87, (0.83, 0.91, 1.0)),
        ReflectionCard("Sky / side glass reflection",
                       (center_x - span * 0.62, facade_y - span * 0.49, center_z + span * 0.24),
                       (center_x, facade_y, center_z),
                       max(0.7, span * 0.095), max(2.5, span * 0.44),
                       span * 43, (0.92, 0.94, 1.0)),
    )


def add_architectural_reflections(bpy: Any, collection: Any, *, center_x: float,
                                 facade_y: float, center_z: float, span: float) -> tuple[Any, ...]:
    """Create specular-dominant Cycles softboxes in the studio-only collection."""
    objects = []
    for spec in plan_architectural_reflections(center_x=center_x, facade_y=facade_y,
                                               center_z=center_z, span=span):
        light = bpy.data.lights.new(spec.name, type="AREA")
        light.shape = "RECTANGLE"
        light.size = spec.width
        light.size_y = spec.height
        light.energy = spec.energy
        light.color = spec.color
        light.diffuse_factor = 0.18
        light.specular_factor = 1.0
        obj = bpy.data.objects.new(spec.name, light)
        collection.objects.link(obj)
        obj.location = spec.location
        aim_at(obj, spec.target)
        objects.append(obj)
    return tuple(objects)
