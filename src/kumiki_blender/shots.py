"""Renderer-independent, deterministic quality-inspection camera views."""

from __future__ import annotations

from dataclasses import dataclass

from .lattice import Vec3
from .recipe import TowerRecipe


@dataclass(frozen=True)
class QualityShot:
    name: str
    position: Vec3
    target: Vec3
    ortho_scale: float


def plan_quality_shots(recipe: TowerRecipe) -> tuple[QualityShot, ...]:
    """Cover the full silhouette at 16:9 and one close-up construction detail.

    Blender's orthographic scale describes the horizontal field for landscape
    renders. A tall tower needs a much larger value than its vertical height
    to fit within a 16:9 image, including its antenna and ground clearance.
    """
    h = recipe.height_m
    return (
        QualityShot("hero", (h * 0.8, -h * 1.25, h * 0.73), (0, 0, h * 0.57), h * 2.65),
        QualityShot("front", (0, -h * 1.4, h * 0.56), (0, 0, h * 0.57), h * 2.5),
        QualityShot("side", (h * 1.4, 0, h * 0.56), (0, 0, h * 0.57), h * 2.5),
        QualityShot("detail", (h * 0.3, -h * 0.45, h * 0.86), (0, 0, h * 0.72), h * 0.46),
    )


def plan_portico_shots(bounds_min: Vec3, bounds_max: Vec3) -> tuple[QualityShot, ...]:
    """Frame a modular arcade from one to twenty bays in landscape previews."""
    span = bounds_max[0] - bounds_min[0]
    height = bounds_max[2] - bounds_min[2]
    center_z = (bounds_min[2] + bounds_max[2]) / 2
    # The 16:9 vertical field must clear a taller pointed-arch cornice even
    # for the narrowest single-bay configuration.
    vertical_fit = height * 1.28 * (16 / 9)
    return (
        QualityShot("hero", (span * 0.65, -span * 1.55, center_z + height),
                    (0, 0, center_z), max(13, span * 1.38, vertical_fit)),
        QualityShot("front", (0, -span * 1.5, center_z + 1.3),
                    (0, 0, center_z), max(12, span * 1.32, vertical_fit)),
        QualityShot("detail", (min(2.1, span/4) + 3, -9, center_z + 3.5),
                    (min(2.1, span/4), 0, center_z + 1.0), 4.8),
    )


def plan_capital_shot(bays: int, *, pitch: float = 4.2) -> QualityShot:
    """Target an actual central column, not empty space between arcade bays."""
    if type(bays) is not int or not 1 <= bays <= 20:
        raise ValueError("Column closeup needs 1–20 bays")
    column_x = -pitch * bays / 2 + (bays // 2) * pitch
    return QualityShot("capital", (column_x + 0.8, -4.5, 5.2),
                       (column_x, 0.15, 4.32), 2.9)
