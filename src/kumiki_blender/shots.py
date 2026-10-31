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
    return (
        QualityShot("hero", (span * 0.65, -span * 1.55, 10.5),
                    (0, 0, 3.7), max(13, span * 1.38)),
        QualityShot("front", (0, -span * 1.5, 5.0),
                    (0, 0, 3.6), max(12, span * 1.32)),
        QualityShot("detail", (4.2, -9, 7.5), (0, 0, 5.1), 4.8),
    )


def plan_capital_shot(bays: int, *, pitch: float = 4.2) -> QualityShot:
    """Target an actual central column, not empty space between arcade bays."""
    if type(bays) is not int or not 1 <= bays <= 20:
        raise ValueError("Column closeup needs 1–20 bays")
    column_x = -pitch * bays / 2 + (bays // 2) * pitch
    return QualityShot("capital", (column_x + 0.8, -4.5, 5.2),
                       (column_x, 0.15, 4.32), 2.9)
