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
    """Cover the overall silhouette and one close-up construction detail."""
    h = recipe.height_m
    return (
        QualityShot("hero", (h * 0.8, -h * 1.25, h * 0.73), (0, 0, h * 0.52), h * 1.72),
        QualityShot("front", (0, -h * 1.4, h * 0.56), (0, 0, h * 0.54), h * 1.55),
        QualityShot("side", (h * 1.4, 0, h * 0.56), (0, 0, h * 0.54), h * 1.55),
        QualityShot("detail", (h * 0.3, -h * 0.45, h * 0.86), (0, 0, h * 0.72), h * 0.46),
    )
