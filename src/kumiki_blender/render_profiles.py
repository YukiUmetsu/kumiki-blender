"""Deterministic render-budget profiles for production and CI smoke renders."""

from __future__ import annotations

import os
from dataclasses import dataclass


class RenderProfileError(ValueError):
    """An unsupported or unsafe render profile was requested."""


@dataclass(frozen=True)
class RenderProfile:
    name: str
    width: int
    height: int
    samples: int


def select_render_profile(width: int, height: int, samples: int, *, mode: str | None = None) -> RenderProfile:
    """Keep production quality, or cap CI to <=640x360 and <=8 samples.

    The CI profile changes only render quality, never scene geometry, mesh
    modifiers, export settings, material definitions or asset provenance.
    """
    if any(type(x) is not int or not 1 <= x <= 16_384 for x in (width, height)):
        raise RenderProfileError("Render width and height must be positive integers")
    if type(samples) is not int or not 1 <= samples <= 1024:
        raise RenderProfileError("Render samples must be an integer in [1, 1024]")
    name = os.environ.get("KUMIKI_RENDER_PROFILE", "production") if mode is None else mode
    if name == "production":
        return RenderProfile(name, width, height, samples)
    if name == "smoke":
        scale = min(1.0, 640 / width, 360 / height)
        return RenderProfile(name, max(1, round(width * scale)), max(1, round(height * scale)), min(samples, 8))
    raise RenderProfileError("KUMIKI_RENDER_PROFILE must be 'production' or 'smoke'")
