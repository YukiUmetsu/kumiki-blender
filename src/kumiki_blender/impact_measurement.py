"""Conservative impact cues from *simulated* Blender rigid-body trajectories.

This does not expose Bullet's native contact manifold or impulse. It only
reports a floor encounter when geometry approaches the authored collision
plane, measured downward velocity is significant and subsequent simulation
frames show support/rebound. No contact means no invented SFX timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence


@dataclass(frozen=True)
class GroundSample:
    frame: int
    lowest_z: float
    center_z: float
    x: float
    y: float


@dataclass(frozen=True)
class PairRelativeSample:
    """Evaluated center-of-mass proxy: position of body A relative to B."""

    frame: int
    dx: float
    dy: float
    dz: float


def pair_impact_frames(samples: Sequence[PairRelativeSample], near_surface: Callable[[int], bool],
                       *, fps: int = 30, minimum_speed_mps: float = .25) -> list[tuple[int, float]]:
    """Conservative body-body candidate from simulated relative motion + true mesh proximity.

    `near_surface` is a *narrowphase geometry check* supplied by Blender. A
    fast pass-through without a post-contact slowdown is never an impact cue.
    Rotational-only/glancing hits may be omitted rather than misidentified.
    """
    if not 4 <= len(samples) <= 3601 or not 1 <= fps <= 240:
        return []
    cues = []
    last_impact = -100
    for i in range(2, len(samples) - 1):
        past, previous, current, upcoming = samples[i-2:i+2]
        if (current.frame - previous.frame != 1 or previous.frame - past.frame != 1
                or upcoming.frame - current.frame != 1 or current.frame - last_impact < 5):
            continue
        values = [value for item in (past, previous, current, upcoming)
                  for value in (item.dx, item.dy, item.dz)]
        if not all(math.isfinite(value) for value in values):
            continue
        separation = math.sqrt(current.dx**2 + current.dy**2 + current.dz**2)
        if separation < 1e-5:
            continue
        direction = (current.dx/separation, current.dy/separation, current.dz/separation)
        pre_speed = -fps * sum((b-a)*n for a, b, n in zip(
            (past.dx, past.dy, past.dz), (previous.dx, previous.dy, previous.dz), direction))
        if pre_speed < minimum_speed_mps:
            continue
        post_speed = fps * sum((b-a)*n for a, b, n in zip(
            (current.dx, current.dy, current.dz), (upcoming.dx, upcoming.dy, upcoming.dz), direction))
        if post_speed < -max(.3, .5*pre_speed):
            continue  # Bodies keep closing: possible pass-through, not verified impact.
        if not near_surface(i):
            continue
        cues.append((current.frame, round(pre_speed, 4)))
        last_impact = current.frame
    return cues


def ground_impact_frames(samples: Sequence[GroundSample], *, height_m: float,
                         half_extent_m: float, fps: int = 30,
                         minimum_speed_mps: float = .25) -> list[tuple[float, float]]:
    """Return estimated fractional contact frames and pre-contact speeds.

    Rejection conditions deliberately favor missing a quiet contact over
    falsely claiming an impact where a fast-moving object passes the plane.
    """
    if not 2 <= len(samples) <= 3601 or not 1 <= fps <= 240:
        return []
    tolerance = .012
    cues: list[tuple[float, float]] = []
    last_impact = -100
    for i in range(2, len(samples) - 1):
        past, previous, current, upcoming = samples[i-2:i+2]
        if (current.frame - previous.frame != 1 or previous.frame - past.frame != 1
                or upcoming.frame - current.frame != 1 or current.frame - last_impact < 5):
            continue
        if (abs(current.x) > half_extent_m or abs(current.y) > half_extent_m
                or any(not math.isfinite(value) for row in (past, previous, current, upcoming)
                       for value in (row.lowest_z, row.center_z, row.x, row.y))):
            continue
        previous_bottom = previous.lowest_z - height_m
        bottom = current.lowest_z - height_m
        if previous_bottom <= tolerance or bottom > tolerance:
            continue
        pre_speed = max(0.0, (past.center_z - previous.center_z)*fps)
        if pre_speed < minimum_speed_mps:
            continue
        # A body just falling through the plane will keep moving down; that
        # is *not* a solver contact and must never trigger a sound cue.
        if (upcoming.lowest_z < height_m - .018
                or (upcoming.center_z - current.center_z)*fps < -max(.3, .5*pre_speed)):
            continue
        interval = previous.lowest_z - current.lowest_z
        alpha = min(1.0, max(0.0, (previous_bottom - tolerance)/interval)) if interval > 1e-6 else 1.0
        cues.append((round(previous.frame + alpha, 4), round(pre_speed, 4)))
        last_impact = current.frame
    return cues
