"""Conservative impact cues from *simulated* Blender rigid-body trajectories.

This does not expose Bullet's native contact manifold or impulse. It only
reports a floor encounter when geometry approaches the authored collision
plane, measured downward velocity is significant and subsequent simulation
frames show support/rebound. No contact means no invented SFX timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class GroundSample:
    frame: int
    lowest_z: float
    center_z: float
    x: float
    y: float


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
