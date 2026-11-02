"""NPS head/feature dimension ratios for an editable anatomical sculpt guide.

These transfer authentic *relative* measurements to the procedural study's
existing head height. They do not measure the existing mesh, establish the
unseen depth, or turn a generic sculpture into the original monument.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class LibertyHeadTargets:
    height: float
    width: float
    eye_width: float
    nose_length: float
    mouth_width: float


def scaled_nps_head_targets(height: float = 3.52) -> LibertyHeadTargets:
    """Transfer NPS's 5.26-metre cranium ratios to an illustrative head."""
    if isinstance(height, bool) or not isinstance(height, (int, float)) or (
            not math.isfinite(height) or not .1 <= height <= 100.):
        raise ValueError("Head height must be finite and within [0.1, 100] metres")
    ratio = float(height)/5.26
    return LibertyHeadTargets(float(height), 3.05*ratio, .76*ratio,
                              1.37*ratio, .91*ratio)
