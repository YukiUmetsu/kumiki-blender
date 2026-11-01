"""Source-neutral, rights-declared Sphinx scan placement, independent of Blender.

The source's *front axis* describes the orientation after Blender's glTF
coordinate conversion. We cannot infer anatomical orientation from a bbox.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .asset_import import (
    AssetImportError, AssetProvenance, inspect_source_glb, load_asset_provenance,
)


FRONT_ANGLES = {"-X": 0.0, "+X": math.pi, "-Y": -math.pi / 2, "+Y": math.pi / 2}
# Egyptian State Information Service's published full monument envelope.
# Bounds are only an initial screening signal: a bronze replica can pass
# these ratios while its facial anatomy is still entirely inaccurate.
SPHINX_SURVEY_LENGTH_M = 73.5
SPHINX_SURVEY_WIDTH_M = 19.3
SPHINX_SURVEY_HEIGHT_M = 20.0


@dataclass(frozen=True)
class SphinxScan:
    source: Path
    provenance: AssetProvenance
    inspection: dict[str, Any]
    front_axis: str


@dataclass(frozen=True)
class ScanAlignment:
    rotation_z_rad: float
    uniform_scale: float
    translation_m: tuple[float, float, float]
    source_bounds_m: tuple[tuple[float, float, float], tuple[float, float, float]]
    aligned_bounds_m: tuple[tuple[float, float, float], tuple[float, float, float]]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "rotation_z_deg": round(math.degrees(self.rotation_z_rad), 6),
            "uniform_scale": self.uniform_scale,
            "translation_m": self.translation_m,
            "source_bounds_m": self.source_bounds_m,
            "aligned_bounds_m": self.aligned_bounds_m,
        }


def sphinx_proportion_review(alignment: ScanAlignment, *,
                             tolerance: float = 0.25) -> dict[str, Any]:
    """Surface a reproducible geometry-fidelity *warning*, not an approval.

    Bounding-box dimensions cannot verify nose, head, nemes, paws, flank or
    repair geometry. This report is intentionally distinct from asset safety
    and watertightness checks and never claims archaeological certification.
    """
    if not isinstance(alignment, ScanAlignment):
        raise AssetImportError("Sphinx proportion review requires an aligned GLB scan")
    if type(tolerance) not in (float, int) or not math.isfinite(tolerance) or not 0 < tolerance < 1:
        raise AssetImportError("Sphinx tolerance must be between 0 and 1")
    lo, hi = alignment.aligned_bounds_m
    length, width, height = (hi[axis]-lo[axis] for axis in range(3))
    observed = (width/length, height/length)
    reference = (SPHINX_SURVEY_WIDTH_M/SPHINX_SURVEY_LENGTH_M,
                 SPHINX_SURVEY_HEIGHT_M/SPHINX_SURVEY_LENGTH_M)
    delta = tuple((actual/expected)-1 for actual, expected in zip(observed, reference))
    plausible = all(abs(difference) <= tolerance for difference in delta)
    return {
        "reference": "Egyptian State Information Service, Great Sphinx dimensional envelope",
        "reference_dimensions_m": {
            "length": SPHINX_SURVEY_LENGTH_M,
            "width": SPHINX_SURVEY_WIDTH_M,
            "height": SPHINX_SURVEY_HEIGHT_M,
        },
        "ratios": {
            "width_to_length": round(observed[0], 5),
            "height_to_length": round(observed[1], 5),
            "reference_width_to_length": round(reference[0], 5),
            "reference_height_to_length": round(reference[1], 5),
        },
        "relative_deviation": {
            "width": round(delta[0], 5),
            "height": round(delta[1], 5),
        },
        "passes_bounding_envelope_screen": plausible,
        "status": ("bounds-plausible / anatomical source verification still required"
                   if plausible else
                   "review-required / bbox deviates from real monument proportions"),
        "unverified_details": [
            "head and nemes morphology", "face anatomy and missing nose",
            "foreleg/paw geometry", "hindquarters and damaged masonry repairs",
            "provenance as an actual monument scan rather than a replica",
        ],
    }


def prepare_sphinx_scan(source: str | Path, record: str | Path, *, front_axis: str = "-X") -> SphinxScan:
    """Preflight provenance and static GLB before creating a Blender scene."""
    if front_axis not in FRONT_ANGLES:
        raise AssetImportError(f"Unknown Sphinx front axis {front_axis!r}")
    provenance = load_asset_provenance(record)
    license_upper = provenance.license.upper()
    if provenance.commercial_use_allowed is not True or "-NC" in license_upper or "-ND" in license_upper:
        raise AssetImportError(
            "Sphinx scan needs commercial_use_allowed: true and a license without NC/ND restrictions")
    inspection = inspect_source_glb(source, provenance)
    return SphinxScan(Path(source).resolve(), provenance, inspection, front_axis)


def fit_sphinx_scan(
    bounds_min: tuple[float, float, float],
    bounds_max: tuple[float, float, float],
    *,
    front_axis: str,
    target_length_m: float,
    target_center_xy: tuple[float, float],
    ground_z: float = 0.02,
) -> ScanAlignment:
    """Uniformly fit a tightly cropped static scan to a landmark site.

    Orientation is explicit; anisotropic scaling would distort the statue.
    Recompute bounds for 90/180-degree rotations rather than reusing world AABB.
    """
    if front_axis not in FRONT_ANGLES:
        raise AssetImportError("Sphinx front axis must be -X, +X, -Y, or +Y")
    values = (*bounds_min, *bounds_max, target_length_m, *target_center_xy, ground_z)
    if any(type(number) not in (float, int) or not math.isfinite(number) for number in values):
        raise AssetImportError("Sphinx alignment requires finite numeric bounds and dimensions")
    if target_length_m <= 0 or target_length_m > 1000:
        raise AssetImportError("Sphinx target length must be in (0, 1000] meters")
    sizes = tuple(b - a for a, b in zip(bounds_min, bounds_max))
    if any(size <= 1e-5 for size in sizes):
        raise AssetImportError("Sphinx scan has a zero-sized or inverted dimension")
    angle = FRONT_ANGLES[front_axis]
    cosine, sine = math.cos(angle), math.sin(angle)
    rotated = [(cosine*x-sine*y, sine*x+cosine*y)
               for x in (bounds_min[0], bounds_max[0])
               for y in (bounds_min[1], bounds_max[1])]
    rx0, rx1 = min(p[0] for p in rotated), max(p[0] for p in rotated)
    ry0, ry1 = min(p[1] for p in rotated), max(p[1] for p in rotated)
    length = rx1-rx0
    width_ratio, height_ratio = (ry1-ry0)/length, sizes[2]/length
    if not (0.06 <= width_ratio <= 0.9 and 0.06 <= height_ratio <= 0.9):
        raise AssetImportError(
            "Scan bounds are implausible for a cropped recumbent Sphinx; "
            "check front axis and remove excess site terrain")
    scale = target_length_m/length
    tx = target_center_xy[0] - scale*(rx0+rx1)/2
    ty = target_center_xy[1] - scale*(ry0+ry1)/2
    tz = ground_z - scale*bounds_min[2]
    aligned = ((scale*rx0+tx, scale*ry0+ty, ground_z),
               (scale*rx1+tx, scale*ry1+ty, scale*bounds_max[2]+tz))
    return ScanAlignment(angle, scale, (tx, ty, tz), (bounds_min, bounds_max), aligned)
