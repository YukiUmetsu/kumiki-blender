"""Small, immutable contracts shared by the statue-face evidence stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


VIEW_LABELS = frozenset({"front", "front-three-quarter", "profile", "rear-three-quarter",
                         "rear", "unknown"})
VISIBILITY = frozenset({"visible", "occluded", "uncertain", "not-visible"})
ORIGINS = frozenset({"manual", "proposed"})
LANDMARK_IDS = frozenset({
    "chin-tip", "jaw-left", "jaw-right", "cheek-left", "cheek-right",
    "eye-inner-left", "eye-inner-right", "eye-outer-left", "eye-outer-right",
    "brow-left", "brow-right", "nose-tip", "nose-bridge", "nose-base",
    "nostril-left", "nostril-right", "mouth-left", "mouth-right",
    "upper-lip", "lower-lip", "forehead-center", "temple-left", "temple-right",
    "ear-left", "ear-right", "hairline-center", "hairline-left", "hairline-right",
    "hair-mass-left", "hair-mass-right", "head-top", "neck-base",
})


@dataclass(frozen=True)
class Photo:
    image_id: str
    path: Path
    width: int
    height: int
    sha256: str
    view: str = "unknown"
    provenance: str = "user-supplied / rights-unverified"


@dataclass(frozen=True)
class Annotation:
    image_id: str
    landmark_id: str
    xy_px: tuple[float, float] | None
    origin: str
    visibility: str


@dataclass(frozen=True)
class ViewAnnotations:
    image_id: str
    view: str
    landmarks: tuple[Annotation, ...]
    outline_px: tuple[tuple[float, float], ...] | None = None


@dataclass(frozen=True)
class AnnotationSet:
    images: dict[str, ViewAnnotations]
