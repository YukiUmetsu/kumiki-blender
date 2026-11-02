"""Private photo-set intake and strictly observed 2D sculpture features."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from ..reconstruction.references import ReferenceSet
from .schema import (Annotation, AnnotationSet, LANDMARK_IDS, ORIGINS, Photo,
                     VIEW_LABELS, VISIBILITY, ViewAnnotations)


_DOSSIER_VIEWS = {
    "front": "front", "front-left": "front-three-quarter",
    "front-right": "front-three-quarter", "left-profile": "profile",
    "right-profile": "profile", "rear-left": "rear-three-quarter",
    "rear-right": "rear-three-quarter", "rear": "rear", "top": "unknown",
    "unknown": "unknown",
}


def _image_id(path: Path) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")
    if not slug or len(slug) > 96:
        raise ValueError(f"Invalid image_id derived from {path.name}")
    return slug


def load_photo_set(paths: Sequence[Path]) -> tuple[Photo, ...]:
    """Read original dimensions and hashes without changing or publishing images."""
    if not 1 <= len(paths) <= 256:
        raise ValueError("Photo set requires 1–256 images")
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as error:
        raise ValueError("Photo intake needs Pillow (install .[qa])") from error
    photos = []
    seen: set[str] = set()
    for input_path in paths:
        path = Path(input_path)
        if path.is_symlink():
            raise ValueError(f"Photo symlink is not permitted: {path.name}")
        if not path.is_file() or path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            raise ValueError(f"Photo must be an existing PNG/JPEG file: {path.name}")
        if path.stat().st_size > 50 * 1024 * 1024:
            raise ValueError(f"Photo exceeds 50 MiB: {path.name}")
        image_id = _image_id(path)
        if image_id in seen:
            raise ValueError(f"duplicate image_id: {image_id}")
        seen.add(image_id)
        try:
            with Image.open(path) as image:
                if image.format not in ("PNG", "JPEG"):
                    raise ValueError(f"Photo format does not match PNG/JPEG: {path.name}")
                width, height = image.size
                if width < 64 or height < 64 or width * height > 40_000_000:
                    raise ValueError(f"Photo dimensions outside 64px–40Mpx: {path.name}")
                image.load()
        except (OSError, UnidentifiedImageError) as error:
            raise ValueError(f"Cannot decode photo {path.name}: {error}") from error
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        photos.append(Photo(image_id, path.resolve(), width, height, digest))
    return tuple(photos)


def photos_from_reference_set(reference_set: ReferenceSet) -> tuple[Photo, ...]:
    """Adapt a *previously validated* reconstruction photo dossier, retaining original IDs.

    The existing reference loader owns file/hash validation; this adapter only
    maps its more detailed camera-angle vocabulary to face-analysis views.
    """
    if not isinstance(reference_set, ReferenceSet):
        raise ValueError("Expected a validated ReferenceSet")
    return tuple(Photo(image.image_id, image.path, image.dimensions[0], image.dimensions[1],
                       image.sha256, _DOSSIER_VIEWS[image.view],
                       "validated-reference-manifest / rights-unverified")
                 for image in reference_set.images)


def _point(value: Any, photo: Photo, label: str) -> tuple[float, float]:
    if (not isinstance(value, list) or len(value) != 2
            or any(type(component) not in (int, float) or not math.isfinite(component)
                   for component in value)):
        raise ValueError(f"{label} must contain two finite pixel coordinates")
    x, y = float(value[0]), float(value[1])
    if not (0 <= x < photo.width and 0 <= y < photo.height):
        raise ValueError(f"{label} coordinate outside {photo.image_id} photo dimensions")
    return (x, y)


def load_annotations(path: Path, photos: Sequence[Photo]) -> AnnotationSet:
    """Load reviewed 2D evidence; rear views may contain only head outlines."""
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Annotation file is missing, linked, or exceeds 2 MiB")
    try:
        data = json.loads(path.read_text(encoding="utf-8"),
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(f"Invalid annotation JSON: {error}") from error
    return parse_annotations_data(data, photos)


def parse_annotations_data(data: Any, photos: Sequence[Photo]) -> AnnotationSet:
    """Validate annotations from a private file or a recorded evidence bundle."""
    if not isinstance(data, dict) or set(data) != {"schema_version", "images"} or data["schema_version"] != 1:
        raise ValueError("Annotation JSON requires schema_version=1 and images")
    rows = data["images"]
    if not isinstance(rows, list) or len(rows) > 256:
        raise ValueError("Annotation images must be a list with at most 256 entries")
    photo_map = {photo.image_id: photo for photo in photos}
    images = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) - {"image_id", "view", "landmarks", "outline_px"}:
            raise ValueError(f"Invalid annotation image record {index}")
        name = row.get("image_id")
        if name not in photo_map or name in images:
            raise ValueError(f"Unknown or duplicate annotation image_id: {name}")
        view = row.get("view", "unknown")
        if view not in VIEW_LABELS:
            raise ValueError(f"Invalid view for image {name}")
        points = row.get("landmarks", [])
        if not isinstance(points, list) or len(points) > len(LANDMARK_IDS):
            raise ValueError(f"Invalid landmark list for image {name}")
        seen: set[str] = set()
        observations = []
        for entry in points:
            if not isinstance(entry, dict) or set(entry) != {
                    "landmark_id", "xy_px", "origin", "visibility"}:
                raise ValueError(f"Invalid landmark record for image {name}")
            landmark_id = entry["landmark_id"]
            if landmark_id not in LANDMARK_IDS:
                raise ValueError(f"Unknown landmark {landmark_id}")
            if landmark_id in seen:
                raise ValueError(f"duplicate landmark {landmark_id} in {name}")
            seen.add(landmark_id)
            if entry["origin"] not in ORIGINS or entry["visibility"] not in VISIBILITY:
                raise ValueError(f"Invalid origin/visibility for landmark {landmark_id}")
            if view == "rear" and entry["visibility"] == "visible":
                raise ValueError(f"Facial landmark cannot be visible in rear view: {landmark_id}")
            if entry["visibility"] == "not-visible":
                if entry["xy_px"] is not None:
                    raise ValueError(f"not-visible landmark must have null coordinate: {landmark_id}")
                xy = None
            else:
                xy = _point(entry["xy_px"], photo_map[name], landmark_id)
            observations.append(Annotation(name, landmark_id, xy, entry["origin"],
                                           entry["visibility"]))
        outline = row.get("outline_px")
        if outline is not None:
            if not isinstance(outline, list) or not 3 <= len(outline) <= 2048:
                raise ValueError(f"Outline must have 3–2048 vertices: {name}")
            outline = tuple(_point(point, photo_map[name], "outline") for point in outline)
            signed_area = sum(a[0] * b[1] - b[0] * a[1]
                              for a, b in zip(outline, outline[1:] + outline[:1]))
            if abs(signed_area) < 1:
                raise ValueError(f"Degenerate outline for image {name}")
        images[name] = ViewAnnotations(name, view, tuple(observations), outline)
    return AnnotationSet(images)


def inspect_coverage(photos: Sequence[Photo], annotations: AnnotationSet | None) -> dict[str, Any]:
    """Summarize *reviewed* view evidence, never infer independent cameras."""
    rows = annotations.images if annotations else {}
    counts = Counter(rows[p.image_id].view if p.image_id in rows else p.view for p in photos)
    measured = sorted(p.image_id for p in photos if p.image_id in rows and any(
        observation.visibility == "visible" and observation.origin == "manual"
        for observation in rows[p.image_id].landmarks))
    return {
        "schema_version": 1,
        "images_total": len(photos),
        "image_ids": [p.image_id for p in photos],
        "view_counts": dict(sorted(counts.items())),
        "facial_landmark_views": measured,
        "view_coverage_status": "reviewed" if rows and len(rows) == len(photos)
                                and all(view.view != "unknown" for view in rows.values()) else "incomplete",
        "independent_camera_count": None,
        "limitations": "View labels are human-supplied; related frames are not proven independent camera observations.",
    }
