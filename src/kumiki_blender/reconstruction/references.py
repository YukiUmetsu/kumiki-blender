"""Validated reference-only photographs and conservative coverage diagnostics.

The images stay local and are *never* interpreted as a grant to redistribute,
texture a deliverable, or claim survey-quality reconstruction.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


class ReferenceError(ValueError):
    """A photo dossier is untrustworthy or missing required evidence."""


MAX_MANIFEST_BYTES = 128 * 1024
MAX_IMAGES = 256
MAX_IMAGE_BYTES = 50 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
VIEWS = frozenset(("front", "front-left", "left-profile", "rear-left", "rear",
                   "rear-right", "right-profile", "front-right", "top", "unknown"))
_SLUG = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_SHA = re.compile(r"[a-f0-9]{64}\Z")


@dataclass(frozen=True)
class ReferenceImage:
    image_id: str
    path: Path
    source_url: str
    view: str
    regions: tuple[str, ...]
    mask: Path | None
    mask_sha256: str | None
    sha256: str
    dimensions: tuple[int, int]
    dhash: int


@dataclass(frozen=True)
class ReferenceSet:
    asset_id: str
    subject: str
    required_views: tuple[str, ...]
    required_regions: tuple[str, ...]
    images: tuple[ReferenceImage, ...]
    manifest_sha256: str


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReferenceError(f"Reference manifest has duplicate JSON key: {key}")
        result[key] = value
    return result


def _slug(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) > 96 or not _SLUG.fullmatch(value):
        raise ReferenceError(f"{label} must be a lowercase hyphenated slug")
    return value


def _string(value: Any, label: str, *, limit: int) -> str:
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= limit:
        raise ReferenceError(f"{label} must be nonempty text (max {limit} characters)")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ReferenceError(f"{label} contains control characters")
    return value.strip()


def _tags(value: Any, label: str, *, allowed: frozenset[str] | None = None) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or len(value) > 32:
        raise ReferenceError(f"{label} must contain 1–32 tags")
    tags = tuple(_slug(item, label) for item in value)
    if len(set(tags)) != len(tags):
        raise ReferenceError(f"{label} contains duplicate tags")
    if allowed is not None and any(tag not in allowed for tag in tags):
        raise ReferenceError(f"{label} contains an unsupported view")
    return tags


def _path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ReferenceError(f"{label} must be a relative local path")
    relative = Path(value)
    if relative.is_absolute() or any(part in ("..", ".") for part in relative.parts):
        raise ReferenceError(f"{label} must be a relative local path without traversal")
    try:
        resolved = (root / relative).resolve(strict=True)
    except OSError as error:
        raise ReferenceError(f"{label} file is unavailable: {value}") from error
    if not resolved.is_relative_to(root):
        raise ReferenceError(f"{label} escapes the manifest directory")
    if not resolved.is_file():
        raise ReferenceError(f"{label} must be a regular local file")
    if resolved.stat().st_size > MAX_IMAGE_BYTES:
        raise ReferenceError(f"{label} exceeds 50 MiB")
    if resolved.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
        raise ReferenceError(f"{label} must be a JPEG, PNG or WebP image")
    return resolved


def _decode(path: Path, label: str) -> tuple[tuple[int, int], int]:
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as error:
        raise ReferenceError("Reference image inspection requires: pip install 'kumiki-blender[qa]'") from error
    try:
        with Image.open(path) as source:
            width, height = source.size
            if min(width, height) < 64 or width * height > MAX_IMAGE_PIXELS:
                raise ReferenceError(f"{label} dimensions outside 64px–40Mpx safety bounds")
            source.load()
            # dHash is only a near-duplicate hint, never a geometric similarity score.
            scaled = source.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
            pixels = list(scaled.tobytes())
            digest = 0
            for row in range(8):
                for col in range(8):
                    digest = (digest << 1) | (pixels[row * 9 + col] > pixels[row * 9 + col + 1])
            return (width, height), digest
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        if isinstance(error, ReferenceError):
            raise
        raise ReferenceError(f"Could not decode {label}: {path.name}") from error


def _url(value: Any) -> str:
    result = _string(value, "source_url", limit=2048)
    try:
        parts = urlsplit(result)
    except ValueError as error:
        raise ReferenceError("source_url must be a valid HTTPS reference") from error
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
        raise ReferenceError("source_url must be HTTPS without embedded credentials")
    return result


def load_reference_set(path: str | Path) -> ReferenceSet:
    """Inspect a local dossier without changing the referenced files."""
    manifest = Path(path)
    try:
        if not manifest.is_file() or manifest.stat().st_size > MAX_MANIFEST_BYTES:
            raise ReferenceError("Reference manifest is missing or exceeds 128 KiB")
        contents = manifest.read_bytes()
        raw = json.loads(contents.decode("utf-8"), object_pairs_hook=_unique_object,
                         parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (OSError, UnicodeError, ValueError) as error:
        if isinstance(error, ReferenceError):
            raise
        raise ReferenceError(f"Invalid reference manifest JSON: {error}") from error
    if not isinstance(raw, dict) or set(raw) != {
            "schema_version", "asset_id", "subject", "required_views", "required_regions", "images"}:
        raise ReferenceError("Reference manifest has missing or unexpected fields")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ReferenceError("Reference manifest schema_version must be 1")
    asset_id = _slug(raw["asset_id"], "asset_id")
    subject = _string(raw["subject"], "subject", limit=200)
    required_views = _tags(raw["required_views"], "required_views", allowed=VIEWS)
    required_regions = _tags(raw["required_regions"], "required_regions")
    entries = raw["images"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_IMAGES:
        raise ReferenceError("images must contain 1–256 entries")
    root = manifest.resolve().parent
    images = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ReferenceError(f"images[{index}] must be an object")
        required = {"id", "file", "source_url", "view", "regions"}
        optional = {"mask", "expected_sha256"}
        if not required.issubset(entry) or set(entry) - (required | optional):
            raise ReferenceError(f"images[{index}] has missing or unexpected fields")
        image_id = _slug(entry["id"], f"images[{index}].id")
        if image_id in seen:
            raise ReferenceError(f"images has duplicate id: {image_id}")
        seen.add(image_id)
        file = _path(root, entry["file"], f"images[{index}].file")
        view = entry["view"]
        if not isinstance(view, str) or view not in VIEWS:
            raise ReferenceError(f"images[{index}].view must be a supported viewpoint")
        regions = _tags(entry["regions"], f"images[{index}].regions")
        mask = _path(root, entry["mask"], f"images[{index}].mask") if "mask" in entry else None
        dimensions, dhash = _decode(file, f"images[{index}].file")
        if mask is not None and _decode(mask, f"images[{index}].mask")[0] != dimensions:
            raise ReferenceError(f"images[{index}] mask dimensions differ from photograph")
        with file.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        mask_digest = None
        if mask is not None:
            with mask.open("rb") as stream:
                mask_digest = hashlib.file_digest(stream, "sha256").hexdigest()
        expected = entry.get("expected_sha256")
        if expected is not None and (not isinstance(expected, str) or not _SHA.fullmatch(expected)
                                     or expected != digest):
            raise ReferenceError(f"images[{index}].expected_sha256 differs from actual SHA-256")
        images.append(ReferenceImage(image_id, file, _url(entry["source_url"]), view,
                                     regions, mask, mask_digest, digest, dimensions, dhash))
    return ReferenceSet(asset_id, subject, required_views, required_regions,
                        tuple(images), hashlib.sha256(contents).hexdigest())


def audit_reference_set(reference_set: ReferenceSet) -> dict[str, Any]:
    """Report *available evidence*, not a guaranteed successful reconstruction."""
    images = reference_set.images
    exact_groups: dict[str, list[ReferenceImage]] = {}
    for image in images:
        exact_groups.setdefault(image.sha256, []).append(image)
    # Exact duplicates can safely be excluded from matching. Perceptual hashes
    # are only advisory: visually similar views may have useful parallax.
    independent = [group[0] for group in exact_groups.values()]
    parents = list(range(len(independent)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for i, image in enumerate(independent):
        for j in range(i):
            previous = independent[j]
            if (image.view == previous.view and
                    (image.dhash ^ previous.dhash).bit_count() <= 3):
                parents[root(i)] = root(j)
                break
    groups: dict[int, list[ReferenceImage]] = {}
    for index, image in enumerate(independent):
        groups.setdefault(root(index), []).append(image)
    representatives = [group[0] for group in groups.values()]
    view_counts = {view: sum(image.view == view for image in representatives)
                   for view in sorted({image.view for image in representatives})}
    region_counts = {region: sum(region in image.regions for image in representatives)
                     for region in sorted(set().union(*(set(image.regions) for image in representatives)))}
    return {
        "schema_version": 1,
        "asset_id": reference_set.asset_id,
        "manifest_sha256": reference_set.manifest_sha256,
        "source_images": len(images),
        "independent_images": len(independent),
        "conservative_coverage_images": len(representatives),
        "view_counts": view_counts,
        "region_counts": region_counts,
        "missing_views": [view for view in reference_set.required_views if view not in view_counts],
        "missing_regions": [region for region in reference_set.required_regions if region not in region_counts],
        "duplicate_groups": [[image.image_id for image in group] for group in exact_groups.values()
                             if len(group) > 1],
        "near_duplicate_groups": [[image.image_id for image in group] for group in groups.values()
                                  if len(group) > 1],
        "status": "review-required",
        "historical_fidelity_verified": False,
        "photogrammetry_readiness": "unverified: requires geometric feature matching and camera registration",
        "limitations": "Photo metadata/coverage and perceptual hashes cannot establish overlapping views, correct anatomy, camera calibration or mesh fidelity.",
    }
