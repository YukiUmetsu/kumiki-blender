"""Private multiview-photo intake and OpenSplat dataset preparation."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from .dense import DenseSettings, _check_regular, _json, _sha256, plan_dense_reconstruction
from .references import (MAX_IMAGE_BYTES, MAX_IMAGES, ReferenceError, ReferenceImage,
                         ReferenceSet, _decode, _slug, audit_reference_set)


MIN_SPLAT_VIEWS = 8
_PHOTO_SUFFIXES = frozenset((".jpg", ".jpeg", ".png", ".webp"))


def load_photo_folder(path: str | Path, asset_id: str) -> ReferenceSet:
    """Accept local photos without pretending they have published source rights.

    These photos remain private. Camera viewpoint is deliberately unknown until
    COLMAP registers overlapping pixels; image file order is not camera pose.
    """
    _slug(asset_id, "asset_id")
    folder = Path(path)
    if folder.is_symlink():
        raise ReferenceError("Photo folder must not be a symbolic link")
    if not folder.is_dir():
        raise ReferenceError("Photo folder is missing or is not a directory")
    photographs = sorted((entry for entry in folder.iterdir()
                          if entry.suffix.lower() in _PHOTO_SUFFIXES),
                         key=lambda entry: entry.name)
    if not MIN_SPLAT_VIEWS <= len(photographs) <= MAX_IMAGES:
        raise ReferenceError(f"Photo folder requires {MIN_SPLAT_VIEWS}–{MAX_IMAGES} images")
    images: list[ReferenceImage] = []
    unique_hashes: set[str] = set()
    identities = []
    for number, photograph in enumerate(photographs, start=1):
        if photograph.is_symlink():
            raise ReferenceError(f"Photo must not be a symbolic link: {photograph.name}")
        if (not photograph.is_file() or photograph.stat().st_size > MAX_IMAGE_BYTES):
            raise ReferenceError(f"Photo is not a regular file <=50 MiB: {photograph.name}")
        dimensions, dhash = _decode(photograph, photograph.name)
        with photograph.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        images.append(ReferenceImage(
            f"photo-{number:04d}", photograph.resolve(), "local-unattributed",
            "unknown", ("object",), None, None, digest, dimensions, dhash))
        unique_hashes.add(digest)
        identities.append((photograph.name, digest))
    if len(unique_hashes) < MIN_SPLAT_VIEWS:
        raise ReferenceError(f"Photo folder requires {MIN_SPLAT_VIEWS} independent images, not duplicates")
    serialized = json.dumps({"asset_id": asset_id, "files": identities},
                            sort_keys=True, separators=(",", ":")).encode("utf-8")
    return ReferenceSet(asset_id, asset_id.replace("-", " "), ("unknown",),
                        ("object",), tuple(images), hashlib.sha256(serialized).hexdigest())


def plan_splat_input(references: ReferenceSet) -> dict:
    """Describe independently available images, not guessed camera overlap."""
    result = audit_reference_set(references)
    if result["independent_images"] < MIN_SPLAT_VIEWS:
        raise ReferenceError(f"Gaussian splats require {MIN_SPLAT_VIEWS} independent photographs")
    return {**result, "status": "candidate / camera-registration-required",
            "distribution_allowed": False,
            "missing_photo_rights": any(image.source_url == "local-unattributed"
                                        for image in references.images),
            "caution": "Image count and metadata cannot prove parallax or object fidelity."}


def plan_splat_dataset(reconstruction: str | Path, output: str | Path, *,
                       rectified_masks: str | Path | None = None) -> dict:
    """Reuse the dense-MVS camera/photo integrity gate without requiring OpenMVS."""
    settings = DenseSettings(refine=False, texture=False)
    return plan_dense_reconstruction(reconstruction, output, settings,
                                     rectified_masks=rectified_masks, executables={})


def stage_splat_dataset(reconstruction: str | Path, output: str | Path, *,
                        rectified_masks: str | Path | None = None) -> dict:
    """Copy hash-checked, undistorted COLMAP cameras and images for OpenSplat."""
    plan = plan_splat_dataset(reconstruction, output, rectified_masks=rectified_masks)
    source = Path(plan["source_revision"])
    target = Path(plan["output"])
    source_images = source / "undistorted" / "images"
    source_sparse = source / "undistorted" / "sparse"
    original_index = _json(source / "undistorted-reference-index.json")
    expected_photos = {entry["filename"]: entry["sha256"]
                       for entry in original_index["images"]}
    selected_sparse: list[tuple[Path, str]] = []
    for component in ("cameras", "images", "points3D"):
        file = next((source_sparse / f"{component}.{suffix}" for suffix in ("bin", "txt")
                    if (source_sparse / f"{component}.{suffix}").is_file()), None)
        if file is None or file.is_symlink() or file.stat().st_size > 512 * 1024 * 1024:
            raise ReferenceError(f"Unsafe or missing COLMAP sparse file: {component}")
        selected_sparse.append((file, _sha256(file)))
    target.mkdir(parents=True, exist_ok=False)
    images_target = target / "images"
    images_target.mkdir()
    sparse_target = target / "sparse" / "0"
    sparse_target.mkdir(parents=True)
    copied = []
    for name in plan["undistorted_images"]:
        origin = source_images / name
        _check_regular(origin, source)
        expected = expected_photos[name]
        if _sha256(origin) != expected:
            raise ReferenceError(f"Rectified source photo changed during staging: {name}")
        destination = images_target / name
        shutil.copyfile(origin, destination)
        actual = _sha256(destination)
        if actual != expected:
            raise ReferenceError(f"Rectified source photo changed during staging: {name}")
        copied.append({"name": name, "sha256": actual})
    for origin, expected in selected_sparse:
        destination = sparse_target / origin.name
        shutil.copyfile(origin, destination)
        if _sha256(destination) != expected:
            raise ReferenceError(f"COLMAP camera model changed during staging: {origin.name}")
    mask_count = 0
    if plan["rectified_masks"]:
        mask_target = target / "masks"
        mask_target.mkdir()
        for entry in plan["rectified_masks"]:
            image_name = entry["filename"]
            source_mask = Path(entry["path"])
            target_mask = mask_target / f"{Path(image_name).stem}.png"
            if target_mask.exists():
                raise ReferenceError("Rectified masks have conflicting file stems")
            if _sha256(source_mask) != entry["sha256"]:
                raise ReferenceError(f"Rectified mask changed during staging: {image_name}")
            shutil.copyfile(source_mask, target_mask)
            if _sha256(target_mask) != entry["sha256"]:
                raise ReferenceError(f"Rectified mask copy changed: {image_name}")
            mask_count += 1
    report = {"schema_version": 1, "source_revision": str(source),
              "asset_id": plan["asset_id"], "registered_images": plan["registered_images"],
              "images": len(copied), "masks": mask_count,
              "photo_hashes": copied, "distribution_allowed": False,
              "status": "staged / OpenSplat-training-required"}
    (target / "dataset-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
