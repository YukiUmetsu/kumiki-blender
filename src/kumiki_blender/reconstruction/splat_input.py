"""Private multiview-photo intake and OpenSplat dataset preparation."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import struct
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
    plan = plan_dense_reconstruction(reconstruction, output, settings,
                                     rectified_masks=rectified_masks, executables={})
    report = _json(Path(plan["source_revision"]) / "reconstruction-report.json")
    evidence = report.get("sparse_feature_evidence")
    if (not isinstance(evidence, dict) or
            type(evidence.get("triangulated_points")) is not int or
            evidence["triangulated_points"] < 1):
        raise ReferenceError("Gaussian splats require triangulated sparse 3D points, not cameras alone")
    plan["sparse_feature_evidence"] = evidence
    plan["viewpoint_qa"] = audit_splat_viewpoints(
        Path(plan["source_revision"]), plan["undistorted_images"])
    return plan


def audit_splat_viewpoints(reconstruction: Path, image_names: list[str]) -> dict:
    """Check actual registered camera poses, not inferred filename/view labels.

    A narrow-angle warning is advisory: valid stereo can have parallel optical
    axes. Camera-centre baseline, identities and intrinsics are hard integrity
    requirements. COLMAP's arbitrary scale is never presented as metres.
    """
    from .cameras import _quat_matrix, _read_intrinsics, _read_poses

    root = reconstruction / "undistorted" / "text-model"
    if root.is_symlink() or not root.is_dir():
        raise ReferenceError("Undistorted COLMAP text camera model is missing")
    poses_file = root / "images.txt"
    camera_file = root / "cameras.txt"
    for file, maximum in ((poses_file, 256 * 1024 * 1024),
                          (camera_file, 1024 * 1024)):
        _check_regular(file, reconstruction)
        if file.stat().st_size > maximum:
            raise ReferenceError(f"COLMAP text camera model is unexpectedly large: {file.name}")
    poses = _read_poses(poses_file)
    # The splat trainer preserves independent fx/fy; only the Blender camera
    # matching path requires square-pixel assumptions.
    intrinsics = _read_intrinsics(camera_file, require_square_pixels=False)
    if len(poses) != len(image_names) or set(item[2] for item in poses) != set(image_names):
        raise ReferenceError("Registered camera model names do not match indexed images")
    sparse_dir = reconstruction / "undistorted" / "sparse"
    binary_counts = {}
    binary_hashes = {}
    # Minimum bytes per record in COLMAP's binary sparse formats; variable
    # image names and feature tracks make actual records larger than this.
    minimum_record_bytes = {"cameras": 48, "images": 73, "points3D": 51}
    for name, expected in (("cameras", len(intrinsics)), ("images", len(poses)),
                           ("points3D", None)):
        source = sparse_dir / f"{name}.bin"
        if source.exists() or source.is_symlink():
            _check_regular(source, reconstruction)
            with source.open("rb") as stream:
                prefix = stream.read(8)
            if len(prefix) != 8:
                raise ReferenceError(f"COLMAP sparse binary header is truncated: {name}")
            count = struct.unpack("<Q", prefix)[0]
            if (expected is not None and count != expected) or (expected is None and count < 1):
                raise ReferenceError(f"COLMAP registered sparse model {name} count does not match verified evidence")
            if source.stat().st_size < 8 + count * minimum_record_bytes[name]:
                raise ReferenceError(f"COLMAP sparse binary {name} has truncated records")
            binary_counts[name] = count
            binary_hashes[source.name] = _sha256(source)
    index = _json(reconstruction / "undistorted-reference-index.json")
    expected_dimensions = {entry["filename"]: entry.get("dimensions")
                           for entry in index["images"]}
    centers = []
    forward = []
    for _, camera_id, filename, quaternion, translation in poses:
        camera = intrinsics.get(camera_id)
        if camera is None:
            raise ReferenceError(f"Registered camera missing intrinsics: {filename}")
        if [camera["width"], camera["height"]] != expected_dimensions[filename]:
            raise ReferenceError(f"Registered camera dimensions differ from rectified photo: {filename}")
        rotation = _quat_matrix(quaternion)
        center = tuple(-sum(rotation[row][axis] * translation[row]
                            for row in range(3)) for axis in range(3))
        centers.append(center)
        forward.append(tuple(rotation[2]))

    def distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
        return math.dist(a, b)

    span = max(distance(a, b) for i, a in enumerate(centers)
               for b in centers[i + 1:])
    if not math.isfinite(span) or span < 1e-9:
        raise ReferenceError("Registered cameras have no usable translation baseline")
    unique = []
    tolerance = max(span * 1e-6, 1e-10)
    for center in centers:
        if all(distance(center, existing) > tolerance for existing in unique):
            unique.append(center)
    maximum_angle = max(math.degrees(math.acos(max(-1.0, min(1.0,
                        sum(x * y for x, y in zip(a, b))))))
                        for i, a in enumerate(forward) for b in forward[i + 1:])
    warnings = []
    if maximum_angle < 20:
        warnings.append("narrow optical-axis coverage; review front/profile/rear image coverage")
    if len(unique) < len(centers):
        warnings.append("some photos share near-identical camera centers")
    return {"registered_cameras": len(poses), "intrinsic_camera_models": len(intrinsics),
            "unique_camera_centers": len(unique),
            "camera_center_span_colmap_units": round(span, 5),
            "maximum_optical_axis_angle_deg": round(maximum_angle, 2),
            "text_model_hashes": {"images.txt": _sha256(poses_file),
                                  "cameras.txt": _sha256(camera_file)},
            "sparse_binary_counts": binary_counts,
            "sparse_binary_hashes": binary_hashes,
            "warnings": warnings,
            "limitation": "Optical-axis spread and camera baseline do not prove target silhouette, depth, or scan fidelity"}


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
    for origin, digest in selected_sparse:
        expected = plan["viewpoint_qa"]["sparse_binary_hashes"].get(origin.name)
        if expected is not None and digest != expected:
            raise ReferenceError(f"COLMAP camera model changed after preflight: {origin.name}")
    text_model = source / "undistorted" / "text-model"
    for name, expected in plan["viewpoint_qa"]["text_model_hashes"].items():
        if _sha256(text_model / name) != expected:
            raise ReferenceError(f"COLMAP text camera model changed after preflight: {name}")
    target.mkdir(parents=True, exist_ok=False)
    images_target = target / "images"
    images_target.mkdir()
    sparse_target = target / "sparse" / "0"
    sparse_target.mkdir(parents=True)
    # Keep the exact undistorted text cameras used at training time. The
    # registered photos can then be compared against the resulting splat
    # without depending on a mutable external reconstruction directory.
    camera_target = target / "cameras"
    camera_target.mkdir()
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
    for name, expected in plan["viewpoint_qa"]["text_model_hashes"].items():
        source_camera = text_model / name
        destination = camera_target / name
        shutil.copyfile(source_camera, destination)
        if _sha256(destination) != expected:
            raise ReferenceError(f"COLMAP text camera model changed during staging: {name}")
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
              "viewpoint_qa": plan["viewpoint_qa"],
              "camera_text_hashes": plan["viewpoint_qa"]["text_model_hashes"],
              "photo_hashes": copied, "distribution_allowed": False,
              "status": "staged / OpenSplat-training-required"}
    (target / "dataset-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
