"""Optional, source-neutral COLMAP sparse-reconstruction worker.

This deliberately stops before dense meshing and texturing: online photos are
often non-overlapping, and even a successful SfM model cannot authenticate a
statue, resolve missing anatomy, or clear source-media redistribution rights.
"""

from __future__ import annotations

import hashlib
import io
import importlib.util
import json
import sqlite3
import math
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from .references import ReferenceError, ReferenceImage, ReferenceSet, audit_reference_set


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _probe_colmap(executable: str | list[str]) -> tuple[dict[str, str], tuple[str, str]]:
    """Negotiate CPU flags across legacy SIFT and newer Feature* CLIs."""
    help_texts = []
    try:
        prefix = [executable] if isinstance(executable, str) else executable
        for command in ("feature_extractor", "exhaustive_matcher"):
            result = subprocess.run([*prefix, command, "--help"],
                                    capture_output=True, text=True, timeout=20, check=False)
            help_texts.append(result.stdout + result.stderr)
        version = subprocess.run([*prefix, "--version"], capture_output=True,
                                 text=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ReferenceError(f"COLMAP CLI preflight failed: {error}") from error
    modern = ("--FeatureExtraction.use_gpu" in help_texts[0] and
              "--FeatureMatching.use_gpu" in help_texts[1])
    legacy = ("--SiftExtraction.use_gpu" in help_texts[0] and
              "--SiftMatching.use_gpu" in help_texts[1])
    if not modern and not legacy:
        raise ReferenceError("Unsupported COLMAP CLI: cannot determine CPU feature/matching flags")
    family = "feature" if modern else "sift"
    flags = (("--FeatureExtraction.use_gpu", "--FeatureMatching.use_gpu") if modern else
             ("--SiftExtraction.use_gpu", "--SiftMatching.use_gpu"))
    version_line = (version.stdout + version.stderr).strip().splitlines()
    return {"executable": (str(Path(executable).resolve()) if isinstance(executable, str)
                           else " ".join(prefix)),
            "version": version_line[0][:200] if version_line else "unreported",
            "cpu_flag_family": family}, flags


def _write_json(path: Path, content: dict[str, Any]) -> None:
    path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")


def _staged_images(reference_set: ReferenceSet, report: dict[str, Any]) -> list[ReferenceImage]:
    # Representative-only matching: duplicate images cannot become fake camera baselines.
    excluded = {name for group in report["duplicate_groups"] for name in group[1:]}
    return [image for image in reference_set.images if image.image_id not in excluded]


def _repair_jpeg_orientation(data: bytes) -> tuple[bytes, bool]:
    """Repair invalid EXIF orientation without recompressing a single pixel.

    Some real camera exports store EXIF Orientation=0. COLMAP/PyCOLMAP 4.0
    aborts the entire process at C++ level on such images. Correct the inline
    TIFF SHORT tag on a *staged copy* only; all other JPEG bytes stay identical.
    Rotated but valid orientations 1–8 are left untouched.
    """
    if not data.startswith(b"\xff\xd8"):
        return data, False
    offset = 2
    while offset + 4 < len(data):
        if data[offset] != 0xff:
            break
        while offset < len(data) and data[offset] == 0xff:
            offset += 1
        if offset >= len(data):
            break
        marker = data[offset]
        offset += 1
        if marker == 0xda:  # Start of compressed image scan
            break
        if marker in (0xd8, 0xd9, 0x01) or 0xd0 <= marker <= 0xd7:
            continue
        if offset + 2 > len(data):
            break
        size = int.from_bytes(data[offset:offset + 2], "big")
        if size < 2 or offset + size > len(data):
            break
        begin, end = offset + 2, offset + size
        offset = end
        if marker != 0xe1 or data[begin:begin + 6] != b"Exif\x00\x00":
            continue
        base = begin + 6
        if base + 8 > end or data[base:base + 2] not in (b"II", b"MM"):
            continue
        endian = "little" if data[base:base + 2] == b"II" else "big"
        if int.from_bytes(data[base + 2:base + 4], endian) != 42:
            continue
        table = base + int.from_bytes(data[base + 4:base + 8], endian)
        if table < base + 8 or table + 2 > end:
            continue
        count = int.from_bytes(data[table:table + 2], endian)
        for index in range(count):
            tag = table + 2 + index * 12
            if tag + 12 > end:
                break
            if int.from_bytes(data[tag:tag + 2], endian) != 274:
                continue
            # EXIF orientation must be SHORT (type 3), count 1, inline.
            if (int.from_bytes(data[tag + 2:tag + 4], endian) != 3 or
                    int.from_bytes(data[tag + 4:tag + 8], endian) != 1):
                return data, False
            old = int.from_bytes(data[tag + 8:tag + 10], endian)
            if 1 <= old <= 8:
                return data, False
            corrected = bytearray(data)
            corrected[tag + 8:tag + 10] = (1).to_bytes(2, endian)
            return bytes(corrected), True
    return data, False


def _orient_mask(mask, orientation: int):
    """Apply the same EXIF pixel transform used on the source photograph."""
    from PIL import Image

    transforms = {
        2: Image.Transpose.FLIP_LEFT_RIGHT,
        3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM,
        5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_270,
        7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_90,
    }
    return mask.transpose(transforms[orientation]) if orientation in transforms else mask


def _prepare_images(references: list[ReferenceImage], output: Path, *,
                    normalize_for_colmap: bool = False) -> bool:
    """Stage immutable photos and masks; optionally normalize to RGB PNG.

    PyCOLMAP's image decoder cannot reliably import WebP or some JPEG variants
    that Pillow accepts. For real camera reconstruction, transcode all inputs
    into standard RGB PNG (including unusual JPEGs) without modifying originals.
    EXIF orientation and alpha are resolved explicitly; the staged hash,
    dimensions, source hash and original filename remain auditable.
    """
    from PIL import Image, ImageOps

    image_root = output / "images"
    image_root.mkdir()
    has_masks = any(image.mask is not None for image in references)
    mask_root = output / "masks"
    if has_masks:
        mask_root.mkdir()
    index = []
    for image in references:
        filename = image.image_id + (".png" if normalize_for_colmap
                                     else image.path.suffix.lower())
        staged = image_root / filename
        corrected_exif = False
        orientation = 1
        source_format = None
        source_mode = None
        alpha_composited = False
        if normalize_for_colmap:
            # Decode the same exact bytes that were validated during intake.
            data = image.path.read_bytes()
            if hashlib.sha256(data).hexdigest() != image.sha256:
                raise ReferenceError(f"Reference {image.image_id} changed during image staging")
            try:
                with Image.open(io.BytesIO(data)) as source:
                    source.load()
                    source_format, source_mode = source.format, source.mode
                    orientation = source.getexif().get(274, 1)
                    orientation = orientation if orientation in range(1, 9) else 1
                    upright = ImageOps.exif_transpose(source)
                    if "A" in upright.getbands() or "transparency" in upright.info:
                        rgba = upright.convert("RGBA")
                        normalized = Image.new("RGB", rgba.size, (255, 255, 255))
                        normalized.paste(rgba, mask=rgba.getchannel("A"))
                        alpha_composited = True
                    else:
                        normalized = upright.convert("RGB")
                    dimensions = normalized.size
                    normalized.save(staged, format="PNG")
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                raise ReferenceError(f"Cannot normalize reference {image.image_id} for COLMAP: {error}") from error
        else:
            shutil.copyfile(image.path, staged)
            if _sha256(staged) != image.sha256:
                raise ReferenceError(f"Reference {image.image_id} changed during image staging")
            if staged.suffix.lower() in (".jpg", ".jpeg"):
                clean, corrected_exif = _repair_jpeg_orientation(staged.read_bytes())
                if corrected_exif:
                    staged.write_bytes(clean)
            dimensions = image.dimensions
        if has_masks:
            if image.mask is not None:
                if _sha256(image.mask) != image.mask_sha256:
                    raise ReferenceError(f"Reference mask {image.image_id} changed during image staging")
                with Image.open(image.mask) as original:
                    mask = original.convert("L")
                if normalize_for_colmap:
                    mask = _orient_mask(mask, orientation)
            else:
                mask = Image.new("L", dimensions, 255)
            if mask.size != dimensions:
                raise ReferenceError(f"Staged mask {image.image_id} dimensions differ from photograph")
            mask.save(mask_root / (filename + ".png"))
        index.append({"image_id": image.image_id, "filename": filename,
                      "original_filename": image.path.name,
                      "source_url": image.source_url, "view": image.view,
                      "regions": image.regions, "sha256": _sha256(staged),
                      "original_sha256": image.sha256,
                      "normalized_for_colmap": normalize_for_colmap,
                      "source_format": source_format, "source_mode": source_mode,
                      "alpha_composited_on_white": alpha_composited,
                      "exif_orientation_applied": orientation if normalize_for_colmap else None,
                      "invalid_exif_orientation_normalized": corrected_exif,
                      "staged_mask_sha256": (_sha256(mask_root / (filename + ".png"))
                                             if has_masks else None),
                      "original_dimensions": list(image.dimensions),
                      "dimensions": list(dimensions), "usage": "reference-only"})
    _write_json(output / "staged-reference-index.json", {"images": index,
        "notes": "Local source references only. COLMAP-normalized pixels are private, not redistributable."})
    return has_masks


def _feature_import_report(database: Path, staged_index: Path) -> dict:
    """Audit what the COLMAP decoder *actually* imported, not what was staged."""
    staged = json.loads(staged_index.read_text(encoding="utf-8"))
    expected = {item["filename"] for item in staged["images"]}
    try:
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
            imported = {row[0] for row in db.execute("SELECT name FROM images")}
    except sqlite3.DatabaseError as error:
        # Some synthetic COLMAP test fixtures do not create SQLite databases.
        return {"status": "unverified / database-unreadable",
                "staged_images": len(expected), "error": str(error)[:300]}
    missing = sorted(expected - imported)
    unexpected = sorted(imported - expected)
    return {"status": "complete" if not missing and not unexpected else "incomplete",
            "staged_images": len(expected), "imported_images": len(imported),
            "missing_images": missing, "unexpected_images": unexpected}



def _run(executable: str | list[str], stage: str, arguments: list[str], output: Path,
         number: int, timeout_seconds: int) -> tuple[int, list[str]]:
    prefix = [executable] if isinstance(executable, str) else executable
    command = [*prefix, stage, *arguments]
    log = output / "logs" / f"{number:02d}-{stage}.log"
    try:
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                    check=False, timeout=timeout_seconds)
        return result.returncode, [stage, *arguments]
    except subprocess.TimeoutExpired:
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"\nCOLMAP stage timed out after {timeout_seconds} seconds\n")
        return 124, [stage, *arguments]
    except OSError as error:
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"\nFailed to start COLMAP stage: {error}\n")
        return 127, [stage, *arguments]


def _registered_image_count(text_model: Path) -> int:
    """Parse COLMAP's alternating pose/observations lines, not numeric points."""
    images = text_model / "images.txt"
    if not images.is_file():
        return 0
    camera_ids: set[int] = set()
    lines = iter(images.read_text(encoding="utf-8").splitlines())
    for line in lines:
        if not line or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if (len(parts) >= 10 and parts[0].isdigit() and parts[8].isdigit()
                and all(_finite_float(value) for value in parts[1:8])):
            camera_ids.add(int(parts[0]))
            # The following row contains 2D feature triples, which can also
            # consist entirely of integers and look like a camera pose.
            next(lines, None)
    return len(camera_ids)


def _finite_float(value: str) -> bool:
    try:
        from math import isfinite
        return isfinite(float(value))
    except ValueError:
        return False


def _sparse_point_metrics(text_model: Path) -> dict[str, Any] | None:
    """Describe triangulation evidence, never statue surface or likeness accuracy.

    COLMAP stores each point's *mean pixel reprojection error* and 2D feature
    observation pairs in points3D.txt. A tight reprojection fit is necessary
    for credible cameras but does not establish a clean foreground mesh.
    """
    path = text_model / "points3D.txt"
    if path.is_symlink() or not path.is_file():
        return None
    errors: list[float] = []
    track_sum = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            if (len(fields) < 10 or (len(fields) - 8) % 2 or
                    not fields[0].isdigit() or not _finite_float(fields[7])):
                continue
            error = float(fields[7])
            if error < 0:
                continue
            errors.append(error)
            track_sum += (len(fields) - 8) // 2
    if not errors:
        return None
    return {
        "triangulated_points": len(errors),
        "mean_track_length": round(track_sum / len(errors), 3),
        "median_reprojection_error_px": round(statistics.median(errors), 3),
        "mean_reprojection_error_px": round(math.fsum(errors) / len(errors), 3),
        "interpretation": "Sparse matched-feature fit only; not surface, anatomical or monument fidelity",
    }


def build_sparse_reconstruction(reference_set: ReferenceSet, output: str | Path,
                                *, colmap: str = "colmap",
                                engine: str = "colmap",
                                timeout_seconds: int = 3600,
                                undistort: bool = False,
                                max_image_size: int = 4000) -> dict[str, Any]:
    """Produce auditable camera/sparse data using a locally installed COLMAP.

    External tools may fail on historical or unrelated internet photos. Their
    return codes are reported as review-required artifacts, not silent success.
    Preflight errors never create an output directory.
    """
    target = Path(output).resolve()
    if Path(output).is_symlink() or target.exists():
        raise ReferenceError(f"Output already exists: {target}. Choose an unused revision directory")
    if type(timeout_seconds) is not int or not 30 <= timeout_seconds <= 86400:
        raise ReferenceError("COLMAP timeout_seconds must be 30–86400")
    if type(undistort) is not bool or type(max_image_size) is not int or not 512 <= max_image_size <= 16384:
        raise ReferenceError("undistort must be bool and max_image_size must be 512–16384")
    if engine not in ("colmap", "pycolmap"):
        raise ReferenceError("Sparse engine must be colmap or pycolmap")
    audit = audit_reference_set(reference_set)
    references = _staged_images(reference_set, audit)
    if len(references) < 3:
        raise ReferenceError("COLMAP requires at least three independent reference images")
    if engine == "pycolmap":
        if importlib.util.find_spec("pycolmap") is None:
            raise ReferenceError("PyCOLMAP is not installed: pip install pycolmap")
        executable: str | list[str] = [sys.executable, "-m",
            "kumiki_blender.reconstruction.pycolmap_worker"]
    else:
        found = shutil.which(colmap)
        if not found:
            raise ReferenceError(f"COLMAP executable not found: {colmap}")
        executable = found
    colmap_info, cpu_flags = _probe_colmap(executable)
    colmap_info["engine"] = engine
    # Re-check before creating anything: photo files can be edited after intake.
    for image in references:
        if not image.path.is_file() or _sha256(image.path) != image.sha256:
            raise ReferenceError(f"Reference {image.image_id} changed since validation")
        if image.mask is not None and (
                not image.mask.is_file() or _sha256(image.mask) != image.mask_sha256):
            raise ReferenceError(f"Reference {image.image_id} mask changed since validation")
    target.mkdir(parents=True, exist_ok=False)
    (target / "logs").mkdir()
    report: dict[str, Any] = {
        "schema_version": 1, "asset_id": reference_set.asset_id,
        "manifest_sha256": reference_set.manifest_sha256,
        "source_images": len(reference_set.images),
        "staged_independent_images": len(references),
        "registered_images": 0,
        "undistortion_requested": undistort,
        "colmap_info": colmap_info,
        "commands": [], "failed_stage": None, "exit_code": None,
        "status": "review-required", "historical_fidelity_verified": False,
        "limitations": "Sparse camera reconstruction is not a face/mesh scan or a historical likeness certificate.",
    }

    def checkpoint() -> dict[str, Any]:
        _write_json(target / "reconstruction-report.json", report)
        return report

    checkpoint()
    try:
        _write_json(target / "reference-audit.json", audit)
        has_masks = _prepare_images(references, target, normalize_for_colmap=True)
    except (ReferenceError, OSError, ValueError) as error:
        report["failed_stage"], report["exit_code"] = "staging", 2
        report["error"] = str(error)
        return checkpoint()
    database = target / "database.db"
    images = target / "images"
    sparse = target / "sparse"
    sparse.mkdir()
    stages = [
        ("feature_extractor", ["--database_path", str(database), "--image_path", str(images)] +
         (["--ImageReader.mask_path", str(target / "masks")] if has_masks else []) +
         [cpu_flags[0], "0"] +
         (["--max_image_size", str(max_image_size)] if engine == "pycolmap" else [])),
        ("exhaustive_matcher", ["--database_path", str(database), cpu_flags[1], "0"]),
        ("mapper", ["--database_path", str(database), "--image_path", str(images),
                    "--output_path", str(sparse)]),
    ]

    for number, (stage, args) in enumerate(stages, start=1):
        code, command = _run(executable, stage, args, target, number, timeout_seconds)
        report["commands"].append(command)
        if code:
            report["failed_stage"], report["exit_code"] = stage, code
            return checkpoint()
        if stage == "feature_extractor":
            report["feature_import"] = _feature_import_report(
                database, target / "staged-reference-index.json")
            if report["feature_import"]["status"] == "incomplete":
                report.update(
                    failed_stage="feature_extraction_incomplete", exit_code=2,
                    error="COLMAP could not import every staged photo; inspect feature_import.missing_images")
                return checkpoint()
        checkpoint()
    models = sorted((directory for directory in sparse.iterdir() if directory.is_dir()),
                    key=lambda entry: entry.name)
    if not models:
        report["failed_stage"] = "no_sparse_models"
        return checkpoint()
    counts = []
    for index, model in enumerate(models):
        text_model = target / "text-models" / model.name
        text_model.mkdir(parents=True, exist_ok=False)
        args = ["--input_path", str(model), "--output_path", str(text_model),
                "--output_type", "TXT"]
        code, command = _run(executable, "model_converter", args, target, index + 4,
                             timeout_seconds)
        report["commands"].append(command)
        if code:
            report["failed_stage"], report["exit_code"] = "model_converter", code
            return checkpoint()
        counts.append({"model": model.name, "registered_images": _registered_image_count(text_model)})
    report["models"] = counts
    report["registered_images"] = max(item["registered_images"] for item in counts)
    if report["registered_images"] >= 3:
        report["status"] = "candidate / visual-review-required"
    else:
        report["failed_stage"] = "insufficient_registered_views"
        return checkpoint()
    best = max(counts, key=lambda item: item["registered_images"])["model"]
    report["sparse_feature_evidence"] = _sparse_point_metrics(target / "text-models" / best)
    checkpoint()
    if undistort:
        # A production COLMAP default is SIMPLE_RADIAL; a Blender pinhole camera
        # cannot reproduce its curved optics. Rectify *both* pixels and cameras.
        rectified = target / "undistorted"
        rectified.mkdir()
        args = ["--image_path", str(images), "--input_path", str(sparse / best),
                "--output_path", str(rectified), "--output_type", "COLMAP",
                "--max_image_size", str(max_image_size)]
        code, command = _run(executable, "image_undistorter", args, target,
                             len(report["commands"]) + 1, timeout_seconds)
        report["commands"].append(command)
        if code:
            report["status"] = "review-required"
            report["failed_stage"], report["exit_code"] = "image_undistorter", code
            return checkpoint()
        text_model = rectified / "text-model"
        text_model.mkdir()
        args = ["--input_path", str(rectified / "sparse"), "--output_path", str(text_model),
                "--output_type", "TXT"]
        code, command = _run(executable, "model_converter", args, target,
                             len(report["commands"]) + 1, timeout_seconds)
        report["commands"].append(command)
        if code:
            report["status"] = "review-required"
            report["failed_stage"], report["exit_code"] = "undistorted_model_converter", code
            return checkpoint()
        try:
            from PIL import Image
            from .cameras import _read_intrinsics, _read_poses

            corrected_cameras = _read_intrinsics(text_model / "cameras.txt")
            corrected_poses = _read_poses(text_model / "images.txt")
            if len(corrected_poses) != report["registered_images"]:
                raise ReferenceError("Undistorted model changed registered-image count")
            original_index = json.loads((target / "staged-reference-index.json").read_text())
            originals = {entry["filename"]: entry for entry in original_index["images"]}
            rectified_index = []
            for _, camera_id, filename, _, _ in corrected_poses:
                if filename not in originals or camera_id not in corrected_cameras:
                    raise ReferenceError("Undistorted COLMAP image is not in source index")
                path = rectified / "images" / filename
                if path.is_symlink() or not path.is_file():
                    raise ReferenceError(f"Missing rectified image {filename}")
                with Image.open(path) as image:
                    size = image.size
                    image.verify()
                camera = corrected_cameras[camera_id]
                if list(size) != [camera["width"], camera["height"]]:
                    raise ReferenceError(f"Rectified image dimensions differ from camera {filename}")
                rectified_index.append({**originals[filename], "dimensions": list(size),
                                        "sha256": _sha256(path),
                                        "mask_requires_rectification": True})
            _write_json(target / "undistorted-reference-index.json", {"images": rectified_index})
            report["undistorted_model"] = "undistorted/text-model"
            report["undistorted_images"] = len(rectified_index)
            report["source_sparse_model"] = best
        except (ReferenceError, OSError, ValueError, KeyError) as error:
            report["status"] = "review-required"
            report["failed_stage"], report["exit_code"] = "undistortion_validation", 2
            report["error"] = str(error)
            return checkpoint()
    return checkpoint()
