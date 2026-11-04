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
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from .references import ReferenceError, ReferenceImage, ReferenceSet, audit_reference_set


# Pairs come from COLMAP's documented compatible extractor/matcher combinations.
# Each revision uses a fresh database: descriptors from different extractors must
# never be mixed, even when their dimensionality happens to agree.
MATCHING_PRESETS: dict[str, tuple[str, str]] = {
    "sift": ("SIFT", "SIFT_BRUTEFORCE"),
    "sift-lightglue": ("SIFT", "SIFT_LIGHTGLUE"),
    "aliked-lightglue": ("ALIKED_N16ROT", "ALIKED_LIGHTGLUE"),
    "loma": ("LOMA_B", "LOMA_B"),
    "loma128": ("LOMA_B128", "LOMA_B128"),
}


def _validate_matching_preset(preset: str, engine: str) -> None:
    if preset not in MATCHING_PRESETS:
        raise ReferenceError("Matching preset must be one of: " + ", ".join(MATCHING_PRESETS))
    if preset != "sift" and engine != "colmap":
        raise ReferenceError("Learned matching requires --engine colmap with a recent ONNX-enabled COLMAP CLI")


def _preflight_matching_preset(executable: str, preset: str) -> bool:
    """Detect COLMAP's learned-feature CLI switches without guessing enum help text.

    Some packaged COLMAP builds print only option names (not the list of
    accepted feature/matcher values) and some return a nonzero exit status for
    --help. Let the real COLMAP stage validate the selected enum: its failure
    will be captured in the immutable reconstruction report and stage log.
    """
    if preset == "sift":
        return False
    try:
        extraction = subprocess.run([executable, "feature_extractor", "--help"],
                                    capture_output=True, text=True, check=False, timeout=20)
        matching = subprocess.run([executable, "exhaustive_matcher", "--help"],
                                  capture_output=True, text=True, check=False, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ReferenceError(f"Learned COLMAP preflight failed: {error}") from error
    extractor_help = extraction.stdout + extraction.stderr
    matcher_help = matching.stdout + matching.stderr
    if ("--FeatureExtraction.type" not in extractor_help or
            "--FeatureMatching.type" not in matcher_help):
        raise ReferenceError(
            "COLMAP lacks learned feature-selection CLI flags; "
            "install COLMAP 4.2+ with ONNX enabled, or choose --matching-preset sift")
    return "--FeatureExtraction.max_image_size" in extractor_help


def _preflight_mps_worker() -> None:
    """Check optional neural dependencies in an isolated process before extraction.

    Importing PyTorch in the COLMAP orchestrator would retain its memory while
    native feature extraction runs. The lightweight worker check exits without
    constructing LightGlue or downloading model weights.
    """
    command = [sys.executable, "-m", "kumiki_blender.reconstruction.mps_lightglue_worker",
               "check", "--device", "mps"]
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=90, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ReferenceError(f"MPS LightGlue preflight failed: {error}") from error
    if result.returncode:
        details = (result.stderr or result.stdout).strip().splitlines()
        reason = details[-1][:400] if details else f"worker exited {result.returncode}"
        raise ReferenceError(f"MPS LightGlue preflight failed: {reason}")


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
    # A hard stop during matching must not leave a half-written resume file.
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp",
                                         delete=False) as stream:
            temporary = stream.name
            stream.write(json.dumps(content, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


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


def _cap_colmap_sift_features(database: Path, maximum: int) -> dict[str, int]:
    """Enforce an exact neural input cap on a *fresh* COLMAP database.

    COLMAP's --FeatureMatching.max_num_matches does not limit SIFT_LIGHTGLUE's
    inputs. It is safe to replace keypoints/descriptors only before matching
    starts, otherwise saved match indices would refer to the wrong feature.
    """
    import struct
    from .mps_lightglue_worker import select_spatial_keypoints

    total = 0
    reduced = 0
    max_original = 0
    max_selected = 0
    with sqlite3.connect(database) as db:
        if db.execute("SELECT COUNT(*) FROM matches").fetchone()[0] or db.execute(
                "SELECT COUNT(*) FROM two_view_geometries").fetchone()[0]:
            raise ReferenceError("Cannot change keypoint indexing after any matching results exist")
        rows = db.execute(
            "SELECT i.image_id, c.width, c.height, k.rows, k.cols, k.data, "
            "d.rows, d.cols, d.data FROM images i "
            "JOIN cameras c ON i.camera_id=c.camera_id "
            "JOIN keypoints k ON k.image_id=i.image_id "
            "JOIN descriptors d ON d.image_id=i.image_id").fetchall()
        image_count = db.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        if len(rows) != image_count:
            raise ReferenceError("Missing SIFT features; cannot apply a neural keypoint cap")
        changes: list[tuple[int, bytes, bytes, int]] = []
        for image_id, width, height, n, kc, kdata, dn, dc, ddata in rows:
            if (n != dn or kc != 4 or dc != 128 or
                    len(kdata) != 4 * kc * n or len(ddata) != dc * dn):
                raise ReferenceError("Invalid SIFT feature dimensions; refusing to rewrite database")
            max_original = max(max_original, n)
            total += n
            if n <= maximum:
                max_selected = max(max_selected, n)
                continue
            selected = select_spatial_keypoints(
                [(x, y, scale) for x, y, scale, _ in struct.iter_unpack("<ffff", kdata)],
                width, height, maximum)
            if not selected:
                raise ReferenceError("No valid SIFT keypoints remain after selection")
            new_kdata = b"".join(kdata[i * kc * 4:(i + 1) * kc * 4] for i in selected)
            new_ddata = b"".join(ddata[i * dc:(i + 1) * dc] for i in selected)
            changes.append((image_id, new_kdata, new_ddata, len(selected)))
            reduced += n - len(selected)
            max_selected = max(max_selected, len(selected))
        for image_id, kdata, ddata, count in changes:
            db.execute("UPDATE keypoints SET rows=?, data=? WHERE image_id=?",
                       (count, kdata, image_id))
            db.execute("UPDATE descriptors SET rows=?, data=? WHERE image_id=?",
                       (count, ddata, image_id))
    return {"images": len(rows), "original_keypoints": total,
            "removed_keypoints": reduced, "max_original": max_original,
            "max_selected": max_selected, "limit": maximum}



def _run(executable: str | list[str], stage: str, arguments: list[str], output: Path,
         number: int, timeout_seconds: int, *,
         max_rss_mb: int | None = None) -> tuple[int, list[str]]:
    prefix = [executable] if isinstance(executable, str) else executable
    command = [*prefix, stage, *arguments]
    log = output / "logs" / f"{number:02d}-{stage}.log"
    try:
        with log.open("w", encoding="utf-8") as stream:
            if max_rss_mb is None:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                        check=False, timeout=timeout_seconds)
                return result.returncode, [stage, *arguments]
            with subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT) as process:
                deadline = time.monotonic() + timeout_seconds
                peak_rss_kib = 0
                while True:
                    try:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise subprocess.TimeoutExpired(command, timeout_seconds)
                        code = process.wait(timeout=min(0.25, remaining))
                        stream.write(f"\nMemory guard sampled peak RSS: {peak_rss_kib // 1024} MiB\n")
                        return code, [stage, *arguments]
                    except subprocess.TimeoutExpired:
                        pass
                    if time.monotonic() >= deadline:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        stream.write(f"\nCOLMAP stage timed out after {timeout_seconds} seconds\n")
                        return 124, [stage, *arguments]
                    # On macOS/Linux ps reports resident memory in KiB. A
                    # runaway native worker must never silently disable the
                    # guard when memory measurement fails.
                    rss_kib = _process_rss_kib(process.pid)
                    if rss_kib is not None:
                        peak_rss_kib = max(peak_rss_kib, rss_kib)
                    if rss_kib is None or rss_kib >= max_rss_mb * 1024:
                        code = process.poll()
                        if code is not None:
                            return code, [stage, *arguments]
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        detail = (f"RSS {rss_kib / 1024:.0f} MiB >= {max_rss_mb} MiB"
                                  if rss_kib is not None else "RSS monitoring unavailable")
                        stream.write(f"\nMemory guard stopped COLMAP: {detail}\n")
                        return 125, [stage, *arguments]
    except subprocess.TimeoutExpired:
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"\nCOLMAP stage timed out after {timeout_seconds} seconds\n")
        return 124, [stage, *arguments]
    except OSError as error:
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"\nFailed to start COLMAP stage: {error}\n")
        return 127, [stage, *arguments]


def _process_rss_kib(pid: int) -> int | None:
    """Return process RSS in KiB using the platform ps utility, or None."""
    try:
        result = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                                capture_output=True, text=True, timeout=3, check=False)
        if result.returncode != 0:
            return None
        return int(result.stdout.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def _memory_guard_reason(log: Path) -> str:
    """Retrieve the small guard message appended after terminating COLMAP."""
    try:
        with log.open("rb") as stream:
            size = stream.seek(0, 2)
            stream.seek(max(0, size - 2048))
            tail = stream.read().decode("utf-8", errors="replace")
    except OSError:
        return "details unavailable"
    for line in reversed(tail.splitlines()):
        if line.startswith("Memory guard stopped COLMAP: "):
            return line.partition(": ")[2]
    return "details unavailable"


def _next_stage_number(target: Path, commands: list[list[str]]) -> int:
    """Avoid overwriting a native log after an unexpected process interruption."""
    numbers = [len(commands)]
    for path in (target / "logs").glob("*-*.log"):
        prefix = path.name.split("-", 1)[0]
        if prefix.isdecimal():
            numbers.append(int(prefix))
    for path in (target / "pairs").glob("*-pairs.txt"):
        prefix = path.name.split("-", 1)[0]
        if prefix.isdecimal():
            numbers.append(int(prefix))
    return max(numbers) + 1


def _resolve_matching_pairs(staged_names: list[str], matching_pair_list: Path | None,
                            report: dict[str, Any]) -> tuple[list[tuple[str, str]], str]:
    """Validate the immutable pair selection and checkpoint without writing it."""
    names = sorted(staged_names)
    if len(names) != len(set(names)):
        raise ReferenceError("Duplicate staged image names cannot be matched safely")
    pairs = [(first, second) for i, first in enumerate(names)
             for second in names[i + 1:]]
    if matching_pair_list is not None:
        from .mps_lightglue_worker import read_pairs
        requested = read_pairs(matching_pair_list)
        known = set(names)
        if not requested or any(a not in known or b not in known for a, b in requested):
            raise ReferenceError("Matching pair list must contain only known staged image names")
        selected = {(min(a, b), max(a, b)) for a, b in requested}
        if len(selected) != len(requested):
            raise ReferenceError("Matching pair list contains duplicate pairs")
        pairs = [pair for pair in pairs if pair in selected]
    fingerprint = hashlib.sha256(
        "".join(f"{a}\t{b}\n" for a, b in pairs).encode("utf-8")).hexdigest()
    old_fingerprint = report.get("matching_pair_fingerprint")
    if old_fingerprint is not None and old_fingerprint != fingerprint:
        raise ReferenceError("Matching pair selection changed; cannot resume existing progress")
    # Old full-exhaustive checkpoints did not have a fingerprint. Their
    # accumulated pair count and order must not be interpreted as a subset.
    if old_fingerprint is None and report.get("matching_progress") and matching_pair_list is not None:
        raise ReferenceError("Cannot apply a new pair list to an existing matching checkpoint")
    progress = report.get("matching_progress")
    if progress is not None:
        if (type(progress) is not dict or progress.get("total_pairs") != len(pairs) or
                type(progress.get("next_pair_index")) is not int or
                not 0 <= progress["next_pair_index"] <= len(pairs) or
                type(progress.get("batch_size")) is not int or
                not 1 <= progress["batch_size"] <= 1024):
            raise ReferenceError("Invalid or incompatible matching checkpoint; refusing resume")
    return pairs, fingerprint


def _match_in_bounded_processes(
    executable: str | list[str], target: Path, database: Path,
    match_args: list[str], timeout_seconds: int, max_rss_mb: int | None,
    staged_names: list[str], pairs_per_process: int,
    report: dict[str, Any], checkpoint: Callable[[], dict[str, Any]],
    *, matching_device: str = "colmap-cpu", max_keypoints: int = 4096,
    matching_pair_list: Path | None = None,
) -> bool:
    """Persist pair-matching progress across independent COLMAP workers.

    A failed group is safe to retry: COLMAP skips previously verified pairs in
    the shared database. Chunk boundaries and progress are saved after every
    completed group, allowing --resume after an interrupted process.
    """
    pairs, fingerprint = _resolve_matching_pairs(staged_names, matching_pair_list, report)
    report["matching_pair_fingerprint"] = fingerprint
    report["matching_pair_selection"] = "manual" if matching_pair_list else "exhaustive"
    progress = report.setdefault("matching_progress", {
        "total_pairs": len(pairs), "next_pair_index": 0,
        "completed_pairs": 0, "completed_batches": 0,
        "batch_size": pairs_per_process, "status": "in-progress",
    })
    # On resume, allow the caller to tighten (but not silently loosen) a
    # previously established safe batch-size ceiling.
    if progress.get("matching_device", "colmap-cpu") != matching_device:
        # A new backend has a new memory profile. Keep already completed pairs,
        # but don't inherit the old CPU ONNX worker's one-pair ceiling.
        progress["batch_size"] = pairs_per_process
    else:
        progress["batch_size"] = min(progress["batch_size"], pairs_per_process)
    progress["matching_device"] = matching_device
    pairs_dir = target / "pairs"
    if pairs_dir.is_symlink():
        raise ReferenceError("Pair-list directory cannot be a symlink")
    pairs_dir.mkdir(exist_ok=True)
    checkpoint()
    while progress["next_pair_index"] < len(pairs):
        start = progress["next_pair_index"]
        count = min(progress["batch_size"], len(pairs) - start)
        number = _next_stage_number(target, report["commands"])
        pair_file = pairs_dir / f"{number:04d}-pairs.txt"
        if pair_file.exists() or pair_file.is_symlink():
            raise ReferenceError(f"Pair list already exists: {pair_file}")
        pair_file.write_text("".join(f"{a} {b}\n" for a, b in pairs[start:start + count]),
                             encoding="utf-8")
        if matching_device == "colmap-cpu":
            arguments = (["--database_path", str(database), "--match_list_path", str(pair_file),
                          "--match_type", "pairs"] + match_args)
            stage = "matches_importer"
            worker = executable
        else:
            arguments = (["--database", str(database), "--pairs", str(pair_file),
                          "--max-keypoints", str(max_keypoints),
                          "--device", matching_device])
            stage = "match"
            worker = [sys.executable, "-m", "kumiki_blender.reconstruction.mps_lightglue_worker"]
        code, command = _run(worker, stage, arguments, target, number,
                             timeout_seconds, max_rss_mb=max_rss_mb)
        report["commands"].append(command)
        progress["last_log"] = f"logs/{number:02d}-{stage}.log"
        if code == 0:
            progress["next_pair_index"] += count
            progress["completed_pairs"] = progress["next_pair_index"]
            progress["completed_batches"] += 1
            checkpoint()
            if (progress["completed_batches"] == 1 or
                    progress["completed_batches"] % 10 == 0 or
                    progress["completed_pairs"] == len(pairs)):
                print(f"COLMAP matching: {progress['completed_pairs']}/{len(pairs)} pairs "
                      f"completed (up to {progress['batch_size']} per worker)",
                      file=sys.stderr, flush=True)
            continue
        reason = (_memory_guard_reason(target / progress["last_log"])
                  if code == 125 else None)
        progress["last_failure"] = {"code": code, "pair_index": start,
                                    "pairs_in_batch": count, "reason": reason}
        if code == 125 and count > 1 and reason and reason.startswith("RSS "):
            progress["batch_size"] = max(1, count // 2)
            checkpoint()
            print(f"COLMAP matching: {reason}; retrying with "
                  f"{progress['batch_size']} pair(s) per worker",
                  file=sys.stderr, flush=True)
            continue
        report["failed_stage"], report["exit_code"] = stage, code
        if code == 125:
            report["memory_guard_reason"] = reason
            report["error"] = (f"Matching stopped at pair {start + 1}/{len(pairs)}: {reason}. "
                               f"See {progress['last_log']}")
        else:
            report["error"] = (f"Matching failed at pair {start + 1}/{len(pairs)} "
                               f"(exit {code}); see {progress['last_log']}")
        checkpoint()
        return False
    progress["status"] = "complete"
    checkpoint()
    return True


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


def _registered_image_names(text_model: Path) -> list[str]:
    """Read image names from camera-pose rows, skipping alternating observations."""
    path = text_model / "images.txt"
    if path.is_symlink() or not path.is_file():
        return []
    names: list[str] = []
    lines = iter(path.read_text(encoding="utf-8").splitlines())
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split(maxsplit=9)
        if (len(parts) == 10 and parts[0].isdigit() and parts[8].isdigit()
                and all(_finite_float(value) for value in parts[1:8])):
            names.append(parts[9])
            next(lines, None)
    return sorted(set(names))


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
                                max_image_size: int = 4000,
                                registration_profile: str = "standard",
                                matching_preset: str = "sift",
                                feature_threads: int | None = None,
                                matching_threads: int | None = None,
                                sift_first_octave: int | None = None,
                                max_colmap_rss_mb: int | None = None,
                                matching_block_size: int | None = None,
                                matching_max_features: int | None = None,
                                matching_pairs_per_process: int | None = None,
                                matching_device: str = "colmap-cpu",
                                matching_pair_list: Path | None = None,
                                resume: bool = False) -> dict[str, Any]:
    """Produce auditable camera/sparse data using a locally installed COLMAP.

    External tools may fail on historical or unrelated internet photos. Their
    return codes are reported as review-required artifacts, not silent success.
    Preflight errors never create an output directory.
    """
    target = Path(output).resolve()
    if Path(output).is_symlink():
        raise ReferenceError("Reconstruction output cannot be a symlink")
    if resume and not target.is_dir():
        raise ReferenceError(f"Cannot resume missing reconstruction: {target}")
    if not resume and target.exists():
        raise ReferenceError(f"Output already exists: {target}. Choose an unused revision directory")
    if type(timeout_seconds) is not int or not 30 <= timeout_seconds <= 86400:
        raise ReferenceError("COLMAP timeout_seconds must be 30–86400")
    if type(undistort) is not bool or type(max_image_size) is not int or not 512 <= max_image_size <= 16384:
        raise ReferenceError("undistort must be bool and max_image_size must be 512–16384")
    if engine not in ("colmap", "pycolmap"):
        raise ReferenceError("Sparse engine must be colmap or pycolmap")
    if registration_profile not in ("standard", "high-recall"):
        raise ReferenceError("Registration profile must be standard or high-recall")
    if registration_profile == "high-recall" and engine != "pycolmap":
        raise ReferenceError("High-recall registration currently requires --engine pycolmap")
    _validate_matching_preset(matching_preset, engine)
    if matching_device not in ("colmap-cpu", "mps"):
        raise ReferenceError("Matching device must be colmap-cpu or mps")
    if matching_device == "mps" and (matching_preset != "sift-lightglue" or
                                     engine != "colmap"):
        raise ReferenceError("MPS matching requires --engine colmap --matching-preset sift-lightglue")
    if matching_device == "mps" and matching_pairs_per_process is None:
        raise ReferenceError("MPS matching requires --matching-pairs-per-process (recommended: 8)")
    if matching_pair_list is not None:
        if matching_pairs_per_process is None:
            raise ReferenceError("--matching-pair-list requires --matching-pairs-per-process")
        matching_pair_list = Path(matching_pair_list)
        if not matching_pair_list.is_file():
            raise ReferenceError(f"Missing matching pair list: {matching_pair_list}")
        from .mps_lightglue_worker import read_pairs
        try:
            requested_pairs = read_pairs(matching_pair_list)
        except (ValueError, OSError) as error:
            raise ReferenceError(f"Invalid matching pair list: {error}") from error
        allowed_names = {image.image_id + ".png" for image in reference_set.images}
        if not requested_pairs or any(a not in allowed_names or b not in allowed_names
                                      for a, b in requested_pairs):
            raise ReferenceError("Matching pair list must contain only known staged image names")
        if len({tuple(sorted(pair)) for pair in requested_pairs}) != len(requested_pairs):
            raise ReferenceError("Matching pair list contains duplicate pairs")
    for name, count in (("feature_threads", feature_threads),
                        ("matching_threads", matching_threads)):
        if count is not None and (type(count) is not int or not 1 <= count <= 64):
            raise ReferenceError(f"{name} must be 1–64")
    if sift_first_octave not in (None, -1, 0):
        raise ReferenceError("sift_first_octave must be -1 or 0")
    if (max_colmap_rss_mb is not None and
            (type(max_colmap_rss_mb) is not int or
             (max_colmap_rss_mb != 0 and not 1024 <= max_colmap_rss_mb <= 131072))):
        raise ReferenceError("max_colmap_rss_mb must be 0 (disabled) or 1024–131072")
    if (matching_block_size is not None and
            (type(matching_block_size) is not int or not 1 <= matching_block_size <= 50)):
        raise ReferenceError("matching_block_size must be 1–50")
    if (matching_max_features is not None and
            (type(matching_max_features) is not int or not 256 <= matching_max_features <= 32768)):
        raise ReferenceError("matching_max_features must be 256–32768")
    if (matching_pairs_per_process is not None and
            (type(matching_pairs_per_process) is not int or
             not 1 <= matching_pairs_per_process <= 1024)):
        raise ReferenceError("matching_pairs_per_process must be 1–1024")
    if type(resume) is not bool or (resume and matching_pairs_per_process is None):
        raise ReferenceError("--resume requires --matching-pairs-per-process")
    if engine != "colmap" and any(value is not None for value in
                                  (feature_threads, matching_threads, sift_first_octave,
                                   max_colmap_rss_mb, matching_block_size,
                                   matching_max_features, matching_pairs_per_process)):
        raise ReferenceError("COLMAP memory controls require --engine colmap")
    if sift_first_octave is not None and matching_preset not in ("sift", "sift-lightglue"):
        raise ReferenceError("SIFT first octave applies only to SIFT-based matching")
    learned = matching_preset != "sift"
    effective_feature_threads = feature_threads if feature_threads is not None else (1 if learned else None)
    effective_matching_threads = matching_threads if matching_threads is not None else (1 if learned else None)
    effective_sift_octave = (sift_first_octave if sift_first_octave is not None else
                             0 if matching_preset == "sift-lightglue" else None)
    effective_rss_mb = (max_colmap_rss_mb if max_colmap_rss_mb is not None else
                        12288 if learned else 0)
    # Limit *actual input keypoints* before CPU ONNX matching, or per pair for
    # MPS matching. Preserve the 4K pixels; do not mistake a SIFT output-match
    # limit for a cap on LightGlue's neural attention input.
    effective_block_size = (matching_block_size if matching_block_size is not None else
                            5 if learned else None)
    effective_match_features = (matching_max_features if matching_max_features is not None else
                                4096 if learned else None)
    if effective_rss_mb:
        if not shutil.which("ps") or _process_rss_kib(os.getpid()) is None:
            raise ReferenceError("Memory guard needs a working ps command; refusing unmonitored COLMAP execution")
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
    supports_extraction_size = (_preflight_matching_preset(executable, matching_preset)
                                if engine == "colmap" else False)
    if matching_device == "mps":
        _preflight_mps_worker()
    colmap_info["engine"] = engine
    # Re-check before creating anything: photo files can be edited after intake.
    for image in references:
        if not image.path.is_file() or _sha256(image.path) != image.sha256:
            raise ReferenceError(f"Reference {image.image_id} changed since validation")
        if image.mask is not None and (
                not image.mask.is_file() or _sha256(image.mask) != image.mask_sha256):
            raise ReferenceError(f"Reference {image.image_id} mask changed since validation")
    initial_report: dict[str, Any] = {
        "schema_version": 1, "asset_id": reference_set.asset_id,
        "manifest_sha256": reference_set.manifest_sha256,
        "source_images": len(reference_set.images),
        "staged_independent_images": len(references),
        "registration_profile": registration_profile,
        "matching_preset": matching_preset,
        "matching_device": matching_device,
        "max_image_size": max_image_size,
        "memory_controls": {"feature_threads": effective_feature_threads,
                            "matching_threads": effective_matching_threads,
                            "sift_first_octave": effective_sift_octave,
                            "max_colmap_rss_mb": effective_rss_mb,
                            "matching_block_size": effective_block_size,
                            "matching_max_features": effective_match_features,
                            "matching_pairs_per_process": matching_pairs_per_process},
        "registered_images": 0,
        "undistortion_requested": undistort,
        "colmap_info": colmap_info,
        "commands": [], "failed_stage": None, "exit_code": None,
        "status": "review-required", "historical_fidelity_verified": False,
        "limitations": "Sparse camera reconstruction is not a face/mesh scan or a historical likeness certificate.",
    }
    if resume:
        report_path = target / "reconstruction-report.json"
        index_path = target / "staged-reference-index.json"
        database_path = target / "database.db"
        for path in (report_path, index_path, database_path):
            if path.is_symlink() or not path.is_file():
                raise ReferenceError(f"Cannot resume: missing or unsafe {path.name}")
        for folder in (target / "images", target / "logs", target / "sparse"):
            if folder.is_symlink() or not folder.is_dir():
                raise ReferenceError(f"Cannot resume: missing or unsafe {folder.name}")
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            stored_index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ReferenceError(f"Cannot resume invalid reconstruction metadata: {error}") from error
        for key in ("asset_id", "manifest_sha256", "registration_profile",
                    "matching_preset", "undistortion_requested"):
            if report.get(key) != initial_report[key]:
                raise ReferenceError(f"Cannot resume: {key} differs from source revision")
        # Older interrupted revisions recorded the extraction image size only
        # in the command; preserve that validation while supporting upgrades.
        recorded_size = report.get("max_image_size")
        if recorded_size is None:
            commands = report.get("commands", [])
            extract = next((command for command in commands
                            if command and command[0] == "feature_extractor"), [])
            if "--FeatureExtraction.max_image_size" in extract:
                recorded_size = int(extract[extract.index("--FeatureExtraction.max_image_size") + 1])
        if recorded_size != max_image_size:
            raise ReferenceError("Cannot resume: input image resolution has changed")
        if (report.get("feature_import", {}).get("status") != "complete" or
                report.get("failed_stage") not in ("exhaustive_matcher", "matches_importer",
                                                   "match", "geometric_verifier", None)):
            raise ReferenceError("Cannot resume: feature extraction is incomplete or stage is incompatible")
        entries = stored_index.get("images", [])
        expected = {image.image_id + ".png": image.sha256 for image in references}
        if (type(entries) is not list or len(entries) != len(expected) or
                {entry.get("filename"): entry.get("original_sha256")
                 for entry in entries} != expected):
            raise ReferenceError("Cannot resume: staged image provenance differs")
        for entry in entries:
            staged = target / "images" / entry["filename"]
            if (staged.is_symlink() or not staged.is_file() or
                    _sha256(staged) != entry["sha256"]):
                raise ReferenceError(f"Cannot resume: staged image changed ({entry['filename']})")
        matching_complete = report.get("matching_progress", {}).get("status") == "complete"
        if ((matching_complete and report.get("failed_stage") != "geometric_verifier") or
                any((target / "sparse").iterdir())):
            raise ReferenceError("Cannot resume a completed matching or mapping stage")
        # Reject incompatible pair selections before clearing errors or writing
        # the next checkpoint. Failed resumes must preserve the prior report.
        _resolve_matching_pairs([entry["filename"] for entry in entries],
                                matching_pair_list, report)
        previous_error = report.get("error")
        if previous_error:
            report.setdefault("prior_failures", []).append({
                "stage": report.get("failed_stage"), "error": previous_error,
                "exit_code": report.get("exit_code"),
            })
        report.update(failed_stage=None, exit_code=None, status="review-required")
        report.pop("error", None)
        report.pop("memory_guard_reason", None)
        report["memory_controls"] = initial_report["memory_controls"]
        report.setdefault("matching_device_history", []).append(
            {"from": report.get("matching_device", "colmap-cpu"), "to": matching_device})
        report["matching_device"] = matching_device
        report["max_image_size"] = max_image_size
        report["resume_count"] = report.get("resume_count", 0) + 1
    else:
        target.mkdir(parents=True, exist_ok=False)
        (target / "logs").mkdir()
        report = initial_report

    def checkpoint() -> dict[str, Any]:
        _write_json(target / "reconstruction-report.json", report)
        return report

    checkpoint()
    if resume:
        has_masks = (target / "masks").is_dir()
    else:
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
    if not resume:
        sparse.mkdir()
    profile_args = (["--registration-profile", "high-recall"]
                    if registration_profile == "high-recall" else [])
    extractor, matcher = MATCHING_PRESETS[matching_preset]
    extract_resource_args = (["--FeatureExtraction.num_threads", str(effective_feature_threads)]
                             if effective_feature_threads is not None else [])
    if effective_sift_octave is not None:
        extract_resource_args += ["--SiftExtraction.first_octave", str(effective_sift_octave)]
    match_resource_args = (["--FeatureMatching.num_threads", str(effective_matching_threads)]
                           if effective_matching_threads is not None else [])
    if effective_match_features is not None:
        match_resource_args += ["--FeatureMatching.max_num_matches", str(effective_match_features)]
    if effective_block_size is not None:
        match_resource_args += ["--ExhaustiveMatching.block_size", str(effective_block_size)]
    extract_args = (["--FeatureExtraction.type", extractor]
                    if matching_preset != "sift" else [])
    if supports_extraction_size:
        extract_args += ["--FeatureExtraction.max_image_size", str(max_image_size)]
    match_args = (["--FeatureMatching.type", matcher]
                  if matching_preset != "sift" else [])
    def run_stage(stage: str, args: list[str]) -> bool:
        number = _next_stage_number(target, report["commands"])
        code, command = _run(executable, stage, args, target, number, timeout_seconds,
                             max_rss_mb=effective_rss_mb or None)
        report["commands"].append(command)
        if code:
            report["failed_stage"], report["exit_code"] = stage, code
            if code == 125:
                reason = _memory_guard_reason(target / "logs" / f"{number:02d}-{stage}.log")
                report["memory_guard_reason"] = reason
                report["error"] = (f"COLMAP memory guard stopped {stage}: {reason}; "
                                   f"see logs/{number:02d}-{stage}.log")
            elif code == -9:
                report["error"] = "COLMAP received SIGKILL; inspect macOS memory pressure and the stage log"
            checkpoint()
            return False
        if stage == "feature_extractor":
            report["feature_import"] = _feature_import_report(
                database, target / "staged-reference-index.json")
            if report["feature_import"]["status"] == "incomplete":
                report.update(
                    failed_stage="feature_extraction_incomplete", exit_code=2,
                    error="COLMAP could not import every staged photo; inspect feature_import.missing_images")
                checkpoint()
                return False
        checkpoint()
        return True

    if not resume:
        extract_command = (["--database_path", str(database), "--image_path", str(images)] +
                           (["--ImageReader.mask_path", str(target / "masks")]
                            if has_masks else []) +
                           [cpu_flags[0], "0"] +
                           (["--max_image_size", str(max_image_size)] if engine == "pycolmap" else []) +
                           profile_args + extract_resource_args + extract_args)
        if not run_stage("feature_extractor", extract_command):
            return report
        if matching_preset == "sift-lightglue" and matching_device == "colmap-cpu":
            try:
                report["keypoint_cap"] = _cap_colmap_sift_features(
                    database, effective_match_features or 4096)
                checkpoint()
            except (ReferenceError, sqlite3.Error, ValueError) as error:
                report.update(failed_stage="feature_cap", exit_code=2, error=str(error))
                return checkpoint()
    if matching_pairs_per_process is None:
        args = (["--database_path", str(database), cpu_flags[1], "0"] +
                profile_args + match_resource_args + match_args)
        if not run_stage("exhaustive_matcher", args):
            return report
    else:
        staged = json.loads((target / "staged-reference-index.json").read_text(encoding="utf-8"))
        # Only exhaustive_matcher accepts ExhaustiveMatching.block_size.
        # matches_importer instead consumes exactly the supplied pair list.
        bounded_match_args = [cpu_flags[1], "0"] + profile_args
        if effective_matching_threads is not None:
            bounded_match_args += ["--FeatureMatching.num_threads", str(effective_matching_threads)]
        if effective_match_features is not None:
            bounded_match_args += ["--FeatureMatching.max_num_matches", str(effective_match_features)]
        bounded_match_args += match_args
        if not _match_in_bounded_processes(
                executable, target, database, bounded_match_args,
                timeout_seconds, effective_rss_mb or None,
                [entry["filename"] for entry in staged["images"]],
                matching_pairs_per_process, report, checkpoint,
                matching_device=matching_device, max_keypoints=effective_match_features or 4096,
                matching_pair_list=matching_pair_list):
            return report
    if matching_device == "mps" or any(
            command and command[0] == "match" for command in report["commands"]):
        # The worker stores raw COLMAP correspondence rows. Without two-view
        # geometric verification the mapper will silently see zero matches.
        # This also applies when switching to CPU while resuming MPS-produced
        # raw matches (including a previous failed verification attempt).
        if not run_stage("geometric_verifier", ["--database_path", str(database),
                                                "--FeatureMatching.num_threads",
                                                str(effective_matching_threads or 1)]):
            return report
    if not run_stage("mapper", ["--database_path", str(database),
                                "--image_path", str(images), "--output_path", str(sparse)]):
        return report
    models = sorted((directory for directory in sparse.iterdir() if directory.is_dir()),
                    key=lambda entry: entry.name)
    if not models:
        report["failed_stage"] = "no_sparse_models"
        return checkpoint()
    counts = []
    staged_index = json.loads((target / "staged-reference-index.json").read_text(encoding="utf-8"))
    staged_names = {entry["filename"] for entry in staged_index["images"]}
    for index, model in enumerate(models):
        text_model = target / "text-models" / model.name
        text_model.mkdir(parents=True, exist_ok=False)
        args = ["--input_path", str(model), "--output_path", str(text_model),
                "--output_type", "TXT"]
        code, command = _run(executable, "model_converter", args, target,
                             _next_stage_number(target, report["commands"]),
                             timeout_seconds, max_rss_mb=effective_rss_mb or None)
        report["commands"].append(command)
        if code:
            report["failed_stage"], report["exit_code"] = "model_converter", code
            return checkpoint()
        registered_names = _registered_image_names(text_model)
        if not set(registered_names).issubset(staged_names):
            report.update(failed_stage="camera_provenance_mismatch", exit_code=2,
                          error="Sparse model refers to images absent from staged sources")
            return checkpoint()
        counts.append({"model": model.name, "registered_images": len(registered_names),
                       "registered_filenames": registered_names})
    report["models"] = counts
    report["registered_images"] = max(item["registered_images"] for item in counts)
    if report["registered_images"] >= 3:
        report["status"] = "candidate / visual-review-required"
    else:
        report["failed_stage"] = "insufficient_registered_views"
        return checkpoint()
    best = max(counts, key=lambda item: item["registered_images"])["model"]
    best_names = next(item["registered_filenames"] for item in counts if item["model"] == best)
    report["registered_filenames"] = best_names
    report["unregistered_filenames"] = sorted(
        entry["filename"] for entry in staged_index["images"]
        if entry["filename"] not in best_names)
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
                             len(report["commands"]) + 1, timeout_seconds,
                             max_rss_mb=effective_rss_mb or None)
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
                             len(report["commands"]) + 1, timeout_seconds,
                             max_rss_mb=effective_rss_mb or None)
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
