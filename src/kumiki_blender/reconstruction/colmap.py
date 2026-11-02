"""Optional, source-neutral COLMAP sparse-reconstruction worker.

This deliberately stops before dense meshing and texturing: online photos are
often non-overlapping, and even a successful SfM model cannot authenticate a
statue, resolve missing anatomy, or clear source-media redistribution rights.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .references import ReferenceError, ReferenceImage, ReferenceSet, audit_reference_set


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _probe_colmap(executable: str) -> tuple[dict[str, str], tuple[str, str]]:
    """Negotiate CPU flags across legacy SIFT and newer Feature* CLIs."""
    help_texts = []
    try:
        for command in ("feature_extractor", "exhaustive_matcher"):
            result = subprocess.run([executable, command, "--help"],
                                    capture_output=True, text=True, timeout=20, check=False)
            help_texts.append(result.stdout + result.stderr)
        version = subprocess.run([executable, "--version"], capture_output=True,
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
    return {"executable": str(Path(executable).resolve()),
            "version": version_line[0][:200] if version_line else "unreported",
            "cpu_flag_family": family}, flags


def _write_json(path: Path, content: dict[str, Any]) -> None:
    path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")


def _staged_images(reference_set: ReferenceSet, report: dict[str, Any]) -> list[ReferenceImage]:
    # Representative-only matching: duplicate images cannot become fake camera baselines.
    excluded = {name for group in report["duplicate_groups"] for name in group[1:]}
    return [image for image in reference_set.images if image.image_id not in excluded]


def _prepare_images(references: list[ReferenceImage], output: Path) -> bool:
    from PIL import Image

    image_root = output / "images"
    image_root.mkdir()
    has_masks = any(image.mask is not None for image in references)
    mask_root = output / "masks"
    if has_masks:
        mask_root.mkdir()
    index = []
    for image in references:
        filename = image.image_id + image.path.suffix.lower()
        staged = image_root / filename
        shutil.copyfile(image.path, staged)
        if _sha256(staged) != image.sha256:
            raise ReferenceError(f"Reference {image.image_id} changed during image staging")
        if has_masks:
            if image.mask is not None:
                with Image.open(image.mask) as original:
                    mask = original.convert("L")
            else:
                mask = Image.new("L", image.dimensions, 255)
            mask.save(mask_root / (filename + ".png"))
        index.append({"image_id": image.image_id, "filename": filename,
                      "source_url": image.source_url, "view": image.view,
                      "regions": image.regions, "sha256": image.sha256,
                      "dimensions": image.dimensions, "usage": "reference-only"})
    _write_json(output / "staged-reference-index.json", {"images": index,
        "notes": "Local source references only. Do not bundle photographs into distributable GLB assets."})
    return has_masks


def _run(executable: str, stage: str, arguments: list[str], output: Path,
         number: int, timeout_seconds: int) -> tuple[int, list[str]]:
    command = [executable, stage, *arguments]
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


def build_sparse_reconstruction(reference_set: ReferenceSet, output: str | Path,
                                *, colmap: str = "colmap",
                                timeout_seconds: int = 3600) -> dict[str, Any]:
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
    audit = audit_reference_set(reference_set)
    references = _staged_images(reference_set, audit)
    if len(references) < 3:
        raise ReferenceError("COLMAP requires at least three independent reference images")
    executable = shutil.which(colmap)
    if not executable:
        raise ReferenceError(f"COLMAP executable not found: {colmap}")
    colmap_info, cpu_flags = _probe_colmap(executable)
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
        has_masks = _prepare_images(references, target)
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
         [cpu_flags[0], "0"]),
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
