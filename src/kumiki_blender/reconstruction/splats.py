"""Immutable, private OpenSplat training revisions for Apple Metal or CUDA.

OpenSplat is a separately installed, AGPL-licensed executable. This module
does not bundle or modify it and never claims that a splat is a polygon mesh.
"""

from __future__ import annotations

import json
import importlib.util
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .dense import _sha256
from .references import ReferenceError, ReferenceSet
from .splat_input import plan_splat_dataset, plan_splat_input, stage_splat_dataset


_SCALAR_BYTES = {"char": 1, "uchar": 1, "short": 2, "ushort": 2,
                 "int": 4, "uint": 4, "float": 4, "float32": 4,
                 "double": 8, "float64": 8}
_GAUSSIAN_FIELDS = frozenset(("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2",
                              "opacity", "scale_0", "scale_1", "scale_2",
                              "rot_0", "rot_1", "rot_2", "rot_3"))


def validate_camera_settings(engine: str, timeout_seconds: int,
                             max_image_size: int) -> None:
    if engine not in ("colmap", "pycolmap"):
        raise ReferenceError("Camera registration engine must be colmap or pycolmap")
    if type(timeout_seconds) is not int or not 30 <= timeout_seconds <= 86_400:
        raise ReferenceError("Camera registration timeout must be 30–86,400 seconds")
    if type(max_image_size) is not int or not 512 <= max_image_size <= 16_384:
        raise ReferenceError("Camera registration maximum image size must be 512–16,384")


@dataclass(frozen=True)
class SplatSettings:
    iterations: int = 7000
    timeout_seconds: int = 14_400
    output_format: str = "ply"
    opensplat: str = "opensplat"

    def validate(self) -> "SplatSettings":
        if type(self.iterations) is not int or not 100 <= self.iterations <= 50_000:
            raise ReferenceError("OpenSplat iterations must be 100–50,000")
        if type(self.timeout_seconds) is not int or not 1 <= self.timeout_seconds <= 86_400:
            raise ReferenceError("OpenSplat timeout must be 1–86,400 seconds")
        if self.output_format not in ("ply", "spz"):
            raise ReferenceError("OpenSplat output format must be ply or spz")
        if not isinstance(self.opensplat, str) or not self.opensplat.strip() or "\0" in self.opensplat:
            raise ReferenceError("OpenSplat executable must be a nonempty path or command")
        return self


def audit_splat_artifact(path: str | Path) -> dict:
    """Validate Gaussian vertex attributes and binary length, not photorealism."""
    file = Path(path)
    if file.is_symlink() or not file.is_file() or file.stat().st_size < 16:
        raise ReferenceError("Gaussian splat artifact is missing, linked or empty")
    if file.suffix == ".spz":
        return {"format": "spz", "bytes": file.stat().st_size,
                "sha256": _sha256(file), "structure_verified": False,
                "warning": "Compressed SPZ requires an independent format decoder for structural QA"}
    if file.suffix != ".ply":
        raise ReferenceError("Gaussian splat artifact must be .ply or .spz")
    with file.open("rb") as stream:
        header = stream.read(65_536)
    terminator = b"end_header\n"
    end = header.find(terminator)
    if end < 0 or not header.startswith(b"ply\n"):
        raise ReferenceError("Gaussian PLY header is missing or oversized")
    header_size = end + len(terminator)
    try:
        lines = header[:end].decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ReferenceError("Gaussian PLY header must be ASCII") from error
    if "format binary_little_endian 1.0" not in lines:
        raise ReferenceError("Gaussian PLY must use binary little-endian encoding")
    vertex_count: int | None = None
    vertex_fields: dict[str, int] = {}
    in_vertex = False
    for line in lines:
        words = line.split()
        if len(words) == 3 and words[0] == "element":
            in_vertex = words[1] == "vertex"
            if in_vertex:
                try:
                    vertex_count = int(words[2])
                except ValueError as error:
                    raise ReferenceError("Gaussian PLY vertex count is invalid") from error
            elif words[1] != "vertex" and words[2] != "0":
                raise ReferenceError("Gaussian PLY contains unsupported nonvertex elements")
        elif in_vertex and words and words[0] == "property":
            if len(words) != 3 or words[1] not in _SCALAR_BYTES or words[2] in vertex_fields:
                raise ReferenceError("Gaussian PLY has an unsupported vertex property")
            vertex_fields[words[2]] = _SCALAR_BYTES[words[1]]
    if vertex_count is None or not 1 <= vertex_count <= 10_000_000:
        raise ReferenceError("Gaussian PLY must contain 1–10,000,000 vertices")
    if not _GAUSSIAN_FIELDS.issubset(vertex_fields):
        raise ReferenceError("Gaussian PLY is missing positions, opacity, SH color, rotation or scales")
    minimum = header_size + vertex_count * sum(vertex_fields.values())
    if file.stat().st_size < minimum:
        raise ReferenceError("Gaussian PLY contains truncated vertex data")
    return {"format": "ply", "bytes": file.stat().st_size,
            "sha256": _sha256(file), "gaussians": vertex_count,
            "structure_verified": True,
            "warning": "Header/data length cannot certify Gaussian finiteness or photographic likeness"}


def plan_splat_revision(reconstruction: str | Path, output: str | Path,
                        settings: SplatSettings, *,
                        rectified_masks: str | Path | None = None) -> dict:
    """Run all preflight checks before creating an OpenSplat revision."""
    settings.validate()
    target = Path(output).resolve()
    if Path(output).is_symlink() or target.exists():
        raise ReferenceError(f"Output revision already exists: {target}")
    if not shutil.which(settings.opensplat):
        raise ReferenceError(f"OpenSplat executable not available: {settings.opensplat}")
    dataset = plan_splat_dataset(reconstruction, target / "dataset",
                                 rectified_masks=rectified_masks)
    return {"source_revision": dataset["source_revision"], "output": str(target),
            "asset_id": dataset["asset_id"],
            "registered_images": dataset["registered_images"],
            "rectified_masks": dataset["rectified_masks"],
            "opensplat_executable": str(Path(shutil.which(settings.opensplat) or "").resolve()),
            "iterations": settings.iterations, "timeout_seconds": settings.timeout_seconds,
            "output_format": settings.output_format,
            "status": "preflight-only / GPU-training-and-review-required"}


def build_splat_revision(reconstruction: str | Path, output: str | Path,
                         settings: SplatSettings, *,
                         rectified_masks: str | Path | None = None) -> dict:
    """Run separately installed OpenSplat and record an honest private candidate."""
    plan = plan_splat_revision(reconstruction, output, settings,
                               rectified_masks=rectified_masks)
    target = Path(plan["output"])
    target.mkdir(parents=True, exist_ok=False)
    return _train_in_revision(reconstruction, target, settings, plan,
                              rectified_masks=rectified_masks)


def build_multiview_splat_revision(references: ReferenceSet, output: str | Path,
                                   settings: SplatSettings, *,
                                   engine: str = "pycolmap", colmap: str = "colmap",
                                   sparse_timeout_seconds: int = 3600,
                                   max_image_size: int = 4000) -> dict:
    """Register photo cameras then train splats in one self-contained revision.

    Sparse camera evidence stays inside the result; it never lives in a temp
    directory that disappears after training. Failures are written to the same
    revision, without invoking OpenSplat when camera registration failed.
    """
    settings.validate()
    validate_camera_settings(engine, sparse_timeout_seconds, max_image_size)
    intake = plan_splat_input(references)
    target = Path(output).resolve()
    if Path(output).is_symlink() or target.exists():
        raise ReferenceError(f"Output revision already exists: {target}")
    trainer = shutil.which(settings.opensplat)
    if trainer is None:
        raise ReferenceError(f"OpenSplat executable not available: {settings.opensplat}")
    if engine == "pycolmap" and importlib.util.find_spec("pycolmap") is None:
        raise ReferenceError("PyCOLMAP is not installed: pip install pycolmap")
    if engine == "colmap" and shutil.which(colmap) is None:
        raise ReferenceError(f"COLMAP executable not found: {colmap}")
    from .colmap import build_sparse_reconstruction

    target.mkdir(parents=True, exist_ok=False)
    sparse = target / "sparse-reconstruction"
    report: dict = {
        "schema_version": 1, "asset_id": references.asset_id,
        "source_mode": "multiview-photos", "source_images": intake["independent_images"],
        "distribution_allowed": False, "failed_stage": None, "exit_code": None,
        "status": "in-progress / camera-registration-required",
        "limitations": "Multiple photos do not guarantee overlapping camera poses or a faithful object."
    }

    def checkpoint() -> dict:
        (target / "splat-report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    checkpoint()
    try:
        camera_report = build_sparse_reconstruction(
            references, sparse, colmap=colmap, engine=engine,
            timeout_seconds=sparse_timeout_seconds, undistort=True,
            max_image_size=max_image_size)
        report["camera_reconstruction"] = camera_report
        if (not camera_report["status"].startswith("candidate") or
                not camera_report.get("undistorted_images")):
            report.update(status="failed / review-required", failed_stage="camera-registration",
                          exit_code=3)
            return checkpoint()
        # The outer revision already exists, so run the dataset gate directly.
        dataset = plan_splat_dataset(sparse, target / "dataset")
        plan = {"source_revision": dataset["source_revision"], "output": str(target),
                "asset_id": dataset["asset_id"],
                "registered_images": dataset["registered_images"],
                "opensplat_executable": str(Path(trainer).resolve())}
    except (ReferenceError, OSError, ValueError, KeyError) as error:
        report.update(status="failed / review-required", failed_stage="camera-registration",
                      exit_code=2, error=str(error))
        return checkpoint()
    return _train_in_revision(sparse, target, settings, plan, extra_report=report)


def _train_in_revision(reconstruction: str | Path, target: Path,
                       settings: SplatSettings, plan: dict, *,
                       rectified_masks: str | Path | None = None,
                       extra_report: dict | None = None) -> dict:
    """Common training worker for pre-registered and just-registered inputs."""
    logs = target / "logs"
    logs.mkdir()
    report = {"schema_version": 1, "asset_id": plan["asset_id"],
              "source_revision": plan["source_revision"], "settings": {
                  "iterations": settings.iterations,
                  "timeout_seconds": settings.timeout_seconds,
                  "output_format": settings.output_format,
                  "opensplat_executable": plan["opensplat_executable"]},
              "registered_images": plan["registered_images"],
              "accelerator": "OpenSplat build-selected / not independently verified",
              "distribution_allowed": False,
              "status": "in-progress / review-required", "failed_stage": None,
              "exit_code": None,
              "limitations": "A splat is not a mesh or verified monument likeness. Image and model rights must be reviewed independently."}
    if extra_report:
        report.update({"source_mode": extra_report["source_mode"],
                       "source_images": extra_report["source_images"],
                       "camera_reconstruction": extra_report["camera_reconstruction"]})

    def checkpoint() -> dict:
        (target / "splat-report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    checkpoint()
    try:
        dataset_report = stage_splat_dataset(reconstruction, target / "dataset",
                                             rectified_masks=rectified_masks)
        report["dataset"] = dataset_report
    except (ReferenceError, OSError, ValueError) as error:
        report.update(status="failed / review-required", failed_stage="dataset-staging",
                      error=str(error), exit_code=2)
        return checkpoint()
    checkpoint()
    artifact = target / f"splat.{settings.output_format}"
    command = [plan["opensplat_executable"], str(target / "dataset"),
               "-n", str(settings.iterations), "-o", str(artifact)]
    report["command"] = command
    try:
        with (logs / "opensplat.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=settings.timeout_seconds, check=False)
        report["exit_code"] = completed.returncode
        if completed.returncode:
            report.update(status="failed / review-required", failed_stage="opensplat")
            return checkpoint()
    except subprocess.TimeoutExpired:
        report.update(status="failed / review-required", failed_stage="opensplat-timeout",
                      exit_code=124)
        return checkpoint()
    except OSError as error:
        report.update(status="failed / review-required", failed_stage="opensplat",
                      exit_code=127, error=str(error))
        return checkpoint()
    try:
        report["artifact"] = audit_splat_artifact(artifact)
    except ReferenceError as error:
        report.update(status="failed / review-required", failed_stage="artifact-audit",
                      exit_code=3, error=str(error))
        return checkpoint()
    report["status"] = "candidate / visual-review-required"
    report["files"] = {"splat": artifact.name, "dataset": "dataset",
                       "training_log": "logs/opensplat.log"}
    return checkpoint()
