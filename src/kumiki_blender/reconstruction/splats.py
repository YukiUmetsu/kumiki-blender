"""Immutable, private OpenSplat training revisions for Apple Metal or CUDA.

OpenSplat is a separately installed, AGPL-licensed executable. This module
does not bundle or modify it and never claims that a splat is a polygon mesh.
"""

from __future__ import annotations

import gzip
import json
import importlib.util
import math
import shutil
import struct
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
_FLOAT_TYPES = {"float": "f", "float32": "f", "double": "d", "float64": "d"}


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
        return _audit_spz_header(file)
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
    vertex_properties: list[tuple[str, str]] = []
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
            vertex_properties.append((words[2], words[1]))
    if vertex_count is None or not 1 <= vertex_count <= 10_000_000:
        raise ReferenceError("Gaussian PLY must contain 1–10,000,000 vertices")
    if not _GAUSSIAN_FIELDS.issubset(vertex_fields):
        raise ReferenceError("Gaussian PLY is missing positions, opacity, SH color, rotation or scales")
    minimum = header_size + vertex_count * sum(vertex_fields.values())
    if file.stat().st_size < minimum:
        raise ReferenceError("Gaussian PLY contains truncated vertex data")
    # Check a bounded, deterministic sample of records. This catches the
    # common case of invalid GPU output without loading multi-GB splats into RAM.
    offsets: dict[str, tuple[int, str]] = {}
    stride = 0
    for name, scalar in vertex_properties:
        if name in _GAUSSIAN_FIELDS:
            if scalar not in _FLOAT_TYPES:
                raise ReferenceError(f"Gaussian PLY field {name} must be floating-point")
            offsets[name] = (stride, _FLOAT_TYPES[scalar])
        stride += _SCALAR_BYTES[scalar]
    sample_count = min(vertex_count, 256)
    indices = {i * (vertex_count - 1) // max(1, sample_count - 1)
               for i in range(sample_count)}
    with file.open("rb") as stream:
        for index in sorted(indices):
            stream.seek(header_size + index * stride)
            row = stream.read(stride)
            for name, (offset, scalar) in offsets.items():
                if not math.isfinite(struct.unpack_from("<" + scalar, row, offset)[0]):
                    raise ReferenceError(f"Gaussian PLY contains non-finite {name} values")
    return {"format": "ply", "bytes": file.stat().st_size,
            "sha256": _sha256(file), "gaussians": vertex_count,
            "structure_verified": True, "sampled_vertices": len(indices),
            "warning": "Finite-value samples and PLY structure cannot certify every vertex or photographic likeness"}


def _audit_spz_header(file: Path) -> dict:
    """Verify SPZ framing without claiming that we decoded compressed splats.

    Legacy versions 1–3 are gzip-wrapped; v4 uses plaintext NGSP followed by
    a bounded table of contents. This is deliberately *not* a ZSTD decoder.
    """
    size = file.stat().st_size
    with file.open("rb") as stream:
        prefix = stream.read(32)
    if prefix[:4] == b"NGSP":
        if len(prefix) != 32:
            raise ReferenceError("SPZ v4 header is truncated")
        magic, version, points, degree, bits, flags, streams, toc_offset, reserved = (
            struct.unpack("<IIIBBBBI12s", prefix))
        if (magic != 0x5053474e or version != 4 or not 1 <= points <= 10_000_000
                or degree > 4 or bits > 31 or not 1 <= streams <= 32
                or reserved != bytes(12) or toc_offset < 32 or toc_offset > size):
            raise ReferenceError("SPZ v4 header contains invalid metadata")
        if toc_offset + streams * 16 > size:
            raise ReferenceError("SPZ v4 table of contents is truncated")
        with file.open("rb") as stream:
            stream.seek(toc_offset)
            table = stream.read(streams * 16)
        total_compressed = 0
        for index in range(streams):
            compressed, uncompressed = struct.unpack_from("<QQ", table, index * 16)
            if not compressed or not uncompressed:
                raise ReferenceError("SPZ v4 has empty compressed attribute streams")
            total_compressed += compressed
        if toc_offset + streams * 16 + total_compressed != size:
            raise ReferenceError("SPZ v4 attribute stream sizes do not match file length")
    elif prefix[:2] == b"\x1f\x8b":
        try:
            with gzip.open(file, "rb") as stream:
                header = stream.read(16)
        except (OSError, EOFError) as error:
            raise ReferenceError("SPZ legacy gzip header is invalid") from error
        if len(header) != 16:
            raise ReferenceError("SPZ legacy gzip header is truncated")
        magic, version, points = struct.unpack_from("<III", header)
        if magic != 0x5053474e or version not in (1, 2, 3) or not 1 <= points <= 10_000_000:
            raise ReferenceError("SPZ legacy gzip metadata is invalid")
    else:
        raise ReferenceError("SPZ must begin with an NGSP v4 or legacy gzip header")
    return {"format": "spz", "bytes": size, "sha256": _sha256(file),
            "spz_version": version, "gaussians": points,
            "structure_verified": False,
            "warning": "SPZ header and stream lengths checked; compressed data still requires an independent decoder"}


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
            "sparse_feature_evidence": dataset["sparse_feature_evidence"],
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
                "sparse_feature_evidence": dataset["sparse_feature_evidence"],
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
              "sparse_feature_evidence": plan["sparse_feature_evidence"],
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
