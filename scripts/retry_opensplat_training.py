#!/usr/bin/env python3
"""Retry OpenSplat from an existing verified COLMAP dataset, without rerunning SfM.

Copies the immutable staged dataset to a new revision: OpenSplat may create
image caches next to its input, so running against the source is unsafe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
from pathlib import Path

from kumiki_blender.reconstruction.splats import audit_splat_artifact


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_file(root: Path, relative: str, expected: str) -> None:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Invalid relative dataset path")
    if any(part in ("..", ".") for part in Path(relative).parts):
        raise ValueError("Unsafe dataset path")
    path = root / relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Missing, linked or unsafe dataset file: {relative}")
    if _sha256(path) != expected:
        raise ValueError(f"Dataset integrity mismatch: {relative}")


def validate_dataset(source: Path) -> dict:
    dataset = source / "dataset"
    if source.is_symlink() or dataset.is_symlink() or not dataset.is_dir():
        raise ValueError("Source must contain a regular staged dataset directory")
    report_file = dataset / "dataset-report.json"
    if report_file.is_symlink() or not report_file.is_file():
        raise ValueError("Missing dataset-report.json")
    report = json.loads(report_file.read_text(encoding="utf-8"))
    photos = report.get("photo_hashes")
    qa = report.get("viewpoint_qa") or {}
    if (not isinstance(photos, list) or len(photos) < 8 or
            len(photos) != report.get("registered_images")):
        raise ValueError("Dataset must have >=8 verified registered photos")
    seen = set()
    for entry in photos:
        name = entry["name"]
        if name in seen or Path(name).name != name:
            raise ValueError("Duplicate or unsafe dataset photo name")
        seen.add(name)
        _checked_file(dataset, "images/" + name, entry["sha256"])
    for name, digest in qa.get("sparse_binary_hashes", {}).items():
        if name not in ("cameras.bin", "images.bin", "points3D.bin"):
            raise ValueError("Unexpected COLMAP binary filename")
        _checked_file(dataset, "sparse/0/" + name, digest)
    for name, digest in qa.get("text_model_hashes", {}).items():
        if name not in ("cameras.txt", "images.txt"):
            raise ValueError("Unexpected COLMAP camera filename")
        _checked_file(dataset, "cameras/" + name, digest)
    if set(qa.get("sparse_binary_hashes", {})) != {
            "cameras.bin", "images.bin", "points3D.bin"}:
        raise ValueError("Incomplete sparse COLMAP binary hashes")
    if set(qa.get("text_model_hashes", {})) != {"cameras.txt", "images.txt"}:
        raise ValueError("Incomplete COLMAP camera hashes")
    for directory, subdirs, files in os.walk(dataset, followlinks=False):
        for name in subdirs + files:
            if (Path(directory) / name).is_symlink():
                raise ValueError("Dataset contains a symbolic link")
    return report


def train_existing(revision: Path, output: Path, *, opensplat: str = "opensplat",
                   iterations: int = 300, downscale: float = 4.0,
                   cpu: bool = False, timeout: int = 14400) -> dict:
    if not 100 <= iterations <= 50000 or not isinstance(iterations, int):
        raise ValueError("Iterations must be 100..50000")
    if not math.isfinite(downscale) or not 1 <= downscale <= 16:
        raise ValueError("Downscale must be between 1 and 16")
    if not 60 <= timeout <= 86400:
        raise ValueError("Timeout must be 60..86400 seconds")
    source = revision.resolve(strict=True)
    target = output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")
    if target == source or target.is_relative_to(source):
        raise ValueError("Output must not be inside the source revision")
    executable = shutil.which(opensplat)
    if not executable:
        raise FileNotFoundError(f"OpenSplat executable not found: {opensplat}")
    executable = str(Path(executable).resolve())
    if not os.access(executable, os.X_OK):
        raise ValueError(f"OpenSplat is not executable: {executable}")
    source_report = validate_dataset(source)
    target.mkdir(parents=True, exist_ok=False)
    logs = target / "logs"
    logs.mkdir()
    dataset = target / "dataset"
    shutil.copytree(source / "dataset", dataset, symlinks=False)
    validate_dataset(target)  # Detect copy errors before invoking the native trainer.
    command = [executable, str(dataset), "-n", str(iterations),
               "-d", str(downscale), "-o", str(target / "splat.ply")]
    if cpu:
        command.append("--cpu")
    report = {
        "schema_version": 1,
        "source_revision": str(source),
        "source_dataset_report_sha256": _sha256(source / "dataset/dataset-report.json"),
        "registered_images": source_report["registered_images"],
        "mode": "cpu" if cpu else "build-selected-gpu",
        "downscale_factor": downscale,
        "iterations": iterations,
        "command": command,
        "status": "in-progress",
        "exit_code": None,
        "training_log": "logs/opensplat.log",
        "input_copy": "dataset",
        "limitations": "A trained splat is not a polygon mesh or verified statue likeness.",
    }
    def save() -> dict:
        (target / "retry-report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                  encoding="utf-8")
        return report
    save()
    try:
        with (logs / "opensplat.log").open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                       timeout=timeout, check=False)
        report["exit_code"] = completed.returncode
        if completed.returncode:
            report["status"] = "failed / native-trainer"
            if completed.returncode < 0:
                report["signal"] = -completed.returncode
                report["signal_name"] = (
                    signal.Signals(-completed.returncode).name
                    if -completed.returncode in signal.Signals._value2member_map_
                    else "unknown")
    except subprocess.TimeoutExpired:
        report.update(status="failed / timeout", exit_code=124)
    except OSError as error:
        report.update(status="failed / launch", exit_code=127, error=str(error))
    if report["status"] == "in-progress":
        try:
            report["artifact"] = audit_splat_artifact(target / "splat.ply")
            report["status"] = "candidate / visual-review-required"
        except (OSError, ValueError) as error:
            report.update(status="failed / artifact-audit", error=str(error),
                          exit_code=3)
    if report["status"].startswith("failed"):
        log_file = logs / "opensplat.log"
        if log_file.exists():
            # Read only a bounded tail; native trainer logs can be very large.
            with log_file.open("rb") as stream:
                stream.seek(max(0, log_file.stat().st_size - 16384))
                report["log_tail"] = stream.read().decode("utf-8", "replace")[-8000:]
        report["debug_command"] = shlex.join([
            "lldb", "--batch", "-o", "run", "-o", "thread backtrace all",
            "--", *command[:-1], command[-1]
        ])
        report["next_action"] = (
            "Inspect logs/opensplat.log. If MPS crashed, retry from the original "
            "source revision with --cpu to isolate GPU-specific failures. "
            "Run debug_command manually for a native stack trace."
        )
    return save()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision", type=Path, help="Existing splat revision with dataset/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opensplat", default="opensplat")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--downscale", type=float, default=4)
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = train_existing(
            args.revision, args.output, opensplat=args.opensplat,
            iterations=args.iterations, downscale=args.downscale,
            cpu=args.cpu, timeout=args.timeout)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.exit(2, f"ERROR: {error}\n")
    print(f"Source cameras: {report['registered_images']} (no COLMAP rerun)")
    print(f"Status: {report['status']} | exit code: {report['exit_code']}")
    print(f"Report: {args.output / 'retry-report.json'}")
    print(f"Log: {args.output / 'logs/opensplat.log'}")
    if report.get("signal_name"):
        print(f"Native crash: {report['signal_name']}")
    if report.get("next_action"):
        print(report["next_action"])
    return 0 if report["status"].startswith("candidate") else 1


if __name__ == "__main__":
    raise SystemExit(main())
