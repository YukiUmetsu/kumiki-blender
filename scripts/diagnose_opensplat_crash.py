#!/usr/bin/env python3
"""Collect a native OpenSplat crash backtrace from a copied, registered dataset.

Requires LLDB on the Mac. Does not rerun COLMAP or modify the source revision.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import signal
import subprocess
from pathlib import Path

from retry_opensplat_training import validate_dataset


def _short_probe(argv: list[str], *, timeout: int = 15) -> dict:
    try:
        p = subprocess.run(argv, capture_output=True, text=True, errors="replace",
                           timeout=timeout, check=False)
        return {"exit_code": p.returncode,
                "output": (p.stdout + p.stderr)[-6000:]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def _debug_command(source: Path, target: Path, report: dict, lldb: str) -> list[str]:
    original = report.get("command")
    if not isinstance(original, list) or len(original) < 8 or not all(
            isinstance(part, str) for part in original):
        raise ValueError("Invalid retry command")
    binary = Path(original[0]).resolve(strict=True)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError("OpenSplat binary is not executable")
    if Path(original[1]).resolve() != (source / "dataset").resolve():
        raise ValueError("Retry report references a different input dataset")
    if original[2:3] != ["-n"] or original[4:5] != ["-d"]:
        raise ValueError("Unexpected retry argument layout")
    if original[6:7] != ["-o"] or len(original) not in (8, 9):
        raise ValueError("Unexpected retry argument layout")
    if len(original) == 9 and original[8] != "--cpu":
        raise ValueError("Unexpected retry option")
    if Path(original[7]).resolve() != (source / "splat.ply").resolve():
        raise ValueError("Retry report references an unexpected output file")
    # Reuse the exact CPU/GPU setting and training parameters that crashed.
    training = [str(binary), str(target / "dataset"),
                "-n", original[3], "-d", original[5],
                "-o", str(target / "splat.ply"), *original[8:]]
    return [lldb, "--batch", "-o", "run", "-k", "thread backtrace all",
            "--", *training]


def diagnose(retry_revision: Path, output: Path, *, lldb: str = "lldb",
             timeout: int = 600) -> dict:
    source = retry_revision.resolve(strict=True)
    if not (source / "retry-report.json").is_file():
        raise ValueError("Expected a failed OpenSplat retry revision")
    if not 30 <= timeout <= 3600:
        raise ValueError("Timeout must be between 30 and 3600 seconds")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")
    target = output.resolve()
    if target == source or target.is_relative_to(source):
        raise ValueError("Debug output cannot be inside the source retry revision")
    report = json.loads((source / "retry-report.json").read_text(encoding="utf-8"))
    if report.get("status") != "failed / native-trainer":
        raise ValueError("Only debug failed native-trainer retries")
    validate_dataset(source)
    debugger = shutil.which(lldb)
    if debugger is None:
        raise FileNotFoundError(f"LLDB not available: {lldb}; run xcode-select --install")
    debugger = str(Path(debugger).resolve())
    command = _debug_command(source, target, report, debugger)
    target.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source / "dataset", target / "dataset", symlinks=False)
    validate_dataset(target)
    log_path = target / "lldb.log"
    summary = {
        "schema_version": 1,
        "status": "in-progress",
        "source_retry": str(source),
        "original_exit_code": report.get("exit_code"),
        "original_mode": report.get("mode"),
        "registered_images": report.get("registered_images"),
        "host": {"system": platform.system(), "machine": platform.machine(),
                 "macos": platform.mac_ver()[0]},
        "binary_version_probe": _short_probe([command[7], "--version"]),
        "binary_linkage_probe": (
            _short_probe(["otool", "-L", command[7]])
            if platform.system() == "Darwin" else {"status": "not-macOS"}),
        "lldb_command": command,
        "lldb_log": "lldb.log",
        "dataset": "copied-and-hash-verified",
        "status_note": "LLDB exit code alone does not identify the native crash cause.",
    }
    def save() -> dict:
        (target / "crash-qa.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary
    save()
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            summary["lldb_exit_code"] = process.wait(timeout=timeout)
            summary["status"] = "trace-captured / inspect-log"
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            summary["lldb_exit_code"] = 124
            summary["status"] = "debugger-timeout"
    # Scan the whole log for the first crashing frames. Long backtraces may
    # push the actual fault out of the last 128 KiB, especially with many
    # worker threads. Keep only bounded excerpts in JSON.
    crash_pattern = re.compile(
        r"EXC_BAD_ACCESS|SIGSEGV|Segmentation fault", re.IGNORECASE)
    frame_pattern = re.compile(r"frame #\d+|stop reason =", re.IGNORECASE)
    launch_failure_pattern = re.compile(
        r"process launch failed|error:.*(?:launch|attach).*failed|"
        r"not allowed to attach|Operation not permitted", re.IGNORECASE)
    saw_crash = False
    launch_failed = False
    frames = []
    with log_path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            saw_crash |= bool(crash_pattern.search(line))
            launch_failed |= bool(launch_failure_pattern.search(line))
            if len(frames) < 50 and (frame_pattern.search(line) or
                                     launch_failure_pattern.search(line)):
                frames.append(line.strip()[:500])
    with log_path.open("rb") as stream:
        stream.seek(max(0, log_path.stat().st_size - 128 * 1024))
        tail = stream.read().decode("utf-8", "replace")
    summary["saw_native_crash"] = saw_crash and not launch_failed
    summary["frame_excerpt"] = frames
    summary["log_tail"] = tail[-12000:]
    if summary["status"] != "debugger-timeout":
        if launch_failed:
            summary["status"] = "debugger-launch-failed"
        elif not saw_crash:
            summary["status"] = "trace-inconclusive"
        elif not any(re.search(r"frame #\d+", frame, re.IGNORECASE)
                     for frame in frames):
            summary["status"] = "crash-seen / no-backtrace"
    return save()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("retry_revision", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lldb", default="lldb")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args(argv)
    try:
        report = diagnose(args.retry_revision, args.output,
                          lldb=args.lldb, timeout=args.timeout)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.exit(2, f"ERROR: {error}\n")
    print(f"LLDB status: {report['status']}")
    print(f"Original mode: {report['original_mode']}; "
          f"registered cameras: {report['registered_images']}")
    print(f"Native crash observed in trace: {report['saw_native_crash']}")
    print(f"Report: {args.output / 'crash-qa.json'}")
    print(f"Trace: {args.output / 'lldb.log'}")
    for line in report["frame_excerpt"][:12]:
        print(line)
    return 0 if report["status"].startswith("trace-captured") else 1


if __name__ == "__main__":
    raise SystemExit(main())
