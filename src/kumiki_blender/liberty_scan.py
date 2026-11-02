"""Item-level provenance and integrity checks for Liberty's licensed scan base.

The photogrammetry was made from a nineteenth-century *miniature*, so it is an
artistic geometry reference, not a measured scan of the New York monument.
Nothing in this module quietly substitutes an unrelated or unlicensed model.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .liberty_face_sculpt import face_relief_offset


SOURCE_RECORD = (Path(__file__).resolve().parent / "assets" / "landmarks" /
                 "statues" / "statue-of-liberty" / "scan_source.json")


class ScanSourceError(ValueError):
    """Missing, different, or incompletely attributed scan geometry."""


def load_scan_source() -> dict[str, Any]:
    """Load the reviewed source/licence record included with this project."""
    try:
        record = json.loads(SOURCE_RECORD.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ScanSourceError(f"Scan provenance is unreadable: {error}") from error
    if (not isinstance(record, dict) or record.get("schema_version") != 1
            or record.get("license") != "CC-BY-4.0"
            or record.get("author") != "jerryfisher"
            or not str(record.get("original_url", "")).startswith("https://sketchfab.com/3d-models/")
            or record.get("license_url") != "https://creativecommons.org/licenses/by/4.0/"
            or not re.fullmatch(r"[a-f0-9]{64}", str(record.get("sha256", "")))
            or type(record.get("size_bytes")) is not int
            or record["size_bytes"] <= 0
            or not record.get("attribution")):
        raise ScanSourceError("Scan source record lacks confirmed CC BY 4.0 attribution or checksum")
    return record


def verify_scan_source(path: str | Path, record: dict[str, Any] | None = None) -> dict[str, Any]:
    """Refuse anonymous, truncated, or substituted GLB files before Blender runs."""
    requested = Path(path)
    if requested.is_symlink():
        raise ScanSourceError(f"Licensed Liberty scan must not be a symlink: {requested}")
    source = requested.resolve()
    metadata = record if record is not None else load_scan_source()
    if not source.is_file():
        raise ScanSourceError(f"Licensed Liberty scan not found: {source}")
    if source.stat().st_size != metadata["size_bytes"]:
        raise ScanSourceError("Licensed scan size does not match the recorded source")
    digest = hashlib.sha256()
    with source.open("rb") as binary:
        if binary.read(4) != b"glTF":
            raise ScanSourceError("Licensed scan is not a GLB binary")
        binary.seek(0)
        for chunk in iter(lambda: binary.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != metadata["sha256"]:
        raise ScanSourceError("Licensed scan SHA-256 differs from the reviewed source")
    return metadata

__all__ = ("ScanSourceError", "load_scan_source", "verify_scan_source", "face_relief_offset")
