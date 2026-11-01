"""Reviewed CC0 Blender Studio anatomical head, distinct from the miniature scan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


RECORD_PATH = (Path(__file__).resolve().parent / "assets" / "landmarks" /
               "statues" / "statue-of-liberty" / "anatomy_source.json")


class AnatomySourceError(ValueError):
    """The Blender Studio CC0 mesh is missing or not integrity-verified."""


def load_anatomy_source() -> dict[str, Any]:
    record = json.loads(RECORD_PATH.read_text(encoding="utf8"))
    if (record.get("schema_version") != 1 or record.get("license") != "CC0-1.0"
            or record.get("author") != "Dan Ulrich / Blender Studio"
            or not record.get("download_url", "").startswith(
                "https://download.blender.org/demo/asset-bundles/")
            or len(record.get("sha256", "")) != 64
            or len(record.get("archive_sha256", "")) != 64):
        raise AnatomySourceError("CC0 head metadata is missing or lacks reviewed provenance")
    return record


def verify_anatomy_source(path: str | Path,
                          record: dict[str, Any] | None = None) -> dict[str, Any]:
    source = Path(path)
    metadata = record or load_anatomy_source()
    if source.is_symlink() or not source.is_file():
        raise AnatomySourceError("The original CC0 Blender Studio .blend is missing or is a symlink")
    if source.stat().st_size != metadata["size_bytes"]:
        raise AnatomySourceError("CC0 head source binary size differs from the reviewed record")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        prefix = stream.read(7)
        # Blender stores some .blend assets as zstd-compressed containers.
        if not (prefix.startswith(b"BLENDER") or
                prefix.startswith(b"\x28\xb5\x2f\xfd")):
            raise AnatomySourceError("CC0 head source lacks a Blender or zstd header")
        stream.seek(0)
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != metadata["sha256"]:
        raise AnatomySourceError("CC0 head source SHA-256 mismatch")
    return metadata


__all__ = ("AnatomySourceError", "load_anatomy_source", "verify_anatomy_source")
