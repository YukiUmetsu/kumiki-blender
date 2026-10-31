"""Lightweight native artifact existence and magic-byte checks."""

from __future__ import annotations

from pathlib import Path, PurePosixPath


class ExportAuditError(ValueError):
    """A required output is absent, empty or has an invalid signature."""


_SIGNATURES = {
    ".blend": b"BLENDER",
    ".png": b"\x89PNG\r\n\x1a\n",
}


def audit_export_files(root: Path, filenames: list[str]) -> dict[str, int]:
    """Read only explicitly contained, regular output files; return sizes."""
    destination = Path(root).resolve()
    sizes: dict[str, int] = {}
    for filename in filenames:
        relative = PurePosixPath(filename)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts or "\\" in filename:
            raise ExportAuditError(f"Expected a contained relative output name: {filename}")
        target = destination.joinpath(*relative.parts)
        if not target.resolve().is_relative_to(destination):
            raise ExportAuditError(f"Output path escapes artifact root: {filename}")
        if target.is_symlink() or not target.is_file():
            raise ExportAuditError(f"Required output is missing or not a regular file: {filename}")
        size = target.stat().st_size
        if size == 0:
            raise ExportAuditError(f"Required output is empty: {filename}")
        expected = _SIGNATURES.get(target.suffix.lower())
        if expected:
            with target.open("rb") as file:
                if file.read(len(expected)) != expected:
                    raise ExportAuditError(f"Invalid {target.suffix.upper()[1:]} signature in {filename}")
        sizes[filename] = size
    return sizes
