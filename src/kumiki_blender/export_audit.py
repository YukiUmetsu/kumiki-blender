"""Native artifact checks, including complete PNG chunk integrity."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import struct
import zlib


class ExportAuditError(ValueError):
    """A required output is absent, empty or has an invalid signature."""


_SIGNATURES = {
    ".blend": b"BLENDER",
    ".png": b"\x89PNG\r\n\x1a\n",
}


def _verify_png(target: Path) -> None:
    """Reject cut-off Cycles renders that still have a valid PNG signature.

    PNG uses length-prefixed chunks with CRCs; streaming their checks avoids
    Pillow as a runtime dependency and detects damaged IDAT/IEND chunks.
    """
    with target.open("rb") as stream:
        if stream.read(8) != _SIGNATURES[".png"]:
            raise ExportAuditError(f"Invalid PNG signature in {target.name}")
        first=True
        seen_data=False
        while True:
            prefix=stream.read(8)
            if len(prefix)!=8:
                raise ExportAuditError(f"Truncated PNG chunks in {target.name}")
            length,tag=struct.unpack(">I4s",prefix)
            if length>100_000_000:
                raise ExportAuditError(f"Unreasonably large PNG chunk in {target.name}")
            if first and (tag!=b"IHDR" or length!=13):
                raise ExportAuditError(f"Missing PNG image header in {target.name}")
            first=False
            checksum=zlib.crc32(tag)
            remaining=length
            while remaining:
                chunk=stream.read(min(remaining,1024*1024))
                if not chunk:
                    raise ExportAuditError(f"Truncated PNG pixel data in {target.name}")
                checksum=zlib.crc32(chunk,checksum)
                remaining-=len(chunk)
            footer=stream.read(4)
            if len(footer)!=4 or struct.unpack(">I",footer)[0]!=checksum:
                raise ExportAuditError(f"PNG checksum mismatch in {target.name}")
            if tag==b"IDAT":
                seen_data=True
            if tag==b"IEND":
                if length or not seen_data or stream.read(1):
                    raise ExportAuditError(f"Incomplete PNG image in {target.name}")
                return


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
            if target.suffix.lower()==".png":
                _verify_png(target)
        sizes[filename] = size
    return sizes
