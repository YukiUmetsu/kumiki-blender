"""Strict ingestion of MakeHuman's official CC0 graphical *basemesh* OBJ.

The fixed upstream commit and file digest protect against a replaced model;
helper, joint, and clothing OBJ groups do not become anatomical mesh faces.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen


SOURCE_COMMIT = "a8bc2d54ff0ac92e78ff71431b1023eda42bf482"
SOURCE_URL = ("https://raw.githubusercontent.com/makehumancommunity/makehuman/"
              f"{SOURCE_COMMIT}/makehuman/data/3dobjs/base.obj")
SOURCE_SHA256 = "8e761e6624b8f54536409135d1636da63b32486a90d4897f84e121d144f6fb4c"
SOURCE_LICENSE = "CC0-1.0"
MAX_SOURCE_BYTES = 8 * 1024 * 1024


class MakeHumanSourceError(ValueError):
    """Unapproved or invalid anatomy source."""


@dataclass(frozen=True)
class MeshSource:
    path: Path
    sha256: str
    upstream_url: str = SOURCE_URL
    license_id: str = SOURCE_LICENSE


def verify_makehuman_obj(path: str | Path) -> MeshSource:
    """Require exactly the pinned, CC0-labeled upstream graphical model."""
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.stat().st_size > MAX_SOURCE_BYTES:
        raise MakeHumanSourceError("MakeHuman OBJ is missing or exceeds the size limit")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise MakeHumanSourceError(f"MakeHuman OBJ SHA-256 mismatch: {digest}; expected {SOURCE_SHA256}")
    with path.open("r", encoding="utf-8") as stream:
        header = "".join(stream.readline() for _ in range(16))
    if "This is a mesh file for MakeHuman" not in header or "released as CC0" not in header:
        raise MakeHumanSourceError("MakeHuman OBJ missing the expected CC0 source header")
    return MeshSource(path, digest)


def load_body_mesh(source: MeshSource) -> tuple[tuple[tuple[float, float, float], ...],
                                                tuple[tuple[int, ...], ...]]:
    """Read actual surface topology from `g body`, ignoring all helper groups."""
    vertices: list[tuple[float, float, float]] = []
    body_faces: list[tuple[int, ...]] = []
    group = ""
    with source.path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            if parts[0] == "v":
                if len(parts) != 4:
                    raise MakeHumanSourceError(f"Invalid vertex at line {line_number}")
                try:
                    point = tuple(float(value) for value in parts[1:])
                except ValueError as error:
                    raise MakeHumanSourceError(f"Invalid vertex at line {line_number}") from error
                if any(not math.isfinite(value) for value in point):
                    raise MakeHumanSourceError(f"Non-finite vertex at line {line_number}")
                vertices.append(point)
            elif parts[0] == "g":
                group = " ".join(parts[1:])
            elif parts[0] == "f" and group == "body":
                if not 3 <= len(parts) - 1 <= 8:
                    raise MakeHumanSourceError(f"Invalid body polygon at line {line_number}")
                try:
                    face = tuple(int(value.split("/")[0]) - 1 for value in parts[1:])
                except ValueError as error:
                    raise MakeHumanSourceError(f"Invalid face index at line {line_number}") from error
                if any(index < 0 or index >= len(vertices) for index in face) or len(set(face)) != len(face):
                    raise MakeHumanSourceError(f"Invalid face indices at line {line_number}")
                body_faces.append(face)
    if not body_faces:
        raise MakeHumanSourceError("No anatomical faces found in OBJ `g body` group")
    used_indices = sorted({index for face in body_faces for index in face})
    compact_indices = {index: compact for compact, index in enumerate(used_indices)}
    return (tuple(vertices[index] for index in used_indices),
            tuple(tuple(compact_indices[index] for index in face) for face in body_faces))


def fetch_makehuman_obj(cache_path: str | Path) -> MeshSource:
    """Explicitly fetch a fixed-version OBJ into a local, reusable cache.

    A failed download/hash check leaves no approved cache and never replaces a
    user's existing (possibly incorrect) cache file.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() or cache_path.is_symlink():
        return verify_makehuman_obj(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache_path.parent, prefix=".makehuman-",
                                         suffix=".obj", delete=False) as target:
            temporary = Path(target.name)
            with urlopen(SOURCE_URL, timeout=45) as response:
                remaining = MAX_SOURCE_BYTES + 1
                while True:
                    block = response.read(min(64 * 1024, remaining))
                    if not block:
                        break
                    remaining -= len(block)
                    if remaining <= 0:
                        raise MakeHumanSourceError("MakeHuman download exceeds approved size limit")
                    target.write(block)
        source = verify_makehuman_obj(temporary)
        if cache_path.exists() or cache_path.is_symlink():
            return verify_makehuman_obj(cache_path)
        os.rename(temporary, cache_path)
        temporary = None
        return MeshSource(cache_path.resolve(), source.sha256)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
