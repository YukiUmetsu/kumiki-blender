"""Pure-stdlib, lossless GLB JSON-chunk renaming for a single animation.

glTF exporters may call the sole active action "Animation" even when Blender
uses a meaningful Action name. Change only the JSON chunk and length headers;
do not re-encode buffers, geometry, skinning, textures, or keyframe values.
"""
from __future__ import annotations

import json
import os
import re
import struct
from pathlib import Path

_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,95}$")
_JSON = 0x4E4F534A


def _chunks(path: Path):
    blob = path.read_bytes()
    if len(blob) < 20 or blob[:4] != b"glTF":
        raise ValueError("Not a binary glTF asset")
    magic, version, size = struct.unpack_from("<4sII", blob, 0)
    if version != 2 or size != len(blob):
        raise ValueError("GLB version/size is not valid")
    pos, chunks = 12, []
    while pos < size:
        if pos + 8 > size:
            raise ValueError("Truncated GLB chunk header")
        n, kind = struct.unpack_from("<II", blob, pos)
        pos += 8
        if n % 4 or pos + n > size:
            raise ValueError("Truncated/misaligned GLB chunk")
        chunks.append((kind, blob[pos:pos + n]))
        pos += n
    if not chunks or chunks[0][0] != _JSON or sum(kind == _JSON for kind, _ in chunks) != 1:
        raise ValueError("GLB must contain exactly one leading JSON chunk")
    return chunks


def read_single_animation(path: str | Path) -> dict:
    doc = json.loads(_chunks(Path(path))[0][1])
    if not isinstance(doc.get("animations"), list) or len(doc["animations"]) != 1:
        raise ValueError("GLB must contain exactly one animation")
    return doc


def rewrite_single_animation_name(path: str | Path, clip: str) -> dict:
    """Atomically rename only animation[0] and preserve every non-JSON byte."""
    if not isinstance(clip, str) or not _NAME.fullmatch(clip):
        raise ValueError("Animation clip name must be safe and non-empty")
    path = Path(path)
    chunks = _chunks(path)
    doc = json.loads(chunks[0][1])
    if not isinstance(doc.get("animations"), list) or len(doc["animations"]) != 1:
        raise ValueError("Only single-action GLB exports are supported")
    if not doc["animations"][0].get("channels"):
        raise ValueError("Refusing to label an animation without channels")
    previous = doc["animations"][0].get("name")
    doc["animations"][0]["name"] = clip
    serialized = json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    serialized += b" " * ((-len(serialized)) % 4)
    rebuilt = [(_JSON, serialized), *chunks[1:]]
    size = 12 + sum(8 + len(data) for _, data in rebuilt)
    with_path = struct.pack("<4sII", b"glTF", 2, size)
    with_path += b"".join(struct.pack("<II", len(data), kind) + data for kind, data in rebuilt)
    candidate = path.with_suffix(path.suffix + ".tmp")
    if candidate.exists():
        raise FileExistsError(f"Stale temporary animation asset: {candidate}")
    try:
        candidate.write_bytes(with_path)
        check = read_single_animation(candidate)
        if check["animations"][0]["name"] != clip or _chunks(candidate)[1:] != chunks[1:]:
            raise ValueError("GLB binary chunk changed while renaming animation")
        os.replace(candidate, path)
    finally:
        candidate.unlink(missing_ok=True)
    return {"previous_name": previous, "clip_name": clip, "binary_chunks_unchanged": True,
            "size_bytes": size}
