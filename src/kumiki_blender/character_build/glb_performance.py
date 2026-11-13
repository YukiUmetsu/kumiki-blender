"""Merge simultaneous native Blender skeletal and morph Actions into one GLB clip.

Works ONLY on a single already-exported GLB scene containing both the rig and
the facial mesh. Animation samplers/channels are reindexed in JSON; all buffer
chunks (keyframes, geometry and textures) remain byte-for-byte unchanged.
"""
from __future__ import annotations

import json
import math
import os
import re
import struct
from pathlib import Path

from .glb_animation import _chunks, read_single_animation, rewrite_single_animation_name

_JSON = 0x4E4F534A
_SAFE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,95}$")
_SKIN_PATHS = frozenset(("translation", "rotation", "scale"))


def _animated_targets(doc: dict, animation: dict) -> tuple[int, int, set[tuple[int, str]]]:
    joints = {index for skin in doc.get("skins", [])
              for index in skin.get("joints", [])}
    meshes = doc.get("meshes", [])
    morph_nodes = set()
    for index, node in enumerate(doc.get("nodes", [])):
        mesh_id = node.get("mesh")
        if type(mesh_id) is int and 0 <= mesh_id < len(meshes):
            if any(primitive.get("targets") for primitive in meshes[mesh_id].get("primitives", [])):
                morph_nodes.add(index)
    bone_channels = morph_channels = 0
    targets: set[tuple[int, str]] = set()
    samplers = animation.get("samplers")
    channels = animation.get("channels")
    if not isinstance(samplers, list) or not isinstance(channels, list) or not channels:
        raise ValueError("Animation must contain valid samplers and nonempty channels")
    for channel in channels:
        target = channel.get("target", {})
        node, path = target.get("node"), target.get("path")
        sampler = channel.get("sampler")
        if (type(node) is not int or type(sampler) is not int
                or not 0 <= sampler < len(samplers) or
                path not in _SKIN_PATHS | {"weights"}):
            raise ValueError("Animation contains unsupported or invalid channel targets")
        pair = (node, path)
        if pair in targets:
            raise ValueError(f"Duplicate channel target {pair} in one animation")
        targets.add(pair)
        if path in _SKIN_PATHS and node in joints:
            bone_channels += 1
        if path == "weights" and node in morph_nodes:
            morph_channels += 1
    return bone_channels, morph_channels, targets


def _times(doc: dict, animation: dict) -> tuple[float, float]:
    accessors = doc.get("accessors", [])
    minimum, maximum = math.inf, -math.inf
    for sampler in animation["samplers"]:
        if not isinstance(sampler, dict) or type(sampler.get("input")) is not int:
            raise ValueError("Animation sampler has invalid input")
        accessor_index = sampler["input"]
        if not 0 <= accessor_index < len(accessors):
            raise ValueError("Animation references unknown input accessor")
        accessor = accessors[accessor_index]
        low, high = accessor.get("min"), accessor.get("max")
        if (accessor.get("type") != "SCALAR" or accessor.get("componentType") != 5126
                or not isinstance(low, list) or len(low) != 1
                or not isinstance(high, list) or len(high) != 1
                or any(type(v) not in (float, int) or not math.isfinite(v) for v in low + high)
                or low[0] > high[0]):
            raise ValueError("GLB animation input must have finite float scalar min/max")
        minimum = min(minimum, low[0])
        maximum = max(maximum, high[0])
    if not math.isfinite(minimum) or maximum <= minimum:
        raise ValueError("Animation has no measurable duration")
    return minimum, maximum


def _replace_json(path: Path, doc: dict, old_chunks: list) -> None:
    serialized = json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    serialized += b" " * ((-len(serialized)) % 4)
    chunks = [(_JSON, serialized), *old_chunks[1:]]
    size = 12 + sum(8 + len(payload) for _, payload in chunks)
    output = struct.pack("<4sII", b"glTF", 2, size)
    output += b"".join(struct.pack("<II", len(payload), kind) + payload
                       for kind, payload in chunks)
    tmp = path.with_suffix(path.suffix + ".performance.tmp")
    if tmp.exists():
        raise FileExistsError(f"Stale temporary performance GLB: {tmp}")
    try:
        tmp.write_bytes(output)
        if _chunks(tmp)[1:] != old_chunks[1:]:
            raise ValueError("Animation merge unexpectedly changed binary GLB chunks")
        if len(read_single_animation(tmp).get("animations", [])) != 1:
            raise ValueError("Animation merge failed to produce one clip")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def finalize_performance_glb(path: str | Path, clip: str, *,
                             time_alignment_tolerance_s: float = .05) -> dict:
    """Combine skeletal and facial tracks in place only on a derived GLB."""
    if not isinstance(clip, str) or not _SAFE.fullmatch(clip):
        raise ValueError("Invalid clip name")
    if (type(time_alignment_tolerance_s) not in (float, int)
            or not math.isfinite(time_alignment_tolerance_s)
            or not 0 < time_alignment_tolerance_s <= .2):
        raise ValueError("Invalid time alignment tolerance")
    path = Path(path)
    chunks = _chunks(path)
    doc = json.loads(chunks[0][1])
    animations = doc.get("animations", [])
    if not isinstance(animations, list) or not 1 <= len(animations) <= 2:
        raise ValueError("Expected one combined clip or exactly two simultaneous Blender Actions")
    union: set[tuple[int, str]] = set()
    body = face = 0
    ranges = []
    previous = []
    for animation in animations:
        if not isinstance(animation, dict) or set(animation) - {"name", "samplers", "channels"}:
            raise ValueError("Cannot safely merge animation extensions/extras")
        b, f, targets = _animated_targets(doc, animation)
        if union & targets:
            raise ValueError("Overlapping animation node/path targets would conflict")
        union.update(targets)
        body += b
        face += f
        ranges.append(_times(doc, animation))
        previous.append(animation.get("name"))
    if body < 1 or face < 1:
        raise ValueError("A combined performance needs both skinned-joint and facial morph channels")
    if len(ranges) == 2:
        a, b = ranges
        if (abs(a[0] - b[0]) > time_alignment_tolerance_s
                or abs(a[1] - b[1]) > time_alignment_tolerance_s):
            raise ValueError(f"Facial and skeletal Actions are not time aligned: {ranges}")
    if len(animations) == 2:
        merged = {"name": clip, "samplers": [], "channels": []}
        for animation in animations:
            offset = len(merged["samplers"])
            merged["samplers"].extend(animation["samplers"])
            for channel in animation["channels"]:
                merged["channels"].append({
                    **channel, "sampler": channel["sampler"] + offset
                })
        doc["animations"] = [merged]
        _replace_json(path, doc, chunks)
    else:
        rewrite_single_animation_name(path, clip)
    result = read_single_animation(path)
    b, f, _ = _animated_targets(result, result["animations"][0])
    if b != body or f != face or result["animations"][0]["name"] != clip:
        raise ValueError("Combined performance lost skin/morph channels")
    return {
        "clip": clip, "input_animation_names": previous,
        "input_tracks": len(animations), "output_tracks": 1,
        "deform_joint_channels": b, "morph_weight_channels": f,
        "samplers": len(result["animations"][0]["samplers"]),
        "time_ranges_s": ranges,
        "binary_chunks_preserved": True,
        "status": "combined-performance-review-required",
    }
