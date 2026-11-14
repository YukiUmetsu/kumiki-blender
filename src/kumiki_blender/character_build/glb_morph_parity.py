"""Independently verify exported glTF morph-weight animation against Blender samples.

This reads the actual GLB binary accessors (not merely channel names) and
rejects missing targets, duplicate channels, wrong target order, invalid
timestamps, interpolation drift and morph amplitudes. Standard-library only.
"""
from __future__ import annotations

import bisect
import json
import math
import struct
from pathlib import Path

from .glb_animation import _chunks

_BIN = 0x004E4942


def _scalar_accessor(doc: dict, binary: bytes, index: int) -> list[float]:
    if type(index) is not int or not 0 <= index < len(doc.get("accessors", [])):
        raise ValueError("Missing or invalid GLB animation accessor")
    accessor = doc["accessors"][index]
    if (accessor.get("type") != "SCALAR" or accessor.get("componentType") != 5126 or
            "sparse" in accessor or "bufferView" not in accessor):
        raise ValueError("Morph animation requires dense float32 SCALAR accessors")
    views = doc.get("bufferViews", [])
    view_idx = accessor["bufferView"]
    if type(view_idx) is not int or not 0 <= view_idx < len(views):
        raise ValueError("Invalid morph bufferView index")
    view = views[view_idx]
    if view.get("buffer", 0) != 0:
        raise ValueError("Morph accessor references a nonbinary external buffer")
    count = accessor.get("count")
    if type(count) is not int or not 1 <= count <= 1_000_000:
        raise ValueError("Invalid or excessive morph sample count")
    offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    stride = view.get("byteStride", 4)
    limit = view.get("byteOffset", 0) + view["byteLength"]
    if (type(offset) is not int or offset < 0 or
            type(stride) is not int or stride < 4 or stride % 4 or
            offset + (count - 1) * stride + 4 > min(limit, len(binary))):
        raise ValueError("Morph accessor samples exceed binary buffer bounds")
    values = [struct.unpack_from("<f", binary, offset + stride * i)[0]
              for i in range(count)]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Morph accessor contains NaN or Infinity")
    return values


def _interpolated(times: list[float], values: list[float], t: float,
                  interpolation: str) -> float:
    index = bisect.bisect_right(times, t)
    if index <= 1:
        return values[0]
    if index >= len(times):
        return values[-1]
    left, right = index - 1, index
    if interpolation == "STEP":
        return values[left]
    factor = (t - times[left]) / (times[right] - times[left])
    return values[left] * (1 - factor) + values[right] * factor


def verify_glb_morph_samples(path: str | Path, expected: dict, *,
                             fps: int, tolerance: float = 0.025) -> dict:
    """Compare glTF morph channels with source Blender shape-key driver samples.

    expected: {mesh object name: {shape key name: [(frame, value), ...]}}.
    A morph channel animates *all* shape-key targets at once; this handles
    multiple target shapes and separate mesh nodes without assuming target order.
    """
    if type(fps) is not int or not 1 <= fps <= 120:
        raise ValueError("Invalid morph animation FPS")
    if (type(tolerance) not in (int, float) or not math.isfinite(tolerance)
            or not 0 < tolerance <= .05):
        raise ValueError("Invalid morph parity threshold")
    if not expected:
        raise ValueError("No source shape-key driver values were supplied")
    chunks = _chunks(Path(path))
    if len(chunks) != 2 or chunks[1][0] != _BIN:
        raise ValueError("Morph parity requires exactly one internal GLB BIN chunk")
    doc = json.loads(chunks[0][1])
    animations = doc.get("animations", [])
    if not isinstance(animations, list) or len(animations) != 1:
        raise ValueError("Corrective and bone channels must share a single animation")
    animation = animations[0]
    channels = animation.get("channels", [])
    samplers = animation.get("samplers", [])
    nodes = doc.get("nodes", [])
    meshes = doc.get("meshes", [])
    if not channels or not samplers:
        raise ValueError("GLB animation has no channels or samplers")
    reports = []
    for object_name, named_samples in sorted(expected.items()):
        candidates = [(i, n) for i, n in enumerate(nodes)
                      if n.get("name") == object_name and "mesh" in n]
        if len(candidates) != 1:
            raise ValueError(f"{object_name}: expected exactly one named exported GLB mesh node")
        node_index, node = candidates[0]
        mesh_index = node["mesh"]
        if type(mesh_index) is not int or not 0 <= mesh_index < len(meshes):
            raise ValueError(f"{object_name}: invalid exported mesh reference")
        mesh = meshes[mesh_index]
        target_names = mesh.get("extras", {}).get("targetNames")
        if (not isinstance(target_names, list) or
                len(target_names) != len(set(target_names)) or
                not all(isinstance(name, str) for name in target_names)):
            raise ValueError(f"{object_name}: exporter lacks unique ordered morph target names")
        if not target_names or not set(named_samples) <= set(target_names):
            raise ValueError(f"{object_name}: authored corrective shape targets missing in GLB")
        for primitive in mesh.get("primitives", []):
            if len(primitive.get("targets", [])) != len(target_names):
                raise ValueError(f"{object_name}: GLB primitive does not contain every morph target")
        matched = [
            channel for channel in channels
            if channel.get("target", {}).get("node") == node_index
            and channel.get("target", {}).get("path") == "weights"
        ]
        if len(matched) != 1:
            raise ValueError(f"{object_name}: missing or duplicate glTF morph animation channel")
        sampler_index = matched[0]["sampler"]
        if type(sampler_index) is not int or not 0 <= sampler_index < len(samplers):
            raise ValueError(f"{object_name}: invalid morph animation sampler")
        sampler = samplers[sampler_index]
        interp = sampler.get("interpolation", "LINEAR")
        if interp not in ("STEP", "LINEAR"):
            raise ValueError(f"{object_name}: unsupported morph sampler interpolation {interp}")
        times = _scalar_accessor(doc, chunks[1][1], sampler["input"])
        weights = _scalar_accessor(doc, chunks[1][1], sampler["output"])
        if len(weights) != len(times) * len(target_names):
            raise ValueError(f"{object_name}: morph float stream does not match target count")
        if any(t < 0 or (i and t <= times[i - 1])
               for i, t in enumerate(times)):
            raise ValueError(f"{object_name}: nonmonotonic or negative animation timestamps")
        if any(v < -tolerance or v > 1 + tolerance for v in weights):
            raise ValueError(f"{object_name}: exported shape weights exceed [0,1]")
        maxima = {}
        for name, samples in sorted(named_samples.items()):
            if not isinstance(samples, list) or not samples:
                raise ValueError(f"{object_name}/{name}: no expected source-frame weights")
            shape_index = target_names.index(name)
            sequence = [weights[frame * len(target_names) + shape_index]
                        for frame in range(len(times))]
            max_error = 0.0
            for frame, expected_weight in samples:
                if (type(frame) is not int or
                        type(expected_weight) not in (float, int) or
                        not math.isfinite(expected_weight) or
                        not 0 <= expected_weight <= 1):
                    raise ValueError(f"{object_name}/{name}: invalid expected morph sample")
                instant = frame / fps
                if instant < times[0] - 1e-4 or instant > times[-1] + 1e-4:
                    raise ValueError(f"{object_name}/{name}: source sample outside exported clip range")
                actual = _interpolated(times, sequence, instant, interp)
                max_error = max(max_error, abs(actual - expected_weight))
            if max_error > tolerance + 1e-7:
                raise ValueError(
                    f"{object_name}/{name}: morph weight drift {max_error:.6f} "
                    f"exceeds {tolerance:.6f}")
            maxima[name] = round(max_error, 7)
        reports.append({
            "object": object_name, "targets": target_names,
            "source_correctives": sorted(named_samples),
            "animation_samples": len(times),
            "max_weight_error_by_key": maxima,
            "interpolation": interp,
        })
    return {
        "status": "exported-morph-parity-verified",
        "target_meshes": len(reports),
        "checked_correctives": sum(len(x["source_correctives"]) for x in reports),
        "max_weight_error": max((error for row in reports
                                  for error in row["max_weight_error_by_key"].values()), default=0),
        "meshes": reports,
    }
