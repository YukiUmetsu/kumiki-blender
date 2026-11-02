"""Convert Blender-baked glTF node clips to deterministic Kumiki PM2 actions.

Read the *exported GLB* rather than raw bpy transforms: this preserves the
exact Y-up glTF local basis, hierarchy and semantic node IDs consumed by
Kumiki-UI. stdlib only; no Blender or third-party animation SDK is required.
"""

from __future__ import annotations

from bisect import bisect_right
import json
import math
from pathlib import Path
import re
import struct
from typing import Any


class MotionExportError(ValueError):
    """An animation cannot be exported faithfully to Kumiki PM2."""


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,100}\Z")
_RATE = 30
_MAX_GLB = 512 * 1024 * 1024
_MAX_KEYS = 120_000


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MotionExportError(f"{label} must be an object")
    return value


def _index(value: Any, length: int, label: str) -> int:
    if type(value) is not int or not 0 <= value < length:
        raise MotionExportError(f"{label} must be an in-range index")
    return value


def _real(value: Any, label: str, minimum: float, maximum: float) -> float:
    if (type(value) not in (float, int) or not math.isfinite(value)
            or value < minimum or value > maximum):
        raise MotionExportError(f"{label} must be finite in [{minimum}, {maximum}]")
    return float(value)


def _vector(value: Any, length: int, label: str, minimum: float, maximum: float) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise MotionExportError(f"{label} must be a {length}-element array")
    return [_real(v, f"{label}[{i}]", minimum, maximum) for i, v in enumerate(value)]


def _quat(value: list[float]) -> list[float]:
    norm = math.sqrt(sum(x*x for x in value))
    if not .9 <= norm <= 1.1:
        raise MotionExportError("glTF animation contains a non-unit rotation quaternion")
    return [v/norm for v in value]


def _slerp(a: list[float], b: list[float], t: float) -> list[float]:
    dot = sum(x*y for x, y in zip(a, b))
    aligned = [v * (-1 if dot < 0 else 1) for v in b]
    cosine = min(1.0, abs(dot))
    if cosine > .9995:
        q = [(1-t)*x + t*y for x, y in zip(a, aligned)]
    else:
        angle = math.acos(cosine)
        denom = math.sin(angle)
        q = [(math.sin((1-t)*angle)*x + math.sin(t*angle)*y)/denom
             for x, y in zip(a, aligned)]
    return _quat(q)


class _Glb:
    def __init__(self, path: str | Path) -> None:
        source = Path(path)
        try:
            size = source.stat().st_size
            if not 32 <= size <= _MAX_GLB:
                raise MotionExportError("GLB must be 32 bytes to 512 MiB")
            with source.open("rb") as stream:
                header = stream.read(12)
                if len(header) != 12 or struct.unpack("<4sII", header) != (b"glTF", 2, size):
                    raise MotionExportError("Expected a complete glTF 2.0 GLB")
                length, kind = struct.unpack("<I4s", stream.read(8))
                if kind != b"JSON" or length > 16 * 1024 * 1024 or 20 + length + 8 > size:
                    raise MotionExportError("Missing or oversized GLB JSON chunk")
                payload = stream.read(length)
                if len(payload) != length:
                    raise MotionExportError("Truncated GLB JSON")
                binary_length, kind = struct.unpack("<I4s", stream.read(8))
                if kind != b"BIN\x00" or stream.tell() + binary_length > size:
                    raise MotionExportError("GLB must have an embedded binary chunk")
                self.bin_offset = stream.tell()
                self.bin_length = binary_length
            self.document = _object(json.loads(payload.decode("utf-8").rstrip(" \x00")), "glTF")
            if _object(self.document.get("asset"), "glTF.asset").get("version") != "2.0":
                raise MotionExportError("GLB asset must declare glTF 2.0")
            buffers = self.document.get("buffers")
            if (not isinstance(buffers, list) or len(buffers) != 1
                    or not isinstance(buffers[0], dict) or "uri" in buffers[0]
                    or type(buffers[0].get("byteLength")) is not int
                    or buffers[0]["byteLength"] < 0
                    or buffers[0]["byteLength"] > self.bin_length):
                raise MotionExportError("Animation requires one embedded GLB buffer with valid length")
            self.source = source
        except (OSError, UnicodeError, struct.error, json.JSONDecodeError) as error:
            raise MotionExportError(f"Cannot read GLB animation: {error}") from error

    def accessor(self, number: Any, shape: str) -> list[list[float]]:
        doc = self.document
        accessors = doc.get("accessors", [])
        accessor = _object(accessors[_index(number, len(accessors), "accessor")], "accessor")
        count = _index(accessor.get("count"), 1_000_001, "accessor.count")
        if count == 0:
            raise MotionExportError("Animation accessors must have at least one sample")
        if accessor.get("componentType") != 5126 or accessor.get("type") != shape or "sparse" in accessor:
            raise MotionExportError("Animation accessors must use embedded non-sparse FLOAT data")
        views = doc.get("bufferViews", [])
        view = _object(views[_index(accessor.get("bufferView"), len(views), "bufferView")], "bufferView")
        if view.get("buffer") != 0:
            raise MotionExportError("Animation accessors must reference GLB buffer 0")
        dimensions = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[shape]
        step = dimensions * 4
        stride = view.get("byteStride", step)
        if type(stride) is not int or stride < step or stride % 4:
            raise MotionExportError("Invalid animation bufferView stride")
        view_offset = view.get("byteOffset", 0)
        view_length = view.get("byteLength")
        local = accessor.get("byteOffset", 0)
        if (any(type(v) is not int or v < 0 for v in (view_offset, view_length, local))
                or local + (count-1)*stride + step > view_length
                or view_offset + view_length > self.bin_length):
            raise MotionExportError("Animation accessor exceeds its embedded bufferView")
        offset = self.bin_offset + view_offset + local
        values = []
        with self.source.open("rb") as stream:
            for index in range(count):
                stream.seek(offset + index*stride)
                payload = stream.read(step)
                if len(payload) != step:
                    raise MotionExportError("Truncated animation accessor")
                values.append([float(x) for x in struct.unpack("<" + "f"*dimensions, payload)])
        return values


def _semantic_ids(doc: dict[str, Any]) -> dict[int, str]:
    """Match the GLTFLoader.scene DFS order used by Kumiki's per-role part index."""
    nodes = doc.get("nodes", [])
    scenes = doc.get("scenes", [])
    if not isinstance(nodes, list) or not isinstance(scenes, list):
        raise MotionExportError("GLB nodes and scenes must be arrays")
    scene = _object(scenes[_index(doc.get("scene", 0), len(scenes), "scene")], "scene")
    tagged: dict[int, str] = {}
    counts: dict[str, int] = {}
    seen: set[int] = set()

    def visit(index: Any) -> None:
        i = _index(index, len(nodes), "node")
        if i in seen:
            raise MotionExportError("Repeated/cyclic glTF scene node cannot be indexed")
        seen.add(i)
        node = _object(nodes[i], f"node[{i}]")
        part = _object(node.get("extras", {}), f"node[{i}].extras").get("kumiki_part")
        if part is not None:
            if not isinstance(part, str) or not _ID.fullmatch(part):
                raise MotionExportError(f"node[{i}] has an invalid kumiki_part role")
            counts[part] = counts.get(part, 0) + 1
            tagged[i] = f"kumiki:{part}:{counts[part]}"
        for child in node.get("children", []):
            visit(child)

    for root in scene.get("nodes", []):
        visit(root)
    return tagged


def canonical_ids_by_node_name(path: str | Path) -> dict[str, str]:
    """Resolve the exported glTF node names to canonical UI part IDs.

    Refuse ambiguous names; Blender's object names are unique but another
    exporter may collapse or rename them during its hierarchy conversion.
    """
    glb = _Glb(path)
    ids = _semantic_ids(glb.document)
    result: dict[str, str] = {}
    for index, canonical in ids.items():
        name = glb.document["nodes"][index].get("name")
        if not isinstance(name, str) or not name or name in result:
            raise MotionExportError("Semantic glTF node names must be unique for contact cue alignment")
        result[name] = canonical
    return result


def _sample(times: list[float], values: list[list[float]], path: str,
            seconds: float, interpolation: str) -> list[float]:
    index = max(0, min(len(times)-1, bisect_right(times, seconds)-1))
    before = values[index]
    if index == len(times)-1 or interpolation == "STEP":
        return before
    interval = times[index+1] - times[index]
    t = min(1.0, max(0.0, (seconds-times[index])/interval))
    if path == "rotation":
        return _slerp(before, values[index+1], t)
    return [(1-t)*a + t*b for a, b in zip(before, values[index+1])]


def export_glb_motion(path: str | Path, *, clip_name: str | None = None, action_id: str | None = None,
                      contacts: list[dict[str, Any]] | None = None,
                      merge_clips: bool = False, resample_mixed: bool = False) -> list[dict[str, Any]]:
    """Export a glTF clip, or merge disjoint physics-node clips, to PM2.

    Rigid body physics should be baked to keyframes in Blender before GLB
    export. Unsupported morph, bones without part tags, and cubic interpolation
    fail explicitly so no video silently diverges from Blender's animation.
    """
    glb = _Glb(path)
    doc = glb.document
    clips = doc.get("animations", [])
    if not isinstance(clips, list):
        raise MotionExportError("glTF animations must be an array")
    matches = [a for a in clips if isinstance(a, dict) and (clip_name is None or a.get("name") == clip_name)]
    if not matches or (len(matches) != 1 and not merge_clips):
        raise MotionExportError(f"Expected exactly one glTF animation for {clip_name or 'automatic clip selection'!r}; specify --clip when several exist")
    clip_name = (matches[0].get("name") if len(matches) == 1 else "combined-physics") or "physics"
    if not isinstance(clip_name, str):
        raise MotionExportError("glTF animation name must be text")
    slug = re.sub(r"[^A-Za-z0-9._:-]+", "-", clip_name).strip("-:")[:80] or "physics"
    name = action_id or f"{slug}-baked"
    if not _ID.fullmatch(name):
        raise MotionExportError("PM2 action id must be a safe identifier")
    semantic = _semantic_ids(doc)
    nodes = doc["nodes"]
    samples_by_node: dict[int, dict[str, tuple[list[float], list[list[float]], str]]] = {}
    duration = 0.0
    for animation in matches:
        tracks = animation.get("channels", [])
        if not isinstance(tracks, list) or not 1 <= len(tracks) <= 384:
            raise MotionExportError("Animation requires 1–384 channels")
        samplers = animation.get("samplers", [])
        if not isinstance(samplers, list):
            raise MotionExportError("Animation samplers must be an array")
        for channel in tracks:
            channel = _object(channel, "channel")
            target = _object(channel.get("target"), "channel.target")
            node = _index(target.get("node"), len(nodes), "channel.target.node")
            if node not in semantic:
                raise MotionExportError(f"Animated node[{node}] needs a kumiki_part tag on the node itself")
            component = target.get("path")
            if component not in ("translation", "rotation", "scale"):
                raise MotionExportError(f"Unsupported animated path {component!r}; PM2 requires node TRS")
            if "matrix" in nodes[node]:
                raise MotionExportError(f"Animated node[{node}] uses matrix instead of glTF TRS")
            sampler = _object(samplers[_index(channel.get("sampler"), len(samplers), "sampler")], "sampler")
            interp = sampler.get("interpolation", "LINEAR")
            if interp not in ("LINEAR", "STEP"):
                raise MotionExportError("Only LINEAR and STEP glTF clips can be faithfully converted to PM2")
            times = [row[0] for row in glb.accessor(sampler.get("input"), "SCALAR")]
            values = glb.accessor(sampler.get("output"), "VEC4" if component == "rotation" else "VEC3")
            if not times or len(times) != len(values) or times[0] < 0 or any(
                    not math.isfinite(t) or t < 0 or t > 120 or (i and t <= times[i-1])
                    for i, t in enumerate(times)):
                raise MotionExportError("Animation timestamps must increase strictly within 120 seconds")
            for value in values:
                _vector(value, len(value), "animation transform", -1000, 1000)
            if component == "rotation":
                values = [_quat(_vector(v, 4, "rotation", -1, 1)) for v in values]
            assigned = samples_by_node.setdefault(node, {})
            if component in assigned:
                raise MotionExportError(f"node[{node}] repeats an animated {component} channel")
            assigned[component] = (times, values, interp)
            duration = max(duration, times[-1])
    duration_frames = max(1, math.ceil(duration*_RATE))
    if duration_frames > 3600 or len(samples_by_node) > 128 or (duration_frames+1)*len(samples_by_node) > _MAX_KEYS:
        raise MotionExportError("Animation exceeds the PM2 duration, track or key budget")
    output_tracks = []
    for node, channels in sorted(samples_by_node.items(), key=lambda item: semantic[item[0]]):
        source = nodes[node]
        modes = {c[2] for c in channels.values()}
        if len(modes) > 1 and not resample_mixed:
            raise MotionExportError(f"node[{node}] mixes STEP and LINEAR channels; PM2 interpolates per track")
        defaults = {
            "translation": _vector(source.get("translation", [0, 0, 0]), 3, "node translation", -1000, 1000),
            "rotation": _quat(_vector(source.get("rotation", [0, 0, 0, 1]), 4, "node rotation", -1, 1)),
            "scale": _vector(source.get("scale", [1, 1, 1]), 3, "node scale", 0, 100),
        }
        keys = []
        for frame in range(duration_frames+1):
            seconds = frame/_RATE
            def value(component: str) -> list[float]:
                if component not in channels:
                    return defaults[component]
                ts, values, mode = channels[component]
                return _sample(ts, values, component, seconds, mode)
            keys.append({"frame": frame, "position": value("translation"),
                         "rotation": value("rotation"), "scale": value("scale")})
        # When explicitly opted in, sample mixed glTF channels at every 30-fps
        # output boundary. LINEAR between adjacent baked PM2 frames preserves
        # the exact values at rendered frames; no sub-frame parity is promised.
        interpolation = "linear" if len(modes) > 1 else modes.pop().lower()
        output_tracks.append({"target": semantic[node], "interpolation": interpolation, "keys": keys})

    cues = []
    if contacts is not None:
        if not isinstance(contacts, list) or len(contacts) > 4096:
            raise MotionExportError("Contact cues must be an array of at most 4096 records")
        for i, raw in enumerate(contacts):
            item = _object(raw, f"contacts[{i}]")
            if set(item) not in ({"frame", "target", "speedMps"}, {"frame", "target", "speedMps", "material"}):
                raise MotionExportError("Contact cue requires frame, target, speedMps and optional material")
            target = item["target"]
            if not isinstance(target, str) or target not in semantic.values():
                raise MotionExportError("Contact cue must name one exported canonical semantic part")
            cue = {"frame": _real(item["frame"], "contact.frame", 0, duration_frames),
                   "target": target, "speedMps": _real(item["speedMps"], "contact.speedMps", 0, 500)}
            if "material" in item:
                material = item["material"]
                if not isinstance(material, str) or not _ID.fullmatch(material):
                    raise MotionExportError("Contact material must be a safe semantic label")
                cue["material"] = material
            cues.append(cue)
    return [{"schemaVersion": 1, "id": name, "label": f"Blender bake: {clip_name}",
             "durationFrames": duration_frames, "tracks": output_tracks,
             "contacts": sorted(cues, key=lambda x: x["frame"])}]


def write_glb_motion(path: str | Path, output: str | Path, *, clip_name: str | None = None,
                     action_id: str | None = None, contacts_path: str | Path | None = None,
                     merge_clips: bool = False,
                     contacts: list[dict[str, Any]] | None = None,
                     resample_mixed: bool = False) -> dict[str, Any]:
    """Produce a new standalone PM2 JSON sidecar; never overwrite user output."""
    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise MotionExportError(f"Output already exists: {destination}")
    if contacts is not None and contacts_path is not None:
        raise MotionExportError("Provide either in-memory measured contacts or --contacts, not both")
    cues = contacts
    if contacts_path is not None:
        source = Path(contacts_path)
        if source.stat().st_size > 256 * 1024:
            raise MotionExportError("Contact cue JSON exceeds 256 KiB")
        try:
            cues = json.loads(source.read_text(encoding="utf-8"),
                              parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        except (ValueError, UnicodeError) as error:
            raise MotionExportError(f"Malformed contact cues: {error}") from error
    actions = export_glb_motion(path, clip_name=clip_name, action_id=action_id, contacts=cues,
                                merge_clips=merge_clips, resample_mixed=resample_mixed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(actions, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return {"output": str(destination), "action": actions[0]["id"],
            "tracks": len(actions[0]["tracks"]),
            "frames": actions[0]["durationFrames"], "contacts": len(actions[0]["contacts"])}
