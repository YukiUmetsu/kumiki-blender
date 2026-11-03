"""Blender evaluated deforming-mesh → native glTF morph animation bridge.

The source .blend retains its cloth, soft-body or other evaluated modifier.
The export copy records topology-stable evaluated positions as shape keys and
animates their weights; Kumiki/Three.js then scrubs the standard GLB clip.
This is not a runtime cloth solver or a PM2 rigid-transform sidecar.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import struct
from typing import Any

from .deformation_plan import DeformationPlanError, DeformationShot
from .glb_audit import audit_glb


def _resolve(scene: Any, selector: str) -> Any:
    kind, label = selector.split(":", 1)
    found = [obj for obj in scene.objects if obj.type == "MESH" and
             (obj.name == label if kind == "name" else obj.get("kumiki_part") == label)]
    if len(found) != 1:
        raise DeformationPlanError(f"{selector!r} must resolve to exactly one editable mesh")
    obj = found[0]
    if obj.library is not None or obj.data.library is not None:
        raise DeformationPlanError("Deformation target must be a local, editable mesh")
    if obj.data.shape_keys is not None:
        raise DeformationPlanError("Existing shape keys need artist-directed compositing; use a clean modifier-driven mesh")
    return obj


def _sample_evaluated_vertices(bpy: Any, scene: Any, obj: Any, shot: DeformationShot) -> list[tuple[int, list[Any]]]:
    """Evaluate forward, including intermediate physics cache frames.

    Direct random seeks on an unbaked cloth cache are non-deterministic. A
    forward-only pass preserves the original solver evaluation order.
    """
    base_polygons = [tuple(poly.vertices) for poly in obj.data.polygons]
    if not base_polygons or not 3 <= len(obj.data.vertices) <= shot.max_vertices:
        raise DeformationPlanError("Target needs a surfaced 3–max_vertices mesh")
    included = set(shot.frames)
    sampled: list[tuple[int, list[Any]]] = []
    for frame in range(shot.duration_frames + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        graph = bpy.context.evaluated_depsgraph_get()
        evaluated = obj.evaluated_get(graph)
        data = evaluated.to_mesh(preserve_all_data_layers=False, depsgraph=graph)
        try:
            if (len(data.vertices) != len(obj.data.vertices) or
                    [tuple(poly.vertices) for poly in data.polygons] != base_polygons):
                raise DeformationPlanError("Evaluated topology differs from source; mesh vertex caches require identical indices/faces")
            if frame in included:
                points = [vertex.co.copy() for vertex in data.vertices]
                if any(not all(math.isfinite(v) and abs(v) <= 1000 for v in point) for point in points):
                    raise DeformationPlanError("Evaluated deformation has a nonfinite or out-of-bounds vertex")
                sampled.append((frame, points))
        finally:
            evaluated.to_mesh_clear()
    if tuple(frame for frame, _ in sampled) != shot.frames:
        raise DeformationPlanError("Missing ordered deformation samples")
    return sampled


def _to_morph_animation(bpy: Any, obj: Any, sampled: list[tuple[int, list[Any]]], shot: DeformationShot) -> float:
    # The editable source was already saved. Mutate only the export working
    # copy and never evaluate the original simulation after removing modifiers.
    obj.modifiers.clear()
    obj.data = obj.data.copy()
    basis_points = sampled[0][1]
    for vertex, point in zip(obj.data.vertices, basis_points):
        vertex.co = point
    basis = obj.shape_key_add(name="Basis", from_mix=False)
    basis.value = 0
    key_blocks = []
    max_displacement = 0.0
    for frame, points in sampled[1:]:
        key = obj.shape_key_add(name=f"km_frame_{frame:04d}", from_mix=False)
        key.value = 0.0
        for index, point in enumerate(points):
            key.data[index].co = point
            max_displacement = max(max_displacement, (point - basis_points[index]).length)
        key_blocks.append(key)

    # Keyframe every sampled pose on every channel. LINEAR interpolation gives
    # exact linear blending of adjacent evaluated samples and no Bezier bulges.
    for frame, _ in sampled:
        for i, key in enumerate(key_blocks):
            key.value = float(frame == sampled[i + 1][0])
            key.keyframe_insert(data_path="value", frame=frame, group="Kumiki Deformation")
    shape_keys = obj.data.shape_keys
    if shape_keys is None or shape_keys.animation_data is None or shape_keys.animation_data.action is None:
        raise DeformationPlanError("Blender failed to register deforming shape-key animation")
    shape_keys.animation_data.action.name = shot.action_id
    action = shape_keys.animation_data.action
    for fcurve in action.fcurves:
        for keypoint in fcurve.keyframe_points:
            keypoint.interpolation = "LINEAR"
    return max_displacement


def _audit_morph_animation(path: Path, shot: DeformationShot) -> dict[str, Any]:
    """Refuse a GLB lacking a usable complete native glTF morph animation."""
    with path.open("rb") as stream:
        header = stream.read(20)
        if len(header) != 20 or header[:4] != b"glTF" or struct.unpack_from("<I", header, 4)[0] != 2:
            raise DeformationPlanError("Export did not produce glTF 2.0 GLB")
        json_size = struct.unpack_from("<I", header, 12)[0]
        if header[16:20] != b"JSON" or json_size > 16 * 1024 * 1024:
            raise DeformationPlanError("Exported GLB lacks a bounded JSON chunk")
        document = json.loads(stream.read(json_size).decode("utf-8"))
    meshes = document.get("meshes", [])
    counts = [len(primitive.get("targets", [])) for mesh in meshes for primitive in mesh.get("primitives", [])]
    expected = len(shot.frames) - 1
    if counts != [expected]:
        raise DeformationPlanError(f"glTF morph exporter lost target poses: {counts}, expected {expected}")
    animations = [anim for anim in document.get("animations", []) if anim.get("name") == shot.action_id]
    if len(animations) != 1:
        raise DeformationPlanError("glTF export lost or renamed the expected deformation clip")
    action = animations[0]
    channels = [channel for channel in action.get("channels", []) if channel.get("target", {}).get("path") == "weights"]
    if len(channels) != 1:
        raise DeformationPlanError("Expected exactly one native glTF morph-weights animation channel")
    sampler = action["samplers"][channels[0]["sampler"]]
    accessors = document["accessors"]
    time = accessors[sampler["input"]]
    weights = accessors[sampler["output"]]
    if (sampler.get("interpolation") != "LINEAR" or time.get("type") != "SCALAR" or
            time.get("count", 0) < 2 or weights.get("count") != time["count"] * expected or
            abs(time.get("min", [-1])[0]) > 1e-5 or
            abs(time.get("max", [-1])[0] - shot.duration_frames / 30) > 1e-4):
        raise DeformationPlanError("glTF morph interpolation, key count or 30-fps timeline is inconsistent")
    return {"glb_weight_channels": len(channels), "glb_animation_samples": time["count"]}


def build_deformation_shot(bpy: Any, shot: DeformationShot, output: str | Path) -> dict[str, Any]:
    """Bake one saved .blend modifier-driven mesh into an immutable GLB revision."""
    destination = Path(output).resolve()
    if destination.exists() or destination.is_symlink():
        raise DeformationPlanError(f"Output revision already exists: {destination}")
    if not bpy.app.background:
        raise DeformationPlanError("Deformation bake requires Blender --background")
    source = getattr(bpy.data, "filepath", "")
    if not source or not Path(source).is_file():
        raise DeformationPlanError("Load a saved .blend before build-deformation")
    scene = bpy.context.scene
    if scene is None:
        raise DeformationPlanError("Blender has no active source scene")
    obj = _resolve(scene, shot.target)
    if obj.get("kumiki_part") is None:
        obj["kumiki_part"] = obj.name
    if len(obj.data.vertices) * (len(shot.frames) - 1) > 450_000:
        raise DeformationPlanError("Actual morph vertex budget exceeds 450000")

    scene.frame_start, scene.frame_end, scene.render.fps = 0, shot.duration_frames, 30
    sampled = _sample_evaluated_vertices(bpy, scene, obj, shot)
    destination.mkdir(parents=True, exist_ok=False)
    bpy.ops.wm.save_as_mainfile(filepath=str(destination / "source.blend"))
    max_displacement = _to_morph_animation(bpy, obj, sampled, shot)
    scene.frame_set(0)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.gltf(filepath=str(destination / "model.glb"), export_format="GLB",
                              use_selection=True, export_materials="EXPORT", export_animations=True,
                              export_extras=True, export_cameras=False, export_lights=False,
                              export_apply=False, export_morph=True)
    glb = audit_glb(destination / "model.glb")
    morph_audit = _audit_morph_animation(destination / "model.glb", shot)
    spec = asdict(shot)
    (destination / "deformation-plan.json").write_text(json.dumps({"schema_version": 1, **spec}, indent=2) + "\n")
    sizes = {name: (destination / name).stat().st_size for name in ("source.blend", "model.glb", "deformation-plan.json")}
    manifest = {
        "schema_version": 1, "asset_id": shot.asset_id, "units": "meters",
        "generator": "blender-deformation-morph-bake", "status": "deformation-bake-needs-visual-qa",
        "blender_version": bpy.app.version_string,
        "provenance": {"source_blend": Path(source).name,
                       "simulation": "forward-evaluated Blender modifier deformation"},
        "files": {"source": "source.blend", "model": "model.glb", "deformation_plan": "deformation-plan.json"},
        "output_bytes": sizes, "glb_audit": glb,
        "named_parts": {obj.get("kumiki_part"): 1},
        "deformation": {"action_id": shot.action_id, "duration_frames": shot.duration_frames,
                        "sampled_frames": list(shot.frames), "morph_targets": len(shot.frames) - 1,
                        "vertex_count": len(obj.data.vertices),
                        "max_displacement_m": round(max_displacement, 6),
                        "interpolation": "linear-neighbor-samples", "deformation_solver": "Blender evaluated scene",
                        **morph_audit},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
