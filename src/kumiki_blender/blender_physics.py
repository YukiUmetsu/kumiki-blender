"""Generic Blender rigid-body shot builder for agent-authored PhysicsShot JSON.

Runs only inside Blender with a saved .blend already loaded. Writes a fresh
motion bundle; the original source is never overwritten. The authored scene's
meshes and PBR materials are reused, not reconstructed procedurally.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import re
from typing import Any

from .glb_audit import audit_glb
from .glb_motion import write_glb_motion
from .physics_plan import PhysicsPlanError, PhysicsShot


_ROLE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,100}\Z")


def _tag(obj: Any) -> str | None:
    value = obj.get("kumiki_part")
    if value is None:
        return None
    if not isinstance(value, str) or not _ROLE.fullmatch(value):
        raise PhysicsPlanError(f"Blender object {obj.name!r} has an invalid kumiki_part tag")
    return value


def _model_objects(scene: Any) -> list[Any]:
    tagged = [obj for obj in scene.objects if _tag(obj) is not None and obj.type in {"MESH", "EMPTY"}]
    keep = set(tagged)
    # Include untagged descendants of tagged controls, preserving the existing
    # model hierarchy while excluding studio-only lights, cameras and ground.
    for obj in scene.objects:
        parent = obj.parent
        while parent is not None:
            if _tag(parent) is not None and obj.type in {"MESH", "EMPTY"}:
                keep.add(obj)
                break
            parent = parent.parent
    return [obj for obj in scene.objects if obj in keep]


def _resolve_bodies(scene: Any, shot: PhysicsShot) -> list[tuple[Any, Any]]:
    by_name = {obj.name: obj for obj in scene.objects}
    selected: dict[Any, str] = {}
    assignments = []
    for body in shot.bodies:
        kind, label = body.target.split(":", 1)
        found = ([by_name[label]] if kind == "name" and label in by_name
                 else [obj for obj in scene.objects if _tag(obj) == label] if kind == "role" else [])
        if not found:
            raise PhysicsPlanError(f"Physics target {body.target!r} matches no Blender objects")
        for obj in found:
            if obj.type != "MESH" or getattr(obj, "library", None) is not None:
                raise PhysicsPlanError(f"Rigid body {obj.name!r} must be an editable mesh")
            if obj in selected:
                raise PhysicsPlanError(f"Blender object {obj.name!r} selected by both {selected[obj]!r} and {body.target!r}")
            selected[obj] = body.target
            assignments.append((obj, body))
    return assignments


def _add_ground_collider(bpy: Any, shot: PhysicsShot) -> None:
    """Create a simulation-only floor, not an exported part or replacement mesh."""
    if shot.ground is None:
        return
    if bpy.data.objects.get("Kumiki_Physics_Ground") is not None:
        raise PhysicsPlanError("Source scene already defines Kumiki_Physics_Ground")
    ground = shot.ground
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, ground.height_m - ground.thickness_m/2))
    obj = bpy.context.object
    obj.name = "Kumiki_Physics_Ground"
    obj.scale = (ground.half_extent_m, ground.half_extent_m, ground.thickness_m/2)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj["kumiki_physics_helper"] = "collision-only-ground"
    obj.hide_render = True
    bpy.ops.rigidbody.object_add(type="PASSIVE")
    rigid = obj.rigid_body
    rigid.collision_shape = "BOX"
    rigid.friction = ground.friction
    rigid.restitution = ground.restitution
    rigid.use_margin = True
    rigid.collision_margin = min(.001, ground.thickness_m/10)


def _bake_evaluated_transforms(bpy: Any, scene: Any, objects: list[Any]) -> None:
    """Bake Bullet's evaluated world transforms without UI-only Blender ops.

    bpy.ops.rigidbody.bake_to_keyframes internally invokes a keying-set UI
    operator that fails in Blender 4.5 headless. Evaluated depsgraph sampling
    is reproducible in background mode and retains actual rigid-body contacts.
    """
    if not objects:
        raise PhysicsPlanError("Cannot bake a physics shot without active bodies")
    samples: dict[Any, list[tuple[int, Any, Any | None]]] = {obj: [] for obj in objects}
    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        graph = bpy.context.evaluated_depsgraph_get()
        for obj in objects:
            evaluated = obj.evaluated_get(graph)
            parent = obj.parent.evaluated_get(graph).matrix_world.copy() if obj.parent else None
            samples[obj].append((frame, evaluated.matrix_world.copy(), parent))

    # Remove simulation ownership *after* sampling. Keyframe insertion uses the
    # raw RNA API rather than a VIEW_3D/context-dependent keying-set operator.
    for obj in objects:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.rigidbody.object_remove()
        obj.rotation_mode = "QUATERNION"
        for frame, world_matrix, parent_world in samples[obj]:
            local = (obj.matrix_parent_inverse.inverted() @ parent_world.inverted() @ world_matrix
                     if parent_world is not None else world_matrix)
            location, rotation, scale = local.decompose()
            obj.location = location
            obj.rotation_quaternion = rotation
            obj.scale = scale
            obj.keyframe_insert(data_path="location", frame=frame, group="Kumiki Physics")
            obj.keyframe_insert(data_path="rotation_quaternion", frame=frame, group="Kumiki Physics")
            # Rigid bodies translate and rotate; their mesh scale is static.
            # Baking the constant scale emits glTF STEP channels alongside
            # LINEAR motion, which the PM2 track contract cannot mix.
    scene.frame_set(scene.frame_start)


def build_physics_shot(bpy: Any, shot: PhysicsShot, output: str | Path) -> dict[str, Any]:
    """Configure/bake real Blender rigid bodies and create a PM2-ready GLB bundle."""
    destination = Path(output).resolve()
    if destination.exists() or destination.is_symlink():
        raise PhysicsPlanError(f"Output already exists: {destination}")
    if not bpy.app.background:
        raise PhysicsPlanError("Physics baking must run in a separate Blender --background process")
    source = getattr(bpy.data, "filepath", "")
    if not source or not Path(source).is_file():
        raise PhysicsPlanError("Load a saved .blend file with blender --background source.blend before baking")
    scene = bpy.context.scene
    if scene is None:
        raise PhysicsPlanError("Blender has no active source scene")
    assignments = _resolve_bodies(scene, shot)
    for obj, _ in assignments:
        if _tag(obj) is None and not _ROLE.fullmatch(obj.name):
            raise PhysicsPlanError(f"Untagged object {obj.name!r} needs a safe name or kumiki_part role")
    if not assignments:
        raise PhysicsPlanError("Physics shot resolved no rigid bodies")

    # glTF stores Blender frame N at N/fps. Start at frame zero so the
    # exported GLB and PM2 both span exactly duration_frames, not N+1.
    scene.frame_start = 0
    scene.frame_end = shot.duration_frames
    scene.render.fps = 30
    scene.gravity = shot.gravity_mps2
    for obj, body in assignments:
        if _tag(obj) is None:
            obj["kumiki_part"] = obj.name
        if any(body.offset_m):
            obj.location = tuple(obj.location[i] + body.offset_m[i] for i in range(3))
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        if obj.rigid_body is None:
            bpy.ops.rigidbody.object_add(type=body.kind.upper())
        rigid = obj.rigid_body
        rigid.type = body.kind.upper()
        rigid.collision_shape = body.shape
        rigid.mass = body.mass_kg
        rigid.friction = body.friction
        rigid.restitution = body.restitution
        rigid.use_margin = body.margin_m > 0
        rigid.collision_margin = body.margin_m
    _add_ground_collider(bpy, shot)
    if scene.rigidbody_world is None:
        bpy.ops.rigidbody.world_add()
    world = scene.rigidbody_world
    if world is None:
        raise PhysicsPlanError("Blender failed to create a rigid body world")
    world.substeps_per_frame = shot.substeps
    world.solver_iterations = shot.solver_iterations
    world.point_cache.frame_start = 1
    world.point_cache.frame_end = scene.frame_end
    scene.frame_set(0)
    bpy.context.view_layer.update()

    destination.mkdir(parents=True, exist_ok=False)
    (destination / "physics-plan.json").write_text(
        json.dumps({"schema_version": 1, **asdict(shot)}, indent=2) + "\n", encoding="utf-8")
    editable_source = destination / "source.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(editable_source), copy=True)

    active = [obj for obj, body in assignments if body.kind == "active"]
    if (shot.duration_frames + 1) * len(active) > 120_000:
        raise PhysicsPlanError("Physics bake exceeds the portable PM2 120,000-key budget")
    _bake_evaluated_transforms(bpy, scene, active)

    portable = _model_objects(scene)
    if not portable or not any(obj.type == "MESH" for obj in portable):
        raise PhysicsPlanError("No semantic meshes available for portable glTF export")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in portable:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = portable[0]
    model_path = destination / "model.glb"
    bpy.ops.export_scene.gltf(
        filepath=str(model_path), export_format="GLB", use_selection=True,
        export_materials="EXPORT", export_animations=True, export_extras=True,
        export_cameras=False, export_lights=False, export_apply=False,
    )
    audit = audit_glb(model_path)
    # Blender can export a distinct action for every selected rigid body. PM2
    # can combine them only when their animated channels do not overlap.
    motion = write_glb_motion(model_path, destination / "baked-actions.json",
                              action_id=shot.action_id, merge_clips=True)
    role_counts = Counter(_tag(obj) for obj in portable if _tag(obj) is not None)
    if sum(role_counts.values()) != audit["semantic_nodes"]:
        raise PhysicsPlanError("Portable glTF semantic role count differs from the source .blend")

    files = ("source.blend", "model.glb", "baked-actions.json", "physics-plan.json")
    manifest = {
        "schema_version": 1, "asset_id": shot.asset_id, "units": "meters",
        "generator": "blender-physics-shot-bake", "status": "physics-bake-needs-visual-qa",
        "blender_version": getattr(bpy.app, "version_string", "unknown"),
        "provenance": {"source_blend": Path(source).name, "simulation": "Blender Rigid Body"},
        "files": {"source": "source.blend", "model": "model.glb", "motion": "baked-actions.json",
                  "physics_plan": "physics-plan.json"},
        "output_bytes": {name: (destination / name).stat().st_size for name in files},
        "glb_audit": audit, "named_parts": dict(role_counts),
        "physics": {"action_id": shot.action_id, "duration_frames": shot.duration_frames,
                    "substeps": shot.substeps, "solver_iterations": shot.solver_iterations,
                    "has_collision_ground": shot.ground is not None,
                    "animated_tracks": motion["tracks"], "contact_cues": motion["contacts"]},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"output": str(destination), "asset_id": shot.asset_id,
            "animated_tracks": motion["tracks"], "duration_frames": shot.duration_frames,
            "requires_visual_qa": True}
