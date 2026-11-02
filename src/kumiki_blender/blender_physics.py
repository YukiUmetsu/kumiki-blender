"""Generic Blender rigid-body shot builder for agent-authored PhysicsShot JSON.

Runs only inside Blender with a saved .blend already loaded. Writes a fresh
motion bundle; the original source is never overwritten. The authored scene's
meshes and PBR materials are reused, not reconstructed procedurally.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import re
from typing import Any

from .glb_audit import audit_glb
from .glb_motion import canonical_ids_by_node_name, write_glb_motion
from .impact_measurement import GroundSample, ground_impact_frames
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


def _add_constraints(bpy: Any, shot: PhysicsShot, assignments: list[tuple[Any, Any]]) -> None:
    """Build Blender-native helpers anchored in world space, never exported."""
    if not shot.constraints:
        return
    from mathutils import Vector

    def find(selector: str) -> Any:
        kind, label = selector.split(":", 1)
        matches = [obj for obj, _ in assignments if (obj.name == label if kind == "name" else _tag(obj) == label)]
        if len(matches) != 1:
            raise PhysicsPlanError(f"Constraint target {selector!r} must select exactly one configured rigid body")
        return matches[0]

    for rule in shot.constraints:
        a, b = find(rule.body_a), find(rule.body_b)
        if a == b or (a.rigid_body.type != "ACTIVE" and b.rigid_body.type != "ACTIVE"):
            raise PhysicsPlanError(f"Constraint {rule.id!r} needs two distinct bodies, at least one active")
        name = f"Kumiki_Constraint_{rule.id}"
        if bpy.data.objects.get(name) is not None:
            raise PhysicsPlanError(f"Source scene already defines {name!r}")
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.object.empty_add(type="ARROWS", location=rule.pivot_m)
        helper = bpy.context.object
        helper.name = name
        helper["kumiki_physics_helper"] = "collision-only-constraint"
        helper.hide_render = True
        axis = Vector(rule.axis)
        # Hinge uses the helper's Z axis; slider moves on its X axis.
        track_axis, up_axis = ("Z", "Y") if rule.kind == "hinge" else ("X", "Z")
        helper.rotation_euler = axis.to_track_quat(track_axis, up_axis).to_euler()
        bpy.ops.rigidbody.constraint_add(type=rule.kind.upper())
        native = helper.rigid_body_constraint
        if native is None:
            raise PhysicsPlanError(f"Blender failed to create constraint {rule.id!r}")
        native.object1, native.object2 = a, b
        native.disable_collisions = rule.disable_collisions
        if rule.limits is not None and rule.kind == "hinge":
            native.use_limit_ang_z = True
            native.limit_ang_z_lower = math.radians(rule.limits[0])
            native.limit_ang_z_upper = math.radians(rule.limits[1])
        if rule.limits is not None and rule.kind == "slider":
            native.use_limit_lin_x = True
            native.limit_lin_x_lower, native.limit_lin_x_upper = rule.limits


def _bake_evaluated_transforms(bpy: Any, scene: Any, objects: list[Any],
                               shot: PhysicsShot, assignments: list[tuple[Any, Any]]) -> dict[Any, list[dict[str, Any]]]:
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

    measured: dict[Any, list[dict[str, Any]]] = {}
    if shot.ground is not None:
        from mathutils import Vector
        body_by_object = {obj: body for obj, body in assignments}
        for obj in objects:
            curve = []
            for frame, matrix, _ in samples[obj]:
                lowest = min((matrix @ Vector(corner)).z for corner in obj.bound_box)
                center = matrix.translation
                curve.append(GroundSample(frame, lowest, center.z, center.x, center.y))
            hits = ground_impact_frames(curve, height_m=shot.ground.height_m,
                                       half_extent_m=shot.ground.half_extent_m)
            measured[obj] = [dict(frame=frame, speedMps=speed,
                                  **({"material": body_by_object[obj].impact_material}
                                     if body_by_object[obj].impact_material else {}))
                             for frame, speed in hits]

    # Blender 4.5 Bullet constraint helpers retain pointers to body objects.
    # Detach/remove them from the *working copy* before stripping rigid-body
    # components for keyframe export. The pre-bake source.blend was already
    # copied with fully editable native constraints intact. Leaving dangling
    # helpers can segfault Blender on process exit (observed in CI).
    for helper in list(scene.objects):
        native = getattr(helper, "rigid_body_constraint", None)
        if native is None or (native.object1 not in objects and native.object2 not in objects):
            continue
        bpy.ops.object.select_all(action="DESELECT")
        helper.select_set(True)
        bpy.context.view_layer.objects.active = helper
        bpy.ops.rigidbody.constraint_remove()
        if helper.get("kumiki_physics_helper") == "collision-only-constraint":
            bpy.data.objects.remove(helper, do_unlink=True)

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
    return measured


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
    _add_constraints(bpy, shot, assignments)
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
    observations = _bake_evaluated_transforms(bpy, scene, active, shot, assignments)

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
    cues = []
    if any(observations.values()):
        ids = canonical_ids_by_node_name(model_path)
        for obj, events in observations.items():
            if events and obj.name not in ids:
                raise PhysicsPlanError(f"Exported GLB lacks semantic node for measured contact {obj.name!r}")
            cues.extend({"target": ids[obj.name], **event} for event in events)
        cues.sort(key=lambda event: (event["frame"], event["target"]))
    # Blender can export a distinct action for every selected rigid body. PM2
    # can combine them only when their animated channels do not overlap.
    motion = write_glb_motion(model_path, destination / "baked-actions.json",
                              action_id=shot.action_id, merge_clips=True, contacts=cues,
                              resample_mixed=True)
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
                    "constraints": len(shot.constraints),
                    "contact_evidence": "evaluated-geometry-ground-proximity" if cues else "none",
                    "animated_tracks": motion["tracks"], "contact_cues": motion["contacts"]},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"output": str(destination), "asset_id": shot.asset_id,
            "animated_tracks": motion["tracks"], "duration_frames": shot.duration_frames,
            "requires_visual_qa": True}
