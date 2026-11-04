"""Render one shot from a set, a cast and a Kumiki per-frame bake.

Frames are written to a work directory so a retry resumes. The final bundle is
committed by rename, so a failed or cancelled build never looks complete.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

from .actor_plan import load_actor
from .blender_actor import ActorRig, apply_state, build_actor
from .blender_set import build_set
from .shot_plan import (
    RENDER_PROFILES, ShotPlan, ShotPlanError, to_blender_object_quaternion, to_blender_position, to_blender_quaternion,
)

CONTACT_TOLERANCE_M = .002


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan_fingerprint(plan: ShotPlan, spec_path: Path) -> str:
    digest = hashlib.sha256()
    for path in (spec_path, plan.set_path, *(actor.profile for actor in plan.actors)):
        digest.update(path.read_bytes())
    for actor in plan.actors:
        digest.update((actor.pack / "character.json").read_bytes())
    digest.update(json.dumps(plan.bake.__dict__, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def _camera(bpy: Any, plan: ShotPlan) -> Any:
    data = bpy.data.cameras.new("shot-camera")
    data.sensor_fit = "VERTICAL"
    data.sensor_height = 24
    data.dof.use_dof = True
    data.dof.aperture_fstop = plan.look.get("aperture_fstop", 4.0)
    obj = bpy.data.objects.new("shot-camera", data)
    bpy.context.collection.objects.link(obj)
    obj.rotation_mode = "QUATERNION"
    bpy.context.scene.camera = obj
    return obj


def _key_camera(camera: Any, plan: ShotPlan) -> None:
    track = plan.bake.camera
    for index in range(plan.bake.duration_frames):
        frame = index + 1
        camera.location = to_blender_position(track["position"][index])
        camera.rotation_quaternion = to_blender_quaternion(track["orientation"][index])
        camera.data.lens = 12 / math.tan(math.radians(track["fov"][index]) / 2)
        camera.data.dof.focus_distance = track["focus_distance"][index]
        camera.keyframe_insert("location", frame=frame)
        camera.keyframe_insert("rotation_quaternion", frame=frame)
        camera.data.keyframe_insert("lens", frame=frame)
        camera.data.dof.keyframe_insert("focus_distance", frame=frame)


def _key_objects(objects: dict[str, Any], plan: ShotPlan) -> None:
    for name, channels in plan.bake.objects.items():
        if name not in objects:
            raise ShotPlanError(f"Bake moves unknown set object {name}")
        obj = objects[name]
        obj.rotation_mode = "QUATERNION"
        bend = obj.modifiers.get("bend")
        if "bend" in channels and bend is None:
            raise ShotPlanError(f"Object {name} cannot bend")
        for index in range(plan.bake.duration_frames):
            frame = index + 1
            obj.location = to_blender_position(channels["position"][index])
            obj.rotation_quaternion = to_blender_object_quaternion(channels["orientation"][index])
            obj.keyframe_insert("location", frame=frame)
            obj.keyframe_insert("rotation_quaternion", frame=frame)
            if "bend" in channels:
                bend.angle = channels["bend"][index]
                bend.keyframe_insert("angle", frame=frame)


def _key_lights(objects: dict[str, Any], plan: ShotPlan) -> None:
    for name, levels in plan.bake.lights.items():
        if name not in objects or objects[name].type != "LIGHT":
            raise ShotPlanError(f"Bake drives unknown light {name}")
        data = objects[name].data
        base = data.energy
        for index, level in enumerate(levels):
            data.energy = base * level
            data.keyframe_insert("energy", frame=index + 1)


def _key_actor(rig: ActorRig, channels: dict[str, Any], frames: int) -> None:
    root = rig.root
    root.rotation_mode = "XYZ"
    for index in range(frames):
        frame = index + 1
        root.location = to_blender_position(channels["root"]["position"][index])
        root.rotation_euler = (0, 0, channels["root"]["yaw"][index])
        root.keyframe_insert("location", frame=frame)
        root.keyframe_insert("rotation_euler", frame=frame)
        apply_state(rig, channels["parts"][index], frame)
    for effector, track in (channels.get("effectors") or {}).items():
        if effector not in rig.targets:
            raise ShotPlanError(f"Actor {rig.plan.actor_id} has no effector {effector}")
        target = rig.targets[effector]
        for index in range(frames):
            frame = index + 1
            target.location = to_blender_position(track["position"][index])
            target.keyframe_insert("location", frame=frame)
            for constraint in rig.constraints[effector]:
                constraint.influence = track["influence"][index]
                constraint.keyframe_insert("influence", frame=frame)


def _configure(bpy: Any, plan: ShotPlan) -> None:
    scene = bpy.context.scene
    profile = RENDER_PROFILES[plan.profile]
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = profile["samples"]
    scene.eevee.use_raytracing = True
    scene.eevee.ray_tracing_options.resolution_scale = profile["resolution_scale"]
    scene.render.resolution_x, scene.render.resolution_y = plan.width, plan.height
    scene.render.resolution_percentage = 100
    scene.render.fps, scene.render.fps_base = plan.bake.fps_num, plan.bake.fps_den
    scene.frame_start, scene.frame_end = 1, plan.bake.duration_frames
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = plan.look.get("view_transform", "AgX")
    scene.view_settings.look = plan.look.get("look", "AgX - Medium High Contrast")
    scene.view_settings.exposure = plan.look.get("exposure", 0)
    shutter = plan.look.get("motion_blur_shutter", 0)
    scene.render.use_motion_blur = shutter > 0
    if shutter > 0:
        scene.render.motion_blur_shutter = shutter


def _contact_error(bpy: Any, rig: ActorRig, effector: str) -> float:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    target = rig.targets[effector].evaluated_get(depsgraph).matrix_world.translation
    best = math.inf
    for armature, bone in rig.tips[effector]:
        evaluated = armature.evaluated_get(depsgraph)
        tail = evaluated.matrix_world @ evaluated.pose.bones[bone].tail
        best = min(best, (tail - target).length)
    return best


def _encode(work: Path, plan: ShotPlan, output: Path) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", f"{plan.bake.fps_num}/{plan.bake.fps_den}",
                    "-start_number", "1", "-i", str(work / "frame-%05d.png"), "-c:v", "libx264", "-preset", "slow",
                    "-crf", "14", "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709",
                    "-color_trc", "bt709", "-movflags", "+faststart", "-an", str(output)], check=True, timeout=1800)


def build_shot(bpy: Any, plan: ShotPlan, spec_path: Path, output: Path, *, save_blend: bool = True) -> dict[str, Any]:
    if output.exists():
        raise ShotPlanError(f"Output already exists: {output}")
    fingerprint = plan_fingerprint(plan, spec_path)
    work = output.parent / f".{output.name}.work"
    stamp = work / "fingerprint.txt"
    if work.exists() and (not stamp.exists() or stamp.read_text() != fingerprint):
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    stamp.write_text(fingerprint)
    bpy.ops.wm.read_homefile(use_factory_startup=True, use_empty=True)
    started = time.monotonic()
    objects = build_set(bpy, plan.set_path)
    rigs = {}
    for actor in plan.actors:
        rigs[actor.id] = build_actor(bpy, load_actor(actor.pack, actor.profile))
    camera = _camera(bpy, plan)
    _key_camera(camera, plan)
    _key_objects(objects, plan)
    _key_lights(objects, plan)
    for actor_id, channels in plan.bake.actors.items():
        if channels:
            _key_actor(rigs[actor_id], channels, plan.bake.duration_frames)
    _configure(bpy, plan)
    scene = bpy.context.scene
    # Measure every contact before rendering, so a miss costs seconds, not a full render.
    errors: dict[str, float] = {}
    for index in range(plan.bake.duration_frames):
        scene.frame_set(index + 1)
        for actor_id, channels in plan.bake.actors.items():
            for effector, track in (channels.get("effectors") or {}).items():
                if track["influence"][index] > .99:
                    key = f"{actor_id}.{effector}"
                    errors[key] = max(errors.get(key, 0), _contact_error(bpy, rigs[actor_id], effector))
    failures = {key: value for key, value in errors.items() if value > CONTACT_TOLERANCE_M}
    if failures:
        raise ShotPlanError(f"Contact missed its target: {failures}")
    rendered = 0
    for index in range(plan.bake.duration_frames):
        frame = index + 1
        path = work / f"frame-{frame:05d}.png"
        if not path.exists():
            scene.frame_set(frame)
            temporary = path.with_suffix(".tmp.png")
            scene.render.filepath = str(temporary)
            bpy.ops.render.render(write_still=True)
            temporary.rename(path)
            rendered += 1
        print(json.dumps({"event": "frame", "done": frame, "total": plan.bake.duration_frames}), flush=True)
    render_seconds = time.monotonic() - started
    bundle = output.parent / f".{output.name}.bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir()
    _encode(work, plan, bundle / "shot.mp4")
    if save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(bundle / "shot.blend"), compress=True)
    manifest = {
        "schema_version": 1, "shot_id": plan.shot_id, "fingerprint": fingerprint,
        "blender": bpy.app.version_string, "frames": plan.bake.duration_frames,
        "fps": {"num": plan.bake.fps_num, "den": plan.bake.fps_den},
        "width": plan.width, "height": plan.height, "profile": plan.profile,
        "frames_rendered": rendered, "render_seconds": round(render_seconds, 2),
        "contact_error_m": {key: round(value, 6) for key, value in errors.items()},
        "video_sha256": _sha256(bundle / "shot.mp4"),
    }
    (bundle / "shot-manifest.json").write_text(json.dumps(manifest, indent=2))
    bundle.rename(output)
    shutil.rmtree(work)
    return manifest
