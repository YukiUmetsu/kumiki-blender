"""Blender-only helpers: turn reviewed joint drivers into sampled morph keyframes.

Samples drivers while the source IK/FK controls are active, checks sampled
weights after the deform-bone bake, and then removes the drivers and writes
ordinary Blender shape-key actions that the glTF exporter can serialize.
"""
from __future__ import annotations

import math
import re

import bpy

_CORRECTIVE_PATH = re.compile(r'^key_blocks\["(CORR_[A-Za-z0-9_.:-]{1,95})"\]\.value$')


def collect_corrective_drivers(bound, rig, enabled: bool, *, allow_facial_actions: bool = False):
    """Fail closed on unsupported drivers or facial Action that was not opted in."""
    tracked = {}
    for obj in bound:
        keys = obj.data.shape_keys
        if keys is None or keys.animation_data is None:
            continue
        animation = keys.animation_data
        if animation.nla_tracks:
            raise ValueError(f"{obj.name}: shape-key NLA cannot be merged into a single clip")
        if animation.action and not allow_facial_actions:
            raise ValueError(f"{obj.name}: facial shape-key Action requires --bake-visemes")
        if not enabled:
            if animation.drivers:
                raise ValueError(
                    f"{obj.name}: joint-driven shapes require --bake-correctives; "
                    "skeleton-only GLB would discard the sculpt correction")
            continue
        for curve in animation.drivers:
            match = _CORRECTIVE_PATH.fullmatch(curve.data_path)
            if match is None or curve.array_index != 0 or not curve.is_valid:
                raise ValueError(f"{obj.name}: invalid/unreviewed shape-key driver {curve.data_path}")
            name = match.group(1)
            if name not in keys.key_blocks:
                raise ValueError(f"{obj.name}: joint correction has no sculpt key {name}")
            driver = curve.driver
            if driver.type != "SCRIPTED" or len(driver.variables) != 1:
                raise ValueError(f"{obj.name}/{name}: expected a single reviewed rotation driver")
            var = driver.variables[0]
            if var.type != "TRANSFORMS" or len(var.targets) != 1:
                raise ValueError(f"{obj.name}/{name}: expected a bone transform driver")
            target = var.targets[0]
            bone = rig.data.bones.get(target.bone_target)
            if (target.id != rig or bone is None or not bone.use_deform or
                    target.transform_space != "LOCAL_SPACE" or
                    target.transform_type not in ("ROT_X", "ROT_Y", "ROT_Z")):
                raise ValueError(f"{obj.name}/{name}: only local deform-bone rotation is supported")
            tracked.setdefault(obj.name, {})[name] = keys.key_blocks[name]
        if tracked.get(obj.name):
            for key in list(keys.key_blocks)[1:]:
                if (key.name not in tracked[obj.name] and
                        not (allow_facial_actions and key.name in {
                            f"viseme_{code}" for code in "ABCDEFGH"}) and
                        abs(key.value) > 1e-6):
                    raise ValueError(
                        f"{obj.name}/{key.name}: non-corrective face shape must be neutral")
    if enabled and not tracked:
        raise ValueError("Requested corrective morph baking but no CORR_ joint drivers exist")
    return tracked


def sample_correctives(tracked, frames) -> dict:
    """Read actual evaluated driver values at each integer source frame."""
    result = {obj: {shape: [] for shape in shapes}
              for obj, shapes in tracked.items()}
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        for obj_name, shapes in tracked.items():
            for shape_name, key in shapes.items():
                value = float(key.value)
                if not math.isfinite(value) or value < -1e-5 or value > 1 + 1e-5:
                    raise ValueError(
                        f"{obj_name}/{shape_name}: out-of-range corrective {value} at {frame}")
                result[obj_name][shape_name].append(
                    (frame, max(0.0, min(1.0, value))))
    return result


def compare_samples(before, after, *, tolerance=.02) -> float:
    maximum = 0.0
    for obj_name, shapes in before.items():
        for shape_name, earlier in shapes.items():
            current = after[obj_name][shape_name]
            if len(earlier) != len(current):
                raise ValueError("Corrective sample count changed during animation bake")
            for (frame, first), (new_frame, value) in zip(earlier, current):
                if frame != new_frame:
                    raise ValueError("Corrective frame identity changed during animation bake")
                maximum = max(maximum, abs(first - value))
    if maximum > tolerance + 1e-8:
        raise ValueError(
            f"Corrective morph drift {maximum:.6f} exceeds {tolerance:.6f}; "
            "visual bone bake changed the authored joint correction")
    return round(maximum, 7)


def keyframe_correctives(tracked, samples, frames, clip):
    """Remove Blender-only drivers and create ordinary keyed morph animation."""
    for obj_name, shapes in tracked.items():
        keys = bpy.data.objects[obj_name].data.shape_keys
        for key in shapes.values():
            if not key.driver_remove("value"):
                raise RuntimeError(f"{obj_name}/{key.name}: could not remove joint driver")
        for shape_name, key in shapes.items():
            for frame, value in samples[obj_name][shape_name]:
                key.value = value
                if not key.keyframe_insert(
                        data_path="value", frame=frame, group="KumikiJointCorrectives"):
                    raise RuntimeError(
                        f"{obj_name}/{shape_name}: Blender refused corrective frame {frame}")
        if keys.animation_data is None or keys.animation_data.action is None:
            raise RuntimeError(f"{obj_name}: no shape-key animation Action was created")
        if keys.animation_data.drivers:
            raise RuntimeError(f"{obj_name}: Blender-only drivers remain after sampled bake")
        keys.animation_data.action.name = f"{clip}_joint_correctives_{obj_name}"
    bpy.context.scene.frame_set(frames[0])
