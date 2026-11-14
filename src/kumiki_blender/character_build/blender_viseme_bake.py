"""Blender-only authored facial Action validation for unified GLB character clips.

The worker does not synthesize mouth geometry or infer audio timings. It
preserves genuine, user-authored viseme_A..H shape-key animation and verifies
that the final GLB contains the source evaluated weights alongside skeleton
and optional joint-corrective morphs.
"""
from __future__ import annotations

import math
import re

import bpy

_VISEME_NAME = re.compile(r"^viseme_[A-H]$")


def collect_viseme_actions(bound, enabled: bool):
    """Reject unapproved facial Actions, unsupported tracks, empty mouth shapes."""
    tracked = {}
    for obj in bound:
        keys = obj.data.shape_keys
        if keys is None or keys.animation_data is None:
            continue
        anim = keys.animation_data
        if anim.nla_tracks:
            raise ValueError(f"{obj.name}: facial NLA tracks are not supported")
        if anim.action is None:
            continue
        if not enabled:
            raise ValueError(f"{obj.name}: facial Action requires --bake-visemes")
        if not keys.use_relative or obj.data.users != 1 or obj.library or obj.data.library:
            raise ValueError(f"{obj.name}: facial mesh needs unique editable relative shape keys")
        shapes = {key.name: key for key in list(keys.key_blocks)[1:]
                  if _VISEME_NAME.fullmatch(key.name)}
        if not shapes:
            raise ValueError(f"{obj.name}: facial Action has no named viseme_A..H keys")
        basis = keys.key_blocks[0]
        for name, key in shapes.items():
            moving = sum(
                1 for original, sculpted in zip(basis.data, key.data)
                if (original.co - sculpted.co).length > 1e-6)
            if moving == 0:
                raise ValueError(f"{obj.name}/{name}: viseme contains no authored geometry")
        tracked[obj.name] = shapes
    if enabled and not tracked:
        raise ValueError("Requested --bake-visemes but no rig-bound facial shape-key Action exists")
    return tracked


def merge_morph_tracks(correctives, faces):
    """Merge reviewed morph key sources without replacing a duplicate name."""
    merged = {obj: dict(shapes) for obj, shapes in correctives.items()}
    for obj_name, shapes in faces.items():
        target = merged.setdefault(obj_name, {})
        duplicate = set(target).intersection(shapes)
        if duplicate:
            raise ValueError(f"{obj_name}: duplicate morph sources {sorted(duplicate)}")
        target.update(shapes)
    return merged


def verify_facial_samples(samples, faces):
    """Require visibly varying viseme weights, no excessive simultaneous mouth mix."""
    for obj_name, shapes in faces.items():
        any_movement = False
        for name in shapes:
            values = [weight for _, weight in samples[obj_name][name]]
            if max(values) > .05 and max(values) - min(values) > .05:
                any_movement = True
        if not any_movement:
            raise ValueError(f"{obj_name}: facial Action does not animate any viseme in the requested clip")
        count = len(samples[obj_name][next(iter(shapes))])
        for frame_idx in range(count):
            blend = sum(samples[obj_name][name][frame_idx][1] for name in shapes)
            if blend > 1.05:
                raise ValueError(
                    f"{obj_name}: multiple facial visemes overdrive the mouth at sample {frame_idx}")


def guard_untracked_shape_weights(bound, tracked, frames):
    """No implicit export of unsupported blinking/face morphs as static shapes."""
    untracked = {
        obj.name: [key for key in list(obj.data.shape_keys.key_blocks)[1:]
                   if key.name not in tracked.get(obj.name, {})]
        for obj in bound if obj.data.shape_keys is not None
    }
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        for obj_name, shapes in untracked.items():
            for shape in shapes:
                weight = float(shape.value)
                if not math.isfinite(weight) or abs(weight) > 1e-5:
                    raise ValueError(
                        f"{obj_name}/{shape.name}: untracked facial shape is active at frame {frame}")
