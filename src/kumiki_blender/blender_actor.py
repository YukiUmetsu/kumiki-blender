"""Build a lit, bendable puppet in Blender from an ActorPlan.

Each rig layer becomes an inflated mesh under a transform empty. Empties take
Kumiki character state per frame, so Blender adds no animation of its own. Parts
named by an effector also carry an armature with an IK target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from .actor_geometry import chain_weights, heightfield_mesh, inflate_mask, local_matrix, trace_chain
from .actor_plan import ActorPart, ActorPlan, Effector

_CACHE: dict[int, tuple[np.ndarray, int, int]] = {}


def image_alpha(image: Any) -> np.ndarray:
    """Alpha mask, row 0 at the top of the art (Blender stores rows bottom-up)."""
    width, height = image.size
    pixels = np.empty(width * height * 4, np.float32)
    image.pixels.foreach_get(pixels)
    return pixels.reshape(height, width, 4)[::-1, :, 3] > .5


def _downsample(mask: np.ndarray, step: int) -> np.ndarray:
    rows, columns = mask.shape
    return mask[:rows - rows % step or rows, :columns - columns % step or columns][::step, ::step]


@dataclass
class ActorRig:
    plan: ActorPlan
    root: Any
    scale: float
    origin: tuple[float, float]
    empties: dict[str, Any] = field(default_factory=dict)
    meshes: dict[tuple[str, str], Any] = field(default_factory=dict)
    targets: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, list[Any]] = field(default_factory=dict)
    tips: dict[str, list[tuple[Any, str]]] = field(default_factory=dict)


def _material(bpy: Any, name: str, image: Any, roughness: float) -> Any:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.surface_render_method = "DITHERED"
    nodes = material.node_tree
    shader = nodes.nodes["Principled BSDF"]
    texture = nodes.nodes.new("ShaderNodeTexImage")
    texture.image = image
    shader.inputs["Roughness"].default_value = roughness
    nodes.links.new(texture.outputs["Color"], shader.inputs["Base Color"])
    nodes.links.new(texture.outputs["Alpha"], shader.inputs["Alpha"])
    return material


def _part_mesh(bpy: Any, plan: ActorPlan, name: str, image: Any, mask: np.ndarray, *,
               logical: tuple[float, float], scale: float, layer: float, inflate: float,
               back: float) -> tuple[Any, np.ndarray]:
    step = plan.grid_step
    height = inflate_mask(_downsample(mask, step))
    mesh = heightfield_mesh(height, step)
    art_height, art_width = mask.shape
    sx, sy = logical[0] / art_width, logical[1] / art_height
    design = mesh.pixels * (sx, sy)
    depth = np.where(mesh.sides < 0, -1.0, back) * inflate * mesh.heights
    vertices = np.column_stack((design[:, 0] * scale, layer + depth, -design[:, 1] * scale))
    data = bpy.data.meshes.new(name)
    data.from_pydata(vertices.tolist(), [], mesh.faces.tolist())
    data.update()
    uv = mesh.pixels / (art_width, art_height)
    uv[:, 1] = 1 - uv[:, 1]
    layer_uv = data.uv_layers.new()
    loops = np.empty(len(data.loops), np.int32)
    data.loops.foreach_get("vertex_index", loops)
    layer_uv.data.foreach_set("uv", uv[loops].astype(np.float32).ravel())
    data.polygons.foreach_set("use_smooth", np.ones(len(data.polygons), bool))
    data.materials.append(_material(bpy, name, image, plan.roughness))
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    return obj, design


def _rest_bounds(plan: ActorPlan, images: dict[str, Any]) -> tuple[float, float, float, float]:
    """Design-space bounds of the rest pose, from drawn alpha, including nested parts."""
    parts = plan.by_id
    matrices: dict[str, np.ndarray] = {}

    def world(part: ActorPart) -> np.ndarray:
        if part.id not in matrices:
            rest = part.rest
            local = local_matrix(pivot=part.pivot, x=rest.get("x", 0), y=rest.get("y", 0), rotation=rest.get("rotation", 0),
                                 scale_x=rest.get("scaleX", 1), scale_y=rest.get("scaleY", 1))
            matrices[part.id] = world(parts[part.parent]) @ local if part.parent else local
        return matrices[part.id]

    xs, ys = [], []
    for part in plan.parts:
        image = images[part.asset]
        mask, width, height = _CACHE[id(image)]
        logical = _logical_size(plan, part.asset, image)
        rows, columns = np.nonzero(mask)
        if not len(rows):
            continue
        sx, sy = logical[0] / width, logical[1] / height
        corners = np.array([[columns.min() * sx, rows.min() * sy, 1], [columns.max() * sx, rows.min() * sy, 1],
                            [columns.max() * sx, rows.max() * sy, 1], [columns.min() * sx, rows.max() * sy, 1]])
        placed = (world(part) @ corners.T).T
        xs += placed[:, 0].tolist()
        ys += placed[:, 1].tolist()
    return min(xs), min(ys), max(xs), max(ys)


def _logical_size(plan: ActorPlan, key: str, image: Any) -> tuple[float, float]:
    return plan.logical_sizes.get(key) or (float(image.size[0]), float(image.size[1]))


def _add_chain(bpy: Any, rig: ActorRig, effector: Effector, part: ActorPart, obj: Any, parent: Any,
               mask: np.ndarray, design: np.ndarray, logical: tuple[float, float], layer: float) -> None:
    art_height, art_width = mask.shape
    sx, sy = logical[0] / art_width, logical[1] / art_height
    start = effector.start or part.pivot
    tip = effector.tip
    chain_art = trace_chain(mask, (start[0] / sx, start[1] / sy), joints=effector.joints,
                            tip=(tip[0] / sx, tip[1] / sy) if tip else None)
    chain = [(x * sx, y * sy) for x, y in chain_art]
    armature_data = bpy.data.armatures.new(f"{obj.name}-bones")
    armature = bpy.data.objects.new(armature_data.name, armature_data)
    bpy.context.collection.objects.link(armature)
    armature.parent = parent
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    names = []
    previous = None
    for index, (head, tail) in enumerate(zip(chain, chain[1:])):
        bone = armature_data.edit_bones.new(f"bone{index}")
        bone.head = (head[0] * rig.scale, layer, -head[1] * rig.scale)
        bone.tail = (tail[0] * rig.scale, layer, -tail[1] * rig.scale)
        bone.parent = previous
        bone.use_connect = previous is not None
        previous = bone
        names.append(bone.name)
    bpy.ops.object.mode_set(mode="OBJECT")
    weights = chain_weights(design, chain, blend=max(logical) * .06)
    for index, name in enumerate(names):
        group = obj.vertex_groups.new(name=name)
        for weight in np.unique(np.round(weights[:, index], 2)):
            if weight > 0:
                group.add(np.nonzero(np.round(weights[:, index], 2) == weight)[0].tolist(), float(weight), "REPLACE")
    modifier = obj.modifiers.new("chain", "ARMATURE")
    modifier.object = armature
    target = rig.targets.get(effector.id)
    if target is None:
        target = bpy.data.objects.new(f"target-{rig.plan.actor_id}-{effector.id}", None)
        bpy.context.collection.objects.link(target)
        rig.targets[effector.id] = target
    pose_bone = armature.pose.bones[names[-1]]
    constraint = pose_bone.constraints.new("IK")
    constraint.target = target
    constraint.chain_count = len(names)
    constraint.influence = 0
    rig.constraints.setdefault(effector.id, []).append(constraint)
    rig.tips.setdefault(effector.id, []).append((armature, names[-1]))


def build_actor(bpy: Any, plan: ActorPlan) -> ActorRig:
    images: dict[str, Any] = {}
    for key, path in plan.assets.items():
        image = bpy.data.images.load(str(path), check_existing=True)
        images[key] = image
        if id(image) not in _CACHE:
            _CACHE[id(image)] = (image_alpha(image), image.size[0], image.size[1])
    left, top, right, bottom = _rest_bounds(plan, images)
    scale = plan.height_m / (bottom - top)
    root = bpy.data.objects.new(f"actor-{plan.actor_id}", None)
    bpy.context.collection.objects.link(root)
    rig = ActorRig(plan, root, scale, ((left + right) / 2, bottom))
    ordered = sorted(plan.parts, key=lambda part: part.z)
    for part in plan.parts:
        empty = bpy.data.objects.new(f"{plan.actor_id}-{part.id}", None)
        bpy.context.collection.objects.link(empty)
        empty.rotation_mode = "XYZ"
        rig.empties[part.id] = empty
    for part in plan.parts:
        rig.empties[part.id].parent = rig.empties[part.parent] if part.parent else root
    effector_of = {(effector.part, effector.variant): effector for effector in plan.effectors}
    for rank, part in enumerate(ordered):
        stage = plan.stage.get(part.id)
        layer = stage.layer if stage and stage.layer is not None else -rank * .002
        back = stage.back if stage else .55
        for variant, key in ((None, part.asset), *part.variants.items()):
            if (part.id, key) in rig.meshes:
                continue
            image = images[key]
            mask, _, _ = _CACHE[id(image)]
            if not mask.any():
                continue
            logical = _logical_size(plan, key, image)
            inflate = stage.inflate if stage and stage.inflate is not None else plan.height_m * .05
            name = f"{plan.actor_id}-{part.id}-{key}"
            obj, design = _part_mesh(bpy, plan, name, image, mask, logical=logical, scale=scale, layer=layer,
                                     inflate=inflate, back=back)
            obj.parent = rig.empties[part.id]
            rig.meshes[(part.id, key)] = obj
            effector = effector_of.get((part.id, variant)) or (effector_of.get((part.id, None)) if key == part.asset else None)
            if effector:
                _add_chain(bpy, rig, effector, part, obj, rig.empties[part.id], mask, design, logical, layer)
    apply_state(rig, {}, None)
    root.location = (0, 0, 0)
    return rig


def describe_actor(rig: ActorRig) -> dict[str, Any]:
    """Rest geometry Kumiki needs to plan reaches. Y-up metres, relative to the actor root."""
    rig.root.users_scene[0].view_layers[0].update()
    effectors = {}
    for effector, tips in rig.tips.items():
        armature, bone = tips[0]
        pose = armature.pose.bones
        names = [item.name for item in pose]
        head = armature.matrix_world @ pose[names[0]].head
        tail = armature.matrix_world @ pose[bone].tail
        scale = armature.matrix_world.to_scale().x
        reach = sum(item.length for item in armature.data.bones) * scale
        effectors[effector] = {"rest": [tail.x, tail.z, -tail.y], "shoulder": [head.x, head.z, -head.y], "reach_m": reach}
    return {"actor_id": rig.plan.actor_id, "revision": rig.plan.revision, "height_m": rig.plan.height_m,
            "effectors": effectors, "parts": [part.id for part in rig.plan.parts]}


def _matrix_channels(rig: ActorRig, part: ActorPart, state: dict[str, float]) -> tuple[tuple[float, float, float], float, tuple[float, float, float]]:
    rest = part.rest
    x, y = state.get("x", rest.get("x", 0)), state.get("y", rest.get("y", 0))
    rotation = state.get("rotation", rest.get("rotation", 0))
    sx, sy = state.get("scaleX", rest.get("scaleX", 1)), state.get("scaleY", rest.get("scaleY", 1))
    matrix = local_matrix(pivot=part.pivot, x=x, y=y, rotation=rotation, scale_x=sx, scale_y=sy)
    tx, ty = matrix[0, 2], matrix[1, 2]
    if part.parent is None:
        tx -= rig.origin[0]
        ty -= rig.origin[1]
    return (tx * rig.scale, 0.0, -ty * rig.scale), math.radians(rotation), (sx, 1.0, sy)


def apply_state(rig: ActorRig, parts: dict[str, dict[str, Any]], frame: int | None) -> None:
    """Pose every part from one Kumiki frame state. Keyframes it when `frame` is given."""
    for part in rig.plan.parts:
        state = parts.get(part.id, {})
        location, angle, scale = _matrix_channels(rig, part, state)
        empty = rig.empties[part.id]
        empty.location = location
        empty.rotation_euler = (math.radians(state.get("pitch", 0)), angle, math.radians(state.get("yaw", 0)))
        empty.scale = scale
        if frame is not None:
            for path in ("location", "rotation_euler", "scale"):
                empty.keyframe_insert(path, frame=frame)
        shown = rig.plan.asset_of(part, state.get("variant"))
        for (owner, key), obj in rig.meshes.items():
            if owner != part.id:
                continue
            hidden = key != shown
            obj.hide_render = hidden
            obj.hide_viewport = hidden
            if frame is not None:
                obj.keyframe_insert("hide_render", frame=frame)
                obj.keyframe_insert("hide_viewport", frame=frame)
