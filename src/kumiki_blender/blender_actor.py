"""Build a lit, bendable puppet in Blender from an ActorPlan.

Each rig layer becomes an inflated mesh under a transform empty. Empties take
Kumiki character state per frame, so Blender adds no animation of its own. Parts
named by an effector also carry an armature with an IK target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .actor_geometry import chain_weights, heightfield_mesh, inflate_mask, local_matrix, part_transform, trace_chain, variant_union
from .actor_plan import ActorPart, ActorPlan, ActorPlanError, Effector

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
    meshes: dict[str, Any] = field(default_factory=dict)
    weights: dict[str, dict[str, Any]] = field(default_factory=dict)
    targets: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, list[Any]] = field(default_factory=dict)
    tips: dict[str, list[tuple[Any, str]]] = field(default_factory=dict)


def _socket(node: Any, identifier: str, outputs: bool = False) -> Any:
    for socket in (node.outputs if outputs else node.inputs):
        if socket.identifier == identifier:
            return socket
    raise KeyError(identifier)


def _part_material(bpy: Any, name: str, variants: list[tuple[str, Any]], roughness: float) -> tuple[Any, dict[str, Any]]:
    """One material for every drawing of a part. Keyframed weights choose the drawing.

    Colours are premultiplied, summed, and divided back, so a crossfade between two
    drawings blends their ink, not the junk colour under transparent pixels.
    """
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.surface_render_method = "DITHERED"
    tree = material.node_tree
    shader = tree.nodes["Principled BSDF"]
    shader.inputs["Roughness"].default_value = roughness
    weights: dict[str, Any] = {}
    colour_acc = alpha_acc = None
    for key, image in variants:
        texture = tree.nodes.new("ShaderNodeTexImage")
        texture.image = image
        weight = tree.nodes.new("ShaderNodeValue")
        weight.label = f"weight:{key}"
        weight.outputs[0].default_value = 0.0
        weights[key] = weight
        premultiplied = tree.nodes.new("ShaderNodeMix")
        premultiplied.data_type, premultiplied.blend_type = "RGBA", "MULTIPLY"
        _socket(premultiplied, "Factor_Float").default_value = 1.0
        tree.links.new(texture.outputs["Color"], _socket(premultiplied, "A_Color"))
        tree.links.new(texture.outputs["Alpha"], _socket(premultiplied, "B_Color"))
        add = tree.nodes.new("ShaderNodeMix")
        add.data_type, add.blend_type = "RGBA", "ADD"
        tree.links.new(weight.outputs[0], _socket(add, "Factor_Float"))
        if colour_acc is None:
            _socket(add, "A_Color").default_value = (0, 0, 0, 1)
        else:
            tree.links.new(colour_acc, _socket(add, "A_Color"))
        tree.links.new(_socket(premultiplied, "Result_Color", True), _socket(add, "B_Color"))
        colour_acc = _socket(add, "Result_Color", True)
        alpha = tree.nodes.new("ShaderNodeMath")
        alpha.operation = "MULTIPLY_ADD"
        tree.links.new(texture.outputs["Alpha"], alpha.inputs[0])
        tree.links.new(weight.outputs[0], alpha.inputs[1])
        if alpha_acc is None:
            alpha.inputs[2].default_value = 0.0
        else:
            tree.links.new(alpha_acc, alpha.inputs[2])
        alpha_acc = alpha.outputs[0]
    floor = tree.nodes.new("ShaderNodeMath")
    floor.operation = "MAXIMUM"
    floor.inputs[1].default_value = 1e-4
    tree.links.new(alpha_acc, floor.inputs[0])
    straight = tree.nodes.new("ShaderNodeMix")
    straight.data_type, straight.blend_type = "RGBA", "DIVIDE"
    _socket(straight, "Factor_Float").default_value = 1.0
    tree.links.new(colour_acc, _socket(straight, "A_Color"))
    tree.links.new(floor.outputs[0], _socket(straight, "B_Color"))
    tree.links.new(_socket(straight, "Result_Color", True), shader.inputs["Base Color"])
    tree.links.new(alpha_acc, shader.inputs["Alpha"])
    return material, weights


def _part_mesh(bpy: Any, plan: ActorPlan, name: str, material: Any, mask: np.ndarray, *,
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
    data.materials.append(material)
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    return obj, design


def _rest_bounds(plan: ActorPlan, images: dict[str, Any]) -> tuple[float, float, float, float, float]:
    """Design-space rest bounds from drawn alpha, plus the ground line.

    The ground comes from `ground_parts` when named, because a tail can hang below the feet.
    """
    parts = plan.by_id
    matrices: dict[str, np.ndarray] = {}

    def world(part: ActorPart) -> np.ndarray:
        if part.id not in matrices:
            rest = part.rest
            local = local_matrix(pivot=part.pivot, x=rest.get("x", 0), y=rest.get("y", 0), rotation=rest.get("rotation", 0),
                                 scale_x=rest.get("scaleX", 1), scale_y=rest.get("scaleY", 1))
            matrices[part.id] = world(parts[part.parent]) @ local if part.parent else local
        return matrices[part.id]

    xs, ys, ground = [], [], []
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
        if part.id in plan.ground_parts:
            ground += placed[:, 1].tolist()
    return min(xs), min(ys), max(xs), max(ys), max(ground or ys)


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
    left, top, right, bottom, ground = _rest_bounds(plan, images)
    scale = plan.height_m / (bottom - top)
    root = bpy.data.objects.new(f"actor-{plan.actor_id}", None)
    bpy.context.collection.objects.link(root)
    rig = ActorRig(plan, root, scale, ((left + right) / 2, ground))
    ordered = sorted(plan.parts, key=lambda part: part.z)

    def depth_scale(part: ActorPart) -> float:
        own = part.rest.get("scaleX", 1)
        return own * depth_scale(plan.by_id[part.parent]) if part.parent else own

    for part in plan.parts:
        empty = bpy.data.objects.new(f"{plan.actor_id}-{part.id}", None)
        bpy.context.collection.objects.link(empty)
        empty.rotation_mode = "QUATERNION"
        rig.empties[part.id] = empty
    for part in plan.parts:
        rig.empties[part.id].parent = rig.empties[part.parent] if part.parent else root
    effector_of = {effector.part: effector for effector in plan.effectors}
    for rank, part in enumerate(ordered):
        stage = plan.stage.get(part.id)
        layer = stage.layer if stage and stage.layer is not None else -rank * .002
        back = stage.back if stage else .55
        drawings = [(key, images[key]) for key in dict.fromkeys((part.asset, *part.variants.values()))]
        masks = {key: _CACHE[id(image)][0] for key, image in drawings}
        if not any(mask.any() for mask in masks.values()):
            continue  # A transform-only part, such as a root with a blank drawing.
        try:
            union, _ = variant_union(masks)
        except ValueError as error:
            raise ActorPlanError(f"Part {part.id}: {error}") from error
        reference = next(key for key, mask in masks.items() if mask.any())
        logical = _logical_size(plan, reference, images[reference])
        inflate = stage.inflate if stage and stage.inflate is not None else plan.height_m * .05
        # Profiles use real metres. Part empties scale depth, so divide that back out.
        parent_depth = depth_scale(part)
        name = f"{plan.actor_id}-{part.id}"
        material, weights = _part_material(bpy, name, drawings, plan.roughness)
        obj, design = _part_mesh(bpy, plan, name, material, union, logical=logical, scale=scale, layer=layer / parent_depth,
                                 inflate=inflate / parent_depth, back=back)
        obj.parent = rig.empties[part.id]
        rig.meshes[part.id] = obj
        rig.weights[part.id] = weights
        effector = effector_of.get(part.id)
        if effector:
            # The chain follows the drawing the effector names, or the default drawing.
            traced = plan.asset_of(part, effector.variant) if effector.variant else part.asset
            _add_chain(bpy, rig, effector, part, obj, rig.empties[part.id], masks[traced], design, logical, layer / parent_depth)
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
        part = next(item.part for item in rig.plan.effectors if item.id == effector)
        effectors[effector] = {"rest": [tail.x, tail.z, -tail.y], "shoulder": [head.x, head.z, -head.y],
                               "reach_m": reach, "part": part}
    return {"actor_id": rig.plan.actor_id, "revision": rig.plan.revision, "height_m": rig.plan.height_m,
            "effectors": effectors, "parts": [part.id for part in rig.plan.parts]}


def _matrix_channels(rig: ActorRig, part: ActorPart, state: dict[str, float]) -> tuple[Any, Any, tuple[float, float, float]]:
    from mathutils import Matrix, Vector

    rest = part.rest
    location, rotation, scale = part_transform(
        pivot=part.pivot, x=state.get("x", rest.get("x", 0)), y=state.get("y", rest.get("y", 0)),
        rotation=state.get("rotation", rest.get("rotation", 0)),
        scale_x=state.get("scaleX", rest.get("scaleX", 1)), scale_y=state.get("scaleY", rest.get("scaleY", 1)),
        metres_per_px=rig.scale, yaw=state.get("yaw", 0), pitch=state.get("pitch", 0))
    if part.parent is None:
        location = location - np.array((rig.origin[0] * rig.scale, 0.0, -rig.origin[1] * rig.scale))
    return Vector(location.tolist()), Matrix(rotation.tolist()).to_quaternion(), scale


def apply_state(rig: ActorRig, parts: dict[str, dict[str, Any]], frame: int | None) -> None:
    """Pose every part from one Kumiki frame state. Keyframes it when `frame` is given."""
    for part in rig.plan.parts:
        state = parts.get(part.id, {})
        location, rotation, scale = _matrix_channels(rig, part, state)
        empty = rig.empties[part.id]
        empty.location = location
        empty.rotation_quaternion = rotation
        empty.scale = scale
        if frame is not None:
            for path in ("location", "rotation_quaternion", "scale"):
                empty.keyframe_insert(path, frame=frame)
        # Kumiki's own crossfade: `variantFrom` at 1 - mix, `variant` at mix. Everything else 0.
        shown = rig.plan.asset_of(part, state.get("variant"))
        mix = state.get("variantMix")
        previous = rig.plan.asset_of(part, state.get("variantFrom")) if state.get("variantFrom") else None
        wanted = {shown: 1.0}
        if previous is not None and mix is not None and previous != shown:
            wanted = {shown: float(mix), previous: 1.0 - float(mix)}
        for key, node in rig.weights.get(part.id, {}).items():
            node.outputs[0].default_value = wanted.get(key, 0.0)
            if frame is not None:
                node.outputs[0].keyframe_insert("default_value", frame=frame)


def hold_variants(rig: ActorRig) -> None:
    """Drawing weights step between keys. A blur sample between frames must not half-show two faces."""
    for part_id, weights in rig.weights.items():
        tree = rig.meshes[part_id].data.materials[0].node_tree
        action = tree.animation_data.action if tree.animation_data else None
        if action is None:
            continue
        for curve in _action_curves(action):
            for point in curve.keyframe_points:
                point.interpolation = "CONSTANT"


def _action_curves(action: Any):
    """Every F-curve of an action, for flat (4.x) and layered (4.4+) actions alike."""
    if hasattr(action, "fcurves"):
        yield from action.fcurves
        return
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                yield from bag.fcurves
