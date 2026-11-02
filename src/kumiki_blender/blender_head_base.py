"""Blender-only bridge from an existing human head to a statue sculpt study."""

from __future__ import annotations

from typing import Any

from .reconstruction.head_base import (
    HeadFitError, HeadFitPlan, HeadFitResult, fit_head_landmarks,
    head_deformation_field,
)


def apply_head_base_fit(bpy: Any, scene: Any, imported: list[Any], meshes: list[Any],
                        plan: HeadFitPlan) -> dict[str, Any]:
    """Place the imported head hierarchy and add an editable local sculpt key.

    Source xyz coordinates must be the *imported* GLB's Blender Z-up world
    coordinates, including glTF's axis conversion and any imported root nodes.
    """
    from mathutils import Matrix, Vector  # type: ignore[import-not-found]

    candidates = [obj for obj in meshes if obj.name == plan.mesh_name]
    if len(candidates) != 1:
        raise HeadFitError(
            f"Expected exactly one mesh called {plan.mesh_name!r}; "
            f"imported meshes: {', '.join(obj.name for obj in meshes)}")
    head = candidates[0]
    fit: HeadFitResult = fit_head_landmarks(plan)
    field = head_deformation_field(plan, fit)
    matrix = Matrix(tuple(
        tuple(fit.scale*fit.rotation[row][column] for column in range(3))
        + (fit.translation_m[row],) for row in range(3)
    ) + ((0, 0, 0, 1),))
    imported_objects = set(imported)
    roots = [obj for obj in imported if obj.parent not in imported_objects]
    if not roots:
        raise HeadFitError("Imported head has no transformable hierarchy roots")
    for root in roots:
        root.matrix_world = matrix @ root.matrix_world
    bpy.context.view_layer.update()

    if plan.fit_mode == "local-sculpt":
        if getattr(head.data, "shape_keys", None) is not None:
            raise HeadFitError("Source head already has morph targets; export a neutral static base before local sculpting")
        # glTF can contain multiple nodes sharing one mesh. A local facial
        # sculpt must never mutate an eye/body/duplicate mesh instance.
        if head.data.users > 1:
            head.data = head.data.copy()
        head.shape_key_add(name="Original anatomical base", from_mix=False)
        correction = head.shape_key_add(name="Kumiki / statue landmark corrections", from_mix=False)
        local_to_world = head.matrix_world
        world_to_local = local_to_world.inverted()
        for vertex, key_vertex in zip(head.data.vertices, correction.data):
            world = local_to_world @ vertex.co
            shaped = field.displace((world.x, world.y, world.z))
            key_vertex.co = world_to_local @ Vector(shaped)
        correction.value = 1.0

    markers = bpy.data.collections.new("Sculpt review / measured target landmarks (not GLB)")
    scene.collection.children.link(markers)
    for item in plan.landmarks:
        marker = bpy.data.objects.new(f"Target landmark / {item.name}", None)
        marker.empty_display_type = "SPHERE"
        marker.empty_display_size = max(0.005, item.radius_m * 0.12)
        marker.location = item.target_xyz_m
        marker.hide_render = True
        markers.objects.link(marker)

    return {**fit.to_report(), "fit_mode": plan.fit_mode,
            "sculpt_mesh": head.name, "head_source_asset_id": plan.provenance.asset_id,
            "anatomical_base_preserved_in_blend": True,
            "local_sculpt_shape_key": plan.fit_mode == "local-sculpt",
            "target_landmarks": len(plan.landmarks)}
