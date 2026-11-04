"""Blender-only, non-destructive assembly of evidence-guided 3D candidates."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .blender_import import imported_bounds
from .blender_output import create_model_scene, export_revision
from .reconstruction.completion import (
    CompletionError, CompletionPlan, evaluate_completion, verify_completion_sources,
)
from .render_profiles import select_render_profile
from .shots import QualityShot


def _import_glb(bpy: Any, scene: Any, source: Path) -> tuple[list[Any], list[Any]]:
    """Return all new hierarchy nodes and just its mesh objects."""
    before = {item.as_pointer() for item in bpy.data.objects}
    result = bpy.ops.import_scene.gltf(
        filepath=str(source), import_pack_images=True, import_scene_extras=True,
        import_scene_as_collection=True, import_select_created_objects=True,
    )
    if "FINISHED" not in result:
        raise CompletionError(f"Blender could not import candidate {source.name}")
    imported = [item for item in scene.objects if item.as_pointer() not in before]
    meshes = [item for item in imported if item.type == "MESH"]
    if not meshes or any(not item.data.vertices or not item.data.polygons for item in meshes):
        raise CompletionError(f"No usable mesh surfaces in {source.name}")
    return imported, meshes


def _apply_candidate_fit(bpy: Any, imported: list[Any], transform: dict[str, Any]) -> None:
    """Transform hierarchy roots, never mutate the original base mesh data."""
    from mathutils import Matrix  # type: ignore[import-not-found]

    rotation = transform["rotation_3x3"]
    scale = transform["scale"]
    translation = transform["translation_m"]
    matrix = Matrix(tuple(
        tuple(scale * rotation[row][column] for column in range(3)) + (translation[row],)
        for row in range(3)
    ) + ((0, 0, 0, 1),))
    belonging = set(imported)
    roots = [item for item in imported if item.parent not in belonging]
    if not roots:
        raise CompletionError("Candidate GLB has no transformable hierarchy root")
    for root in roots:
        root.matrix_world = matrix @ root.matrix_world
    bpy.context.view_layer.update()


def tag_base_mesh(mesh: Any, *, mode: str, declared_origin: str, source_sha: str) -> None:
    """Never promote previously inferred components into observed evidence."""
    existing = mesh.get("kumiki_geometry_origin")
    if existing is not None and existing not in ("observed", "inferred"):
        raise CompletionError("Base contains unknown geometry provenance")
    origin = ("inferred" if existing == "inferred" or declared_origin == "inferred"
              else "observed")
    if mode == "historical" and origin == "inferred":
        raise CompletionError("historical mode cannot import an inferred base component")
    if not mesh.get("kumiki_part"):
        mesh["kumiki_part"] = "observed-base" if origin == "observed" else "inferred-base"
    mesh["kumiki_geometry_origin"] = origin
    if not mesh.get("kumiki_source_sha256"):
        mesh["kumiki_source_sha256"] = source_sha


def validate_imported_texture_images(meshes: list[Any]) -> None:
    """Check imported mesh material textures, not unrelated Blender projects."""
    for mesh in meshes:
        for slot in mesh.material_slots:
            material = slot.material
            if material is None or not material.use_nodes or material.node_tree is None:
                continue
            for node in material.node_tree.nodes:
                if node.type != "TEX_IMAGE" or node.image is None:
                    continue
                image = node.image
                if image.source == "FILE" and image.filepath and not image.packed_file:
                    raise CompletionError(f"Imported texture was not packed: {image.name}")


def build_completion_scene(plan: CompletionPlan, report: dict[str, Any],
                           output: str | Path, *, bpy_module: Any) -> dict[str, Any]:
    """Assemble admitted candidate meshes and create a private GLB/Blend revision.

    Candidate parts are separate objects in the Blend. This does not weld or
    claim to recover unseen geometry. Observed base objects remain untouched.
    """
    bpy = bpy_module
    verify_completion_sources(plan)
    if report != evaluate_completion(plan):
        raise CompletionError("Candidate evaluation changed or does not match validated inputs")
    if tuple(bpy.app.version) < (4, 2, 0):
        raise CompletionError("Blender 4.2 or newer is required")
    output = Path(output).resolve()
    if output.exists() or output.is_symlink():
        raise CompletionError(f"Output revision already exists: {output}")
    region_reports = {item["id"]: item for item in report["regions"]}
    if len(region_reports) != len(plan.regions):
        raise CompletionError("Candidate evaluation has missing or duplicated regions")
    profile = select_render_profile(1280, 720, 24)
    scene = create_model_scene(bpy, plan.asset_id)
    scene.render.engine = "CYCLES"
    scene.cycles.samples = profile.samples
    scene.render.resolution_x = profile.width
    scene.render.resolution_y = profile.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "AgX"

    _, base_meshes = _import_glb(bpy, scene, plan.base_glb)
    for mesh in base_meshes:
        tag_base_mesh(mesh, mode=plan.mode, declared_origin=plan.base_origin,
                      source_sha=plan.base_sha256)
    output_meshes = list(base_meshes)
    for region in plan.regions:
        record = region_reports[region.id]
        if not record["included"]:
            continue
        imported, meshes = _import_glb(bpy, scene, region.candidate_glb)
        _apply_candidate_fit(bpy, imported, record["transform"])
        for mesh in meshes:
            mesh["kumiki_part"] = f"completion-{region.id}"
            mesh["kumiki_completion_region"] = region.id
            mesh["kumiki_geometry_origin"] = record["geometry_origin"]
            mesh["kumiki_source_sha256"] = region.candidate_sha256
        output_meshes.extend(meshes)
    bpy.context.view_layer.update()
    validate_imported_texture_images(output_meshes)
    bounds_min, bounds_max = imported_bounds(output_meshes)
    center = tuple((a + b) / 2 for a, b in zip(bounds_min, bounds_max))
    extent = max(b - a for a, b in zip(bounds_min, bounds_max))
    if not math.isfinite(extent) or extent <= 0:
        raise CompletionError("Completion geometry has invalid world bounds")
    frame = max(0.3, extent * 1.9)

    sky = bpy.data.worlds.new("Kumiki / completion review sky")
    sky.use_nodes = True
    background = next(node for node in sky.node_tree.nodes if node.type == "BACKGROUND")
    background.inputs["Color"].default_value = (0.35, 0.40, 0.47, 1.0)
    background.inputs["Strength"].default_value = 0.7
    scene.world = sky
    light_data = bpy.data.lights.new("Completion / softbox", type="AREA")
    light_data.energy = 1000
    light_data.shape = "DISK"
    light_data.size = max(2, extent * 1.1)
    light = bpy.data.objects.new(light_data.name, light_data)
    scene.collection.objects.link(light)
    light.location = (center[0] + 1.5*extent, center[1] - 1.3*extent,
                      center[2] + 1.8*extent)
    from .blender_output import aim_at
    aim_at(light, center)

    camera_data = bpy.data.cameras.new("Completion / review camera")
    camera_data.type = "ORTHO"
    camera = bpy.data.objects.new(camera_data.name, camera_data)
    scene.collection.objects.link(camera)

    def position(x: float, y: float, z: float) -> tuple[float, float, float]:
        return tuple(center[i] + extent * component for i, component in enumerate((x, y, z)))

    shots = (
        QualityShot("hero", position(1.4, -1.4, 0.85), center, frame),
        QualityShot("front", position(0, -2, 0.15), center, frame),
        QualityShot("profile", position(2, 0, 0.15), center, frame),
        QualityShot("rear", position(0, 2, 0.15), center, frame),
        QualityShot("top", position(0.1, -0.2, 2.3), center, frame),
    )
    manifest = export_revision(
        bpy, scene, camera, output_meshes, shots, output,
        samples=profile.samples,
        metadata={
            "asset_id": plan.asset_id,
            "generator": "evidence-guided-completion",
            "mode": plan.mode,
            "status": "completion-candidate / artistic-review-required",
            "base_glb_sha256": plan.base_sha256,
            "base_geometry_preserved": True,
            "historical_fidelity_verified": False,
            "regions": report["regions"],
            "geometry_joined": False,
            "limitations": "Separate candidate meshes, no automatic seam welding or photographic likeness approval.",
        },
    )
    return manifest
