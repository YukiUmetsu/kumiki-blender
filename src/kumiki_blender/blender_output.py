"""Shared Blender revision exporter for procedurally generated model types.

The same export path handles lattice towers, architectural studies and future
landmark generators. Studio objects never enter the portable selected GLB.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

from .export_audit import audit_export_files
from .glb_audit import audit_glb
from .shots import QualityShot


def aim_at(obj: Any, target: tuple[float, float, float]) -> None:
    from mathutils import Vector  # Blender-only dependency.

    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def create_model_scene(bpy: Any, asset_id: str) -> Any:
    """Use the active factory scene for headless exports; isolate UI builds.

    Blender 4.5's glTF exporter can crash when a new background scene is passed
    through ``temp_override``. A background process must therefore begin with
    factory startup, and must not open a user .blend file first.
    """
    name = f"Kumiki / {asset_id}"
    if bpy.app.background:
        if bpy.data.filepath:
            raise RuntimeError("Headless Kumiki builds need an empty factory-startup Blender session")
        scene = bpy.context.scene
        if scene is None:
            raise RuntimeError("Headless Blender has no active scene")
        scene.name = name
        # The default cube, camera and light remain recoverable in source.blend,
        # but cannot contaminate the preview or selected portable GLB.
        for obj in scene.objects:
            obj.hide_render = True
            obj.hide_set(True)
        return scene
    scene = bpy.data.scenes.new(name)
    if bpy.context.window is not None:
        bpy.context.window.scene = scene
    return scene


def bake_export_meshes(bpy: Any, objects: Sequence[Any]) -> None:
    """Bake evaluated bevels/normals after saving the editable source blend.

    The glTF exporter's global Apply Modifiers option has crashed in our
    Blender 4.5 headless smoke test with hundreds of individual members.
    Capture evaluated meshes explicitly, before mutating any source object,
    so GLB still contains the same physical edge details as the Cycles scene.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    baked = []
    for obj in objects:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = bpy.data.meshes.new_from_object(evaluated, preserve_all_data_layers=True, depsgraph=depsgraph)
        if mesh is None or not mesh.vertices or not mesh.polygons:
            raise RuntimeError(f"Failed to bake modified mesh: {obj.name}")
        baked.append((obj, mesh))
    for obj, mesh in baked:
        obj.modifiers.clear()
        obj.data = mesh


def export_revision(
    bpy: Any,
    scene: Any,
    camera: Any,
    objects: Sequence[Any],
    shots: Sequence[QualityShot],
    destination: Path,
    *,
    samples: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Write one immutable revision of an existing staged Blender scene."""
    destination = Path(destination).resolve()
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"Output revision already exists: {destination}")
    if not shots or shots[0].name != "hero" or len({shot.name for shot in shots}) != len(shots):
        raise RuntimeError("Quality shots require a first 'hero' view and unique names")
    if not objects:
        raise RuntimeError("Cannot export an empty model")

    hero = shots[0]
    camera.location = hero.position
    aim_at(camera, hero.target)
    camera.data.ortho_scale = hero.ortho_scale
    scene.camera = camera

    destination.mkdir(parents=True, exist_ok=False)
    scene.render.filepath = str(destination / "preview.png")
    context = (nullcontext() if getattr(bpy.context, "scene", None) is scene
               else bpy.context.temp_override(scene=scene, view_layer=scene.view_layers[0]))
    with context:
        bpy.ops.wm.save_as_mainfile(filepath=str(destination / "source.blend"))

        bake_export_meshes(bpy, objects)

        # Explicit model selection excludes all studio-only lights and ground.
        bpy.ops.object.select_all(action="DESELECT")
        for obj in objects:
            obj.select_set(True, view_layer=scene.view_layers[0])
        bpy.ops.export_scene.gltf(
            filepath=str(destination / "model.glb"), export_format="GLB",
            use_selection=True, export_materials="EXPORT", export_animations=False,
            export_cameras=False, export_lights=False, export_extras=True,
            export_apply=False,  # Explicitly baked above; avoid the global modifier exporter path.
        )
        glb_report = audit_glb(destination / "model.glb")
        bpy.ops.object.select_all(action="DESELECT")

        preview_dir = destination / "previews"
        preview_dir.mkdir(exist_ok=True)
        for shot in shots:
            camera.location = shot.position
            aim_at(camera, shot.target)
            camera.data.ortho_scale = shot.ortho_scale
            scene.cycles.samples = samples if shot.name == "hero" else max(8, samples // 2)
            scene.render.filepath = str(destination / "preview.png" if shot.name == "hero"
                                        else preview_dir / f"{shot.name}.png")
            bpy.ops.render.render(write_still=True, scene=scene.name)

    views = ["preview.png", *(f"previews/{shot.name}.png" for shot in shots[1:])]
    exported_sizes = audit_export_files(destination, ["source.blend", *views])
    manifest = {
        "schema_version": 1,
        **metadata,
        "blender_version": bpy.app.version_string,
        "glb_audit": glb_report,
        "output_bytes": {**exported_sizes, "model.glb": glb_report["bytes"]},
        "files": {"source": "source.blend", "model": "model.glb",
                  "preview": "preview.png", "quality_views": views},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
