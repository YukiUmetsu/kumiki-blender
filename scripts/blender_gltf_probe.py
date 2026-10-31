"""Isolate Blender/glTF exporter availability from Kumiki scene generation.

Run with ``blender --background --factory-startup --python scripts/blender_gltf_probe.py``.
This file is a CI diagnostic, not a deliverable asset.
"""

import sys
from pathlib import Path

import bpy


target = Path(".review/ci/diagnostics/minimal.glb").resolve()
target.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.mesh.primitive_cube_add()
cube = bpy.context.object
cube.name = "kumiki-ci-minimal-cube"
cube["kumiki_part"] = "probe"
print("KUMIKI_PROBE: exporter starting", flush=True)
bpy.ops.export_scene.gltf(
    filepath=str(target),
    export_format="GLB",
    use_selection=True,
    export_materials="EXPORT",
    export_animations=False,
    export_extras=True,
)
print(f"KUMIKI_PROBE: exporter completed, {target.stat().st_size} bytes", flush=True)

# A controlled second case exercises the same new-scene, modifier, material,
# save-and-export sequence as the asset pipeline, with ordinary cube meshes.
# This distinguishes an exporter/context defect from malformed input geometry.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kumiki_blender.blender_output import bake_export_meshes  # noqa: E402
from kumiki_blender.blender_scene import _material  # noqa: E402
from kumiki_blender.surfaces import get_surface  # noqa: E402


scene = bpy.data.scenes.new("Kumiki / isolated exporter probe")
if bpy.context.window is not None:
    bpy.context.window.scene = scene
collection = bpy.data.collections.new("Probe / portable geometry")
scene.collection.children.link(collection)
material = _material(bpy, get_surface("natural-stone"))
objects = []
for index in range(64):
    mesh = bpy.data.meshes.new(f"probe mesh {index}")
    mesh.from_pydata(
        [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
         (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)],
        [],
        [(3, 2, 1, 0), (4, 5, 6, 7), (0, 1, 5, 4),
         (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)],
    )
    mesh.update()
    mesh.materials.append(material)
    obj = bpy.data.objects.new(f"probe cube {index}", mesh)
    obj.location.x = index * 2.5
    obj["kumiki_part"] = "probe"
    collection.objects.link(obj)
    bevel = obj.modifiers.new("Probe bevel", "BEVEL")
    bevel.width = 0.02
    bevel.segments = 2
    obj.modifiers.new("Probe normals", "WEIGHTED_NORMAL")
    objects.append(obj)

with bpy.context.temp_override(scene=scene, view_layer=scene.view_layers[0]):
    print("KUMIKI_PROBE: staged scene save starting", flush=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(target.parent / "probe.blend"))
    print("KUMIKI_PROBE: baking 64 modifier meshes", flush=True)
    bake_export_meshes(bpy, objects)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True, view_layer=scene.view_layers[0])
    print("KUMIKI_PROBE: staged scene exporter starting", flush=True)
    bpy.ops.export_scene.gltf(
        filepath=str(target.parent / "staged.glb"),
        export_format="GLB",
        use_selection=True,
        export_materials="EXPORT",
        export_animations=False,
        export_cameras=False,
        export_lights=False,
        export_extras=True,
        export_apply=False,
    )
    print("KUMIKI_PROBE: staged scene exporter completed", flush=True)
