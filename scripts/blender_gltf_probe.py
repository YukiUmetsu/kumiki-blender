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

# Probe material, modifier baking and exporting from Blender's active scene.
# The separate inactive-scene/temp_override diagnostic previously reproduced
# a native crash even with one cube; production now avoids that path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kumiki_blender.blender_output import bake_export_meshes  # noqa: E402
from kumiki_blender.blender_scene import _material  # noqa: E402
from kumiki_blender.surfaces import get_surface  # noqa: E402


cube.data.materials.append(_material(bpy, get_surface("natural-stone")))
bevel = cube.modifiers.new("Default scene bevel", "BEVEL")
bevel.width = 0.02
bevel.segments = 2
cube.modifiers.new("Default scene normals", "WEIGHTED_NORMAL")
print("KUMIKI_PROBE: default scene save starting", flush=True)
bpy.ops.wm.save_as_mainfile(filepath=str(target.parent / "default.blend"))
bake_export_meshes(bpy, [cube])
print("KUMIKI_PROBE: default scene baked exporter starting", flush=True)
bpy.ops.export_scene.gltf(
    filepath=str(target.parent / "default-baked.glb"),
    export_format="GLB",
    use_selection=True,
    export_materials="EXPORT",
    export_animations=False,
    export_extras=True,
    export_apply=False,
)
print("KUMIKI_PROBE: default scene baked exporter completed", flush=True)
