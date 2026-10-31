"""Isolate Blender/glTF exporter availability from Kumiki scene generation.

Run with ``blender --background --factory-startup --python scripts/blender_gltf_probe.py``.
This file is a CI diagnostic, not a deliverable asset.
"""

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
