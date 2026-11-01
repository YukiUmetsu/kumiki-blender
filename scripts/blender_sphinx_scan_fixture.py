"""Build a first-party GLB proxy to exercise scan placement in CI.

The boxes/sphere deliberately are NOT a replica of the Great Sphinx. They
test GLB hierarchy, source PBR material preservation, and model replacement.
"""

from pathlib import Path

import bpy


target = Path(".review/ci/first-party-sphinx-proxy.glb").resolve()
target.parent.mkdir(parents=True, exist_ok=True)
material = bpy.data.materials.new("Source scan / unmodified ochre")
material.use_nodes = True
shader = next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
shader.inputs["Base Color"].default_value = (.47, .36, .23, 1)
shader.inputs["Roughness"].default_value = .92

objects = []
for name, location, dimensions in (
    ("proxy-body", (.25, 0, 1.15), (9.5, 3.4, 2.3)),
    ("proxy-left-forepaw", (-3.5, -1.25, .42), (2.3, .85, .84)),
    ("proxy-right-forepaw", (-3.5, 1.25, .42), (2.3, .85, .84)),
):
    bpy.ops.mesh.primitive_cube_add(size=2, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    obj.data.materials.append(material)
    obj["kumiki_part"] = "ci-fixture-sculpt"
    objects.append(obj)
bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=14, location=(-3.6, 0, 3.26))
head = bpy.context.object
head.name = "proxy-head"
head.dimensions = (2.25, 2.4, 2.6)
head.data.materials.append(material)
head["kumiki_part"] = "ci-fixture-head"
objects.append(head)

bpy.ops.object.select_all(action="DESELECT")
for obj in objects:
    obj.select_set(True)
bpy.context.view_layer.objects.active = head
result = bpy.ops.export_scene.gltf(
    filepath=str(target), export_format="GLB", use_selection=True,
    export_materials="EXPORT", export_animations=False,
    export_extras=True,
)
if "FINISHED" not in result or target.stat().st_size < 1000:
    raise RuntimeError("Failed to create first-party Sphinx scan placement fixture")
print(f"KUMIKI_SCAN_FIXTURE: wrote {target.stat().st_size} bytes", flush=True)
