"""Real-Blender scan-bake regression using first-party non-historical geometry."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def main() -> None:
    import bpy  # type: ignore[import-not-found]

    output = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    source_dir = output.parent / "scan-lod-first-party-input"
    source_dir.mkdir(parents=True, exist_ok=False)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32)
    sculpture = bpy.context.object
    sculpture.name = "First-party eroded stone relief"
    modifier = sculpture.modifiers.new("Test subdivision", "SUBSURF")
    modifier.levels = 2
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    for vertex in sculpture.data.vertices:
        x, y, z = vertex.co
        amplitude = 1 + 0.035*math.sin(12*x+9*z)*math.sin(16*y-7*z)
        vertex.co *= amplitude
    sculpture.data.update()
    surface = bpy.data.materials.new("Original stone test pigment")
    surface.use_nodes = True
    surface.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (
        0.58, 0.47, 0.37, 1.0)
    sculpture.data.materials.append(surface)
    original = source_dir / "original-sculpt.glb"
    bpy.ops.export_scene.gltf(filepath=str(original), export_format="GLB",
                              use_selection=True, export_animations=False)
    provenance_path = source_dir / "provenance.json"
    provenance_path.write_text(json.dumps({
        "schema_version": 1, "asset_id": "first-party-relief-fixture",
        "creator": "Kumiki Blender CI authors",
        "source_url": "https://github.com/YukiUmetsu/kumiki-blender",
        "license": "MIT", "attribution": "Original synthetic CI relief mesh",
        "redistribution_allowed": True,
    }))
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from kumiki_blender.asset_import import load_asset_provenance
    from kumiki_blender.reconstruction.blender_lods import bake_statue_lods
    from kumiki_blender.reconstruction.lods import ScanLODSettings

    report = bake_statue_lods(original, load_asset_provenance(provenance_path),
                              output, ScanLODSettings(target_triangles=900,
                                                      map_size=512, ray_distance_m=0.08,
                                                      render_previews=True), bpy_module=bpy)
    assert report["status"] == "lod-bake-candidate / artistic-review-required", report
    assert report["high_faces"] >= 4*report["low_faces"], report
    assert report["glb_audit"]["external_references"] == 0, report
    assert (output / "source-high.png").is_file()
    assert (output / "baked-low.png").is_file()
    print("Real Blender PBR high->low scan bake smoke passed", flush=True)


if __name__ == "__main__":
    main()
