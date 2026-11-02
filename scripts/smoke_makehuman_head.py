"""Independent real-Blender QA of the official MakeHuman head GLB output."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import bmesh
import bpy


def main(path: Path) -> None:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["asset_id"] == "makehuman-neutral-head"
    assert manifest["provenance"]["license"] == "CC0-1.0"
    assert manifest["status"].endswith("no-statue-likeness-claim")
    assert manifest["mesh_diagnostic"]["boundary_edges"] == 0
    assert manifest["mesh_diagnostic"]["nonmanifold_edges"] == 0
    provenance = json.loads((path / "head-provenance.json").read_text(encoding="utf-8"))
    assert provenance["license"] == "CC0-1.0" and provenance["redistribution_allowed"]
    for filename in ("source.blend", "model.glb", "preview.png", "previews/front.png",
                     "previews/profile.png", "previews/three-quarter.png", "head-provenance.json"):
        assert (path / filename).stat().st_size > 200, filename
    bpy.ops.import_scene.gltf(filepath=str(path / "model.glb"))
    meshes = [item for item in bpy.context.scene.objects if item.type == "MESH"]
    assert len(meshes) == 1, f"Unexpected additional body/helper meshes: {len(meshes)}"
    head = meshes[0]
    assert head.get("kumiki_part") == "anatomical_head"
    assert len(head.data.vertices) > 3500, "Missing actual face anatomy"
    assert len(head.data.polygons) > 3500, "Missing real polygon topology"
    bm = bmesh.new()
    try:
        bm.from_mesh(head.data)
        assert all(edge.is_manifold for edge in bm.edges), "Portable GLB contains open/nonmanifold edges"
    finally:
        bm.free()
    bounds_min, bounds_max = manifest["bounds_m"]
    width = bounds_max[0] - bounds_min[0]
    height = bounds_max[2] - bounds_min[2]
    assert 0.15 < width < 0.23 and 0.18 < height < 0.28, (width, height)
    print(f"MakeHuman CC0 head smoke passed: {len(head.data.vertices)} vertices, "
          f"{len(head.data.polygons)} faces, {width:.3f}m wide")


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:]
    main(Path(args[0]).resolve())
