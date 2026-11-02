"""Blender 4.2+ synthetic PLY -> private GLB/editable-review regression."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def main() -> None:
    import bpy  # type: ignore[import-not-found]

    output = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    fixture = output.parent / "dense-blender-first-party-input"
    fixture.mkdir(parents=True, exist_ok=False)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24)
    sphere = bpy.context.object
    sphere.name = "Original synthetic private review mesh"
    # Make actual geometry variation, not a flat screenshot.
    for vertex in sphere.data.vertices:
        x, y, z = vertex.co
        vertex.co.z += 0.04*x*y
    sphere.data.update()
    original_vertices = len(sphere.data.vertices)
    original_faces = len(sphere.data.polygons)
    path = fixture / "first-party-source.ply"
    bpy.ops.wm.ply_export(filepath=str(path), export_selected_objects=True)
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    (fixture / "dense-report.json").write_text(json.dumps({
        "status": "dense-mesh-candidate / artistic-and-rights-review-required",
        "distribution_allowed": False,
        "mesh": {"ply": path.name, "sha256": sha, "vertices": original_vertices,
                 "faces": original_faces},
    }))
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from kumiki_blender.reconstruction.blender_dense import build_dense_blender_review

    result = build_dense_blender_review(fixture, output, bpy_module=bpy, render_previews=True)
    assert result["status"].startswith("private-Blender-mesh-candidate"), result
    assert len(result["previews"]) == 3, result
    assert result["distribution_allowed"] is False
    assert (output / "private-dense-review.blend").stat().st_size > 0
    assert (output / "private-dense-review.glb").stat().st_size > 0
    for filename in result["previews"]:
        assert (output / filename).stat().st_size > 0
    print("Real Blender private PLY import, GLB and three-axis review passed", flush=True)


if __name__ == "__main__":
    main()
