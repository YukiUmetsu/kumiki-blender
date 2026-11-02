"""Blender 4.2+ synthetic PLY -> private GLB/editable-review regression."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def main() -> None:
    import bpy  # type: ignore[import-not-found]

    output = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    # Key the fixture by review output so separate smoke invocations do not
    # collide with another independent CI/local revision in the same folder.
    fixture = output.with_name(output.name + "-first-party-input")
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
    path = fixture / "first-party-source.ply"
    bpy.ops.wm.ply_export(filepath=str(path), export_selected_objects=True)
    # Blender's PLY exporter may triangulate polygons and merge identical
    # pole vertices; the real boundary contract is the exported PLY header.
    from kumiki_blender.reconstruction.dense import _ply_counts
    counts = _ply_counts(path)
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    # First-party camera/photo fixture proves actual Blender calibrated renders
    # can inspect dense meshes, with no borrowed online reference pixels.
    sparse = fixture / "sparse"
    model = sparse / "undistorted/text-model"
    model.mkdir(parents=True)
    images = sparse / "undistorted/images"
    images.mkdir()
    (model / "cameras.txt").write_text("1 PINHOLE 640 480 600 600 320 240\n")
    (model / "images.txt").write_text(
        "1 1 0 0 0 0 0 4 1 first-01.jpg\n\n"
        "2 1 0 0 0 0.4 0 4 1 first-02.jpg\n\n")
    entries = []
    for number in (1, 2):
        filename = f"first-{number:02d}.jpg"
        img = bpy.data.images.new(f"Test photo {number}", width=640, height=480)
        img.generated_color = (0.65, 0.42, 0.28, 1.0)
        img.filepath_raw = str(images / filename)
        img.file_format = "JPEG"
        img.save()
        with (images / filename).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        entries.append({"filename": filename, "dimensions": [640, 480],
                        "sha256": digest})
    (sparse / "undistorted-reference-index.json").write_text(
        json.dumps({"images": entries}))
    (sparse / "reconstruction-report.json").write_text(json.dumps({
        "status": "candidate / visual-review-required", "registered_images": 2,
    }))
    textured = fixture / "workspace"
    textured.mkdir()
    obj = textured / "scene_dense_mesh_refined_texture.obj"
    mtl = textured / "scene_dense_mesh_refined_texture.mtl"
    atlas = textured / "scene_dense_mesh_refined_texture_material_00_map_Kd.jpg"
    obj.write_text(f"mtllib {mtl.name}\n"
                   "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
                   "vt 0 0\nvt 1 0\nvt 0 1\nvt 1 1\n"
                   "usemtl material_00\nf 1/1 2/2 3/3\nf 1/1 3/3 4/4\n")
    mtl.write_text(f"newmtl material_00\nmap_Kd {atlas.name}\n")
    atlas.write_bytes((images / "first-01.jpg").read_bytes())
    with obj.open("rb") as stream:
        obj_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    with atlas.open("rb") as stream:
        atlas_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    (fixture / "dense-report.json").write_text(json.dumps({
        "status": "dense-mesh-candidate / artistic-and-rights-review-required",
        "distribution_allowed": False,
        "source_revision": str(sparse),
        "mesh": {"ply": path.name, "sha256": sha, **counts},
        "textured_obj": {"path": str(obj.relative_to(fixture)), "sha256": obj_sha},
        "texture_atlases": [{"path": str(atlas.relative_to(fixture)), "sha256": atlas_sha}],
    }))
    from kumiki_blender.reconstruction.blender_dense import build_dense_blender_review

    result = build_dense_blender_review(fixture, output, bpy_module=bpy,
                                        render_previews=True, matched_views=2)
    assert result["status"].startswith("private-Blender-mesh-candidate"), result
    assert len(result["previews"]) == 3, result
    assert len(result["photo_matched_previews"]) == 2, result
    assert result["distribution_allowed"] is False
    assert (output / "private-dense-review.blend").stat().st_size > 0
    assert (output / "private-dense-review.glb").stat().st_size > 0
    for filename in result["previews"] + result["photo_matched_previews"]:
        assert (output / filename).stat().st_size > 0
    textured_output = output.with_name(output.name + "-textured")
    textured_result = build_dense_blender_review(
        fixture, textured_output, bpy_module=bpy,
        render_previews=False, textured=True)
    assert textured_result["status"].startswith("private-Blender-mesh-candidate"), textured_result
    assert textured_result["textured"] is True
    assert textured_result["glb_audit"]["materials"] >= 1
    assert (textured_output / "private-textured-review.blend").stat().st_size > 0
    assert (textured_output / "private-textured-review.glb").stat().st_size > 0
    print("Real Blender PLY/GLB, 3 axis, 2 calibrated, and private textured OBJ/GLB passed",
          flush=True)


if __name__ == "__main__":
    main()
