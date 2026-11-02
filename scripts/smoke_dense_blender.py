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
    (fixture / "dense-report.json").write_text(json.dumps({
        "status": "dense-mesh-candidate / artistic-and-rights-review-required",
        "distribution_allowed": False,
        "source_revision": str(sparse),
        "mesh": {"ply": path.name, "sha256": sha, **counts},
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
    print("Real Blender private PLY/GLB, 3 axis and 2 calibrated photo views passed", flush=True)


if __name__ == "__main__":
    main()
