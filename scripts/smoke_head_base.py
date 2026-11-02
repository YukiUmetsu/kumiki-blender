"""Headless real-Blender integration fixture; no third-party human mesh shipped."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

import bpy  # type: ignore[import-not-found]  # noqa: E402

from kumiki_blender.cli import main  # noqa: E402


def smoke(destination: Path) -> None:
    stage = destination.parent / "synthetic-anatomical-base-input"
    if stage.exists() or destination.exists():
        raise RuntimeError("Choose unused smoke output directories")
    stage.mkdir(parents=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=16, radius=0.12)
    head = bpy.context.object
    head.name = "Head"
    head.data.name = "HeadMesh"
    head["kumiki_part"] = "anatomical-head"
    original_vertices = [tuple(vertex.co) for vertex in head.data.vertices]
    linked = bpy.data.objects.new("HeadReferenceCopy", head.data)
    linked.location = (0.5, 0, 0)
    bpy.context.scene.collection.objects.link(linked)
    # Source GLB deliberately contains two instances of the same datablock.
    linked.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(stage / "head.glb"),
                              export_format="GLB", use_selection=True,
                              export_animations=False, export_extras=True)
    (stage / "rights.json").write_text(json.dumps({
        "schema_version": 1, "asset_id": "synthetic-head-ci-fixture",
        "creator": "Kumiki Blender test author",
        "source_url": "https://github.com/YukiUmetsu/kumiki-blender/tree/main/scripts",
        "license": "MIT", "attribution": "Original temporary fixture generated for CI",
        "redistribution_allowed": True,
    }))
    landmarks = {
        "nose": (0, -0.12, 0.05), "left-eye": (-0.06, -0.10, 0.04),
        "right-eye": (0.06, -0.10, 0.04), "chin": (0, -0.11, -0.12),
        "right-ear": (0.10, 0.015, 0), "forehead": (0, -0.085, 0.13),
    }
    (stage / "fit.json").write_text(json.dumps({
        "schema_version": 1, "asset_id": "synthetic-head-ci-study",
        "source_glb": "head.glb", "source_provenance": "rights.json",
        "mesh_name": "Head", "fit_mode": "local-sculpt",
        "landmarks": [{
            "id": name, "region": "face", "source_xyz_m": list(point),
            "target_xyz_m": [3-2*point[1]+(0.01 if name == "nose" else 0),
                             -1+2*point[0], 5+2*point[2]],
            "radius_m": 0.12,
        } for name, point in landmarks.items()],
    }))

    bpy.ops.wm.read_factory_settings(use_empty=True)
    status = main(["build-head-base", "--manifest", str(stage / "fit.json"),
                   "--output", str(destination)])
    if status != 0:
        raise RuntimeError(f"Anatomical head-base Blender smoke failed with code {status}")
    report = json.loads((destination / "manifest.json").read_text())
    if (report["head_fit"]["sculpt_mesh"] != "Head"
            or report["head_fit"]["local_sculpt_shape_key"] is not True
            or report["head_fit"]["historical_fidelity_verified"] is not False):
        raise RuntimeError("Real Blender head fit lacks its expected sculpt/QA metadata")
    for filename in ("source.blend", "model.glb", "preview.png",
                     "previews/front.png", "previews/profile.png",
                     "previews/three-quarter.png"):
        if (destination / filename).stat().st_size <= 0:
            raise RuntimeError(f"Anatomical head-base smoke omitted {filename}")

    # Inspect the actual editable Blender file, not only its JSON manifest.
    bpy.ops.wm.open_mainfile(filepath=str(destination / "source.blend"))
    editable_head = bpy.data.objects.get("Head")
    untouched_copy = bpy.data.objects.get("HeadReferenceCopy")
    if editable_head is None or untouched_copy is None:
        raise RuntimeError("Source .blend lost the head or sibling mesh instance")
    if editable_head.data == untouched_copy.data:
        raise RuntimeError("Head sculpt did not isolate its shared mesh datablock")
    keys = getattr(editable_head.data.shape_keys, "key_blocks", ()) if editable_head.data.shape_keys else ()
    if "Original anatomical base" not in keys or "Kumiki / statue landmark corrections" not in keys:
        raise RuntimeError("Editable .blend lost the neutral anatomy or sculpture correction key")
    if untouched_copy.data.shape_keys is not None:
        raise RuntimeError("Facial correction accidentally affected a linked instance")

    # Reimport the *exported* GLB to verify that geometry, not just metadata,
    # moved into the statue's coordinate frame and that the correction baked.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(destination / "model.glb"))
    exported_head = bpy.data.objects.get("Head")
    exported_copy = bpy.data.objects.get("HeadReferenceCopy")
    if exported_head is None or exported_copy is None:
        raise RuntimeError("Portable GLB omitted the target or linked geometry")

    def global_positions(obj):
        return [tuple(obj.matrix_world @ vertex.co) for vertex in obj.data.vertices]

    head_points = global_positions(exported_head)
    copy_points = global_positions(exported_copy)
    center = tuple((min(p[axis] for p in head_points) + max(p[axis] for p in head_points)) / 2
                   for axis in range(3))
    if math.dist(center, (3, -1, 5)) > 0.05:
        raise RuntimeError(f"GLB face geometry failed landmark-space transform: {center}")

    def registered(point, *, linked_offset=0):
        x, y, z = point
        return (3 - 2*y, -1 + 2*(x + linked_offset), 5 + 2*z)

    def maximum_deviation(points, expected):
        return max(min(math.dist(point, reference) for reference in expected)
                   for point in points)

    original_head = [registered(point) for point in original_vertices]
    original_copy = [registered(point, linked_offset=0.5) for point in original_vertices]
    if maximum_deviation(copy_points, original_copy) > 0.002:
        raise RuntimeError("Linked geometry was modified beyond its rigid transform")
    if maximum_deviation(head_points, original_head) < 0.0002:
        raise RuntimeError("Exported GLB did not bake the landmark correction shape key")


if __name__ == "__main__":
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(arguments) != 1:
        raise SystemExit("Usage: blender --python scripts/smoke_head_base.py -- OUTPUT_DIR")
    smoke(Path(arguments[0]).resolve())
