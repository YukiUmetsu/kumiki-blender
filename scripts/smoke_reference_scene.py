"""Real-Blender CI fixture for calibrated multi-view camera import and clay masks.

Creates only first-party synthetic gray reference pixels. No online photograph.
Run inside Blender: --python scripts/smoke_reference_scene.py -- model.glb output
"""

from __future__ import annotations

import json
import struct
import sys
import zlib
from pathlib import Path


def synthetic_png(path: Path, width: int, height: int) -> None:
    def chunk(name: bytes, content: bytes) -> bytes:
        return (struct.pack(">I", len(content)) + name + content +
                struct.pack(">I", zlib.crc32(name + content) & 0xFFFFFFFF))
    raw = b"".join(b"\0" + b"\x80\x80\x80" * width for _ in range(height))
    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    path.write_bytes(data)


def main() -> None:
    arguments = sys.argv[sys.argv.index("--") + 1:]
    model, output = Path(arguments[0]).resolve(), Path(arguments[1]).resolve()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from kumiki_blender.reconstruction.blender_reference import build_reference_scene
    import bpy  # type: ignore[import-not-found]
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector

    input_dir = output.parent / "reference-scene-synthetic-input"
    input_dir.mkdir(parents=True, exist_ok=False)
    cameras = []
    for index, cx in enumerate((60, 75, 45)):
        image_id = f"test-angle-{index}"
        path = input_dir / (image_id + ".png")
        synthetic_png(path, 120, 90)
        cameras.append({
            "image_id": image_id, "source_photo_path": str(path),
            "source_mask_path": None,
            "intrinsics": {"width": 120, "height": 90, "fx_px": 110,
                           "fy_px": 110, "cx_px": cx, "cy_px": 45},
            "translation_m": [0, -8, 2],
            "rotation_3x3": [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        })
    plan = input_dir / "camera-plan.json"
    plan.write_text(json.dumps({"schema_version": 1, "asset_id": "synthetic-ci-fixture",
                                "status": "alignment-candidate / artistic-review-required",
                                "cameras": cameras}))
    result = build_reference_scene(plan, model, output, render=True,
                                   max_cameras=3, bpy_module=bpy)
    assert result["registered_cameras"] == 3, result
    assert len(result["rendered_clay_views"]) == 3, result
    scene = bpy.context.scene
    for index, principal_point in enumerate((60, 75, 45)):
        camera = bpy.data.objects[f"COLMAP / test-angle-{index}"]
        scene.render.resolution_x = 120
        scene.render.resolution_y = 90
        point = world_to_camera_view(scene, camera, Vector((0, 0, 2)))
        assert abs(point.x - principal_point/120) < 0.002, (point.x, principal_point)
        assert abs(point.y - 0.5) < 0.002, (point.y, principal_point)
    print("Real Blender camera intrinsics / axes / clay render smoke passed")


if __name__ == "__main__":
    main()
