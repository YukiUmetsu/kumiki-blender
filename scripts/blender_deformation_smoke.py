"""Real Blender Cloth → shape-key morph GLB, with structural animation QA."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import bpy  # noqa: E402
from kumiki_blender.deformation_plan import load_deformation_plan  # noqa: E402
from kumiki_blender.blender_deformation import build_deformation_shot  # noqa: E402


def main(folder: str) -> None:
    root = Path(folder).resolve()
    root.mkdir(parents=True, exist_ok=False)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=8, y_subdivisions=8, size=1)
    obj = bpy.context.object
    obj.name = "cloth-panel"
    obj["kumiki_part"] = "cloth-panel"
    material = bpy.data.materials.new("woven-cloth-qa")
    material.diffuse_color = (.48, .67, .82, 1)
    obj.data.materials.append(material)
    pinned = obj.vertex_groups.new(name="pinned-hem")
    indices = [vertex.index for vertex in obj.data.vertices if vertex.co.y > .49]
    if len(indices) < 4:
        raise RuntimeError(f"Pinned cloth edge must have multiple vertices, got {len(indices)}")
    pinned.add(indices, 1, "REPLACE")
    cloth = obj.modifiers.new(name="cloth-physics", type="CLOTH")
    cloth.settings.quality = 5
    cloth.settings.vertex_group_mass = pinned.name
    cloth.settings.pin_stiffness = 1
    bpy.ops.wm.save_as_mainfile(filepath=str(root / "original.blend"))

    shot = load_deformation_plan(ROOT / "examples" / "cloth-deformation-shot.json")
    manifest = build_deformation_shot(bpy, shot, root / "deformation-r01")
    glb = (root / "deformation-r01" / "model.glb").read_bytes()
    json_length, kind = struct.unpack_from("<I4s", glb, 12)
    if kind != b"JSON":
        raise RuntimeError("Missing glTF JSON")
    doc = json.loads(glb[20:20 + json_length])
    targets = doc["meshes"][0]["primitives"][0].get("targets", [])
    weight_channels = [channel for animation in doc.get("animations", [])
                       for channel in animation["channels"] if channel["target"]["path"] == "weights"]
    if len(targets) != manifest["deformation"]["morph_targets"] or not weight_channels:
        raise RuntimeError("Real Blender cloth deformations did not export as animated GLB morph targets")
    if manifest["deformation"]["max_displacement_m"] < .025:
        raise RuntimeError("Evaluated cloth physics did not produce meaningful sag")
    if manifest["glb_audit"]["semantic_nodes"] != 1:
        raise RuntimeError("Semantic cloth panel was lost in GLB export")
    print("KUMIKI_DEFORMATION_SMOKE_OK", len(targets), len(weight_channels),
          manifest["deformation"]["max_displacement_m"])


if __name__ == "__main__":
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(arguments) != 1:
        raise SystemExit("usage: blender ... --python scripts/blender_deformation_smoke.py -- <new-output-root>")
    main(arguments[0])
