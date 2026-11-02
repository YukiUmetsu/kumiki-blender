"""Real Blender 4.5 headless smoke: two rigid bodies fall onto a static floor.

Fixture geometry is intentionally primitive; this checks simulation/interchange,
not artistic modeling. Usage: blender --background --factory-startup --python
scripts/blender_physics_smoke.py -- /tmp/new-qa-root
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import bpy  # noqa: E402
from kumiki_blender.blender_physics import build_physics_shot  # noqa: E402
from kumiki_blender.physics_plan import parse_physics_plan  # noqa: E402


def main(folder: str) -> None:
    root = Path(folder).resolve()
    root.mkdir(parents=True, exist_ok=False)
    bpy.ops.wm.read_factory_settings(use_empty=True)

    def cube(name: str, role: str, size: tuple[float, float, float], z: float) -> None:
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, z))
        obj = bpy.context.object
        obj.name = name
        obj.scale = size
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        obj["kumiki_part"] = role
        material = bpy.data.materials.new(name=f"{name}-mat")
        material.diffuse_color = (.45, .53, .65, 1) if role == "hardware" else (.22, .25, .29, 1)
        obj.data.materials.append(material)

    cube("drop-a", "hardware", (.16, .16, .16), 1.3)
    cube("drop-b", "hardware", (.13, .13, .13), 1.8)
    bpy.ops.wm.save_as_mainfile(filepath=str(root / "original.blend"))

    plan = parse_physics_plan({
        "schema_version": 1, "asset_id": "rigid-body-smoke", "action_id": "physics-fall",
        "duration_frames": 45, "substeps": 16, "solver_iterations": 16,
        "ground": {"height_m": 0, "friction": .7, "restitution": .08},
        "bodies": [
            {"target": "role:hardware", "kind": "active", "mass_kg": .3,
             "restitution": .22, "friction": .5, "impact_material": "metal-on-floor"},
        ],
    })
    build_physics_shot(bpy, plan, root / "physics-r01")
    sidecar = json.loads((root / "physics-r01" / "baked-actions.json").read_text())
    if len(sidecar) != 1 or len(sidecar[0]["tracks"]) < 2:
        raise RuntimeError("Real rigid-body bake did not create at least two animated node tracks")
    if sidecar[0]["durationFrames"] != plan.duration_frames:
        raise RuntimeError("GLB/PM2 frame zero or final timestamp is offset from the shot plan")
    first = sidecar[0]["tracks"][0]["keys"][0]["position"][1]
    end = sidecar[0]["tracks"][0]["keys"][-1]["position"][1]
    if first <= end + .25:
        raise RuntimeError(f"Real rigid-body bake failed to fall: {first:.3f} -> {end:.3f} m")
    if any(key["position"][1] < .02 for track in sidecar[0]["tracks"] for key in track["keys"]):
        raise RuntimeError("An animated body penetrated or fell through simulation-only ground")
    contacts = sidecar[0]["contacts"]
    if not contacts or not all(cue["material"] == "metal-on-floor" and .25 <= cue["speedMps"] <= 10
                               and cue["target"].startswith("kumiki:hardware:") for cue in contacts):
        raise RuntimeError("Real solver contact did not produce bounded canonical PM2 SFX cues")
    manifest = json.loads((root / "physics-r01" / "manifest.json").read_text())
    if not manifest["physics"]["has_collision_ground"] or manifest["glb_audit"]["semantic_nodes"] != 2:
        raise RuntimeError("Ground collider must exist in .blend but not leak into GLB")
    print("KUMIKI_PHYSICS_SMOKE_OK", sidecar[0]["id"], len(sidecar[0]["tracks"]),
          sidecar[0]["durationFrames"], round(first, 3), round(end, 3))


if __name__ == "__main__":
    params = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(params) != 1:
        raise SystemExit("usage: blender ... --python scripts/blender_physics_smoke.py -- <new-output-root>")
    main(params[0])
