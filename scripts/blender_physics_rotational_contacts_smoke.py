"""Real Bullet swinging-rod / off-axis obstacle contact QA."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import bpy  # noqa: E402
import kumiki_blender.blender_pair_contacts as pair_contacts  # noqa: E402
from kumiki_blender.blender_physics import build_physics_shot  # noqa: E402
from kumiki_blender.physics_plan import parse_physics_plan  # noqa: E402


def main(folder: str) -> None:
    root = Path(folder).resolve()
    root.mkdir(parents=True, exist_ok=False)
    bpy.ops.wm.read_factory_settings(use_empty=True)

    def cube(name: str, center: tuple[float, float, float], size: tuple[float, float, float]) -> None:
        bpy.ops.mesh.primitive_cube_add(size=1, location=center)
        obj = bpy.context.object
        obj.name = name
        obj.scale = size
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        obj["kumiki_part"] = name

    cube("swinging-rod", (.75, 0, 1.4), (1.5, .12, .12))
    cube("pivot-pin", (0, 0, 1.4), (.08, .12, .08))
    cube("strike-peg", (1.32, 0, .62), (.24, .24, .22))
    bpy.ops.wm.save_as_mainfile(filepath=str(root / "original.blend"))
    shot = parse_physics_plan({
        "schema_version": 1, "asset_id": "rotational-impact-smoke",
        "action_id": "swinging-rod-impact", "duration_frames": 100,
        "substeps": 40, "solver_iterations": 48,
        "bodies": [
            {"target": "name:swinging-rod", "kind": "active", "shape": "BOX", "mass_kg": 1.5,
             "restitution": .5, "friction": .5},
            {"target": "name:pivot-pin", "kind": "passive", "shape": "BOX"},
            {"target": "name:strike-peg", "kind": "passive", "shape": "BOX", "restitution": .35},
        ],
        "constraints": [{"id": "rod-pivot", "kind": "hinge", "body_a": "name:swinging-rod",
                         "body_b": "name:pivot-pin", "pivot_m": [0, 0, 1.4], "axis": [0, 1, 0]}],
        "contact_pairs": [{"body_a": "name:swinging-rod", "body_b": "name:strike-peg",
                           "material": "wood-on-stone", "tolerance_m": .02}],
    })
    # Compare both detectors against the exact same real Bullet shot. This
    # guards against claiming a rotational improvement based on mocked motion.
    with_surface = pair_contacts.surface_impact_speed
    try:
        pair_contacts.surface_impact_speed = lambda *_args, **_kwargs: None
        build_physics_shot(bpy, shot, root / "centers-only")
    finally:
        pair_contacts.surface_impact_speed = with_surface
    baseline = json.loads((root / "centers-only" / "baked-actions.json").read_text())[0]["contacts"]
    bpy.ops.wm.open_mainfile(filepath=str(root / "original.blend"))
    build_physics_shot(bpy, shot, root / "physics-r01")
    action = json.loads((root / "physics-r01" / "baked-actions.json").read_text())[0]
    contacts = action["contacts"]
    if not contacts:
        raise RuntimeError("Real Bullet swing did not produce an independently observed surface impact")
    if not all(6 < contact["frame"] < 90 and contact["speedMps"] > .25 and
               contact["target"] == "kumiki:swinging-rod:1" for contact in contacts):
        raise RuntimeError(f"Invalid swinging-rod contact evidence: {contacts}")
    center_frames = {item["frame"] for item in baseline}
    novel = [item for item in contacts if item["frame"] not in center_frames]
    if not novel:
        raise RuntimeError(f"Surface witness added no true rotational cue over center-only: {contacts}")
    manifest = json.loads((root / "physics-r01" / "manifest.json").read_text())
    if manifest["physics"].get("pair_velocity_method") != "center-and-closest-surface-rigid-motion":
        raise RuntimeError("Revision manifest lacks rotational contact provenance")
    print("KUMIKI_ROTATIONAL_IMPACT_SMOKE_OK", "center", baseline, "surface", contacts)


if __name__ == "__main__":
    params = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(params) != 1:
        raise SystemExit("usage: blender ... --python scripts/blender_physics_rotational_contacts_smoke.py -- <new-root>")
    main(params[0])
