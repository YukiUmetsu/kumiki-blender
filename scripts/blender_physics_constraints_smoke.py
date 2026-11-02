"""Actual Blender 4.5 headless native hinge, slider and fixed constraints QA."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import bpy  # noqa: E402
from kumiki_blender.blender_physics import build_physics_shot  # noqa: E402
from kumiki_blender.physics_plan import load_physics_plan  # noqa: E402


def main(folder: str) -> None:
    root = Path(folder).resolve()
    root.mkdir(parents=True, exist_ok=False)
    bpy.ops.wm.read_factory_settings(use_empty=True)

    def cube(name: str, center: tuple[float, float, float], dimensions: tuple[float, float, float]) -> None:
        bpy.ops.mesh.primitive_cube_add(size=1, location=center)
        obj = bpy.context.object
        obj.name = name
        obj.scale = dimensions
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        obj["kumiki_part"] = name
        mat = bpy.data.materials.new(f"{name}-material")
        mat.diffuse_color = (.58, .42, .24, 1)
        obj.data.materials.append(mat)

    cube("hinge-base", (0, 0, 1), (.2, .2, 2))
    cube("hinge-arm", (.55, 0, 1), (1, .12, .12))
    cube("slider-guide", (2, 0, 1), (.12, .12, 2))
    cube("slider-weight", (2.4, 0, 1.4), (.25, .25, .25))
    cube("fixed-base", (4, 0, 1), (.25, .25, 1.5))
    cube("fixed-piece", (4.4, 0, 1.2), (.25, .25, .25))
    bpy.ops.wm.save_as_mainfile(filepath=str(root / "original.blend"))

    shot = load_physics_plan(ROOT / "examples" / "mechanical-constraints-physics-shot.json")
    build_physics_shot(bpy, shot, root / "physics-r01")
    action = json.loads((root / "physics-r01" / "baked-actions.json").read_text())[0]
    tracks = {track["target"].split(":")[1]: track["keys"] for track in action["tracks"]}
    if set(tracks) != {"hinge-arm", "slider-weight", "fixed-piece"} or action["durationFrames"] != 60:
        raise RuntimeError("Three independent constrained-body tracks were not exported")

    def fall(role: str) -> float:
        keys = tracks[role]
        return keys[0]["position"][1] - keys[30]["position"][1]

    hinge_drop, slider_drop, fixed_drop = fall("hinge-arm"), fall("slider-weight"), fall("fixed-piece")
    if hinge_drop < .10 or not .08 < slider_drop < .65 or abs(fixed_drop) > .08:
        raise RuntimeError(f"Constraint solver deviated: hinge={hinge_drop:.3f}, slider={slider_drop:.3f}, fixed={fixed_drop:.3f}")
    manifest = json.loads((root / "physics-r01" / "manifest.json").read_text())
    if manifest["physics"]["constraints"] != 3 or manifest["glb_audit"]["semantic_nodes"] != 6:
        raise RuntimeError("Native Blender constraint helpers leaked into exported GLB")
    print("KUMIKI_CONSTRAINTS_SMOKE_OK", round(hinge_drop, 3), round(slider_drop, 3), round(fixed_drop, 3))


if __name__ == "__main__":
    params = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(params) != 1:
        raise SystemExit("usage: blender ... --python scripts/blender_physics_constraints_smoke.py -- <new-root>")
    main(params[0])
