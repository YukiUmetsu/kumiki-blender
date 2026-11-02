"""Actual Blender 4.5 body-body collision bake and conservative SFX cue QA."""

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
        material = bpy.data.materials.new(name + "-surface")
        material.diffuse_color = (.63, .63, .66, 1) if name == "impact-block" else (.48, .42, .36, 1)
        obj.data.materials.append(material)

    cube("impact-block", (0, 0, 1.05), (.18, .18, .18))
    cube("stone-anvil", (0, 0, .15), (.65, .65, .3))
    cube("distant-wall", (3, 0, .15), (.5, .5, .3))
    bpy.ops.wm.save_as_mainfile(filepath=str(root / "original.blend"))

    shot = load_physics_plan(ROOT / "examples" / "pair-impact-physics-shot.json")
    build_physics_shot(bpy, shot, root / "physics-r01")
    action = json.loads((root / "physics-r01" / "baked-actions.json").read_text())[0]
    contacts = action["contacts"]
    if len(contacts) < 1 or any(contact["target"] != "kumiki:impact-block:1" or
                                  contact.get("material") != "metal-on-stone" for contact in contacts):
        raise RuntimeError(f"Expected a measured tagged body-body impact: {contacts}")
    if not all(3 < contact["frame"] < 30 and contact["speedMps"] > .25 for contact in contacts):
        raise RuntimeError(f"Pair contact timings/speeds are inconsistent with the Bullet drop: {contacts}")
    track = next(item for item in action["tracks"] if item["target"] == "kumiki:impact-block:1")
    low = min(key["position"][1] for key in track["keys"])
    if low < .295 or low > .48:
        raise RuntimeError(f"Block did not settle on the simulated anvil: min Y={low}")
    manifest = json.loads((root / "physics-r01" / "manifest.json").read_text())
    if manifest["physics"]["requested_contact_pairs"] != 2 or "pair-proximity" not in manifest["physics"]["contact_evidence"]:
        raise RuntimeError("Paired contact evidence was not recorded in the revision manifest")
    print("KUMIKI_PAIR_IMPACT_SMOKE_OK", len(contacts), round(low, 4))


if __name__ == "__main__":
    params = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(params) != 1:
        raise SystemExit("usage: blender ... --python scripts/blender_physics_pair_contacts_smoke.py -- <new-root>")
    main(params[0])
