"""Open an emitted source.blend and exercise the actual moving Blender rig."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy  # Blender's own Python runtime


def verify(source: Path) -> None:
    bpy.ops.wm.open_mainfile(filepath=str(source.resolve()))
    scene = bpy.context.scene
    assert scene.frame_end == 97 and scene.render.fps == 24
    assert bpy.data.objects.get("RIG / aircraft-root") is not None
    channels = (
        "propeller", "upper-left-aileron", "upper-right-aileron",
        "lower-left-aileron", "lower-right-aileron", "elevator",
        "rudder", "left-wheel", "right-wheel",
    )
    controls = {name: bpy.data.objects[f"RIG / {name}"] for name in channels}
    for name, obj in controls.items():
        assert obj.animation_data and obj.animation_data.action, name
        assert obj.parent.name == "RIG / aircraft-root", name
        assert any(child.type == "MESH" for child in obj.children), name
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"
              and isinstance(obj.get("kumiki_part"), str)]
    assert len(meshes) >= 190
    assert any(obj.name == "windscreen-glass" for obj in meshes)
    assert all(image.packed_file is not None for image in bpy.data.images
               if image.source == "FILE")

    def world_vertex(name: str):
        mesh = bpy.data.objects[name]
        # The first propeller vertex is close to its hub; sample the farthest
        # local vertex so half a rotation produces a clearly measurable arc.
        tip = max(mesh.data.vertices, key=lambda vertex: vertex.co.length)
        return mesh.matrix_world @ tip.co

    scene.frame_set(1)
    original = {name: obj.rotation_euler.copy() for name, obj in controls.items()}
    prop_start = world_vertex("propeller-upper-blade")
    left_weight_start = world_vertex("left-wheel-balance-weight")
    right_weight_start = world_vertex("right-wheel-balance-weight")
    axle_start = world_vertex("landing-axle-crossbar")
    scene.frame_set(7)
    # A propeller at the exact end of a revolution looks motionless; sample a
    # half-turn and verify its actual blade mesh changed world position.
    assert (world_vertex("propeller-upper-blade") - prop_start).length > .4
    assert (world_vertex("left-wheel-balance-weight") - left_weight_start).length > .08
    assert (world_vertex("right-wheel-balance-weight") - right_weight_start).length > .08
    assert (world_vertex("landing-axle-crossbar") - axle_start).length < 1e-6
    scene.frame_set(25)
    for name, obj in controls.items():
        axis = 1 if name == "propeller" else 2 if name == "rudder" else 0
        assert abs(obj.rotation_euler[axis] - original[name][axis]) > .05, name
    assert abs(controls["upper-left-aileron"].rotation_euler.x +
               controls["upper-right-aileron"].rotation_euler.x) < 1e-4
    assert abs(controls["upper-left-aileron"].location.z - 2.65) < 1e-6
    assert abs(controls["lower-left-aileron"].location.z - 1.55) < 1e-6
    scene.frame_set(97)
    assert abs(controls["propeller"].rotation_euler.y - original["propeller"].y -
               8*math.tau) < 1e-3
    print(f"Biplane source rig verified: {len(meshes)} named meshes, 9 animated hinges")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    if len(argv) != 1:
        raise SystemExit("Usage: blender --background --python scripts/blender_verify_biplane_rig.py -- <source.blend>")
    verify(Path(argv[0]))
