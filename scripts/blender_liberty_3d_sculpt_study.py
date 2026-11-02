"""One controlled 3D landmark-sculpt *study* on the r59 whole-head base.

This is a reviewable artist-direction experiment, not recovered survey depth.
Its low-amplitude 3D spatial strokes change existing anatomically connected
topology and remain reversible via a Blender shape key. They do NOT reproduce
unique monument anatomy by themselves or lift the photo-faithfulness HOLD.

Blender 4.5::

    blender --background --factory-startup --python-exit-code 1 \
      --python scripts/blender_liberty_3d_sculpt_study.py -- \
      --input .review/liberty-r59-whole-head/whole-head-sculpt.blend \
      --output .review/liberty-r60-spatial-study --resolution 640
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path


HEAD_NAME = "Liberty sculpt candidate / continuous whole-head anatomy"


@dataclass(frozen=True)
class SpatialStroke:
    """A manually chosen whole-head mass sculpt in world-space metres.

    x/z are inspected in the stable front/profile renders; the y/depth anchor
    is found on the *actual* continuous face by front-facing 3D ray casting.
    These are inferred artistic constraints, NOT calibrated photographic data.
    """

    name: str
    x: float
    z: float
    dx: float
    dy: float
    dz: float
    radius: float


# The *actual* HAER side photos show an orbital ledge and substantial lips/chin
# underneath the nose. The previous r59 narrow brow-band deformation created a
# cartoon ledge; these broad, sub-10cm strokes are an independent last trial on
# a continuous CC0 anatomical surface, not a third Gaussian procedural face.
STROKES = (
    SpatialStroke("brow left / softly forward", -.42, 26.85, -.010, -.065, -.012, .79),
    SpatialStroke("brow right / softly forward", .42, 26.85, .010, -.065, -.012, .79),
    SpatialStroke("left cheek / broad zygoma", -.49, 26.22, -.035, -.025, 0., .75),
    SpatialStroke("right cheek / broad zygoma", .49, 26.22, .035, -.025, 0., .75),
    SpatialStroke("upper/lower lips / volume", 0., 25.55, 0., -.085, 0., .52),
    SpatialStroke("chin / forward projection", 0., 25.12, 0., -.105, 0., .66),
    SpatialStroke("left mandible / outer volume", -.41, 25.35, -.045, 0., 0., .72),
    SpatialStroke("right mandible / outer volume", .41, 25.35, .045, 0., 0., .72),
)


def smooth_falloff(distance: float, radius: float) -> float:
    """Bounded 3D volume stroke, zero at its radius and smooth at both ends."""
    if not math.isfinite(distance) or not math.isfinite(radius) or radius <= 0:
        raise ValueError("finite distance and positive radius required")
    t = min(1., max(0., distance / radius))
    s = t * t * (3. - 2. * t)
    return 1. - s


def main() -> None:
    import bpy
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=640)
    cfg = parser.parse_args(argv)
    if not cfg.input.is_file() or not 256 <= cfg.resolution <= 2048:
        parser.error("input must exist; resolution must be in [256, 2048]")
    if cfg.output.exists() and any(cfg.output.iterdir()):
        parser.error("output exists; choose a new isolated sculpt checkpoint")
    cfg.output.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.open_mainfile(filepath=str(cfg.input.resolve()))
    head = bpy.data.objects.get(HEAD_NAME)
    if head is None or head.type != "MESH" or head.get("source_authenticity") != (
            "CC0 generic human anatomy; NOT a real Liberty scan"):
        raise RuntimeError("Expected verified r59 generic whole-head study, not a statue scan")
    if not any(m.type == "MULTIRES" and m.total_levels >= 2 for m in head.modifiers):
        raise RuntimeError("Source is not the reviewed editable multires anatomical head")
    scene = bpy.context.scene
    scene.render.resolution_x = scene.render.resolution_y = cfg.resolution
    scene.render.resolution_percentage = 100

    depsgraph = bpy.context.evaluated_depsgraph_get()
    tree = BVHTree.FromObject(head, depsgraph)
    world = head.matrix_world.copy()
    world_inverse = world.inverted()
    anchors = []
    for stroke in STROKES:
        origin = Vector((stroke.x, -6., stroke.z))
        direction = Vector((0., 1., 0.))
        hit, _, _, _ = tree.ray_cast(
            world_inverse @ origin,
            (world_inverse.to_3x3() @ direction).normalized(), 12.)
        if hit is None:
            raise RuntimeError(f"No 3D facial mesh found for {stroke.name}")
        anchors.append(world @ hit)

    # Original base shape remains addressable in the .blend. Sculpt deformation
    # is separate, reversible, and affects *real 3D vertices* near the rays.
    if head.data.shape_keys is not None:
        raise RuntimeError("Source already has shape keys; refusing to alter it")
    head.shape_key_add(name="r59 donor base / recoverable")
    result = head.shape_key_add(name="r60 inferred 3D anatomy volume study")
    result.value = 1.
    changed, largest = 0, 0.
    for index, vertex in enumerate(head.data.vertices):
        original_world = world @ vertex.co
        displacement = Vector((0., 0., 0.))
        for stroke, anchor in zip(STROKES, anchors):
            radial = (original_world - anchor).length
            weight = smooth_falloff(radial, stroke.radius)
            displacement += weight * Vector((stroke.dx, stroke.dy, stroke.dz))
        result.data[index].co = world_inverse @ (original_world + displacement)
        if displacement.length > .00001:
            changed += 1
            largest = max(largest, displacement.length)
    if changed < 100 or largest > .25:
        raise RuntimeError(f"Unexpected sculpt locality: {changed} vertices, {largest:.3f}m")

    head["r60_stroke_count"] = len(STROKES)
    head["r60_modified_base_vertices"] = changed
    head["r60_max_vertex_displacement_m"] = largest
    head["r60_geometry_evidence"] = "Artist-inferred 3D broad masses, NOT measured HAER data"
    scene["candidate_status"] = "HOLD / sculpt trial needs multi-view actual-monument inspection"
    brief = bpy.data.texts.new("R60 SCULPT STROKE AUDIT / NOT A LIKENESS RELEASE")
    brief.write("This sculpt uses inferred 3D brush volumes on a generic donor.\n"
                "Real installed-head HAER NY-138-236/237/241 camera intrinsics unknown.\n"
                "Reject if either profile or 3/4 fails anatomy; do not export a GLB.\n")
    for stroke in STROKES:
        brief.write(f"{stroke.name}: ({stroke.x:.2f}, {stroke.z:.2f}), "
                    f"move ({stroke.dx:+.3f},{stroke.dy:+.3f},{stroke.dz:+.3f}) "
                    f"radius {stroke.radius:.2f}m\n")

    cameras = {
        name: next((o for o in bpy.data.objects if o.type == "CAMERA"
                    and f" / {name} / " in o.name), None)
        for name in ("front", "threequarter", "profile", "opposite-profile")
    }
    if any(camera is None for camera in cameras.values()):
        raise RuntimeError("The reviewed four fixed cameras are required")
    for name, camera in cameras.items():
        scene.camera = camera
        scene.render.filepath = str(cfg.output / f"{name}.png")
        bpy.ops.render.render(write_still=True)
    scene.camera = cameras["front"]
    bpy.context.view_layer.objects.active = head
    head.select_set(True)
    bpy.ops.wm.save_as_mainfile(filepath=str(cfg.output / "anatomy-stroke-study.blend"))
    print(f"R60_ANATOMY_STUDY_HOLD {changed} vertices, max {largest:.4f}m")


if __name__ == "__main__":
    main()
