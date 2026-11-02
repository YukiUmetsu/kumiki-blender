"""Independent *whole-head* sculpt foundation trial for the Statue of Liberty.

Unlike the rejected r41 donor-face graft, this candidate keeps the entire CC0
anatomical head intact. It is not a scan of the statue or a likeness release.

Blender 4.5+::

    blender --background --factory-startup --python-exit-code 1 \
      --python scripts/blender_liberty_whole_head_sculpt.py -- \
      --source .review/liberty-donor/source.blend \
      --output .review/liberty-r59-whole-head --resolution 640

The output .blend contains a two-level multires sculpt surface and four named
unregistered comparison cameras. No scan graft, face mask or Gaussian relief is
used. The NPS head dimensions constrain only height/width; depth is inferred.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SOURCE_HEAD = "GEO-head_sculpting_realistic"
SOURCE_EYES = (
    "GEO-head_sculpting_realistic.sclera.L",
    "GEO-head_sculpting_realistic.sclera.R",
    "GEO-head_sculpting_realistic.iris.L",
    "GEO-head_sculpting_realistic.iris.R",
)
SHOTS = (
    ("front", (0., -9., 26.69)),
    ("threequarter", (6.5, -8.0, 26.69)),
    ("profile", (9., 0., 26.69)),
    ("opposite-profile", (-9., 0., 26.69)),
)

# CC0 v1.2 donor bust: the full mesh includes neck and shoulders, so those
# bounds must NOT be mistaken for the real cranium. These approximate source
# local-Z landmarks were selected by inspecting the original 3D head contour.
SOURCE_CHIN_Z = 0.17
SOURCE_CRANIAL_WIDTH_Z = (0.24, 0.33)


def main() -> None:
    import bpy
    from mathutils import Matrix, Vector

    from kumiki_blender.liberty_anatomy_source import verify_anatomy_source
    from kumiki_blender.liberty_face_measurements import scaled_nps_head_targets

    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=640)
    cfg = parser.parse_args(args)
    if not 256 <= cfg.resolution <= 2048:
        parser.error("resolution must be in [256, 2048]")
    verify_anatomy_source(cfg.source)
    if cfg.output.exists() and any(cfg.output.iterdir()):
        parser.error("output already contains files; use a new revision path")
    cfg.output.mkdir(parents=True, exist_ok=True)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 24
    scene.cycles.use_denoising = True
    scene.render.resolution_x = cfg.resolution
    scene.render.resolution_y = cfg.resolution
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "AgX"

    with bpy.data.libraries.load(str(cfg.source), link=False) as (source, imported):
        absent = set((SOURCE_HEAD, *SOURCE_EYES)) - set(source.objects)
        if absent:
            raise RuntimeError(f"Reviewed CC0 head objects missing: {sorted(absent)}")
        imported.objects = [SOURCE_HEAD, *SOURCE_EYES]
    head, *eyes = imported.objects

    # The donor is a *full, continuous* anatomical mesh, not a facial patch.
    # Scale its outer head silhouette to NPS height/width ratios but leave all
    # internal anatomical topology and fore-aft depth (unknown) intact.
    low = Vector((min(v.co.x for v in head.data.vertices),
                  min(v.co.y for v in head.data.vertices),
                  min(v.co.z for v in head.data.vertices)))
    high = Vector((max(v.co.x for v in head.data.vertices),
                   max(v.co.y for v in head.data.vertices),
                   max(v.co.z for v in head.data.vertices)))
    cranial_section = [v.co.x for v in head.data.vertices
                       if SOURCE_CRANIAL_WIDTH_Z[0] <= v.co.z <= SOURCE_CRANIAL_WIDTH_Z[1]]
    if len(cranial_section) < 50 or not low.z < SOURCE_CHIN_Z < high.z:
        raise RuntimeError("Reviewed donor head/chin section is not present")
    cranium_width = max(cranial_section) - min(cranial_section)
    # Use approximate *cranium*, not shoulder, section for NPS head ratio.
    midpoint = Vector(((max(cranial_section) + min(cranial_section)) / 2,
                       (low.y + high.y) / 2,
                       (high.z + SOURCE_CHIN_Z) / 2))
    target = scaled_nps_head_targets()
    sx = target.width / cranium_width
    sy = sx  # inferred depth; donor aspect retained, not NPS-measured
    sz = target.height / (high.z - SOURCE_CHIN_Z)
    transform = (Matrix.Translation(Vector((0., 0., 26.69)))
                 @ Matrix.Diagonal((sx, sy, sz, 1.))
                 @ Matrix.Translation(-midpoint))
    # Blender Studio's eyeballs inherit a parent transform from the source.
    # Capture world matrices *before* unlinking parent relations, and remap
    # them through the source head frame so no eyeball floats below the bust.
    for obj in (head, *eyes):
        scene.collection.objects.link(obj)
    bpy.context.view_layer.update()
    donor_head_world = head.matrix_world.copy()
    source_matrices = {obj: obj.matrix_world.copy() for obj in (head, *eyes)}
    target_from_source = transform @ donor_head_world.inverted()

    clay = bpy.data.materials.new("Matte neutral sculpture / geometry only")
    clay.diffuse_color = (.64, .66, .62, 1.)
    clay.use_nodes = True
    clay.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (
        .64, .66, .62, 1.)
    clay.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = .86
    for obj in (head, *eyes):
        obj.parent = None
        obj.matrix_world = target_from_source @ source_matrices[obj]
        obj.data.materials.clear()
        obj.data.materials.append(clay)
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
    head.name = "Liberty sculpt candidate / continuous whole-head anatomy"
    head["source_authenticity"] = "CC0 generic human anatomy; NOT a real Liberty scan"
    head["modeling_strategy"] = "Replace entire head; no r41 graft or seam"
    head["depth_evidence"] = "UNKNOWN - donor proportions retained; not surveyed"
    head["nps_width_ratio_basis_m"] = 3.05
    head["nps_height_ratio_basis_m"] = 5.26
    head["scale_basis"] = "Illustrative 3.52m head height like r58"
    head["donor_cranium_chin_local_z"] = SOURCE_CHIN_Z
    head["donor_cranium_width_band_local_z"] = str(SOURCE_CRANIAL_WIDTH_Z)
    head["sculpt_workflow"] = "Sculpt Mode: adjust entire head, then recheck all four cameras"
    head["sculpt_pass"] = "Original CC0 donor silhouette only; NO Statue-specific sculpt"

    # Multires is actual editable sculpt topology; it does not claim that
    # subdivision or an untouched donor supplies *specific* Liberty likeness.
    bpy.context.view_layer.objects.active = head
    head.select_set(True)
    existing = [m for m in head.modifiers if m.type == "MULTIRES"]
    if len(existing) != 1:
        raise RuntimeError("Expected Blender Studio's one editable multires modifier")
    multires = existing[0]
    multires.name = "Editable multires sculpt (2 levels)"
    while multires.total_levels < 2:
        bpy.ops.object.multires_subdivide(modifier=multires.name, mode="CATMULL_CLARK")
    multires.levels = multires.sculpt_levels = multires.render_levels = 2
    head.select_set(False)
    for obj, name in zip(eyes, ("sclera-left", "sclera-right", "iris-left", "iris-right")):
        obj.name = f"Liberty sculpt candidate / {name}"

    world = bpy.data.worlds.new("Neutral clay grey world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (
        .28, .30, .32, 1.)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = .64
    scene.world = world

    for label, position, energy, size in (
        ("Soft anatomy key", (-6., -7., 31.), 1100., 5.),
        ("Raking side fill", (5., -1.8, 28.), 550., 3.),
    ):
        data = bpy.data.lights.new(label, "AREA")
        data.energy = energy
        data.shape = "DISK"
        data.size = size
        lamp = bpy.data.objects.new(label, data)
        scene.collection.objects.link(lamp)
        lamp.location = position
        lamp.rotation_euler = (Vector((0., 0., 26.69)) - lamp.location).to_track_quat(
            "-Z", "Y").to_euler()

    cameras = {}
    for label, position in SHOTS:
        data = bpy.data.cameras.new(f"Whole-head trial / {label} / UNREGISTERED")
        data.type = "ORTHO"
        data.ortho_scale = 4.10
        camera = bpy.data.objects.new(data.name, data)
        scene.collection.objects.link(camera)
        camera.location = position
        camera.rotation_euler = (Vector((0., 0., 26.69)) - camera.location).to_track_quat(
            "-Z", "Y").to_euler()
        camera["correspondence"] = "Illustrative orthographic; not camera-calibrated HAER"
        cameras[label] = camera

    scene["candidate_status"] = "HOLD / not photo-faithful"
    scene["candidate_note"] = (
        "Anatomical whole-head sculpt base without the rejected graft seams. "
        "Must reshape brow, eyes, nose, lips, chin, jaw, ears against real HAER photos.")
    for name, _ in SHOTS:
        scene.camera = cameras[name]
        scene.render.filepath = str(cfg.output / f"{name}.png")
        bpy.ops.render.render(write_still=True)
    scene.camera = cameras["front"]
    bpy.context.view_layer.objects.active = head
    head.select_set(True)
    bpy.ops.wm.save_as_mainfile(filepath=str(cfg.output / "whole-head-sculpt.blend"))
    print(f"WHOLE_HEAD_SCULPT_HOLD {cfg.output}; {len(head.data.polygons)} base quads")


if __name__ == "__main__":
    main()
