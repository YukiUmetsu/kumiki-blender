"""Stage an editable, non-procedural sculpt head inside the existing Liberty scene.

This is a *HOLD* review adapter, not a statue likeness generator. It imports
five existing CC0 sculpt objects from a reviewed .blend, preserves Multires,
and hides (never deletes) the failed generated face. No facial vertex, fold,
or hair surface is generated, displaced, or reconstructed here.

Blender 4.5+::

    blender --background --python-exit-code 1 \
      --python scripts/blender_liberty_sculpt_integration.py -- \
      --statue-blend .review/liberty-r64-survey/second-pass/source.blend \
      --sculpt-blend .review/liberty-r66-sculpt-studio/liberty-face-sculpt-studio.blend \
      --output .review/liberty-r69-sculpt-assembly --resolution 640

The inputs are geometrically independent: the archival HAER photographs in
the sculpt workspace are *unregistered references*, never copied as texture
projections. The donor is a CC0 generic anatomical head, not the real statue.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


HEAD_NAME = "Liberty sculpt candidate / continuous whole-head anatomy"
EYE_NAMES = (
    "Liberty sculpt candidate / sclera-left",
    "Liberty sculpt candidate / sclera-right",
    "Liberty sculpt candidate / iris-left",
    "Liberty sculpt candidate / iris-right",
)
SOURCE_OBJECTS = (HEAD_NAME, *EYE_NAMES)
REFERENCE_SCENES = (
    "02 HAER POSES / UNREGISTERED",
    "03 HAER ANATOMY / UNREGISTERED",
)
SCULPT_COLLECTION = "SCULPT REVIEW / editable CC0 head / NOT monument likeness"
OLD_HEAD_COLLECTION = "HOLD / previous code-generated head / reversible"


def uniform_head_scale(source_width: float, source_height: float,
                       target_width: float, target_height: float) -> float:
    """Width-anchor a review scaffold without deforming the face or depth.

    A head-and-neck bounding box is not a measured chin-to-cranium height.
    Record a moderate height mismatch, but reject grossly unrelated models.
    Uniform scale cannot substitute for a Statue-specific anatomical sculpt.
    """
    values = (source_width, source_height, target_width, target_height)
    if any(not (0.05 < value < 100.) for value in values):
        raise ValueError("Head dimensions must be plausible, positive metres")
    by_width = target_width / source_width
    by_height = target_height / source_height
    if abs(by_width - by_height) / max(by_width, by_height) > .28:
        raise ValueError("Head width/height ratios disagree; nonuniform anatomical warp prohibited "
                         f"(source {source_width:.3f}×{source_height:.3f}, "
                         f"target {target_width:.3f}×{target_height:.3f}, "
                         f"scales {by_width:.3f}/{by_height:.3f})")
    return by_width


def main() -> None:
    import bpy
    from mathutils import Matrix, Vector

    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statue-blend", type=Path, required=True)
    parser.add_argument("--sculpt-blend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=640)
    cfg = parser.parse_args(arguments)
    if not 384 <= cfg.resolution <= 2048:
        parser.error("resolution must be in [384, 2048]")
    if not cfg.statue_blend.is_file() or not cfg.sculpt_blend.is_file():
        parser.error("both reviewed input .blend files must exist")
    if cfg.statue_blend.resolve() == cfg.sculpt_blend.resolve():
        parser.error("the source sculpt and full statue must be distinct")
    output = cfg.output.resolve()
    if output.exists() or output.is_symlink():
        parser.error("output checkpoint already exists; use a new revision")

    bpy.ops.wm.open_mainfile(filepath=str(cfg.statue_blend.resolve()))
    scene = bpy.context.scene
    legacy = [obj for obj in scene.objects
              if obj.name.startswith("liberty-head-") and obj.type == "MESH"]
    original_face = bpy.data.objects.get("liberty-head-face")
    if len(legacy) < 15 or original_face not in legacy:
        raise RuntimeError("Expected the full statue's independently editable legacy head")

    def bounds(objects):
        corners = [obj.matrix_world @ Vector(corner)
                   for obj in objects for corner in obj.bound_box]
        if not corners:
            raise RuntimeError("Missing head geometry bounds")
        return (Vector([min(p[i] for p in corners) for i in range(3)]),
                Vector([max(p[i] for p in corners) for i in range(3)]))

    target_low, target_high = bounds(legacy)
    face_low, face_high = bounds([original_face])
    source_filepath = str(cfg.sculpt_blend.resolve())
    with bpy.data.libraries.load(source_filepath, link=False) as (source, imported):
        absent = set(SOURCE_OBJECTS) - set(source.objects)
        if absent:
            raise RuntimeError(f"Sculpt source lacks reviewed objects: {sorted(absent)}")
        missing_boards = set(REFERENCE_SCENES) - set(source.scenes)
        if missing_boards:
            raise RuntimeError(f"Sculpt source lacks original-statue photo boards: {sorted(missing_boards)}")
        imported.objects = list(SOURCE_OBJECTS)
        imported.scenes = list(REFERENCE_SCENES)
    objects = list(imported.objects)
    boards = list(imported.scenes)
    for board in boards:
        if (board is None or board.get("camera_registration") !=
                "NONE / photo contact board / pose unknown" or
                board.get("source") != "1984-85 HAER NY-138, actual installed head, NPS/LOC"):
            raise RuntimeError("Photo board provenance or unregistered status was lost")
    head = next(o for o in objects if o.name == HEAD_NAME)
    if (head.get("source_authenticity") !=
            "CC0 generic human anatomy; NOT a real Liberty scan"):
        raise RuntimeError("Sculpt provenance is not the reviewed CC0 anatomical donor")
    if not any(m.type == "MULTIRES" and m.total_levels >= 2 for m in head.modifiers):
        raise RuntimeError("Expected an editable, continuous Multires sculpt foundation")
    if any(obj.parent is not None or obj.type != "MESH" for obj in objects):
        raise RuntimeError("Sculpt import requires parent-free, editable mesh parts")

    sculpt_collection = bpy.data.collections.new(SCULPT_COLLECTION)
    scene.collection.children.link(sculpt_collection)
    for obj in objects:
        sculpt_collection.objects.link(obj)
    bpy.context.view_layer.update()
    sculpt_low, sculpt_high = bounds([head])
    target_width = target_high.x - target_low.x
    target_height = target_high.z - target_low.z
    scale = uniform_head_scale(sculpt_high.x - sculpt_low.x,
                               sculpt_high.z - sculpt_low.z,
                               target_width, target_height)
    center_x = (target_low.x + target_high.x) / 2
    center_y = (face_low.y + face_high.y) / 2
    donor_center = (sculpt_low + sculpt_high) / 2
    translation = Vector((center_x - scale * donor_center.x,
                          center_y - scale * donor_center.y,
                          target_high.z - scale * sculpt_high.z))
    placement = Matrix.Translation(translation) @ Matrix.Scale(scale, 4)
    for obj in objects:
        obj.matrix_world = placement @ obj.matrix_world
        obj["review_only"] = True
        obj["original_sculpt_blend"] = cfg.sculpt_blend.name
        obj["geometry_status"] = "HOLD: CC0 anatomical scaffold, not installed Liberty likeness"
    bpy.context.view_layer.update()
    revised_low, revised_high = bounds([head])
    eyes = [obj for obj in objects if "sclera" in obj.name]
    eye_separation = abs(eyes[0].matrix_world.translation.x - eyes[1].matrix_world.translation.x)
    if not (.40 < eye_separation < 1.60
            and target_low.z - .8 < revised_low.z < target_low.z + .8
            and target_high.z - .2 < revised_high.z < target_high.z + .2
            and all(revised_low.z + .3 * (revised_high.z - revised_low.z)
                    < obj.matrix_world.translation.z < revised_high.z - .15
                    for obj in eyes)):
        raise RuntimeError("Imported sculpt eye/cranium alignment failed")

    # The CC0 source contains the shoulders of a human bust. Keep them
    # recoverable in the source topology, but mask them in this *head-only*
    # candidate: those generic shoulder forms otherwise protrude through the
    # historical statue's copper robe. This is modifier-based source cleanup,
    # NOT a new cloth or facial surface.
    neck_cut_z = target_low.z + .75
    keep_group = head.vertex_groups.new(name="Head and upper neck only / non-destructive")
    kept_vertices = 0
    for vertex in head.data.vertices:
        if (head.matrix_world @ vertex.co).z >= neck_cut_z:
            keep_group.add([vertex.index], 1.0, "REPLACE")
            kept_vertices += 1
    if not 300 < kept_vertices < len(head.data.vertices) - 80:
        raise RuntimeError("Head-versus-shoulders mask did not isolate a plausible neck")
    mask = head.modifiers.new("Remove donor shoulders / reversible / HOLD", "MASK")
    mask.vertex_group = keep_group.name
    mask.threshold = .5

    old_collection = bpy.data.collections.new(OLD_HEAD_COLLECTION)
    scene.collection.children.link(old_collection)
    for obj in legacy:
        old_collection.objects.link(obj)  # Additional collection, no lost source links.
        obj.hide_render = True
        obj.hide_set(True)
        obj["legacy_status"] = "HOLD / recoverable comparison only"

    # Existing crown, hair, body, drapery, torch, tablet and pedestal are
    # deliberately untouched. The user can still compare and sculpt the head.
    scene["head_foundation"] = "CC0 whole-head Multires / manually editable / unverified likeness"
    scene["sculpt_review"] = "HOLD / reference cameras UNREGISTERED"
    scene["legacy_head"] = OLD_HEAD_COLLECTION
    scene["sculpt_input"] = cfg.sculpt_blend.name
    scene["sculpt_uniform_scale"] = round(scale, 6)
    scene["sculpt_height_envelope_residual_m"] = round(
        (revised_high.z - revised_low.z) - target_height, 6)
    scene["donor_shoulder_mask_cut_z"] = round(neck_cut_z, 4)
    scene["head_evidence"] = "HAER NY-138 visual references / not calibrated depth"
    scene["photo_reference_scenes"] = ", ".join(REFERENCE_SCENES)

    note = bpy.data.texts.new("SCULPT INTEGRATION / anatomical HOLD")
    note.write(
        "Non-procedural integration checkpoint; no regenerated anatomy.\n"
        "A CC0 donor head plus its four eyes replaces ONLY visible\n"
        "rendering of 20 previous code-generated head parts. Old meshes\n"
        "are hidden and recoverable in HOLD collection.\n"
        "An editable Mask modifier hides generic donor shoulders below a\n"
        "neck plane; the original donor vertices are never destroyed.\n"
        "The imported mesh is a sculpt scaffold, not actual Liberty geometry.\n"
        "Two packed HAER NY-138 pose/anatomy boards are included as\n"
        "separate Blender scenes. Their cameras are NOT registered.\n"
        "Depth and cameras have not been calibrated: face likeness HOLD.\n"
        "Do not release GLB as photo-faithful.\n"
    )

    clay = bpy.data.materials.new("QA neutral clay / whole-assembly hold")
    clay.diffuse_color = (.69, .71, .70, 1)
    clay.use_nodes = True
    principled = clay.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (.69, .71, .70, 1)
    principled.inputs["Roughness"].default_value = .88
    existing_face_material = original_face.data.materials[0] if original_face.data.materials else None
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(existing_face_material or clay)
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 12
    scene.cycles.use_denoising = True
    scene.render.resolution_x = cfg.resolution
    scene.render.resolution_y = cfg.resolution
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "AgX"

    look_at = Vector((center_x, center_y, (target_low.z + target_high.z) / 2))
    cameras = {}
    for label, position in (
        ("front", (center_x, center_y - 16., look_at.z)),
        ("profile", (center_x + 16., center_y, look_at.z)),
        ("opposite-profile", (center_x - 16., center_y, look_at.z)),
        ("threequarter", (center_x + 12., center_y - 14., look_at.z + .3)),
    ):
        data = bpy.data.cameras.new(f"Head sculpt QA / {label} / UNREGISTERED")
        data.type = "ORTHO"
        data.ortho_scale = 6.9
        camera = bpy.data.objects.new(data.name, data)
        scene.collection.objects.link(camera)
        camera.location = position
        camera.rotation_euler = (look_at - camera.location).to_track_quat("-Z", "Y").to_euler()
        camera["reference_registration"] = "NONE / no recovered HAER camera pose"
        cameras[label] = camera

    for label, location, energy, size in (
        ("Left anatomy key", (center_x - 9., center_y - 11., look_at.z + 7.), 6000., 7.),
        ("Right anatomy fill", (center_x + 9., center_y - 6., look_at.z + 3.), 2300., 6.),
    ):
        light_data = bpy.data.lights.new(label, "AREA")
        light_data.energy = energy
        light_data.shape = "DISK"
        light_data.size = size
        light = bpy.data.objects.new(label, light_data)
        scene.collection.objects.link(light)
        light.location = location
        light.rotation_euler = (look_at - light.location).to_track_quat("-Z", "Y").to_euler()

    output.mkdir(parents=True, exist_ok=False)
    override = scene.view_layers[0].material_override
    # Genuine anatomical QA must not accidentally include the displaced arms,
    # robe, crown rods, tablet, or plinth in the silhouette. The isolated donor
    # is shown with its full reversible source mesh; a separate assembly view
    # reveals exactly where the head-only shoulder mask and generated crown
    # do not interface. Do not crop away those assembly defects.
    backdrop_meshes = [obj for obj in scene.objects
                       if obj.type == "MESH" and obj not in objects]
    backdrop_visibility = {obj: obj.hide_render for obj in backdrop_meshes}
    for obj in backdrop_meshes:
        obj.hide_render = True
    mask.show_render = False
    for label, camera in cameras.items():
        scene.camera = camera
        scene.view_layers[0].material_override = clay
        scene.render.filepath = str(output / f"{label}-clay.png")
        bpy.ops.render.render(write_still=True)
    mask.show_render = True
    for obj, previous in backdrop_visibility.items():
        obj.hide_render = previous
    scene.camera = cameras["front"]
    scene.render.filepath = str(output / "assembly-front-clay.png")
    bpy.ops.render.render(write_still=True)
    scene.view_layers[0].material_override = override
    scene.camera = cameras["threequarter"]
    bpy.ops.object.select_all(action="DESELECT")
    head.select_set(True)
    bpy.context.view_layer.objects.active = head
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "sculpt-in-statue-hold.blend"))

    # An isolated full-statue GLB is useful as an export smoke check, but is
    # NOT published as the new public Statue of Liberty model while HOLD.
    candidate = [obj for obj in scene.objects
                 if obj.type == "MESH" and
                 (obj.name.startswith("liberty-") and obj not in legacy or obj in objects)]
    bpy.ops.object.select_all(action="DESELECT")
    for obj in candidate:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = head
    bpy.ops.export_scene.gltf(filepath=str(output / "hold-export-smoke.glb"),
                              export_format="GLB", use_selection=True,
                              export_materials="EXPORT", export_extras=True,
                              export_apply=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from kumiki_blender.glb_audit import audit_glb

    report = audit_glb(output / "hold-export-smoke.glb")
    (output / "review.json").write_text(json.dumps({
        "status": "HOLD / CC0 anatomical donor is not the actual statue face",
        "new_geometry": "five pre-existing CC0 donor meshes imported / no generated anatomy",
        "source_provenance": "Blender Studio Human Base Meshes 1.2, CC0 1.0",
        "evidence": "unregistered real-target HAER photographs; no triangulated face scan",
        "archival_reference_board_scenes": [board.name for board in boards],
        "target_envelope_m": [round(target_width, 4), round(target_height, 4)],
        "sculpt_envelope_after_fit_m": [round(revised_high.x - revised_low.x, 4),
                                       round(revised_high.z - revised_low.z, 4)],
        "height_envelope_residual_m": round(
            (revised_high.z - revised_low.z) - target_height, 4),
        "uniform_scale": round(scale, 6),
        "donor_shoulders_masked": True,
        "mask_kept_base_vertices": kept_vertices,
        "replaced_head_components_hidden": len(legacy),
        "retained_statue_parts": len(candidate) - len(objects),
        "glb_smoke": report,
        "views": [f"{name}-clay.png" for name in cameras] + ["assembly-front-clay.png"],
        "unresolved": ["exact Statue facial anatomy", "hair/crown contact",
                       "natural copper drapery", "calibrated camera correspondence"],
    }, indent=2) + "\n", encoding="utf8")
    print(f"LIBERTY_SCULPT_INTEGRATION_HOLD scale={scale:.5f} "
          f"legacy={len(legacy)} selected={len(candidate)} {output}")


if __name__ == "__main__":
    main()
