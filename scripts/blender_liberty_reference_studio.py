"""Pack authentic HAER head photographs into an editable Blender sculpt studio.

The archived photographs have unknown camera intrinsics and pose. They are
reference *boards*, deliberately in a separate scene, never projections onto
the illustrative head or registered likeness-gate input.

Run this script with Blender 4.5 after generating a ``clay.blend`` checkpoint.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArchivedView:
    survey_id: str
    filename: str
    label: str
    loc_url: str
    jpeg_url: str


ARCHIVED_VIEWS = (
    ArchivedView(
        "HAER NY-138-236", "haer-236-oblique-large.jpg", "OBLIQUE / RIGHT FACE",
        "https://www.loc.gov/pictures/item/ny1251.color.570005c/",
        "https://cdn.loc.gov/service/pnp/habshaer/ny/ny1200/ny1251/color/570005cv.jpg",
    ),
    ArchivedView(
        "HAER NY-138-237", "haer-237-right-profile-large.jpg", "RIGHT PROFILE",
        "https://www.loc.gov/pictures/item/ny1251.color.570006c/",
        "https://cdn.loc.gov/service/pnp/habshaer/ny/ny1200/ny1251/color/570006cv.jpg",
    ),
    ArchivedView(
        "HAER NY-138-241", "haer-241-left-profile-large.jpg", "LEFT PROFILE",
        "https://www.loc.gov/pictures/item/ny1251.color.570010c/",
        "https://cdn.loc.gov/service/pnp/habshaer/ny/ny1200/ny1251/color/570010cv.jpg",
    ),
)


def resolve_photos(reference_dir: Path) -> list[tuple[ArchivedView, Path]]:
    """Fail closed if any of the three installed-monument source files is absent."""
    if not reference_dir.is_dir():
        raise ValueError(f"missing HAER directory: {reference_dir}")
    resolved = []
    for photo in ARCHIVED_VIEWS:
        path = reference_dir / photo.filename
        if not path.is_file() or path.stat().st_size < 1000:
            raise ValueError(f"missing/empty {photo.survey_id}: {path}; source: {photo.jpeg_url}")
        with path.open("rb") as stream:
            if stream.read(3) != b"\xff\xd8\xff":
                raise ValueError(f"not a JPEG photo for {photo.survey_id}: {path}")
        resolved.append((photo, path.resolve()))
    return resolved


def main() -> None:
    import bpy
    from mathutils import Vector

    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clay-blend", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="New output directory, for the editable .blend and source board PNG")
    cfg = parser.parse_args(argv)
    if not cfg.clay_blend.is_file():
        parser.error(f"missing facial sculpt checkpoint: {cfg.clay_blend}")
    try:
        photos = resolve_photos(cfg.reference_dir)
    except ValueError as error:
        parser.error(str(error))
    if cfg.output.exists() and any(cfg.output.iterdir()):
        parser.error("output revision already exists; select a new checkpoint")
    cfg.output.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.open_mainfile(filepath=str(cfg.clay_blend.resolve()))
    sculpt_scene = bpy.context.scene
    sculpt_scene.name = "01 SCULPT / neutral clay / HOLD"
    sculpt_scene["anatomy_acceptance"] = "HOLD - the installed face has not been reconstructed"
    sculpt_scene["archive_registration"] = "NONE - HAER poses/intrinsics unknown"
    sculpt_scene["reference_board_scene"] = "02 HAER PHOTO BOARD / UNREGISTERED"
    for photo, _path in photos:
        sculpt_scene[photo.survey_id] = photo.loc_url

    brief = bpy.data.texts.new("SCULPT BRIEF / actual Liberty face / HOLD")
    brief.write(
        "ACTUAL STATUE OF LIBERTY FACE - ANATOMICAL SCULPT BRIEF\n"
        "=================================================\n"
        "State: HOLD. The existing illustrative head is not photo-faithful.\n"
        "These photographs are of the installed copper monument (HAER, 1984).\n"
        "Their camera poses and intrinsics are UNKNOWN; do not project them\n"
        "onto the mesh, score pixel matches or claim reconstruction.\n\n"
        "Sculpt order / work from primary mass to secondary folds:\n"
        "1. Establish continuous frontal, temporal, cheek and jaw volumes.\n"
        "   Current profile looks like an upright slab and the chin is squared.\n"
        "2. Study the underside of brow and true hollow orbital transition.\n"
        "   Real eye sits deep under a broad continuous upper lid ridge.\n"
        "3. Rebuild nasal bridge, tip, alae, nostril shadow, and philtrum as\n"
        "   connected masses; the current nose is a pinched protrusion.\n"
        "4. Give both lips genuine forward/underlying volume, a continuous\n"
        "   cupid-to-corner line, and a lower-lip-to-chin break.\n"
        "5. Rebuild masseter, jaw and ear transitions before engraving folds.\n"
        "6. Check front, both sides and three-quarter in *neutral clay*.\n"
        "   Find a licensed authentic scan or calibrate reliable multi-view\n"
        "   camera correspondence before releasing a photo-faithful GLB.\n\n"
        "Reference photos are packed in scene 02 HAER PHOTO BOARD.\n"
        "Reference IDs: HAER NY-138-236, HAER NY-138-237, HAER NY-138-241.\n"
    )

    board = bpy.data.scenes.new("02 HAER PHOTO BOARD / UNREGISTERED")
    board.render.engine = "CYCLES"
    board.cycles.samples = 8
    board.render.resolution_x = 1600
    board.render.resolution_y = 960
    board.render.resolution_percentage = 100
    board.view_settings.view_transform = "Standard"
    board.render.image_settings.file_format = "PNG"
    board["source"] = "1984 HAER NY-138, installed Statue of Liberty head, NPS/LOC"
    board["camera_registration"] = "UNREGISTERED / orthographic board only"
    board["workflow"] = "Reference for human sculpt; no camera-matching or 3D survey claim"

    world = bpy.data.worlds.new("Dark studio / archive board")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (.045, .055, .065, 1)
    board.world = world

    ink = bpy.data.materials.new("Archive board / luminous ivory lettering")
    ink.use_nodes = True
    ink_nodes = ink.node_tree.nodes
    ink_nodes.clear()
    emission_node = ink_nodes.new("ShaderNodeEmission")
    emission_node.inputs["Color"].default_value = (.88, .95, .93, 1.)
    output_node = ink_nodes.new("ShaderNodeOutputMaterial")
    ink.node_tree.links.new(emission_node.outputs["Emission"], output_node.inputs["Surface"])

    def type_label(name: str, content: str, position: tuple[float, float, float],
                   size: float) -> None:
        text = bpy.data.curves.new(name, "FONT")
        text.body = content
        text.size = size
        text.align_x = "CENTER"
        text.extrude = 0.0
        text.materials.append(ink)
        obj = bpy.data.objects.new(name, text)
        board.collection.objects.link(obj)
        obj.location = position
        obj.rotation_euler = (math.pi / 2, 0, 0)

    # A face-on display of photo cards, **not** an attempted projection or
    # camera registration. Image UVs retain the originals and are packed.
    for index, (photo, path) in enumerate(photos):
        image = bpy.data.images.load(str(path), check_existing=False)
        if image.size[0] < 400 or image.size[1] < 500:
            raise ValueError(f"image is unexpectedly small: {path}, {tuple(image.size)}")
        image.pack()
        image.filepath = f"//packed/{photo.filename}"
        h = 3.68
        w = h * image.size[0] / image.size[1]
        x = (index - 1) * 3.2
        mesh = bpy.data.meshes.new(f"Archive photo quad / {photo.survey_id}")
        mesh.from_pydata(
            [(-w / 2, 0, -h / 2), (w / 2, 0, -h / 2),
             (w / 2, 0, h / 2), (-w / 2, 0, h / 2)], [], [(0, 1, 2, 3)])
        mesh.update()
        uv = mesh.uv_layers.new(name="Original photo orientation")
        for polygon in mesh.polygons:
            for loop_index, uv_co in zip(polygon.loop_indices,
                                         ((0, 0), (1, 0), (1, 1), (0, 1))):
                uv.data[loop_index].uv = uv_co
        material = bpy.data.materials.new(f"Photo emission / {photo.survey_id}")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        nodes.clear()
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = image
        emission = nodes.new("ShaderNodeEmission")
        emission.inputs["Strength"].default_value = 1.0
        output = nodes.new("ShaderNodeOutputMaterial")
        color_output = tex.outputs["Color"]
        if photo.survey_id == "HAER NY-138-237":
            # The source transparency is very underexposed. Lift preview
            # shadows without changing the packed, original photo pixels.
            gamma = nodes.new("ShaderNodeGamma")
            gamma.inputs["Gamma"].default_value = .68
            material.node_tree.links.new(color_output, gamma.inputs["Color"])
            color_output = gamma.outputs["Color"]
        material.node_tree.links.new(color_output, emission.inputs["Color"])
        material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
        mesh.materials.append(material)
        obj = bpy.data.objects.new(f"SOURCE {photo.survey_id} | {photo.label}", mesh)
        board.collection.objects.link(obj)
        obj.location.x = x
        obj["public_domain_source"] = photo.loc_url
        obj["camera_status"] = "Unknown: no matched lens, pose or registered pixel grid"
        obj["display_only_gamma"] = .68 if photo.survey_id == "HAER NY-138-237" else 1.0
        type_label(f"Title / {photo.survey_id}", photo.label, (x, -.01, -2.13), .18)
        type_label(f"Accession / {photo.survey_id}", photo.survey_id,
                   (x, -.01, -2.40), .13)
        if photo.survey_id == "HAER NY-138-237":
            type_label("Shadow preview warning", "SHADOWS LIFTED / ORIGINAL PACKED",
                       (x, -.01, -2.59), .10)

    type_label("Banner", "ACTUAL INSTALLED HEAD  |  HAER 1984  |  UNREGISTERED PHOTOGRAPHS",
               (0., -.01, 2.17), .21)
    type_label("Gate warning", "NO FRONT CAMERA CALIBRATION  -  FACE LIKENESS HOLD",
               (0., -.01, -2.86), .18)

    camera_data = bpy.data.cameras.new("Archive contact board / not a survey camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 10.5
    camera = bpy.data.objects.new(camera_data.name, camera_data)
    board.collection.objects.link(camera)
    camera.location = (0., -12., 0.)
    camera.rotation_euler = (Vector((0., 0., 0.)) - camera.location).to_track_quat(
        "-Z", "Y").to_euler()
    board.camera = camera
    bpy.context.window.scene = board
    board.render.filepath = str((cfg.output / "haer-reference-board.png").resolve())
    bpy.ops.render.render(write_still=True, scene=board.name)

    bpy.context.window.scene = sculpt_scene
    bpy.ops.object.select_all(action="DESELECT")
    face = bpy.data.objects.get("liberty-head-face")
    if face:
        face.select_set(True)
        bpy.context.view_layer.objects.active = face
    bpy.ops.wm.save_as_mainfile(filepath=str((cfg.output / "liberty-face-sculpt-studio.blend").resolve()))
    print(f"Editable HAER face studio saved to {cfg.output} (face likeness HOLD)")


if __name__ == "__main__":
    main()
