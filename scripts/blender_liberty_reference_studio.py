"""Pack authentic HAER head photographs into an editable Blender sculpt studio.

The archived photographs have unknown camera intrinsics and pose. They are
reference *boards*, deliberately in a separate scene, never projections onto
the illustrative head or registered likeness-gate input.

Run this script with Blender 4.5 after generating a sculpt ``.blend`` checkpoint.
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
    group: str = "pose"


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
        "HAER NY-138-238", "haer-238-left-ear-large.jpg", "LEFT EAR / REAR MASS",
        "https://www.loc.gov/pictures/item/ny1251.color.570007c/",
        "https://cdn.loc.gov/service/pnp/habshaer/ny/ny1200/ny1251/color/570007cv.jpg",
    ),
    ArchivedView(
        "HAER NY-138-239", "haer-239-left-crown-large.jpg", "LEFT CROWN / HAIR",
        "https://www.loc.gov/pictures/item/ny1251.color.570008c/",
        "https://cdn.loc.gov/service/pnp/habshaer/ny/ny1200/ny1251/color/570008cv.jpg",
    ),
    ArchivedView(
        "HAER NY-138-240", "haer-240-left-profile-large.jpg", "LEFT THREE-QUARTER",
        "https://www.loc.gov/pictures/item/ny1251.color.570009c/",
        "https://cdn.loc.gov/service/pnp/habshaer/ny/ny1200/ny1251/color/570009cv.jpg",
    ),
    ArchivedView(
        "HAER NY-138-241", "haer-241-left-profile-large.jpg", "LEFT PROFILE",
        "https://www.loc.gov/pictures/item/ny1251.color.570010c/",
        "https://cdn.loc.gov/service/pnp/habshaer/ny/ny1200/ny1251/color/570010cv.jpg",
    ),
)

DETAIL_VIEWS = (
    ArchivedView(
        "HAER NY-138-177", "haer-177-repaired-profile-large.jpg", "REPAIRED SIDE / 1985",
        "https://www.loc.gov/pictures/item/ny1251.photos.120435p/",
        "https://cdn.loc.gov/service/pnp/habshaer/ny/ny1200/ny1251/photos/120435pv.jpg",
        "detail",
    ),
    ArchivedView(
        "HAER NY-138-178", "haer-178-face-eye-repair-large.jpg", "LOW VIEW / NOSE / ORBITS",
        "https://www.loc.gov/pictures/item/ny1251.photos.120436p/",
        "https://cdn.loc.gov/service/pnp/habshaer/ny/ny1200/ny1251/photos/120436pv.jpg",
        "detail",
    ),
    ArchivedView(
        "HAER NY-138-179", "haer-179-right-eye-large.jpg", "RIGHT EYE / EYELID",
        "https://www.loc.gov/pictures/item/ny1251.photos.120437p/",
        "https://cdn.loc.gov/service/pnp/habshaer/ny/ny1200/ny1251/photos/120437pv.jpg",
        "detail",
    ),
    ArchivedView(
        "HAER NY-138-180", "haer-180-nose-scale-large.jpg", "LOW VIEW / SCALE STICK",
        "https://www.loc.gov/pictures/item/ny1251.photos.120438p/",
        "https://cdn.loc.gov/service/pnp/habshaer/ny/ny1200/ny1251/photos/120438pv.jpg",
        "detail",
    ),
    ArchivedView(
        "HAER NY-138-181", "haer-181-nose-before-large.jpg", "NOSTRIL / LIP RELIEF",
        "https://www.loc.gov/pictures/item/ny1251.photos.120439p/",
        "https://cdn.loc.gov/service/pnp/habshaer/ny/ny1200/ny1251/photos/120439pv.jpg",
        "detail",
    ),
)

ALL_VIEWS = ARCHIVED_VIEWS + DETAIL_VIEWS

# These are rotating *previews* of other physical objects.  In particular,
# none depicts a verified survey of the full-size copper monument in New York.
# Keep the viewer URLs in the .blend's text datablocks, not imported meshes.
REPLICA_VIEWER_REFERENCES = (
    ("Maurice Svay / Arts et Metiers, Paris / ~3m replica",
     "https://sketchfab.com/3d-models/statue-of-liberty-c461ed8724424ad99500fd058a0ab082",
     "Photogrammetry / 699k tris / study broad cloth and face planes"),
    ("jerryfisher / 19th-century bronze model",
     "https://sketchfab.com/3d-models/statue-of-liberty-6626abe1f3e8469a9d8f4b74d8aa2a71",
     "Photogrammetry / 4M tris / study cast folds and asymmetry"),
    ("3Dystopia / Luxembourg Gardens, Paris / bronze replica",
     "https://sketchfab.com/3d-models/statue-of-liberty-04a8e7b1ba9142ba8cebc6108cd5bb97",
     "Simplified photogrammetry / 20k tris / study full-body proportions"),
    ("3DSCANFR / Bartholdi 1889 Paris replica / premium viewer",
     "https://sketchfab.com/3d-models/statue-of-liberty-auguste-bartholdi-1889-0fc2c8744b654cc6bc44a567158c53aa",
     "580-photo scan / 8K textures / compare front and profile"),
)


def resolve_photos(reference_dir: Path) -> list[tuple[ArchivedView, Path]]:
    """Fail closed if any of the eleven installed-monument photos is absent."""
    if not reference_dir.is_dir():
        raise ValueError(f"missing HAER directory: {reference_dir}")
    resolved = []
    for photo in ALL_VIEWS:
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
    sculpt_scene["reference_sculpt_method"] = "MANUAL: no anatomy equations or copied third-party geometry"
    sculpt_scene["reference_board_scene"] = "02 HAER POSES / UNREGISTERED"
    sculpt_scene["anatomy_detail_scene"] = "03 HAER ANATOMY / UNREGISTERED"
    for photo, _path in photos:
        sculpt_scene[photo.survey_id] = photo.loc_url

    brief = bpy.data.texts.new("SCULPT BRIEF / actual Liberty face / HOLD")
    brief.write(
        "ACTUAL STATUE OF LIBERTY FACE - ANATOMICAL SCULPT BRIEF\n"
        "=================================================\n"
        "State: HOLD. The current head is not photo-faithful.\n"
        "These photographs are of the installed copper monument (HAER, 1984-85).\n"
        "Their camera poses and intrinsics are UNKNOWN; do not project them\n"
        "onto the mesh, score pixel matches or claim reconstruction.\n\n"
        "Sculpt order / work from primary mass to secondary folds:\n"
        "1. Establish real-target frontal, temporal, cheek and jaw volumes.\n"
        "   The old procedural face is slab-like; the CC0 donor is generic.\n"
        "2. Study the underside of brow and true hollow orbital transition.\n"
        "   Real eye sits deep under a broad continuous upper lid ridge.\n"
        "3. Rebuild nasal bridge, tip, alae, nostril shadow, and philtrum as\n"
        "   connected masses; compare actual depth against BOTH profiles.\n"
        "4. Give both lips genuine forward/underlying volume, a continuous\n"
        "   cupid-to-corner line, and a lower-lip-to-chin break.\n"
        "5. Rebuild masseter, jaw and ear transitions before engraving folds.\n"
        "6. Check front, both sides and three-quarter in *neutral clay*.\n"
        "   Find a licensed authentic scan or calibrate reliable multi-view\n"
        "   camera correspondence before releasing a photo-faithful GLB.\n\n"
        "02 HAER POSES: Six whole-head views, including hair and crown.\n"
        "03 HAER ANATOMY: Five details of orbital, alar and mouth surfaces.\n"
        "178-181 show restoration in progress; do not infer untouched\n"
        "copper from damage or repair patches. The 180 scale stick lies\n"
        "in oblique space: it is NOT a calibrated depth measurement.\n"
        "238 and 239 largely repeat a rear/side view. More photographs\n"
        "do not automatically mean a usable parallax baseline.\n"
    )
    brief.write(
        "\nREPLICA SCAN PREVIEWS / VISUAL STUDY ONLY\n"
        "-----------------------------------------\n"
        "The following models are scans of physical REPLICAS. Rotate the public\n"
        "previews to study sculptural anatomy and fold volume, but DO NOT\n"
        "import, trace, extract, redistribute, or blend their geometry.\n"
        "Their faces and folds do not prove the installed monument's shape.\n\n"
    )
    for label, url, note in REPLICA_VIEWER_REFERENCES:
        brief.write(f"{label}\n  {url}\n  {note}\n\n")
    brief.write(
        "NON-PROCEDURAL SCULPT CHECKPOINT\n"
        "-------------------------------\n"
        "Work on an editable mesh in Sculpt Mode; establish primary face\n"
        "planes and true three-dimensional profiles with hand-directed\n"
        "sculpting. Keep the previous code-generated mesh as a visible HOLD\n"
        "comparison, not the foundation of a released replacement.\n"
        "Build proper brow overhang, nasal alae, lip undercuts, jaw, hair\n"
        "volume and deep cross-body copper drapery from multi-angle evidence.\n"
        "Do not export a new final GLB until an independently reviewed\n"
        "front/profile/oblique clay proof has improved substantially.\n"
    )
    sculpt_scene["reference_count"] = len(photos)

    ink = bpy.data.materials.new("Archive board / luminous ivory lettering")
    ink.use_nodes = True
    ink_nodes = ink.node_tree.nodes
    ink_nodes.clear()
    emission_node = ink_nodes.new("ShaderNodeEmission")
    emission_node.inputs["Color"].default_value = (.88, .95, .93, 1.)
    output_node = ink_nodes.new("ShaderNodeOutputMaterial")
    ink.node_tree.links.new(emission_node.outputs["Emission"], output_node.inputs["Surface"])

    def type_label(board, name: str, content: str,
                   position: tuple[float, float, float], size: float) -> None:
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

    def make_board(name: str, sources: list[tuple[ArchivedView, Path]],
                   banner: str, caution: str, filename: str) -> None:
        """Two-dimensional image cards, never fitted photogrammetry cameras."""
        board = bpy.data.scenes.new(name)
        board.render.engine = "CYCLES"
        board.cycles.samples = 8
        board.render.resolution_x = 1920
        board.render.resolution_y = 1280
        board.render.resolution_percentage = 100
        board.view_settings.view_transform = "Standard"
        board.render.image_settings.file_format = "PNG"
        board["source"] = "1984-85 HAER NY-138, actual installed head, NPS/LOC"
        board["camera_registration"] = "NONE / photo contact board / pose unknown"
        board["workflow"] = "Anatomical sculpt aid; not a 3D survey or measured comparison"
        board["photo_count"] = len(sources)
        world = bpy.data.worlds.new(f"Dark archive board / {name}")
        world.use_nodes = True
        world.node_tree.nodes["Background"].inputs["Color"].default_value = (
            .045, .055, .065, 1)
        board.world = world

        for index, (photo, path) in enumerate(sources):
            image = bpy.data.images.load(str(path), check_existing=False)
            if image.size[0] < 400 or image.size[1] < 500:
                raise ValueError(f"image unexpectedly small: {path}, {tuple(image.size)}")
            image.pack()
            image.filepath = f"//packed/{photo.filename}"
            # Leave clear label gutters below tall portrait negatives.
            h = min(3.05, 3.20 * image.size[1] / image.size[0])
            w = min(3.4, h * image.size[0] / image.size[1])
            # Top row of three; bottom row of two or three, with space
            # reserved for the image accession and repair-era qualifiers.
            row = index // 3
            row_count = min(3, len(sources) - 3 * row)
            x = (index % 3 - (row_count - 1) / 2) * 4.1
            z = 2.25 - 4.45 * row
            mesh = bpy.data.meshes.new(f"Archive photo quad / {photo.survey_id}")
            mesh.from_pydata(
                [(-w / 2, 0, -h / 2), (w / 2, 0, -h / 2),
                 (w / 2, 0, h / 2), (-w / 2, 0, h / 2)], [], [(0, 1, 2, 3)])
            mesh.update()
            uv = mesh.uv_layers.new(name="Unwarped archive pixels")
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
                # Display-only lift; original source bytes are packed.
                gamma = nodes.new("ShaderNodeGamma")
                gamma.inputs["Gamma"].default_value = .68
                material.node_tree.links.new(color_output, gamma.inputs["Color"])
                color_output = gamma.outputs["Color"]
            material.node_tree.links.new(color_output, emission.inputs["Color"])
            material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
            mesh.materials.append(material)
            obj = bpy.data.objects.new(f"SOURCE {photo.survey_id} | {photo.label}", mesh)
            board.collection.objects.link(obj)
            obj.location = (x, 0., z)
            obj["public_domain_source"] = photo.loc_url
            obj["camera_status"] = "Unknown: no lens, pose or registered pixel grid"
            obj["display_only_gamma"] = .68 if photo.survey_id == "HAER NY-138-237" else 1.0
            obj["restoration_photo"] = photo.group == "detail"
            type_label(board, f"Title / {photo.survey_id}", photo.label,
                       (x, -.01, z - 1.79), .17)
            type_label(board, f"Accession / {photo.survey_id}", photo.survey_id,
                       (x, -.01, z - 2.02), .13)
        type_label(board, "Banner", banner, (0., -.01, 4.55), .23)
        type_label(board, "Gate warning", caution, (0., -.01, -4.80), .16)

        camera_data = bpy.data.cameras.new(f"Contact board camera / {name}")
        camera_data.type = "ORTHO"
        # Blender's ortho_scale spans the horizontal field for a landscape
        # render; 17 m gives this 3:2 board >11 m of vertical clearance.
        camera_data.ortho_scale = 17.0
        camera = bpy.data.objects.new(camera_data.name, camera_data)
        board.collection.objects.link(camera)
        camera.location = (0., -12., 0.)
        camera.rotation_euler = (Vector((0., 0., 0.)) - camera.location).to_track_quat(
            "-Z", "Y").to_euler()
        board.camera = camera
        bpy.context.window.scene = board
        board.render.filepath = str((cfg.output / filename).resolve())
        bpy.ops.render.render(write_still=True, scene=board.name)

    make_board("02 HAER POSES / UNREGISTERED",
               [item for item in photos if item[0].group == "pose"],
               "INSTALLED LIBERTY HEAD | 1984 | SIX UNREGISTERED VIEWS",
               "NO FRONT SURVEY CAMERA  /  PHOTO-FAITHFUL LIKENESS: HOLD",
               "haer-pose-board.png")
    make_board("03 HAER ANATOMY / UNREGISTERED",
               [item for item in photos if item[0].group == "detail"],
               "INSTALLED LIBERTY FACE | 1985 | RESTORATION DETAILS",
               "DAMAGE / REPAIRS VISIBLE  -  NO DEPTH OR CAMERA CALIBRATION",
               "haer-anatomy-board.png")

    bpy.context.window.scene = sculpt_scene
    bpy.ops.object.select_all(action="DESELECT")
    face = (bpy.data.objects.get("liberty-head-face") or
            bpy.data.objects.get("Liberty sculpt candidate / continuous whole-head anatomy"))
    if face:
        face.select_set(True)
        bpy.context.view_layer.objects.active = face
    bpy.ops.wm.save_as_mainfile(filepath=str((cfg.output / "liberty-face-sculpt-studio.blend").resolve()))
    print(f"Editable HAER face studio saved to {cfg.output} (face likeness HOLD)")


if __name__ == "__main__":
    main()
