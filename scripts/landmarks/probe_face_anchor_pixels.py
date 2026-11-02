"""Probe 2D render annotations against the *actual Blender QA camera*.

The previous script assumed a fixed front-facing world-axis camera and guessed
image size. That silently generated wrong 3D anchors for profile/low/oblique
views, or when a saved .blend had its QA camera removed after rendering.

Use model_shot and model_image_size in the annotation JSON, or provide
--qa-shot and --model-image-size explicitly. The four QA shot presets
must match christ_redeemer.py:views(); all non-default cameras can instead
be selected by --camera-name from a saved Blender scene.

  blender -b study.blend --python scripts/landmarks/probe_face_anchor_pixels.py \
    -- --input annotations.json --output anchors-3d.json

All model pixels are raycast against the evaluated mesh. The hit is projected
back to the model camera to check calibration. Occluded anchors are rejected.
"""

import argparse
import json
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from kumiki_blender.render_ray import pixel_camera_ray

# Keep synchronized with christ_redeemer.py:views().
QA_HEAD_SHOTS = {
    "head_front": ((0,-19,36),(0,0,36),6.4),
    "head_low": ((0,-18,32.5),(0,0,36),6.4),
    "head_profile": ((19,0,36),(0,0,36),6.4),
    "head_three_quarter": ((14,-15,36),(0,0,36),6.4),
}


def choose_camera(scene, options, source):
    if options.camera_name:
        camera = bpy.data.objects.get(options.camera_name)
        if camera is None or camera.type != "CAMERA":
            raise ValueError("Camera not found: " + options.camera_name)
        return camera, "scene_camera"
    shot = options.qa_shot or source.get("model_shot")
    if shot not in QA_HEAD_SHOTS:
        raise ValueError("Specify an authentic --camera-name or a model_shot "
                         "from: " + ", ".join(QA_HEAD_SHOTS))
    position, target, scale = QA_HEAD_SHOTS[shot]
    bpy.ops.object.camera_add(location=position)
    camera = bpy.context.object
    camera.name = "Temporary calibrated QA " + shot
    camera.rotation_euler = (Vector(target)-camera.location).to_track_quat("-Z","Y").to_euler()
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = scale
    return camera, shot


def probe(scene, head, camera, image_size, entries, allow_occluded=False):
    if camera.data.type not in ("ORTHO", "PERSP"):
        raise ValueError("Only orthographic and perspective camera probes are supported")
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = head.evaluated_get(depsgraph)
    matrix = evaluated.matrix_world.copy()
    inverse = matrix.inverted_safe()
    frame = camera.data.view_frame(scene=scene)
    results = []
    for entry in entries:
        local_origin, local_direction = pixel_camera_ray(
            entry["model_uv"], image_size, frame, camera.data.type)
        world_origin = camera.matrix_world @ Vector(local_origin)
        world_direction = (camera.matrix_world.to_3x3() @ Vector(local_direction)).normalized()
        object_origin = inverse @ world_origin
        object_direction = (inverse.to_3x3() @ world_direction).normalized()
        success, hit, normal, _ = evaluated.ray_cast(object_origin, object_direction)
        if not success:
            raise ValueError("No head surface under landmark " + entry["name"])
        world_hit = matrix @ hit
        # Catch wrong view transforms, incorrect image dimensions, and
        # annotations that lie on another visible object such as hair.
        uv = world_to_camera_view(scene, camera, world_hit)
        projected = (uv.x * image_size[0], (1-uv.y) * image_size[1])
        pixel_error = ((projected[0]-entry["model_uv"][0])**2 +
                       (projected[1]-entry["model_uv"][1])**2)**.5
        if pixel_error > 0.6 or uv.z <= 0:
            raise ValueError("Model pixel reprojection failed for %s (%.2f px)" %
                             (entry["name"],pixel_error))
        visible, scene_hit, _, _, scene_object, _ = scene.ray_cast(
            depsgraph, world_origin, world_direction)
        occluded = bool(visible and scene_object.original != head and
                        (scene_hit-world_origin).length < (world_hit-world_origin).length-.002)
        if occluded and not allow_occluded:
            raise ValueError("Anchor %s is occluded by %s; mark a visible face "
                             "pixel instead" % (entry["name"],scene_object.name))
        results.append({"name":entry["name"],
                        "world_xyz":[round(float(x),6) for x in world_hit],
                        "model_uv":entry["model_uv"],
                        "photo_uv":entry["photo_uv"],
                        "model_reprojection_error_px":round(pixel_error,4),
                        "occluded":occluded})
    return results


def main():
    args = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--camera-name",help="Existing scene camera; overrides QA shot")
    parser.add_argument("--qa-shot",choices=tuple(QA_HEAD_SHOTS))
    parser.add_argument("--model-image-size",type=int,nargs=2,metavar=("WIDTH","HEIGHT"))
    parser.add_argument("--allow-occluded",action="store_true",
                        help="Retain flagged, untrustworthy occluded anchors for debugging")
    options = parser.parse_args(args)
    source = json.loads(options.input.read_text())
    size = options.model_image_size or source.get("model_image_size")
    if not size or len(size)!=2 or min(size)<=0:
        parser.error("Model render dimensions required: model_image_size or --model-image-size")
    if not source.get("landmarks"):
        parser.error("At least one annotated landmark is required")
    scene = bpy.context.scene
    head = next((obj for obj in bpy.data.objects if
                 obj.type=="MESH" and obj.name.startswith("Head | anatomical quad-topology")),None)
    if head is None:
        raise ValueError("Load an anatomical clay .blend before probing anchors")
    old_size=(scene.render.resolution_x,scene.render.resolution_y,
              scene.render.resolution_percentage)
    camera=None
    try:
        scene.render.resolution_x=int(size[0])
        scene.render.resolution_y=int(size[1])
        scene.render.resolution_percentage=100
        camera, shot = choose_camera(scene,options,source)
        results = probe(scene,head,camera,size,source["landmarks"],options.allow_occluded)
        report={"model_revision":source.get("model_revision"),
                "subject_photo_url":source.get("subject_photo_url"),
                "image_size":source["image_size"],
                "model_image_size":size,
                "model_shot":shot,
                "model_camera_type":camera.data.type,
                "camera_matrix_world":[[round(float(v),8) for v in row]
                                       for row in camera.matrix_world],
                "landmarks":results,
                "warning":"Unregistered, manually annotated source photograph; "
                          "valid model-ray geometry does not establish target likeness."}
        options.output.parent.mkdir(parents=True,exist_ok=True)
        options.output.write_text(json.dumps(report,indent=2)+"\n")
        print("HEAD_ANCHORS "+json.dumps({"count":len(results),
                                          "model_shot":shot,
                                          "output":str(options.output)}))
    finally:
        scene.render.resolution_x,scene.render.resolution_y,scene.render.resolution_percentage=old_size
        if camera is not None and camera.name.startswith("Temporary calibrated QA "):
            bpy.data.objects.remove(camera,do_unlink=True)


if __name__ == "__main__":
    main()
