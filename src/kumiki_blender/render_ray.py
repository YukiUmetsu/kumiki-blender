"""Convert annotated render pixels into Blender camera-local rays.

Uses the actual Camera.view_frame(scene=...) corners, which encode the
camera's shift, lens, orthographic scale and output aspect ratio. This avoids
hard-coded world-axis assumptions that break for tilted and profile cameras.
No Blender dependency: the geometry can be regression-tested with CPython.
"""

import math


def pixel_camera_ray(pixel, image_size, frame_corners, camera_type):
    """Return (camera-local origin, unit direction) for a render pixel.

    pixel is in continuous upper-left-origin image coordinates (centers at
    integer + 0.5). frame_corners are the four camera-local vectors from
    bpy.types.Camera.view_frame(scene=scene); their order is irrelevant.
    The returned ray must be transformed by camera.matrix_world before
    intersecting the evaluated mesh.
    """
    if camera_type not in ("ORTHO", "PERSP"):
        raise ValueError("Only orthographic and perspective cameras are supported")
    if len(pixel) != 2 or len(image_size) != 2 or len(frame_corners) != 4:
        raise ValueError("Expected a pixel, image size and four camera-frame corners")
    try:
        u, v = (float(pixel[0]), float(pixel[1]))
        width, height = (float(image_size[0]), float(image_size[1]))
        frame = [tuple(float(p) for p in corner) for corner in frame_corners]
    except (ValueError, TypeError) as error:
        raise ValueError("Camera ray inputs must be numeric") from error
    if any(len(corner) != 3 for corner in frame):
        raise ValueError("Camera-frame corners must be three-dimensional")
    if not all(math.isfinite(n) for n in (u, v, width, height, *(x for c in frame for x in c))):
        raise ValueError("Camera ray inputs must be finite")
    if width <= 0 or height <= 0 or not (0 <= u <= width and 0 <= v <= height):
        raise ValueError("Pixel must be inside an image with positive dimensions")
    xs = [p[0] for p in frame]
    ys = [p[1] for p in frame]
    zs = [p[2] for p in frame]
    if max(xs) <= min(xs) or max(ys) <= min(ys) or max(zs)-min(zs) > 1e-5:
        raise ValueError("Invalid planar camera-frame corners")
    x = min(xs) + (u / width) * (max(xs)-min(xs))
    y = max(ys) - (v / height) * (max(ys)-min(ys))
    if camera_type == "ORTHO":
        return (x, y, 0.0), (0.0, 0.0, -1.0)
    z = sum(zs) / len(zs)
    length = math.sqrt(x*x + y*y + z*z)
    if length <= 1e-9 or z >= 0:
        raise ValueError("Perspective frame must lie in front of the camera")
    return (0.0, 0.0, 0.0), (x/length, y/length, z/length)
