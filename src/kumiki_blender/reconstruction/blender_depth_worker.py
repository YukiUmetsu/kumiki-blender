"""Headless Blender BVH worker. Invoked by photographic_qa, not a public CLI."""
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

config = json.loads(Path(sys.argv[sys.argv.index('--') + 1]).read_text())
bpy.ops.wm.ply_import(filepath=config['mesh'])
obj = bpy.context.selected_objects[0]
bvh = BVHTree.FromObject(obj, bpy.context.evaluated_depsgraph_get(), epsilon=0.0)
samples = np.load(config['samples'])
output = Path(config['output'])
for index, camera in enumerate(config['cameras']):
    intr = camera['intrinsics']
    width = min(config['side'], intr['width'])
    height = max(1, round(intr['height'] * width / intr['width']))
    if height > config['side']:
        width = max(1, round(width * config['side'] / height))
        height = config['side']
    rotation = np.asarray(camera['rotation_3x3'])
    center = Vector(camera['location_colmap'])
    silhouette = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            u = (x + .5) * intr['width'] / width
            v = (y + .5) * intr['height'] / height
            local = np.array([(u - intr['cx_px']) / intr['fx_px'],
                              -(v - intr['cy_px']) / intr['fy_px'], -1.0])
            direction = Vector(rotation @ local).normalized()
            if bvh.ray_cast(center, direction)[0] is not None:
                silhouette[y, x] = 255
    visible = np.zeros(len(samples), dtype=np.uint8)
    for i, point in enumerate(samples):
        delta = Vector(point) - center
        distance = delta.length
        if distance == 0:
            continue
        hit = bvh.ray_cast(center, delta / distance, distance * 1.00001)[0]
        if hit is not None and abs((hit - center).length - distance) <= max(distance * 1e-5, 1e-6):
            visible[i] = 1
    np.savez_compressed(output / f'{index:04d}.npz', silhouette=silhouette, visible=visible)
    print(f'QA camera {index + 1}/{len(config["cameras"])}', flush=True)
