"""Synthetic 3D figure with exact orthographic reference views, for character_build tests."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from kumiki_blender.character_build.mesh import Mesh
from kumiki_blender.character_build.views import camera

SKIN, BAG, NOSE = (175, 125, 75), (125, 35, 45), (45, 40, 40)
HEIGHT_M = 1.05  # ear tips; the soles are at z = 0


def ellipsoids(widen: float = 1.0) -> list[tuple[tuple, tuple, int]]:
    """(centre, radii, label) of the figure. Label 1 body, 3 nose."""
    return [((0, 0, .36), (.2 * widen, .15, .3), 1), ((0, 0, .76), (.22, .18, .2), 1),
            ((-.11, 0, .96), (.04, .03, .09), 1), ((.11, 0, .96), (.04, .03, .09), 1),
            ((-.08, -.05, .045), (.06, .09, .045), 1), ((.08, -.05, .045), (.06, .09, .045), 1),
            ((0, -.19, .74), (.03, .04, .03), 3)]


def figure_points(widen: float = 1.0, bag: float = 1.0, step: float = 1.05 / 180):
    """Voxel centres and labels. The bag (label 2) hangs on the figure's right side (-X)."""
    g = np.arange(-0.5, 0.5, step) + step / 2
    x, y, z = np.meshgrid(g, g, np.arange(0, 1.05, step) + step / 2, indexing="ij")
    labels = np.zeros(x.shape, np.uint8)
    for centre, radii, label in ellipsoids(widen):
        inside = sum(((a - c) / r) ** 2 for a, c, r in zip((x, y, z), centre, radii)) <= 1
        labels[inside & (labels == 0)] = label
    sack = (x > -.18 - .12 * bag) & (x < -.17) & (np.abs(y) < .07 * bag) & (z > .22) & (z < .36)
    labels[sack] = 2
    keep = labels > 0
    return np.column_stack([x[keep], y[keep], z[keep]]), labels[keep]


def render_view(points, labels, view: str, *, clay=False, margin=(60, 40), size=(1024, 1536)) -> Image.Image:
    """Exact orthographic view on a flat #D9D9D9 background, nearest point wins."""
    cam = camera(view)
    u, depth = points @ cam.right, points @ cam.toward
    order = np.argsort(depth)
    pixel = 1.06 / 170
    cols = np.floor((u[order] + 0.55) / pixel).astype(int)
    rows = np.floor((1.06 - points[order, 2]) / pixel).astype(int)
    colours = np.array([(0, 0, 0)] + [(150, 150, 150)] * 3 if clay else [(0, 0, 0), SKIN, BAG, NOSE], np.uint8)
    small = np.full((182, 190, 3), 217, np.uint8)
    for dr in (0, 1):
        for dc in (0, 1):
            small[rows + dr, cols + dc] = colours[labels[order]]
    image = Image.fromarray(small).resize((190 * 6, 182 * 6), Image.NEAREST)
    arr = np.asarray(image).astype(int)
    ys, xs = np.nonzero(np.abs(arr - 217).max(axis=2) > 10)
    canvas = Image.new("RGB", size, (217, 217, 217))
    canvas.paste(image.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)), margin)
    return canvas


MARGINS = {"front": (60, 40), "back": (300, 120), "right": (150, 60), "left": (90, 200),
           "front_right": (120, 90), "back_left": (80, 50), "front_left": (100, 70), "back_right": (130, 110)}


def write_views(folder: Path, names=tuple(MARGINS), **figure) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    points, labels = figure_points(**figure)
    for name in names:
        render_view(points, labels, name, margin=MARGINS[name]).save(folder / f"{name}.png")


def ellipsoid_mesh(centre=(0, 0, .5), radii=(.25, .2, .5), subdivisions: int = 3) -> Mesh:
    """Icosphere scaled to an ellipsoid."""
    t = (1 + 5 ** 0.5) / 2
    verts = np.array([(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t),
                      (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)], float)
    faces = np.array([(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
                      (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
                      (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)])
    for _ in range(subdivisions):
        edges = {}
        verts = list(verts)

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in edges:
                verts.append((np.asarray(verts[a]) + np.asarray(verts[b])) / 2)
                edges[key] = len(verts) - 1
            return edges[key]

        new = []
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        verts, faces = np.array(verts), np.array(new)
    verts /= np.linalg.norm(verts, axis=1)[:, None]
    return Mesh(verts * np.array(radii) + np.array(centre), faces)


def signed_volume(mesh: Mesh) -> float:
    a, b, c = (mesh.vertices[mesh.faces[:, i]] for i in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)


def glb_bytes(mesh: Mesh, translation=(0.0, 0.0, 0.0), nested: bool = False) -> bytes:
    """Minimal GLB. The translation sits on the mesh node, or on a parent node when ``nested``.

    Positions are written in glTF Y-up axes.
    """
    points = np.column_stack([mesh.vertices[:, 0], mesh.vertices[:, 2], -mesh.vertices[:, 1]]).astype("<f4")
    indices = mesh.faces.astype("<u4")
    binary = points.tobytes() + indices.tobytes()
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
           "nodes": ([{"translation": list(translation), "children": [1]}, {"mesh": 0}] if nested
                     else [{"mesh": 0, "translation": list(translation)}]),
           "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
           "buffers": [{"byteLength": len(binary)}],
           "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": points.nbytes},
                           {"buffer": 0, "byteOffset": points.nbytes, "byteLength": indices.nbytes}],
           "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(points), "type": "VEC3"},
                         {"bufferView": 1, "componentType": 5125, "count": indices.size, "type": "SCALAR"}]}
    text = json.dumps(doc).encode()
    text += b" " * (-len(text) % 4)
    binary += b"\0" * (-len(binary) % 4)
    body = struct.pack("<II", len(text), 0x4E4F534A) + text + struct.pack("<II", len(binary), 0x004E4942) + binary
    return struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body
