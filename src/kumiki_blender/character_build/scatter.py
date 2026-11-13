"""Scatter feather, fur or scale cards over a mesh region.

The region is chosen by vertex colour (from ``project_colours``), an optional
height band and an optional front-to-back band (Y, the back is +Y). Each card is a tapered, slightly curved quad strip. Its
base sits on the surface, it points along a flow direction (by default down
the surface, like feathers and fur), and it lifts away from the surface by a
small angle so layers overlap. Output is one mesh of cards, separate from the
body, so an artist can delete, groom or convert it to hair curves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .checks import lab
from .mesh import Mesh


@dataclass(frozen=True)
class ScatterSpec:
    colour: tuple[int, int, int] | None = None   # select vertices near this sRGB colour
    max_delta_e: float = 18.0
    z_min_m: float = -np.inf
    z_max_m: float = np.inf
    y_min_m: float = -np.inf  # front (-Y) to back (+Y): a tail is behind the body
    y_max_m: float = np.inf
    density_per_m2: float = 20000.0
    length_m: float = 0.012
    width_m: float = 0.006
    lift_deg: float = 20.0
    flow: tuple[float, float, float] = (0.0, 0.0, -1.0)
    jitter_deg: float = 12.0
    seed: int = 0


def _card(length: float, width: float, segments: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Card in local (along, across, out) space, base at the origin."""
    t = np.linspace(0, 1, segments + 1)
    half = width / 2 * np.sin(np.pi * np.clip(t, 0.08, 1) * 0.92)  # narrow base, round body, near-point tip
    along, out = t * length, 0.25 * length * t ** 2  # curve away from the surface
    left = np.column_stack([along, -half, out])
    right = np.column_stack([along, half, out])
    verts = np.empty((2 * (segments + 1), 3))
    verts[0::2], verts[1::2] = left, right
    faces = []
    for i in range(segments):
        a, b, c, d = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
        faces += [(a, c, b), (b, c, d)]
    return verts, np.array(faces)


def scatter_cards(mesh: Mesh, spec: ScatterSpec) -> tuple[Mesh, dict]:
    rng = np.random.default_rng(spec.seed)
    normals_f, areas = mesh.face_normals()
    centres = mesh.vertices[mesh.faces].mean(axis=1)
    select = ((centres[:, 2] >= spec.z_min_m) & (centres[:, 2] <= spec.z_max_m)
              & (centres[:, 1] >= spec.y_min_m) & (centres[:, 1] <= spec.y_max_m))
    if spec.colour is not None:
        if mesh.colours is None:
            raise ValueError("Colour selection needs a mesh with vertex colours (run project first)")
        face_rgb = mesh.colours[mesh.faces].mean(axis=1).astype(np.uint8)
        distance = np.linalg.norm(lab(face_rgb) - lab(np.array([spec.colour], np.uint8)), axis=1)
        select &= distance <= spec.max_delta_e
    region_area = float(areas[select].sum())
    count = int(rng.poisson(region_area * spec.density_per_m2)) if region_area > 0 else 0
    if count == 0:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), int)), {"cards": 0, "region_area_m2": region_area}
    ids = np.nonzero(select)[0]
    chosen = rng.choice(ids, size=count, p=areas[ids] / areas[ids].sum())
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    tri = mesh.vertices[mesh.faces[chosen]]
    base = tri[:, 0] + u[:, None] * (tri[:, 1] - tri[:, 0]) + v[:, None] * (tri[:, 2] - tri[:, 0])
    normal = normals_f[chosen]
    flow = np.asarray(spec.flow, float)
    along = flow - (normal @ flow)[:, None] * normal
    weak = np.linalg.norm(along, axis=1) < 1e-3  # the surface faces along the flow: use any tangent
    if weak.any():
        helper = np.where(np.abs(normal[weak, 0:1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
        along[weak] = np.cross(normal[weak], helper)
    along /= np.linalg.norm(along, axis=1)[:, None]
    jitter = np.radians(rng.normal(0, spec.jitter_deg, count))
    across = np.cross(normal, along)
    along = np.cos(jitter)[:, None] * along + np.sin(jitter)[:, None] * across
    across = np.cross(normal, along)
    lift = np.radians(spec.lift_deg)
    out_dir = normal
    along_l = np.cos(lift) * along + np.sin(lift) * out_dir  # tilt away from the surface
    out_l = np.cross(along_l, across)
    size = rng.uniform(0.8, 1.2, count)
    card_v, card_f = _card(spec.length_m, spec.width_m)
    local = card_v[None] * size[:, None, None]
    world = (base[:, None] + 0.0005 * normal[:, None]
             + local[..., 0:1] * along_l[:, None] + local[..., 1:2] * across[:, None] + local[..., 2:3] * out_l[:, None])
    faces = (card_f[None] + (np.arange(count) * len(card_v))[:, None, None]).reshape(-1, 3)
    colours = None
    if mesh.colours is not None:
        colours = np.repeat(mesh.colours[mesh.faces[chosen, 0]], len(card_v), axis=0)
    cards = Mesh(world.reshape(-1, 3), faces, colours)
    return cards, {"cards": count, "region_area_m2": round(region_area, 6),
                   "density_per_m2": spec.density_per_m2, "faces": len(faces)}


def fur_mask(basecolor: np.ndarray, exclude: list[tuple[int, int, int]], max_delta_e: float = 18.0,
             grow_px: int = 3) -> np.ndarray:
    """A UV-space mask (uint8, 255 = fur) from a base-colour texture.

    Texels near any ``exclude`` colour (clothing, eyes, nose) get no fur. The excluded
    areas grow by ``grow_px`` so strands do not root on a seam of the vest or scarf.
    Unpainted texels (pure black between UV islands) get no fur either.
    """
    from scipy import ndimage

    h, w = basecolor.shape[:2]
    rgb = basecolor[..., :3].reshape(-1, 3)
    colours = lab(rgb)
    blocked = np.zeros(len(rgb), dtype=bool)
    for colour in exclude:
        blocked |= np.linalg.norm(colours - lab(np.array([colour], np.uint8)), axis=1) <= max_delta_e
    blocked = blocked.reshape(h, w) | (basecolor[..., :3].max(axis=2) < 8)
    if grow_px:
        blocked = ndimage.binary_dilation(blocked, iterations=grow_px)
    return np.where(blocked, 0, 255).astype(np.uint8)


def fur_weights(mesh: Mesh, *, clothing_top_m: float, eye_band_m: tuple[float, float], grow_rounds: int = 2
                ) -> np.ndarray:
    """Per-vertex fur weight (0 or 1) from colours projected from the reference drawings.

    A generator's own texture paints clothing in fur colours, but the drawings separate
    them. No fur grows on: green (a vest), saturated dark red (a scarf), saturated
    orange-brown leather and near-black, below ``clothing_top_m``; and near-black in the
    ``eye_band_m`` height band on the front (pupils). The bare areas grow by
    ``grow_rounds`` rings so strands do not root on a seam.
    """
    if mesh.colours is None:
        raise ValueError("fur_weights needs a mesh with colours projected from the drawings (run project)")
    colours = lab(mesh.colours)
    lightness, chroma = colours[:, 0], np.hypot(colours[:, 1], colours[:, 2])
    hue = np.degrees(np.arctan2(colours[:, 2], colours[:, 1])) % 360
    z, y = mesh.vertices[:, 2], mesh.vertices[:, 1]
    low = z < clothing_top_m
    green = (hue > 90) & (hue < 150) & (chroma > 12)
    red = ((hue < 40) | (hue > 340)) & (chroma > 30) & (lightness < 55) & low
    leather = (hue >= 40) & (hue < 70) & (chroma > 35) & (lightness < 55) & low
    dark = (lightness < 25) & (low | ((z > eye_band_m[0]) & (z < eye_band_m[1]) & (y < 0)))
    bare = green | red | leather | dark
    adjacency = mesh.adjacency()
    for _ in range(grow_rounds):
        bare = bare | ((adjacency @ bare.astype(float)) > 0)
    return np.where(bare, 0.0, 1.0)
