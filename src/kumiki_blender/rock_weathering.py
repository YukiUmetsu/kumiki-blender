"""Deterministic, geometry-level weathering for reusable carved rock solids.

No image textures, random process state or Blender dependency. Noise is smooth
in world-meter coordinates and is combined with horizontal stratification so
large rock forms cast actual raking-light shadows even without normal maps.
"""

from __future__ import annotations

import math

from .architecture_geometry import GeometryError, Mesh, _finite_number


def _hash3(x: int, y: int, z: int, seed: int) -> float:
    n = (x*374761393 + y*668265263 + z*2246822519 + seed*3266489917) & 0xffffffff
    n = ((n ^ (n >> 13))*1274126177) & 0xffffffff
    return (((n ^ (n >> 16)) & 0xffffff) / 0x7fffff) - 1


def _noise3(x: float, y: float, z: float, seed: int) -> float:
    a, b, c = math.floor(x), math.floor(y), math.floor(z)
    u, v, w = x-a, y-b, z-c
    u, v, w = (t*t*(3-2*t) for t in (u, v, w))
    result = 0.0
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                result += _hash3(a+i, b+j, c+k, seed) * (
                    (u if i else 1-u)*(v if j else 1-v)*(w if k else 1-w))
    return result


def rock_height_field(x: float, y: float, z: float, *, seed: int = 0,
                      strata_spacing: float = 0.35) -> float:
    """Bounded, repeatable mineral strata plus two scales of rough stone grain.

    Values are roughly [-1, 1] and vary continuously with 3D position, which
    permits nearby objects to share geological rhythms without seams in noise.
    """
    spacing = _finite_number(strata_spacing, "strata_spacing", 0.03, 8)
    if type(seed) is not int or not 0 <= seed <= 100000:
        raise GeometryError("Rock seed must be an integer from 0 to 100000")
    broad = _noise3(x*.94, y*.94, z*.94, seed)
    fine = _noise3(x*3.7, y*3.7, z*3.7, seed+13)
    grain = _noise3(x*11.4, y*11.4, z*11.4, seed+31)
    # Bedding planes follow physical height with gentle local distortion;
    # sharp/mineral bands interrupt the overly smooth tubular silhouette.
    phase = (z + broad*.105 + fine*.028) / spacing * math.tau
    layers = math.sin(phase) * (0.66 + 0.34*math.sin(phase*.5))
    return .40*broad + .23*fine + .10*grain + .27*layers


def erode_loft(mesh: Mesh, *, axis: str, radial_segments: int,
               strength: float, seed: int = 0, strata_spacing: float = 0.35) -> Mesh:
    """Perturb a closed loft's rings radially, leaving its end caps planar.

    Each vertex remains in the same manifold topology. Flat-cap rings stay
    fixed; interior rings are displaced according to local cross-section
    normals. Use a modest strength relative to the thinnest part radius.
    """
    if axis not in ("x", "y", "z"):
        raise GeometryError("Eroded loft axis must be x, y or z")
    if type(radial_segments) is not int or not 8 <= radial_segments <= 128:
        raise GeometryError("radial_segments must be an integer in [8, 128]")
    amp = _finite_number(strength, "erosion strength", 0, 2)
    _finite_number(strata_spacing, "strata_spacing", 0.03, 8)
    if not isinstance(mesh, Mesh) or len(mesh.vertices) % radial_segments:
        raise GeometryError("Erosion expects complete closed loft rings")
    count = len(mesh.vertices)//radial_segments
    if count < 3:
        raise GeometryError("Eroded loft requires at least three axial rings")
    if type(seed) is not int or not 0 <= seed <= 100000:
        raise GeometryError("Rock seed must be an integer from 0 to 100000")
    if amp == 0:
        return mesh

    altered = []
    for ring in range(count):
        group = mesh.vertices[ring*radial_segments:(ring+1)*radial_segments]
        if axis == "x":
            center_a = sum(v[1] for v in group)/radial_segments
            center_b = sum(v[2] for v in group)/radial_segments
        elif axis == "y":
            center_a = sum(v[2] for v in group)/radial_segments
            center_b = sum(v[0] for v in group)/radial_segments
        else:
            center_a = sum(v[0] for v in group)/radial_segments
            center_b = sum(v[1] for v in group)/radial_segments
        fade = min(1, ring/3, (count-1-ring)/3)
        for x, y, z in group:
            a, b = ((y, z) if axis == "x" else
                    (z, x) if axis == "y" else (x, y))
            va, vb = a-center_a, b-center_b
            magnitude = math.hypot(va, vb)
            value = (amp * fade * rock_height_field(
                x, y, z, seed=seed, strata_spacing=strata_spacing)
                     / max(magnitude, 1e-6))
            aa, bb = a + va*value, b + vb*value
            altered.append(((x, aa, bb) if axis == "x" else
                            (bb, y, aa) if axis == "y" else (aa, bb, z)))
    return Mesh(tuple(altered), mesh.faces)


def rough_bedrock_slab(length: float, width: float, height: float, *,
                       roughness: float = 0.10, seed: int = 0,
                       x_segments: int = 24, y_segments: int = 12) -> Mesh:
    """Watertight topographic limestone plinth with genuinely uneven top/edges.

    A shared grid on the top and bottom is stitched around all four sides.
    The top receives multiscale millimeter-to-decimeter relief; the irregular
    silhouette removes the artificial razor-straight edges of a stone box.
    """
    length = _finite_number(length, "bedrock length", 0.5, 1000)
    width = _finite_number(width, "bedrock width", 0.5, 1000)
    height = _finite_number(height, "bedrock height", 0.1, 50)
    roughness = _finite_number(roughness, "bedrock roughness", 0, 0.5)
    if type(x_segments) is not int or not 4 <= x_segments <= 128:
        raise GeometryError("bedrock x_segments must be in [4, 128]")
    if type(y_segments) is not int or not 4 <= y_segments <= 128:
        raise GeometryError("bedrock y_segments must be in [4, 128]")
    if type(seed) is not int or not 0 <= seed <= 100000:
        raise GeometryError("Rock seed must be an integer from 0 to 100000")
    if roughness > height*.45:
        raise GeometryError("Bedrock roughness must not exceed 45% of height")

    vertices = []
    count_x, count_y = x_segments+1, y_segments+1
    count = count_x*count_y
    for upper in (False, True):
        for j in range(count_y):
            for i in range(count_x):
                x = (i/x_segments-.5)*length
                y = (j/y_segments-.5)*width
                # Only the perimeter changes horizontal position, avoiding
                # artificially straight quarry-stone rectangle silhouettes.
                at_x_edge = i == 0 or i == x_segments
                at_y_edge = j == 0 or j == y_segments
                if at_x_edge:
                    x += roughness*.48*_noise3(x*1.7, y*1.7, .33, seed+44)
                if at_y_edge:
                    y += roughness*.48*_noise3(x*1.7, y*1.7, .71, seed+45)
                relief = roughness*(.55*_noise3(x*1.1, y*1.1, 1.1, seed) +
                                    .28*_noise3(x*4.2, y*4.2, 2.3, seed+9) +
                                    .17*_noise3(x*12, y*12, 3.1, seed+19))
                z = height + relief if upper else 0
                vertices.append((x, y, z))

    def idx(i: int, j: int, upper: bool) -> int:
        return (count if upper else 0) + j*count_x + i

    faces: list[tuple[int, ...]] = []
    for j in range(y_segments):
        for i in range(x_segments):
            a, b = idx(i, j, True), idx(i+1, j, True)
            c, d = idx(i+1, j+1, True), idx(i, j+1, True)
            faces.append((a, b, c, d))
            a, b = idx(i, j, False), idx(i+1, j, False)
            c, d = idx(i+1, j+1, False), idx(i, j+1, False)
            faces.append((d, c, b, a))
    # Bottom/front, right, back and left walls wind outward in that order.
    perimeter = ([idx(i, 0, False) for i in range(count_x)] +
                 [idx(x_segments, j, False) for j in range(1, count_y)] +
                 [idx(i, y_segments, False) for i in range(x_segments-1, -1, -1)] +
                 [idx(0, j, False) for j in range(y_segments-1, 0, -1)])
    for a, b in zip(perimeter, (*perimeter[1:], perimeter[0])):
        faces.append((a, b, b+count, a+count))
    return Mesh(tuple(vertices), tuple(faces))
