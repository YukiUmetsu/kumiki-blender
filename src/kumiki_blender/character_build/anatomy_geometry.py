"""Deterministic, source-anchored 3D anatomy geometry without Blender/ML dependencies.

These are *authoring primitives*, not a claim that an unobserved surface was
reconstructed correctly. All dimensions and stations come from reviewed part
plans; no species default or implicit symmetry is imposed.
"""
from __future__ import annotations

import math
from collections import Counter

Vec = tuple[float, float, float]


def add(a: Vec, b: Vec) -> Vec:
    return tuple(x + y for x, y in zip(a, b))


def sub(a: Vec, b: Vec) -> Vec:
    return tuple(x - y for x, y in zip(a, b))


def mul(a: Vec, n: float) -> Vec:
    return tuple(x * n for x in a)


def dot(a: Vec, b: Vec) -> float:
    return sum(x * y for x, y in zip(a, b))


def cross(a: Vec, b: Vec) -> Vec:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def length(a: Vec) -> float:
    return math.sqrt(dot(a, a))


def unit(a: Vec) -> Vec:
    n = length(a)
    if n < 1e-9:
        raise ValueError("A tangent, direction or normal vector is degenerate")
    return mul(a, 1.0 / n)


def catmull(a, b, c, d, t: float):
    """Interpolating uniform cardinal spline; endpoints are exact control points."""
    return (0.5 * (2*b + (-a+c)*t + (2*a-5*b+4*c-d)*t*t
                   + (-a+3*b-3*c+d)*t*t*t))


def interpolate(points: list, subdivisions: int, closed: bool = False) -> list:
    """Interpolate lists of coordinates or scalars while retaining authored anchors."""
    n = len(points)
    if n < (3 if closed else 2):
        raise ValueError("Spline needs at least 2 points (3 for a closed contour)")
    size = len(points[0]) if isinstance(points[0], (tuple, list)) else 0
    if any((len(p) if isinstance(p, (tuple, list)) else 0) != size for p in points):
        raise ValueError("Spline points must have equal dimensions")

    def interpolate_pair(left, t):
        if not size:
            return catmull(left[0], left[1], left[2], left[3], t)
        return tuple(catmull(left[0][k], left[1][k], left[2][k],
                             left[3][k], t) for k in range(size))

    samples = []
    count = n if closed else n - 1
    for i in range(count):
        if closed:
            indices = ((i-1) % n, i % n, (i+1) % n, (i+2) % n)
        else:
            indices = (max(0, i-1), i, i+1, min(n-1, i+2))
        pair = [points[j] for j in indices]
        for step in range(subdivisions):
            samples.append(interpolate_pair(pair, step / subdivisions))
    if not closed:
        samples.append(points[-1])
    return samples


def _mesh(vertices: list[Vec], faces: list[list[int]], uvs: list[tuple[float, float]],
          *, closed: bool, name: str) -> dict:
    if not vertices or not faces or len(vertices) != len(uvs):
        raise ValueError(f"{name}: invalid vertex/UV/face mesh")
    for v in vertices:
        if not all(math.isfinite(k) for k in v):
            raise ValueError(f"{name}: non-finite mesh coordinate")
    counts = Counter()
    area = 0.0
    signed_volume = 0.0
    for face in faces:
        if len(face) < 3 or len(set(face)) != len(face) or any(i < 0 or i >= len(vertices) for i in face):
            raise ValueError(f"{name}: invalid face indices")
        for j, a in enumerate(face):
            b = face[(j+1) % len(face)]
            counts[tuple(sorted((a, b)))] += 1
        a = vertices[face[0]]
        for i in range(1, len(face)-1):
            b, c = vertices[face[i]], vertices[face[i+1]]
            triangle = cross(sub(b, a), sub(c, a))
            area += length(triangle) / 2
            signed_volume += dot(a, cross(b, c)) / 6
    if area <= 1e-12:
        raise ValueError(f"{name}: degenerate near-zero area")
    if closed and (any(count != 2 for count in counts.values()) or abs(signed_volume) <= 1e-13):
        raise ValueError(f"{name}: expected watertight nonzero-volume part")
    if closed and signed_volume < 0:
        faces = [list(reversed(face)) for face in faces]
        signed_volume *= -1
    return {
        "vertices": [list(p) for p in vertices],
        "faces": faces,
        "uvs": [list(uv) for uv in uvs],
        "stats": {
            "vertices": len(vertices), "faces": len(faces),
            "boundary_edges": sum(c == 1 for c in counts.values()),
            "nonmanifold_edges": sum(c > 2 for c in counts.values()),
            "area_m2": round(area, 10),
            "signed_volume_m3": round(signed_volume, 12) if closed else None,
            "closed_manifold": closed,
        },
    }


def _ring_mesh(rings: list[list[Vec]], *, closed: bool, name: str) -> dict:
    if len(rings) < 2 or any(len(r) != len(rings[0]) for r in rings) or len(rings[0]) < 3:
        raise ValueError(f"{name}: contour rings have mismatched sizes")
    n, m = len(rings[0]), len(rings)
    vertices = [v for ring in rings for v in ring]
    uvs = [(j/n, i/max(1, m-1)) for i in range(m) for j in range(n)]
    faces = []
    for i in range(m-1):
        for j in range(n):
            a, b = i*n+j, i*n+(j+1)%n
            faces.append([a, b, b+n, a+n])
    if closed:
        faces.extend([list(range(n-1, -1, -1)),
                      list(range((m-1)*n, m*n))])
    return _mesh(vertices, faces, uvs, closed=closed, name=name)


def swept_tube(points: list, radii: list, *, radial_segments=20,
               subdivisions=6, reference_up=(0, 0, 1)) -> dict:
    """Watertight curved, varying elliptical cross section.

    Suitable for authored upper/lower eyelids, brows, feather shafts, curved
    lip rims or accessory piping. Uses a minimally twisting transported frame.
    """
    if len(points) != len(radii) or len(points) < 3:
        raise ValueError("Tube needs >=3 centerline anchors and matching radii")
    centers = interpolate([tuple(p) for p in points], subdivisions)
    widths = interpolate([tuple(p) for p in radii], subdivisions)
    if any(len(p) != 2 or min(p) <= 1e-6 for p in widths):
        raise ValueError("Tube radii must remain positive")
    ref = unit(tuple(reference_up))
    basis = None
    rings = []
    for i, center in enumerate(centers):
        tangent = unit(sub(centers[min(len(centers)-1, i+1)],
                           centers[max(0, i-1)]))
        projected = sub(ref if basis is None else basis, mul(tangent, dot(ref if basis is None else basis, tangent)))
        if length(projected) < 1e-8:
            fallback = min([(1, 0, 0), (0, 1, 0), (0, 0, 1)], key=lambda v: abs(dot(v, tangent)))
            projected = sub(fallback, mul(tangent, dot(fallback, tangent)))
        u = unit(projected)
        v = unit(cross(tangent, u))
        basis = u
        width, depth = widths[i]
        rings.append([add(center, add(mul(u, width*math.cos(2*math.pi*j/radial_segments)),
                                      mul(v, depth*math.sin(2*math.pi*j/radial_segments))))
                      for j in range(radial_segments)])
    return _ring_mesh(rings, closed=True, name="swept_tube")


def contour_loft(sections: list, *, section_subdivisions=4,
                 contour_subdivisions=2) -> dict:
    """Real volumetric mesh from authored corresponding 3D anatomical contours.

    Contours are in world metres. The same contour vertex must represent the
    same anatomical feature across all stations; no auto registration/mirroring.
    """
    if len(sections) < 3 or len(sections) > 24:
        raise ValueError("Contour loft needs 3–24 corresponding anatomical stations")
    n = len(sections[0])
    if n < 6 or n > 64 or any(len(section) != n for section in sections):
        raise ValueError("Loft station contours must share 6–64 points")
    smoothed = [interpolate([tuple(v) for v in section], contour_subdivisions, closed=True)
                for section in sections]
    # Shape-smooth across authored axial stations, preserving exact original rings
    flattened = []
    samples = n*contour_subdivisions
    for j in range(samples):
        flattened.append(interpolate([section[j] for section in smoothed],
                                     section_subdivisions))
    rings = [[flattened[j][i] for j in range(samples)]
             for i in range(len(flattened[0]))]
    return _ring_mesh(rings, closed=True, name="contour_loft")


def ocular_globe(center: list, forward: list, up: list, radii: list,
                 *, iris_fraction: float, pupil_fraction: float,
                 radial_segments=40, latitudes=20) -> dict[str, dict]:
    """Sclera ellipsoid, 3D iris annulus, and 3D pupil, with explicit axes.

    Colors/materials are not invented; the native Blender worker assigns
    source-selected materials to separate objects. This does not create lids.
    """
    center = tuple(center)
    f = unit(tuple(forward))
    u = unit(sub(tuple(up), mul(f, dot(tuple(up), f))))
    right = unit(cross(u, f))
    rx, ry, rz = radii
    if not 0 < pupil_fraction < iris_fraction < .95:
        raise ValueError("Eye pupil/iris fractions must be ordered and fit the visible globe")
    def surface(radial, theta, offset):
        a, b = radial*math.cos(theta), radial*math.sin(theta)
        front = math.sqrt(max(0, 1-a*a-b*b))
        return add(center, add(add(mul(right, rx*a), mul(u, rz*b)),
                               mul(f, ry*front+offset)))
    verts = [add(center, mul(u, rz))]
    uvs = [(0.5, 1.0)]
    for i in range(1, latitudes):
        polar = math.pi*i/latitudes
        for j in range(radial_segments):
            angle = 2*math.pi*j/radial_segments
            verts.append(add(center, add(add(mul(right, rx*math.sin(polar)*math.cos(angle)),
                                              mul(f, ry*math.sin(polar)*math.sin(angle))),
                                          mul(u, rz*math.cos(polar)))))
            uvs.append((j/radial_segments, 1-i/latitudes))
    south = len(verts)
    verts.append(sub(center, mul(u, rz)))
    uvs.append((0.5, 0.0))
    faces = []
    for j in range(radial_segments):
        faces.append([0, 1+j, 1+(j+1)%radial_segments])
    for i in range(latitudes-2):
        a = 1 + i*radial_segments
        b = a + radial_segments
        for j in range(radial_segments):
            faces.append([a+j, b+j, b+(j+1)%radial_segments,
                          a+(j+1)%radial_segments])
    last = 1+(latitudes-2)*radial_segments
    for j in range(radial_segments):
        faces.append([last+j, south, last+(j+1)%radial_segments])
    sclera = _mesh(verts, faces, uvs, closed=True, name="eye-sclera")
    # Iris and pupil are thin closed shells, not decals, with small optical
    # stand-offs sized relative to authored depth; avoid coplanar z-fighting.
    def disk_part(inner, outer, offset, thickness, *, name):
        stations = 4
        if inner <= 0:
            front_center = surface(0.0, 0.0, offset)
            back_center = surface(0.0, 0.0, offset-thickness)
            rings_front = [[surface(outer*i/stations, 2*math.pi*j/radial_segments, offset)
                            for j in range(radial_segments)] for i in range(1, stations+1)]
            rings_back = [[surface(outer*i/stations, 2*math.pi*j/radial_segments, offset-thickness)
                           for j in range(radial_segments)] for i in range(1, stations+1)]
            vertices = [front_center, back_center] + [
                v for ring in rings_front+rings_back for v in ring
            ]
            uv = [(0.5, 0.5), (0.5, 0.5)] + [
                (0.5+math.cos(2*math.pi*j/radial_segments)*outer*i/(stations*2),
                 0.5+math.sin(2*math.pi*j/radial_segments)*outer*i/(stations*2))
                for _face in range(2) for i in range(1, stations+1)
                for j in range(radial_segments)
            ]
            faces = []
            start_back = 2+stations*radial_segments
            for j in range(radial_segments):
                k = (j+1)%radial_segments
                faces.append([0, 2+j, 2+k])
                faces.append([1, start_back+k, start_back+j])
            for side in range(2):
                base = 2+side*stations*radial_segments
                for i in range(stations-1):
                    a, b = base+i*radial_segments, base+(i+1)*radial_segments
                    for j in range(radial_segments):
                        k = (j+1)%radial_segments
                        face = [a+j, b+j, b+k, a+k]
                        faces.append(face if side == 0 else list(reversed(face)))
            outer_front = 2+(stations-1)*radial_segments
            outer_back = start_back+(stations-1)*radial_segments
            for j in range(radial_segments):
                k = (j+1)%radial_segments
                faces.append([outer_front+j, outer_back+j, outer_back+k, outer_front+k])
        else:
            fraction = [(inner+(outer-inner)*i/stations) for i in range(stations+1)]
            rings = [[surface(r, 2*math.pi*j/radial_segments, offset-side*thickness)
                      for j in range(radial_segments)]
                     for side in range(2) for r in fraction]
            nring = len(fraction)
            vertices = [v for ring in rings for v in ring]
            uv = [(0.5+math.cos(2*math.pi*j/radial_segments)*r/2,
                   0.5+math.sin(2*math.pi*j/radial_segments)*r/2)
                  for _side in range(2) for r in fraction for j in range(radial_segments)]
            faces = []
            for side in range(2):
                start = side*nring*radial_segments
                for i in range(nring-1):
                    a,b=start+i*radial_segments,start+(i+1)*radial_segments
                    for j in range(radial_segments):
                        k = (j+1)%radial_segments
                        face=[a+j,b+j,b+k,a+k]
                        faces.append(face if side==0 else list(reversed(face)))
            back=nring*radial_segments
            for outer_idx in (0,nring-1):
                a,b=outer_idx*radial_segments,back+outer_idx*radial_segments
                for j in range(radial_segments):
                    k=(j+1)%radial_segments
                    face=[a+j,b+j,b+k,a+k]
                    faces.append(face if outer_idx==0 else list(reversed(face)))
        return _mesh(vertices, faces, uv, closed=True, name=name)
    depth = max(.00007, ry*.018)
    iris = disk_part(pupil_fraction, iris_fraction, depth*2.0, depth*.65, name="eye-iris")
    pupil = disk_part(0, pupil_fraction, depth*3.0, depth*.65, name="eye-pupil")
    return {"sclera": sclera, "iris": iris, "pupil": pupil}
