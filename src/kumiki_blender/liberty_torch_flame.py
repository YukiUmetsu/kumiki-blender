"""Closed sculptural gold flame with a swelling core and overlapping tongues.

This is an original interpretation of the gilded copper torch flame, using
photographs for silhouette reference. All layers have real volume; there are
no zero-thickness triangles or decorative floating lines.
"""

from __future__ import annotations

import math

from .architecture_geometry import Mesh


def _sweep_mesh(rings: list[list[tuple[float, float, float]]]) -> Mesh:
    sectors = len(rings[0])
    vertices = tuple(p for ring in rings for p in ring)
    faces = [tuple(reversed(range(sectors)))]
    for section in range(len(rings)-1):
        a, b = section*sectors, (section+1)*sectors
        for j in range(sectors):
            nxt = (j+1) % sectors
            faces.append((a+j, a+nxt, b+nxt, b+j))
    faces.append(tuple((len(rings)-1)*sectors+j for j in range(sectors)))
    return Mesh(vertices, tuple(faces))


def make_layered_flame() -> dict[str, Mesh]:
    """Return an inspectable 3D core plus eleven offset gold leaf-like lobes."""
    rings = []
    for k in range(19):
        t = k / 18
        z = 1.32 + .54*t
        center_x = .029*math.sin(math.pi*t)*math.sin(2.1*t)
        center_y = .022*math.sin(math.pi*t)*math.cos(2.3*t)
        radius = .176*(1-t)**.72 + .0025
        ring = []
        for j in range(96):
            angle = math.tau*j/96
            flank = (1 + .13*math.sin(5*angle-4.8*t)
                       + .045*math.sin(9*angle+2.7*t))
            r = radius*flank
            ring.append((center_x+r*math.cos(angle+.18*t),
                         center_y+r*math.sin(angle+.18*t), z))
        rings.append(ring)
    parts = {"Flame / swelling gilded core": _sweep_mesh(rings)}

    for j in range(11):
        angle0 = math.tau*j/11
        height = .28 + .23*(.5+.5*math.sin(j*1.7+.5))
        spiral = .38 + .20*math.sin(j*1.31)
        tongue_rings = []
        for k in range(16):
            t = k/15
            angle = angle0 + spiral*t + .09*math.sin(math.pi*t)
            radius = .155 - .064*t + .014*math.sin(math.pi*t+j*.9)
            width = .075*(1-t)**.70 + .0015
            thickness = .034*(1-t)**.85 + .0015
            ring = []
            for sector in range(16):
                a = math.tau*sector/16
                radial = radius+thickness*math.cos(a)
                tangential = width*math.sin(a)
                ring.append((radial*math.cos(angle)-tangential*math.sin(angle),
                             radial*math.sin(angle)+tangential*math.cos(angle),
                             1.32+height*t))
            tongue_rings.append(ring)
        parts[f"Flame / rolled copper tongue {j:02d}"] = _sweep_mesh(tongue_rings)
    return parts
