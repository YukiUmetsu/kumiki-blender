"""One-piece classical forelock sculpt with hair flows embedded in its surface."""

from __future__ import annotations

import math

from .architecture_geometry import Mesh


def sculpted_forelock() -> Mesh:
    """Closed, high-resolution S-shaped hair volume with carved relief waves.

    Longitudinal phase-drifting folds make the copper read as continuous hair
    masses rather than separated thin tubes floating above the forehead.
    """
    rows, sectors = 73, 48
    vertices = []
    for i in range(rows):
        t = i/(rows-1)
        x = -1.22+2.44*t
        broad = math.sin(math.pi*t)**.65
        center_y = -.52-.38*broad-.045*math.sin(math.tau*t)
        center_z = 28.30+.11*math.sin(math.pi*t)+.115*math.sin(math.tau*t-.35)
        radius_y = .115+.195*broad
        radius_z = .10+.115*broad
        for j in range(sectors):
            phi = math.tau*j/sectors
            # The relief wraps and drifts across the mass; no independent
            # strand cylinders or artificial evenly spaced parallel wires.
            relief = (1. + .085*math.sin(5*phi-2.2*math.pi*t)
                      + .035*math.sin(9*phi+3.1*math.pi*t))
            vertices.append((x, center_y+radius_y*math.cos(phi)*relief,
                             center_z+radius_z*math.sin(phi)*relief))
    faces = [tuple(range(sectors))]
    for i in range(rows-1):
        for j in range(sectors):
            faces.append((i*sectors+j, (i+1)*sectors+j,
                          (i+1)*sectors+(j+1)%sectors,
                          i*sectors+(j+1)%sectors))
    faces.append(tuple(reversed(range((rows-1)*sectors, rows*sectors))))
    return Mesh(tuple(vertices), tuple(tuple(reversed(face)) for face in faces))
