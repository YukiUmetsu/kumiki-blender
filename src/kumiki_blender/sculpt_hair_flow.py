"""Reusable physical groove profile for carved stone hair.

This is a geometry displacement primitive, not a procedural hair texture:
the caller moves real mesh vertices along surface normals. All lengths are
in the original anatomical mesh's metres, before statue-scale transforms.
"""

import math


def smoothstep(value):
    t = min(1.0, max(0.0, float(value)))
    return t*t*(3.0-2.0*t)


def stone_hair_groove(distance, arc_fraction, *, depth=0.0018,
                      width=0.0045, shoulder_height=0.00065):
    """Signed normal offset for one hand-guided, finite-length carved flute.

    Negative values cut stone; positive values form an understated bank at
    either side. Taper at both ends avoids clipped incisions and open seams.
    No image pixels, noisy texture or random strands are used.
    """
    if not all(math.isfinite(float(v)) for v in
               (distance, arc_fraction, depth, width, shoulder_height)):
        raise ValueError("Hair groove parameters must be finite")
    if distance < 0 or not 0 <= arc_fraction <= 1:
        raise ValueError("Distance and arc fraction must be in range")
    if depth <= 0 or width <= 0 or shoulder_height < 0:
        raise ValueError("Groove dimensions must be positive")
    taper = smoothstep(arc_fraction / .13) * smoothstep((1-arc_fraction) / .16)
    cut = -depth * math.exp(-.5 * (distance / width)**2)
    bank = shoulder_height * math.exp(-.5 * ((distance-1.8*width)/(.65*width))**2)
    return taper * (cut+bank)


def catmull_rom_3d(points, steps=14):
    """Interpolate manually placed three-dimensional carving guides."""
    if steps < 1 or len(points) < 2 or any(len(p) != 3 for p in points):
        raise ValueError("At least two 3D guide points and positive steps required")
    result = []
    for i in range(len(points)-1):
        a,b,c,d = (points[max(0,i-1)], points[i], points[i+1],
                   points[min(len(points)-1,i+2)])
        for step in range(steps):
            t = step / steps
            result.append(tuple(.5*((2*b[j])+(-a[j]+c[j])*t+
                                      (2*a[j]-5*b[j]+4*c[j]-d[j])*t*t+
                                      (-a[j]+3*b[j]-3*c[j]+d[j])*t*t*t)
                                for j in range(3)))
    result.append(tuple(points[-1]))
    return result
