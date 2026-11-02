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


def polyline_arc_lengths(points):
    """Cumulative Euclidean arc lengths of a 3D sculpting guide.

    Using physical arc length, rather than a Catmull-Rom sample index, keeps
    groove taper uniform when the sculpting guide bends or stretches.
    """
    if len(points) < 2 or any(len(p) != 3 for p in points):
        raise ValueError("A 3D polyline requires at least two points")
    lengths = [0.0]
    for a, b in zip(points, points[1:]):
        if not all(math.isfinite(float(v)) for v in (*a, *b)):
            raise ValueError("Polyline coordinates must be finite")
        segment = math.sqrt(sum((float(y)-float(x))**2 for x,y in zip(a,b)))
        lengths.append(lengths[-1]+segment)
    if lengths[-1] <= 1e-10:
        raise ValueError("Degenerate zero-length sculpting guide")
    return lengths


def nearest_polyline_segment(point, points, lengths, near_index, span=2):
    """Return continuous distance and normalized arc position near a KD hit.

    The KD tree selects a *sample point*, but carving directly from its index
    creates stair-stepped, scalloped flutes. Here the candidate segment is
    projected analytically, and its continuous arc position drives the taper.
    The caller should try several nearby KD hits if guides cross.
    """
    if len(points) != len(lengths) or len(points) < 2:
        raise ValueError("Guide and arc-length table sizes disagree")
    if len(point) != 3 or not all(math.isfinite(float(v)) for v in point):
        raise ValueError("Expected finite 3D query")
    if not 0 <= near_index < len(points) or span < 1:
        raise ValueError("Invalid guide sample index or span")
    best = (math.inf, 0.0)
    for i in range(max(0,near_index-span), min(len(points)-1,near_index+span)):
        a, b = points[i], points[i+1]
        direction = [float(b[j])-float(a[j]) for j in range(3)]
        squared = sum(v*v for v in direction)
        t = (min(1.0,max(0.0,sum((float(point[j])-float(a[j]))*direction[j]
                                    for j in range(3))/squared)) if squared > 1e-18 else 0.0)
        squared_distance = sum((float(point[j])-float(a[j])-t*direction[j])**2
                               for j in range(3))
        if squared_distance < best[0]:
            progress = (lengths[i]+t*(lengths[i+1]-lengths[i]))/lengths[-1]
            best = (squared_distance,progress)
    return math.sqrt(best[0]),best[1]
