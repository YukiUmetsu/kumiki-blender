"""Tension-directed copper drapery study for the Statue of Liberty.

The relief is *artist-inferred*, guided by installed-monument HAER NY-138-252,
253, 254, and 263: broad gravity-sagged copper folds, swept upper-sleeve
pleats, and overlapping cloth planes. It is not a measured drape scan.

Primary folds are connected deformations of the same watertight skin, not
separate thin decorative tubes. Front is -Y, and model-space units are meters.
"""

from __future__ import annotations

import math

from .architecture_geometry import Mesh


def _angular_bump(angle: float, center: float, width: float) -> float:
    delta = math.atan2(math.sin(angle-center), math.cos(angle-center))
    return math.exp(-.5*(delta/width)**2)


# Fold source at the upper robe, its fall direction toward the hem,
# radial height (m), and half-width (radians). These are *artistic guides*,
# not measurements extracted from unregistered archive photographs.
FRONT_TENSION_FOLDS: tuple[tuple[float, float, float, float], ...] = (
    (3.48, 3.40, .24, .112),
    (3.96, 3.79, .31, .134),
    (4.43, 4.28, .35, .154),
    (4.94, 4.88, .32, .138),
    (5.37, 5.43, .29, .132),
    (5.79, 5.95, .25, .112),
)

REAR_TENSION_FOLDS: tuple[tuple[float, float, float, float], ...] = (
    (.27, .42, .17, .18),
    (.94, 1.08, .23, .19),
    (1.64, 1.58, .19, .17),
    (2.32, 2.20, .18, .16),
)


def sculpted_underrobe(*, rows: int = 113, sectors: int = 160) -> Mesh:
    """One closed body volume with broad gravity-fanned copper folds."""
    if rows < 8 or sectors < 32:
        raise ValueError("underrobe needs at least 8 rows and 32 sectors")
    vertices = []
    for row in range(rows):
        t = row/(rows-1)  # hem=0, shoulder=1
        z = 9.83+14.75*t
        # Wider pelvic drape, narrower neck line; major mass is still a single
        # continuous support surface underneath the separate upper overfold.
        rx = 2.28+.40*math.sin(math.pi*t)-.40*t
        ry = 1.71+.22*math.sin(math.pi*t)-.18*t
        for j in range(sectors):
            a = math.tau*j/sectors
            tension = .16+.86*(1-t)**.74
            ripple = 0.
            for top, hem, height, width in FRONT_TENSION_FOLDS:
                angle = top*t+hem*(1-t)
                ripple += tension*height*_angular_bump(a, angle, width)
            for top, hem, height, width in REAR_TENSION_FOLDS:
                angle = top*t+hem*(1-t)
                ripple += tension*height*_angular_bump(a, angle, width)
            # Wide fabric-sag field at the side, pulled below the tablet arm.
            ripple += .12*(1-t)*_angular_bump(a, 5.68-.26*t, .34)
            # Gentle secondary cloth weight, not high-frequency stripes.
            ripple += .028*(1-t)**1.4*math.sin(9*a+2.3*t)
            radius_x = rx+ripple
            radius_y = ry+ripple*.78
            x = radius_x*math.cos(a)+.11*math.sin(math.pi*t)
            y = radius_y*math.sin(a)
            # Uneven edge remains restrained so the plinth/feet don't clip.
            hem_sag = .09*(1-t)**9 * (1-math.cos(3*a+.32))*.5
            vertices.append((x, y, z-hem_sag))
    faces: list[tuple[int, ...]] = [tuple(reversed(range(sectors)))]
    for row in range(rows-1):
        for j in range(sectors):
            k = (j+1) % sectors
            faces.append((row*sectors+j, row*sectors+k,
                          (row+1)*sectors+k, (row+1)*sectors+j))
    last = (rows-1)*sectors
    faces.append(tuple(range(last, last+sectors)))
    return Mesh(tuple(vertices), tuple(faces))


def sculpted_torch_sleeve(*, rows: int = 61, sectors: int = 96) -> Mesh:
    """Swept upper-arm cloth shell with integrated diagonal tension folds."""
    if rows < 8 or sectors < 32:
        raise ValueError("sleeve needs at least 8 rows and 32 sectors")
    vertices = []
    for row in range(rows):
        t = row/(rows-1)  # elbow / hanging cuff to shoulder / arm
        z = 22.45+2.67*t
        # Tube follows the left rising arm's shoulder, becoming fuller below
        # the upper arm where loose cloth hangs under gravity.
        x0 = -2.13-.39*t
        y0 = .05
        radius = .64+.28*math.sin(math.pi*t)**.9-.10*t
        depth = .70+.17*math.sin(math.pi*t)-.18*t
        for j in range(sectors):
            a = math.tau*j/sectors
            pleats = 0.
            # Five diagonal fold paths gather toward the upper-arm tension
            # point; they move in azimuth rather than drawing horizontal rods.
            for i, origin in enumerate((3.46, 3.92, 4.42, 4.90, 5.37)):
                angle = origin+.25*t+(i-2)*.045*t
                pleats += (.105+.025*(i%2))*math.sin(math.pi*(.10+.86*t)) * \
                    _angular_bump(a, angle, .13+.016*(i%2))
            # Underside sag: a broad overhang below the raised arm, visible
            # in HAER 252, not random circumferential noise.
            pleats += .15*(1-t)**.9*_angular_bump(a, 4.65, .52)
            r = max(.08, radius+pleats)
            d = max(.08, depth+pleats*.90)
            x = x0+r*math.cos(a)
            y = y0+d*math.sin(a)
            # Rolled cloth around the cuff, softening the blunt lower cut.
            zz = z+.11*(1-t)**8*(math.cos(2*a+.36)-1)*.5
            vertices.append((x, y, zz))
    faces: list[tuple[int, ...]] = [tuple(reversed(range(sectors)))]
    for row in range(rows-1):
        for j in range(sectors):
            k = (j+1) % sectors
            faces.append((row*sectors+j, row*sectors+k,
                          (row+1)*sectors+k, (row+1)*sectors+j))
    last = (rows-1)*sectors
    faces.append(tuple(range(last, last+sectors)))
    return Mesh(tuple(vertices), tuple(faces))


def sculpted_front_overfold(*, rows: int = 69, columns: int = 73) -> Mesh:
    """Body-following, diagonally hemmed sheet with broad cross-body tension.

    This is a qualitative composition from the installed full-body HAER 232
    photograph, not a reconstruction from matched camera/landmark coordinates.
    Both cloth sides and their four rims form one closed mesh. In particular,
    the leading diagonal hem is part of the sheet, rather than a detached rod.
    """
    if rows < 12 or columns < 16:
        raise ValueError("overfold needs at least 12 rows and 16 columns")
    front = []
    for row in range(rows):
        v = row/(rows-1)  # shoulder origin -> diagonal falling hem
        spread = math.sin(.5*math.pi*v)**.88
        left = -1.13-.79*spread
        right = -.60+2.69*spread
        for col in range(columns):
            u = col/(columns-1)
            x = left+(right-left)*u
            # Left-hand point of the hem hangs 2.65m below the right. The
            # slanted termination prevents a rectangular curtain silhouette.
            z = (24.18-10.64*v+2.65*u*v**1.15
                 +.13*v**8*math.sin(math.tau*u*2.0+.28))
            t = max(0., min(1., (z-9.83)/14.75))
            rx = 2.28+.40*math.sin(math.pi*t)-.40*t
            ry = 1.71+.22*math.sin(math.pi*t)-.18*t
            # Approximate underrobe envelope, then model fabric in relief on
            # its front side. Edges tuck close to the supporting body volume.
            y = -ry*math.sqrt(max(.08, 1-(x/rx)**2))-.13
            y -= .19*math.sin(math.pi*u)**.65*math.sin(math.pi*v)**.8
            # Three substantial lengths of cloth radiate from one shoulder
            # gathering point. They deliberately are not evenly spaced rods.
            for path, width, height in ((.17+.07*v, .085, .44),
                                        (.43-.06*v, .100, .69),
                                        (.76-.03*v, .110, .45)):
                y -= height*(.26+.74*math.sin(math.pi*v)**.70) * \
                    math.exp(-.5*((u-path)/width)**2)
            # A recessed channel between major raised fabric masses provides
            # actual alternating troughs rather than a uniformly swollen bib.
            for path, width, depth in ((.30+.02*v, .055, .18),
                                       (.60-.04*v, .067, .23)):
                y += depth*math.sin(math.pi*v)**.55 * \
                    math.exp(-.5*((u-path)/width)**2)
            # The strongest copper folds cut diagonally across the torso and
            # fall into the long hem, as in the complete installed view.
            for center, width, height in ((.36+.22*u, .080, .48),
                                          (.69+.12*u, .073, .27)):
                y -= height*math.sin(math.pi*u)**.50 * \
                    math.exp(-.5*((v-center)/width)**2)
            front.append((x, y, z))
    count = len(front)
    thickness = .085
    vertices = front + [(x, y+thickness, z) for x, y, z in front]
    faces: list[tuple[int, ...]] = []
    for row in range(rows-1):
        for col in range(columns-1):
            a = row*columns+col
            b = a+columns
            faces.append((a, b, b+1, a+1))
            faces.append((count+a+1, count+b+1, count+b, count+a))
    for col in range(columns-1):
        faces.append((count+col+1, count+col, col, col+1))
        a = (rows-1)*columns+col
        faces.append((count+a, count+a+1, a+1, a))
    for row in range(rows-1):
        a, b = row*columns, (row+1)*columns
        faces.append((a, count+a, count+b, b))
        a, b = row*columns+columns-1, (row+1)*columns+columns-1
        faces.append((b, count+b, count+a, a))
    # Rows run top-down, reversing the legacy bottom-up panel orientation.
    return Mesh(tuple(vertices), tuple(faces))
