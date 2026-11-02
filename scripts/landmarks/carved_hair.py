"""Anatomy-derived, hand-guided stone hair reconstruction candidate."""

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def sculpt_carved_hair_from_anatomy(head, *, curtain_builder, scalp_builder):
    """Study a *continuous* scalp with physically engraved crown flow.

    The monument has a parted hair mass flowing into the temple curtains,
    not six loose ribbon locks glued to a bald donor. This mode derives the
    scalp from the donor's real quad topology, carves shallow guide-aligned
    grooves directly into those vertices, and retains two broad stone
    curtains. It is an unregistered clay study, not accepted GLB geometry.
    """
    from mathutils.kdtree import KDTree
    from kumiki_blender.sculpt_hair_flow import (
        catmull_rom_3d, stone_hair_groove, polyline_arc_lengths,
        nearest_polyline_segment)

    # Construct the curtains first so their ear-covering adjustments are
    # reflected in the extracted scalp topology. Do not create ribbon locks.
    curtains = curtain_builder(head, crown_locks=False)
    scalp = scalp_builder(head)
    bpy.context.view_layer.update()
    bvh = BVHTree.FromPolygons(
        [v.co.copy() for v in scalp.data.vertices],
        [tuple(poly.vertices) for poly in scalp.data.polygons],
        all_triangles=False)

    # Bury the upper roots of the temple curtains *inside* the actual
    # anatomy-derived scalp. A visually plausible overlap must be solved in
    # mesh coordinates; the old standalone panels could float beside the
    # temple, exposing the donor ears in grazing-light closeups.
    def smoothstep(value):
        t = min(1., max(0., value))
        return t*t*(3.-2.*t)

    rooted = 0
    largest_root_gap = 0.
    for curtain in curtains:
        roots = 0
        for vertex in curtain.data.vertices:
            weight = smoothstep((vertex.co.z-.310)/.035)
            if weight < .01:
                continue
            point, normal, _, distance = bvh.find_nearest(vertex.co)
            if point is None or distance > .080:
                raise ValueError("Temple hair root is too far from the anatomical scalp")
            largest_root_gap = max(largest_root_gap, distance)
            # The scalp is solidified inward by .009 donor metres. Embed the
            # root .0035 m into its skin and fade the attachment downwards.
            vertex.co = vertex.co.lerp(point-normal*.0035, weight)
            roots += 1
        if roots < 35:
            raise ValueError("Too few temple curtain vertices rooted in scalp")
        rooted += roots
        curtain.data.update()
        bevel = curtain.modifiers.new("Soft sculpted stone lock edges", "BEVEL")
        bevel.width = .0015
        bevel.segments = 3
        bevel.limit_method = 'ANGLE'
    scalp["rooted_curtain_vertices"] = rooted
    scalp["max_root_snap_distance_m"] = largest_root_gap

    # Hand-directed guides in the donor's metric 3D space. The source is
    # the actual monument's center-parted crown silhouette; individual
    # coordinates are artistic, not surveyed or copied scan vertices.
    guides = [
        [(.008,-.010,.378),(.027,-.078,.363),(.052,-.110,.351),
         (.080,-.100,.334),(.102,-.078,.300),(.112,-.053,.270)],
        [(.005,.018,.374),(.023,-.045,.369),(.050,-.091,.354),
         (.077,-.082,.334),(.102,-.056,.298),(.115,-.031,.262)],
        [(.011,.049,.359),(.028,-.013,.360),(.056,-.062,.351),
         (.086,-.060,.327),(.109,-.025,.288),(.123,-.005,.253)],
    ]
    paths = []
    sample_lookup = []
    samples = []
    for side in (-1, 1):
        for guide in guides:
            guide_points = catmull_rom_3d(
                [(side*x, y, z) for x,y,z in guide], steps=16)
            path = []
            for point in guide_points:
                hit, _, _, _ = bvh.find_nearest(Vector(point))
                if hit is None:
                    raise ValueError("Carved crown guide left anatomical scalp")
                path.append(tuple(hit))
            path_id = len(paths)
            paths.append((path, polyline_arc_lengths(path)))
            for point_index, point in enumerate(path):
                sample_lookup.append((path_id, point_index))
                samples.append(point)
    if not samples:
        raise ValueError("No carved hair guide points")
    tree = KDTree(len(samples))
    for index, position in enumerate(samples):
        tree.insert(position, index)
    tree.balance()
    # Project against the actual guide *segments* instead of snapping to
    # nearest samples. The previous KD lookup quantized the taper and left
    # scalloped channels on a dense anatomical scalp mesh.
    carved = 0
    originals = [(vertex.co.copy(), vertex.normal.copy())
                 for vertex in scalp.data.vertices]
    for vertex, (position, normal) in zip(scalp.data.vertices, originals):
        candidates = tree.find_n(position, 4)
        if not candidates or candidates[0][2] > .025:
            continue
        best = (float("inf"), 0.)
        for _, sample_index, _ in candidates:
            path_id, point_index = sample_lookup[sample_index]
            path, arc_lengths = paths[path_id]
            distance, fraction = nearest_polyline_segment(
                position, path, arc_lengths, point_index, span=2)
            if distance < best[0]:
                best = (distance, fraction)
        if best[0] > .016:
            continue
        offset = stone_hair_groove(*best)
        if abs(offset) > 1e-8:
            vertex.co = position + normal * offset
            carved += 1
    if carved < 150:
        raise ValueError("Insufficient scalp density for visible stone carving")
    scalp.data.update()
    scalp["carved_vertices"] = carved
    scalp["groove_count"] = len(paths)
    scalp["groove_projection"] = "continuous-polyline"
    return [scalp,*curtains]
