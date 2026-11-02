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
    from kumiki_blender.sculpt_hair_flow import catmull_rom_3d, stone_hair_groove

    # Construct the curtains first so their ear-covering adjustments are
    # reflected in the extracted scalp topology. Do not create ribbon locks.
    curtains = curtain_builder(head, crown_locks=False)
    scalp = scalp_builder(head)
    bpy.context.view_layer.update()
    bvh = BVHTree.FromPolygons(
        [v.co.copy() for v in scalp.data.vertices],
        [tuple(poly.vertices) for poly in scalp.data.polygons],
        all_triangles=False)

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
    projected = []
    for side in (-1,1):
        for guide in guides:
            path = catmull_rom_3d([(side*x,y,z) for x,y,z in guide],steps=16)
            for index,point in enumerate(path):
                hit,_,_,_ = bvh.find_nearest(Vector(point))
                if hit is None:
                    raise ValueError("Carved crown guide left anatomical scalp")
                projected.append((hit,index/(len(path)-1)))
    if not projected:
        raise ValueError("No carved hair guide points")
    tree = KDTree(len(projected))
    for index,(position,_) in enumerate(projected):
        tree.insert(position,index)
    tree.balance()
    # Each flute is an actual negative normal displacement with a small
    # raised bank, tapered into unmodified stone at the ends. No texture
    # trick, floating tubes, random strand noise or procedural hood loft.
    carved = 0
    for vertex in scalp.data.vertices:
        _,nearest,distance = tree.find(vertex.co)
        if distance > .016:
            continue
        offset = stone_hair_groove(distance,projected[nearest][1])
        if abs(offset) > 1e-8:
            vertex.co += vertex.normal * offset
            carved += 1
    if carved < 150:
        raise ValueError("Insufficient scalp density for visible stone carving")
    scalp.data.update()
    scalp["carved_vertices"] = carved
    scalp["groove_count"] = 6
    return [scalp,*curtains]
