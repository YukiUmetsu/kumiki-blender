"""Apply unapproved photo-triangulated face offsets to anatomical clay only.

Blender-only entry helper. It never modifies the accepted full-statue GLB.
"""

import math

from mathutils import Vector
from mathutils.bvhtree import BVHTree

from kumiki_blender.sculpt_correction import bounded_sculpt_offsets


def apply_candidate_face_fit(head, report, expected_revision, *, radius_m=.28):
    if not expected_revision or report.get('model_revision') != expected_revision:
        raise ValueError('Facial correction revision does not match the loaded clay head')
    if report.get('status') != 'HOLD_needs_visual_review':
        raise ValueError('Expected a provisional, unapproved face reconstruction report')
    photos=report.get('source_photographs', [])
    if len(photos)<2 or not all(photos) or len(set(photos)) != len(photos):
        raise ValueError('Two distinct documented reference photographs required')
    candidates=report.get('candidates', [])
    if len(candidates)<6 or report.get('candidate_count')!=len(candidates):
        raise ValueError('At least six consistent, counted facial anchors required')
    if len({c.get('name') for c in candidates})!=len(candidates):
        raise ValueError('Facial anchor names must be unique')

    # A correction generated for a different mesh or gaze transform can look
    # numerically plausible but deform the wrong cheek/eye. Require every
    # original 3D anchor to lie on the currently loaded evaluated head.
    import bpy
    bpy.context.view_layer.update()
    depsgraph=bpy.context.evaluated_depsgraph_get()
    bvh=BVHTree.FromObject(head,depsgraph)
    inverse=head.matrix_world.inverted_safe()
    nearest_gap=0.
    for anchor in candidates:
        local=inverse @ Vector(anchor['source_world_xyz'])
        hit,_,_,distance=bvh.find_nearest(local)
        if hit is None:
            raise ValueError('No head surface near anchor '+anchor['name'])
        world_gap=(head.matrix_world.to_3x3() @ (hit-local)).length
        nearest_gap=max(nearest_gap,world_gap)
        if world_gap>.025:
            raise ValueError('HOLD_model_revision_geometry_mismatch: '+anchor['name'])

    positions=[head.matrix_world @ v.co for v in head.data.vertices]
    offsets=bounded_sculpt_offsets(positions,candidates,
                                  influence_radius_m=radius_m,
                                  max_displacement_m=.25)
    inverse_rotation=head.matrix_world.to_3x3().inverted()
    affected=0
    maximum=0.
    for vertex,offset in zip(head.data.vertices,offsets):
        length=math.sqrt(sum(v*v for v in offset))
        if length>1e-8:
            vertex.co+=inverse_rotation @ Vector(offset)
            affected+=1
            maximum=max(maximum,length)
    head.data.update()
    return {'face_correction_status':'HOLD_needs_visual_review',
            'face_correction_features':len(candidates),
            'face_correction_vertices':affected,
            'face_correction_max_vertex_offset_m':round(maximum,5),
            'face_correction_max_anchor_surface_gap_m':round(nearest_gap,5)}
