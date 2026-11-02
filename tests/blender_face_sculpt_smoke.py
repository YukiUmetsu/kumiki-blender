"""Blender-only integration check for bounded facial sculpting on real mesh data.

This uses a synthetic sphere; it is not a Christ the Redeemer likeness test.
"""
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"scripts"/"landmarks"))

import bpy
from registered_face_sculpt import apply_candidate_face_fit

bpy.ops.mesh.primitive_uv_sphere_add(segments=48,ring_count=24,location=(0,0,36))
head=bpy.context.object
head.name="Synthetic anatomical QA mesh"
head.scale=(1.2,1.0,1.6)
bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
bpy.context.view_layer.update()
before=[v.co.copy() for v in head.data.vertices]
count=len(before)
chosen=[v for v in head.data.vertices if v.co.y<-.55 and abs(v.co.x)<.9]
if len(chosen)<6:
    raise AssertionError("Insufficient synthetic surface anchors")
chosen=[chosen[int((len(chosen)-1)*i/5)] for i in range(6)]
candidates=[]
for i,v in enumerate(chosen):
    world=head.matrix_world @ v.co
    candidates.append({"name":"synthetic-%s"%i,
                       "source_world_xyz":list(world),
                       "target_world_xyz":list(world + __import__('mathutils').Vector((0,-.018,0)))})
report={"status":"HOLD_needs_visual_review","model_revision":"synthetic-test",
        "source_photographs":["https://example.org/a.jpg","https://example.org/b.jpg"],
        "candidate_count":len(candidates),"candidates":candidates}
metrics=apply_candidate_face_fit(head,report,"synthetic-test")
assert metrics["face_correction_vertices"]>0,metrics
assert metrics["face_correction_features"]==6,metrics
assert len(head.data.vertices)==count
assert max((v.co-b).length for v,b in zip(head.data.vertices,before))<.025
print("BLENDER_FACE_SCULPT_SMOKE",metrics)
