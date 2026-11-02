# Christ the Redeemer — actual GLB multi-view review (2026-11-02)

## What was inspected

The retained r50 GLB (69 meshes, including the 39,121-vertex procedural
head and 30,720-vertex hair mantle) was recovered from the previous
artifact and rendered from four actual 3D camera views using the
opt-in VTK + trimesh QA script:

```bash
xvfb-run -a python scripts/render_glb_head_qa.py r50.glb \
  --output .review/christ-redeemer/r50 --size 850
```

This is a real mesh render, not a synthetic diagram, but the cameras are
**not** calibrated to the historic photo. The previous neutral-clay
comparison photo is reference-only and is not committed to this repository.

## Actual findings

- **r50 front:** forehead and eye sockets look like a generic, smooth mask.
  Thin eyelid tubes float over blank orbits; mouth is a thin incision; chin
  converges to an unnatural point.
- **r50 profile:** hair mantle wraps the rear skull but its forward edge
  remains too far behind the cheeks; the hair-to-neck transition reads as
  a hollow cape rather than stone locks. The nose profile is insufficiently
  distinctive and the chin is undersculpted.
- **r51 (rejected):** two separate, solid, vertically grooved side hair
  masses frame the face, but render as stiff straight columns. Not a
  likeness improvement; no production export.
- **r52/r53 (rejected):** deforming the existing hair mantle forward
  creates faceted, folded edge topology and an oversized scalp cap. Not
  an acceptable hair reconstruction.
- **r54 (rejected):** a broader chin/eyebrow sculpt applied to the r50 head
  produces an unnatural dark chin shelf. The front/profile render does
  not establish improved likeness.

**Result:** no candidate replaces the r50 production GLB. No r51–r54
experimental geometry was merged into the Blender generator. The four-view
QA renderer was retained because it exposed these failures quickly.

## Quality gate before the next sculpt

1. Use **real-monument** frontal and profile images or a rights-cleared
   authentic survey. The generic donor and unregistered photo do not
   determine target 3D geometry.
2. Calibrate each photo camera with landmarks *withheld* from the feature
   being fitted; see `CHRIST_REDEEMER_FACE_FIT.md`.
3. Build the nose, lips, eyelids and square chin as actual continuous
   anatomical topology. Avoid floating tubes, Gaussian-only masks and
   disconnected hair panels.
4. For hair, preserve a contiguous scalp-to-temple-to-neck topology with
   individually sculpted, nonperiodic stone grooves. **Do not** use more
   lofted vertical columns or blanket mantle warps.
5. Run front/profile/oblique/low neutral-clay renders and reject a sculpt
   whenever its silhouette, facial expression or mesh intersections
   regress. Only then update the approved full GLB.

## Scope of the renderer

`scripts/render_glb_head_qa.py` requires optional `trimesh` and
`vtk` Python packages; on headless Linux use `xvfb-run`.
It reads GLB scene-node transforms, filters head/hair/face meshes,
and renders four consistently framed neutral-shaded views. It is a
rapid **geometry gate**, not Blender Cycles/PBR or photo-registration QA.
