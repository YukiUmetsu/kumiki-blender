# Christ the Redeemer — anatomical full-figure graft, r59 (HOLD)

## Why this exists

The r50 full-figure GLB has a ~39k-vertex continuous parametric face with
nearly closed eyelid slits, a mask-like nasal region and a pinched chin. The
old separate eyelid/mouth guide tubes are visibly detached. Its body, arms,
hands, stone materials and pedestal should not be discarded to rebuild the
head.

The Blender Studio CC0 **generic anatomical donor** provides real facial
topology with eyelids, nostrils, cheeks, jaw, ears, and neck. This is **not**
a photogrammetry scan of Christ the Redeemer. It does not establish likeness.

## Implemented

- \`src/kumiki_blender/head_graft.py\`: scene-node-aware GLB mesh graft,
  preserving source transforms and 65 untouched body/hair nodes; hashes the
  world-space vertices/faces before and after grafting to prevent accidental
  changes to the statue, arms, robe or pedestal.
- \`scripts/preview_anatomical_head_swap.py\`: can generate a donor study via
  Blender CLI automatically, graft it into an existing full-figure GLB,
  write a JSON audit and optionally render four head QA views.
- Two alignment modes: \`world\` for the native Blender study already placed
  on the statue, and \`bounds\` only for diagnostic generic donor GLBs.
  Bounding-box fitting is **not** a measured scan/photo fit.
- Optional hair substitution is off by default. The existing r50 mantle is
  still not photo-faithful, but rejected hair experiments must not silently
  replace it.

## One-command candidate (Blender installed)

\`\`\`bash
python -m pip install -e '.[preview,qa]'
python scripts/preview_anatomical_head_swap.py \
  --base .review/christ_redeemer/r50.glb \
  --blender blender \
  --output .review/christ_redeemer/anatomical-HOLD.glb \
  --views .review/christ_redeemer/anatomical-views
\`\`\`

The \`--blender\` option calls the existing CC0 anatomical head-study
generator. The accepted r50 GLB is **never overwritten**. The tool also
accepts \`--donor <head-study.glb>\` if it was generated previously.

## Real GLB review (r59, 2026-11-02)

We assembled and rendered an actual 343k-vertex full-figure GLB from the
retained r50 GLB and a previous 189,456-vertex Blender Studio anatomy
checkpoint. The donor was uniformly aligned with \`--alignment bounds\`,
which is a diagnostic transform, **not** camera registration.

Audit:
- 189,456 head vertices / 321,728 head faces, replacing 39,121 vertices.
- 65 preserved nodes, verified unchanged by SHA-256 of world-space geometry.
- 66 meshes total after removing the old head and 3 detached face guides.
- 29.4 m arm span and full statue geometry retained; 38 m overall height.
- Four real 3D QA views rendered with \`scripts/render_glb_head_qa.py\`.

**Visible outcome:** the face has much better anatomical topology than r50,
but the eyes, brow, mouth and chin still resemble a generic human donor.
The r50 hair mantle still reads as a hood, and the donor's ears remain too
exposed. r55 hand-carved face grooves were rejected (stacked eyelid ridges);
r57/r58 broad-form donor sculpt experiments were rejected (unnatural brow
and chin shelf). None is photo-faithful enough to replace the approved
full-figure GLB.

Reference photos used for visual review were user-provided, not copied into
the repository. The front and side photographs are **not camera-registered**,
so no pixel-aligned geometric accuracy claim is made.

## Next actual sculpt checkpoint

1. Register the front, low-angle and profile monument photographs with
   measured/annotated 2D-3D landmarks and report residuals per view.
2. Sculpt the **real donor topology** into Landowski's tall square chin,
   angular cheek/jaw planes, deeply hooded eyes and broad squared nose.
   Reject any sharp under-chin shelf or repeated eyelid crests.
3. Rebuild a continuous scalp/temple/neck stone hair mesh, not separate
   cylindrical curtains or blanket hood warps.
4. Render front, profile, oblique and low-angle in Blender and compare
   against genuine photos before approving a replacement GLB.

Status: **HOLD — facial likeness, hair silhouette, collision and Blender
PBR render verification still incomplete.**
