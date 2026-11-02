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

- `src/kumiki_blender/head_graft.py`: scene-node-aware GLB mesh graft,
  preserving source transforms and 65 untouched body/hair nodes; hashes the
  world-space vertices/faces before and after grafting to prevent accidental
  changes to the statue, arms, robe or pedestal.
- `scripts/preview_anatomical_head_swap.py`: can generate a donor study via
  Blender CLI automatically, graft it into an existing full-figure GLB,
  write a JSON audit and optionally render four head QA views.
- Two alignment modes: `world` for the native Blender study already placed
  on the statue, and `bounds` only for diagnostic generic donor GLBs.
  Bounding-box fitting is **not** a measured scan/photo fit.
- Optional hair substitution is off by default. The existing r50 mantle is
  still not photo-faithful, but rejected hair experiments must not silently
  replace it.

## One-command candidate (Blender installed)

```bash
python -m pip install -e '.[preview,head-qa]'
python scripts/preview_anatomical_head_swap.py \
  --base .review/christ_redeemer/r50.glb \
  --blender blender \
  --output .review/christ_redeemer/anatomical-HOLD.glb \
  --views .review/christ_redeemer/anatomical-views
```

The `--blender` option calls the existing CC0 anatomical head-study
generator. The accepted r50 GLB is **never overwritten**. The tool also
accepts `--donor <head-study.glb>` if it was generated previously.

## Real GLB review (r59, 2026-11-02)

We assembled and rendered an actual 343k-vertex full-figure GLB from the
retained r50 GLB and a previous 189,456-vertex Blender Studio anatomy
checkpoint. The donor was uniformly aligned with `--alignment bounds`,
which is a diagnostic transform, **not** camera registration.

Audit:
- 189,456 head vertices / 321,728 head faces, replacing 39,121 vertices.
- 65 preserved nodes, verified unchanged by SHA-256 of world-space geometry.
- 66 meshes total after removing the old head and 3 detached face guides.
- 29.4 m arm span and full statue geometry retained; 38 m overall height.
- Four real 3D QA views rendered with `scripts/render_glb_head_qa.py`.

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

## r61 controlled jaw-cage trial (2026-11-02, HOLD)

A new optional `--experimental-jaw-cage` deforms *complete anatomical
cross-sections*, smoothly tapering the lower-face drop toward the neck
and rear skull. This replaces the r57/r58 front-only chin bulge that
created an unacceptable horizontal shelf. The cage is artist-inferred,
not a surveyed reconstruction.

The r61 preview was actually exported and rendered from the same
189,456-vertex donor topology as r59; its other 65 preserved meshes are
unchanged. We manually marked 17 shared features on **two different
real-monument photos** (user-supplied high-angle and low-angle front views).
Only the pixel coordinates and corresponding generic donor mesh vertices
are stored in `CHRIST_REDEEMER_PHOTO_ANCHORS_R59_HOLD.json`, not the photos.

Weak-perspective independent camera fits, with **no free focal/distance
degeneracy**, gave these *provisional* residuals:

| Photo | r59 RMS | r61 RMS | r59 chin | r61 chin |
|---|---:|---:|---:|---:|
| High-angle front | 24.22 px | 21.12 px | 40.12 px | 16.41 px |
| Low-angle front | 20.34 px | 16.78 px | 56.19 px | 36.69 px |

The chin silhouette moves in the correct direction without adding a
detached chin object, but the overall fit is still **HOLD** (both RMS
values exceed 15 px). The donor remains too generic in the eyelids,
nose, mouth, brow and hair. Annotation uncertainty, unknown lens
distortion and unregistered camera positions mean these numbers are
*not* millimetre-accurate survey results.

Reproduce the provisional jaw variant:

```bash
python scripts/preview_anatomical_head_swap.py \
  --base .review/christ_redeemer/r50.glb --blender blender \
  --experimental-jaw-cage \
  --output .review/christ_redeemer/anatomical-jaw-HOLD.glb \
  --views .review/christ_redeemer/anatomical-jaw-views
```

The photo-residual comparison tool is
`scripts/evaluate_christ_face_photos.py`; it requires two GLBs
with **identical donor vertex topology** and the r59 annotations.
Do not transfer the r59 vertex IDs to a different Blender study revision.

## r62 stone material transfer (2026-11-02, HOLD)

The anatomy checkpoint carried a neutral clay/copper-grey material unrelated
to Christ the Redeemer. The GLB graft now copies the original r50 statue's
**soapstone PBR albedo, normal and roughness maps** onto the donor head and
creates new **metric cylindrical UVs** (0.72 m repeat). This retains the
3 cm tesserae scale instead of stretching a single donor UV island over
the entire 3-metre face. The original body and arm materials remain unchanged.

Run `scripts/render_glb_head_qa.py <preview.glb> --output <folder> --materials`
for a base-color textured QA view. VTK's base-color preview is not a
Cycles-quality material review and does not yet evaluate all normal/roughness
maps or reproduce the source photo's illumination.

The actual r62 GLB was exported, reopened, and inspected:
- Head material: `Warm light grey soapstone | 3 cm tesserae`
- Embedded albedo texture: 768 × 768 pixels
- Head metric UVs: approximately U -5.58..5.58, V 47.25..52.70
- Anatomical mesh: 189,456 vertices; photo likeness and hair still **HOLD**
