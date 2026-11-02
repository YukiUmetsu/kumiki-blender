# Christ the Redeemer — anatomical carved-hair experiment (HOLD)

## Why this mode exists

The bald CC0 donor head was visibly unlike the Rio monument: its ears,
scalp outline, temple hair and long hair mass were missing. Earlier tests
added six detached crown ribbons and two thin temple sheets, or an oversized
ellipsoid mantle. Both approaches left obvious sculptural discontinuities.

The **carved-hair** candidate uses a different approach:

1. Keep the 52k+ vertex quad-topology anatomical head as a clay-only scaffold.
2. Make the original head-derived scalp surface the *continuous* hair foundation.
3. Project six artist-authored 3D crown flow guides onto the scalp's **actual
   polygon surface**, then displace its existing vertices along their normals.
   The center of each channel is physically recessed; a small positive bank
   catches grazing light. Every channel tapers to zero at both ends.
4. Retain two broad, hand-guided temple-to-neck curtains to conceal the donor's
   pinnae and continue the hair silhouette into the shoulders. No loose
   floating crown ribbons are generated in this mode.
5. Leave the original full-figure GLB untouched until multi-view likeness and
   silhouette review establishes that this candidate is an improvement.

This is an **artist-inferred** sculpt, not a licensed monument scan or a
photogrammetrically registered reconstruction. It remains a HOLD.

## Reference sources (view only, do not copy scan geometry)

- [Pix4D / PUC Rio / Aeryon Projeto Redentor white paper](https://cdn.geo-matching.com/4p7aLXRy.pdf):
  actual monument photogrammetry, including front and side **triangle-mesh
  head** illustrations. The images and mesh are reference-only.
- [Baukunst restoration photogrammetry](https://sketchfab.com/3d-models/cristo-redentor-christ-the-redemeer-107e94e081d7464f93267b79fa9372c7):
  authentic monument survey; rights to download and redistribute not established.
- [Wikimedia Commons: Cristo Redentor head](https://commons.wikimedia.org/wiki/File:Cristo_Redentor_head.jpg):
  independently authored real-subject photograph, public domain according to
  its file page; useful for qualitative low-angle front review.
- [Wikimedia Commons: Christ the Redeemer closeup](https://commons.wikimedia.org/wiki/File:Christ_the_Redeemer_closeup.JPG):
  4k close-up of the actual subject, CC BY-SA 3.0; respect attribution if
  reproducing the photo in a public QA sheet.

The reference white paper and survey previews are **not** used as geometry
imports. Sculpt control points are not survey measurements.

## Blender command

```bash
blender -b --factory-startup --python-exit-code 1 \
  --python scripts/landmarks/christ_redeemer.py -- \
  --head-study --face-method anatomical --carved-hair --verify \
  --output .review/christ-redeemer-carved-HOLD.glb \
  --blend .review/christ-redeemer-carved-HOLD.blend \
  --metrics .review/christ-redeemer-carved-metrics.json \
  --views .review/christ-redeemer-carved-views \
  --shot head_front --shot head_profile --shot head_three_quarter \
  --shot head_low --size 850 --samples 24
```

The PR's existing Blender CI job runs this candidate and uploads the GLB,
editable .blend, metrics and four views as a **HOLD** artifact, subject to
CI execution and passing its Blender tests. The file is not a production
replacement and must not silently overwrite the approved r50 full GLB.

## Next quality gates

- Visually inspect frontal, profile, three-quarter and low-angle *real* renders.
  Reject if the scalp edge is stepped, the hair drape intersects the cheeks,
  grooves are too regular, or the hair looks like a detached helmet.
- Re-annotate the exact render pixels and independently sourced real-subject
  photos. Use the camera-correct probe in
  [CHRIST_REDEEMER_CAMERA_REGISTRATION.md](CHRIST_REDEEMER_CAMERA_REGISTRATION.md).
- Compare against the actual head-mesh views in the Pix4D white paper. Rebuild
  the **nose bridge, upper/lower eyelids, mouth corners and strong chin** on
  the same anatomical cage. No facial likeness acceptance until those
  shapes agree in more than one registered camera.
