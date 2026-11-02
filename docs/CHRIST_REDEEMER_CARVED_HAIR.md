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

## Continuous flute and temple-root refinement

The first candidate snapped each scalp vertex to the *nearest sampled guide
point*. That quantized its longitudinal taper, potentially creating scalloped
stone channels. The updated method uses a KD tree only to find candidate
guide segments, then computes an exact closest point and arc-length fraction
on those segments. The flute is therefore continuous between samples.

The upper rows of both temple curtains now snap **inside** the actual
anatomical scalp with a smooth falloff; small physical bevels soften the
otherwise razor-sharp panel edges. The smoke test records the number of
rooted vertices, the largest original root gap, and the groove projection
method. These are still **overlapping meshes, not a single manifold solid**.
A visually convincing seam, facial likeness and watertightness have **not**
been demonstrated until the four-view Blender renders are inspected.

## Rear hairline silhouette guard

The donor scalp mask previously fell below **z = -0.09 donor metres**
around the rear temples, potentially covering the whole neck like a
helmet/cape. The extracted scalp now has a hard lower silhouette bound
at **z = 0.028 donor metres**, aligning it with the temple curtains'
shoulder-length termination. This only constrains geometry: its likeness
still requires actual four-view renders and real-monument comparison.

## Real Blender 4.5 visual QA and nape reconstruction (2026-11-02)

GitHub Actions generated actual `.blend`, GLB, front, profile, oblique
and low-angle clay renders (run 41389926503; artifact
`christ-redeemer-carved-hair-clay-HOLD`). The automated geometry checks
passed, but **visual QA rejected the result**:

- The donor-derived scalp wraps the **entire posterior neck**, creating
  an oversized cape and a rolled lower edge.
- The long temple curtains are broad, almost straight ribbons; their
  connection to the scalp and ears looks broken and exposes dark gaps.
- Crown flow reads like a smooth helmet, not the real monument's distinct
  parted, deeply carved stone locks.
- The face still lacks the real sculpture's heavy brows, angular jaw and
  square chin. Good topology counts are not evidence of likeness.

We isolated the scalp and curtain meshes in the actual GLB. The giant
rolled edge is primarily the **extracted scalp's neck region**, not
the curtains themselves. An offline test trimmed only the rear neck
region and visibly removed the rolled cape, though it exposed a
missing posterior hair mass. Therefore the next carved-hair candidate
uses `trim_nape=True` only for this experimental mode. The other
head studies and the production GLB are unchanged.

**Acceptance gate:** the revised scalp must terminate at the nape
rather than becoming neck skin, but the missing back-of-hair volume
must be reconstructed as an actual connected stone sculpt before any
production promotion. The candidate remains **HOLD** until new
four-view Blender renders pass photo comparison.

### First trim failure and corrected lateral-neck mask

The first `trim_nape` pass still reported scalp minimum Z = **34.57 m**,
so its new Blender test correctly failed. Inspecting the real GLB mesh
showed the lowest remaining scalp vertices at approximately donor
`(x=.108, y=-.010, z=.056)`: **lateral neck**, not rear skull.
A rear-only cutoff cannot remove them. The revised selection now
constrains both the posterior and lateral neck using the actual
anatomical coordinates, with a 0.215-donor-metre upper cutoff
weighted by side exposure. This remains an experimental scalp
selection pending a fresh four-view Blender render.

### Second lateral trim adjustment

Filtering the previous real Blender scalp GLB offline showed that a
lateral ramp starting at donor `|x|=.055` still retained neck vertices
at `|x|≈.08, z≈.08`, with a world minimum near **34.81 m**.
The revised ramp starts at `|x|=.035` and reaches full effect by
`|x|=.080`. The same offline mesh selection predicted a minimum
near **35.40 m** and removed the curled cape, but exposed an abrupt
rear hair edge. **These are diagnostic GLB filtering results, not a
passing Blender reconstruction**; the CI render and a connected
posterior hair volume are still required.
