# Christ the Redeemer — measured face reconstruction (experimental)

## Why the previous face is on HOLD

The existing anatomical clay uses a generic CC0 donor head with manually inferred
jaw, nose and eyelid deformations. A real low-angle photo shows much heavier
orbital arches, a broad nose, a downturned closed mouth and a tall angular chin.
The current donor remains visibly generic. **Neither the previous r16 28.7 px
camera-fit residual nor new uncalibrated measurements prove how far any single
facial vertex must move.**

## New two-photo reconstruction procedure

1. Render a neutral, unchanged anatomical head at a recorded camera pose and
   exact image size; preserve the editable .blend.
2. Independently annotate **at least 12** matching 2D/3D facial landmarks in
   two *different real-subject photographs* (front/low-angle and oblique/profile).
   Use `probe_face_anchor_pixels.py` for the model-side 3D surface anchors.
3. Run `PYTHONPATH=src python scripts/fit_face_geometry.py front.json oblique.json
   --output .review/christ_redeemer/face-corrections-HOLD.json`.
4. For **each** feature, the solver fits both cameras *without that feature*.
   It triangulates the withheld photo observations, rejects parallel rays,
   inconsistent ray intersections, large reprojection residuals, and proposed
   displacements above 25 cm. It outputs named 3D correction candidates and
   explicit held features.
5. Use `--experimental-face-corrections <report.json>` with the Blender
   `--head-study --face-method anatomical --model-revision <revision>` mode
   to produce a **clay-only** comparison. It applies bounded, compact-support,
   C2-smooth mesh displacements to the actual subdivided head vertices without
   replacing the topology.
6. Compare real frontal, profile, oblique and low-angle Blender renders against
   the authentic monument before accepting **any** change. Keep the full
   production GLB unchanged until visual and geometry QA pass.

## Safeguards

- The report status is always `HOLD_needs_visual_review`; it is never an
  automatically approved sculpt or survey reconstruction.
- Different photo URLs, identical model revision, >=10 shared features,
  camera leave-one-out fitting and >=12° camera-ray intersection are required.
- Each proposal is bounded to 25 cm displacement and has a 28 cm localized
  influence radius by default. Facial fit does not alter the hair mesh;
  review possible hair/cheek intersections separately.
- The Blender mode rejects a mismatched revision, missing photo provenance,
  fewer than six accepted facial anchors, and any anchor more than 2.5 cm
  away from the current evaluated head surface. Hair studies cannot be
  combined with facial fitting until collision QA exists.
- A fitted camera is *not* a calibrated survey camera; photographic lens
  distortion, head tilt and manually marked correspondence errors remain.
- The tests use **synthetic camera points**, not a verified monument scan.
  Blender integration and real photo fitting must still be performed.
