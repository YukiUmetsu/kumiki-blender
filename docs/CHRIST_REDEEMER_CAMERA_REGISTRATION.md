# Christ the Redeemer — camera-accurate reconstruction gate

The November r16 front-photo report (28.738 px RMS, 67.384 px max) is **historical diagnostic only**. Its 3D landmark probes used a hard-coded orthographic world-axis camera and an assumed 700×700 render, without reading the actual Blender QA camera. The 3D coordinates must be regenerated before interpreting any numerical mismatch as facial anatomy.

## Stage 1: Annotate the exact render

1. Produce a neutral anatomical clay render with an explicit size and shot. Save the same .blend in that run. The generator removes temporary QA cameras after rendering; the probe tool reconstructs the original camera from the named shot preset.
2. Mark matching *visible surface* landmarks on the render and a photograph of the **real Rio monument**, not on a generic human donor or miniature. Record the image dimensions exactly, and keep the source photo reference-only.
3. The annotation JSON must contain model_shot (e.g. head_front), model_image_size (e.g. [850,850]), image_size for the source photo, subject_photo_url, model_revision, and named model_uv/photo_uv correspondences. The legacy r16 annotation's 700×700 assumption is **unverified**; it must not be relabeled as calibrated.

```bash
blender -b --factory-startup --python scripts/landmarks/christ_redeemer.py -- \
  --head-study --face-method anatomical --guided-hair \
  --views .review/christ_redeemer/qa --blend .review/christ_redeemer/study.blend \
  --shot head_front --shot head_three_quarter --shot head_profile \
  --size 850 --samples 24

blender -b .review/christ_redeemer/study.blend \
  --python scripts/landmarks/probe_face_anchor_pixels.py -- \
  --input .review/christ_redeemer/front-annotations.json \
  --output .review/christ_redeemer/front-anchors-3d.json
```

The probe now reconstructs the *actual QA camera* for front, low, profile and three-quarter shots. It transforms each image pixel through Camera.view_frame(scene), handles perspective/orthographic projection and camera shift, raycasts against the **evaluated** anatomical head, checks model-image reprojection, and rejects occluded head points. For a custom camera already saved in the .blend, pass --camera-name instead of model_shot.

## Stage 2: Independent photos and persistent anatomy mismatch

Repeat with **at least two independently sourced real-monument photographs**, each with its own photo camera, and a minimum of eight 2D–3D correspondences per view. At least six landmark names must overlap. Every shared landmark must resolve to the **same 3D position and same sculpt revision**. Record the distinct source-photo URLs.

```bash
PYTHONPATH=src python scripts/fit_multiview.py \
  .review/christ_redeemer/front-anchors-3d.json \
  .review/christ_redeemer/oblique-anchors-3d.json \
  --output .review/christ_redeemer/multiview-report.json
```

The report independently fits each photo's perspective camera and flags the same features when residuals exceed the threshold across all views. **This is not bundle adjustment.** A camera bound hit, high pixel residual, duplicate photo, inconsistent 3D landmark, or missing provenance blocks any geometry-acceptance claim. Even a low residual is only a provisional fit; manual correspondence and camera calibration still require verification.

## Release gate

- **HOLD:** No new anatomical head or hair is promoted to the full GLB until its multi-view photo/scan registration is validated and its neutral clay likeness is reviewed.
- **Do not mistake projection error for a 3D sculpt offset.** A 67-pixel chin residual could come from wrong model pixel coordinates, camera calibration, photo perspective, annotation error, or actual anatomy. The original r16 number cannot disambiguate them.
- **Do not use the two rejected guide-sheet hair studies as accepted topology.** Reconstruct face, hairline, nape and shoulders as a continuous sculpt fitted to independently validated references, or use a properly licensed authentic monument scan.

Local CPython checks (no Blender required):

```bash
PYTHONPATH=src python -m unittest tests.test_render_ray tests.test_multiview_registration -v
```

Blender integration is required to validate evaluated-mesh raycasts and actual rendered pixels; the pure-Python tests alone do not verify that integration.
