# Source-registered texture revision gate

The existing appearance report describes how well one UV texture matches original color artwork. It does not say whether a *new* texture makes the same character better or worse. This new command compares **two texture images on one unchanged mesh, UV layout and camera set** and rejects source-view color regressions.

## One-command part baking with automatic QA

The part-specific texture baker now optionally runs the source regression gate
automatically after writing a new texture candidate:

    kumiki-character-build texture-parts       references/pip/handoff       models/pip-low.source.obj       .review/pip/recognized-parts-r01       --output .review/pip/basecolor-candidate-r02.png       --qa-baseline .review/pip/basecolor-approved-r01.png       --qa-max-mean-increase 1.5       --qa-max-p90-increase 3.0       --qa-max-within-10-drop 0.02

The original part masks are source-fingerprint checked by texture-parts.
Source camera image presence, baseline image, numeric budgets and output
collisions are preflight-checked **before the expensive multi-part bake**.
The program uses the same mesh and UVs for the candidate/previous texture QA,
with the canonical source-frame filename handling described below.

If the source colors regress, exit code **1** is returned, but the candidate
PNG, bake metadata, four-panel camera images and QA report are **retained**
for revision. Numeric success exits 0 but still requires artist review.
A compact **.source-fidelity.json** receipt beside the candidate is intended
for automated agents. The default QA output folder ends in -source-qa;
override with --qa-output.

Without --qa-baseline, texture-parts keeps its previous behavior, including
its default baking options. A pre-existing output texture, report, reviewed QA
folder or receipt is never overwritten.

## Usage

    kumiki-character-build audit-texture-revision \
      references/pip/handoff \
      models/pip-low.source.obj \
      textures/previous.png \
      textures/candidate.png \
      --max-mean-increase 1.5 \
      --max-p90-increase 3.0 \
      --max-within-10-drop 0.02 \
      --output .review/pip/texture-qa-r01

The default **--max-projection-samples 2000000** guards local RAM usage.
It computes the exact number of mesh-face UV samples that the native software
renderer would allocate **before rendering any source view**. For dense/high-poly
input, the command fails with an actionable error recommending a source-registered,
lower-resolution QA cage. Lowering QA image resolution just to hide oversampling
can miss tiny eyelid or feather changes; use an intentionally approved cage.
The sample budget is reported in report.json so runs can be audited.

The OBJ needs actual UV (vt) records and per-face UV coordinates. Explicit source OBJ filenames (suffix -source.obj or .source.obj) are already Blender **Z-up**; legacy interchange OBJ names are Y-up and automatically converted. Both textures must be square and no larger than 8192 pixels.

Pass --views front,right,front_right to evaluate only those existing source cameras; by default every aligned view in handoff/spec.json is required. Unlike the older appearance command, this gate **fails** if a specified source image is missing or fewer than the required **150 common original/model pixels** are present. The original source drawings must have already been checked for correct camera alignment and identity.

## Independent per-camera rejection conditions

Each candidate renders on the exact **same geometry and UVs** as the baseline. Errors use an identical fixed sampling mask: the eroded intersection of source alpha and mesh projection. Compare CIE76 Lab delta E for each camera.

- Mean delta E increased more than 1.5.
- 90th percentile delta E increased more than 3.0.
- Share of pixels with delta E at most 10 decreased more than 0.02.

These are **regression tolerances**, not acceptable absolute photometric error. An unchanged, bad reference texture can pass its own before/after gate, but is never artist-approved. Separately run the silhouette/contour QA gate to check geometry. This gate cannot hide color errors by shrinking the new mesh because candidate and baseline use one unchanged mesh.

## Immutable reports

The new output contains report.json with per-camera baseline/candidate mean/median/P90 delta E, fraction within delta E 10, signed changes, the fixed sample count, explicit failures and SHA-256 fingerprints of the mesh, source images, handoff spec and both textures. Each camera also has a four-panel PNG: original reference, baseline render, candidate render and signed error heatmap. Red: worsening. Blue: improvement.

The source files are checked **again before finalizing**. Existing output directories are refused, and failing candidates still retain their report and evidence. Exit code 1 rejects a numerical regression; exit code 0 means numerical checks passed but **artistic review is still required**. Evidence or registration errors return nonzero and do not produce a final report.

Entirely local and deterministic using NumPy, SciPy and Pillow. It does not evaluate photorealistic PBR shading, light transport, facial identity, hidden geometry, fur/feather strand quality or animation. Different original views may contain inconsistent painterly lighting; inspect trade-offs instead of relying on a single score.
