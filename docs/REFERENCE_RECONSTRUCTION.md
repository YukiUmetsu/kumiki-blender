# Reference-driven reconstruction foundation

## What this solves

The current statue studies were built mainly from parametric profiles, body
lofts and sculpt displacements. Their renders can be technically valid while
the actual monument's face, hair or drapery is wrong. In particular, repeated
manual adjustments cannot recover an unknown camera pose or unseen 3D depth.

This source-neutral optional workflow provides the **first repeatable evidence
stage** for all monuments: inspect photos, identify weak viewpoints, avoid
inflating confidence with duplicates, and attempt sparse camera registration
with COLMAP. It does **not** generate a finished statue mesh, replace genuine
survey data, or claim that the output is suitable for close-up cinematography.

## Prepare local references

Install the existing image QA dependency (Blender not required):

```bash
pip install -e '.[qa]'
```

Copy [`examples/monument-reference-set.example.json`](../examples/monument-reference-set.example.json)
beside a `photos/` directory and replace its dummy source URLs, filenames,
monument identity and observations with **actual** evidence. Do not commit
third-party photographs or borrowed meshes to the source repository. This
manifest is intentionally a template and cannot be run without real local
image files.

- `id`: unique lower-case slug, independent of filename.
- `file`: JPEG, PNG or WebP relative to the JSON; no absolute or escaping paths.
- `source_url`: actual HTTPS reference/photo page; this is provenance, **not**
  an assertion of commercial/republication rights.
- `view`: `front`, `front-left`, `left-profile`, `rear-left`, `rear`,
  `rear-right`, `right-profile`, `front-right`, `top`, or `unknown`.
- `regions`: known visible parts, such as `face`, `chin`, `ear`, `hair`,
  `headcloth`, `drapery`, `tablet`, `torch`, `paws`, or `torso`.
- `mask` (optional): grayscale mask relative to the JSON, with the **same
  dimensions** as the photograph. Black excludes background keypoints;
  white permits matching. Use a conservative mask that retains the statue.
- `expected_sha256` (optional): exact pinned hash for a photo; the workflow
  always calculates/records the actual digest even without a pin.
- `required_views` and `required_regions`: project-specific evidence coverage
  goals. Missing labels are warnings, not an invention of geometry.

An image is decoded before acceptance (bounded size and dimensions), with
SHA-256 identity and a small perceptual duplicate-hash. Exact-byte duplicates
are excluded from COLMAP; similar but byte-distinct photos are **kept** because
they may contain useful parallax. Similar images of the same declared viewpoint
are conservatively grouped **for coverage reporting only**. The grouping is
a **hint**, not a reliable 3D overlap graph.

```bash
kumiki-blender reference-audit --manifest /path/to/monument/references.json
```

This prints `view_counts`, `region_counts`, `missing_views`,
`missing_regions`, duplicate IDs and an honest `review-required` verdict.
`independent_images` counts distinct SHA-256 photos;
`conservative_coverage_images` counts perceptual groups. Neither measures
reconstructed cameras or verified likeness.

## Try sparse reconstruction when photo overlap is plausible

Install COLMAP separately and ensure `colmap` is on `PATH` (or pass
`--colmap /absolute/path/to/colmap`). A minimum of three independent images
is a safety preflight, **not** a guarantee of enough parallax, overlap or
texture. Photos taken decades apart, smooth stone/copper surfaces, different
focal lengths and environmental clutter can all defeat reconstruction.

```bash
kumiki-blender reconstruct-photos \
  --manifest /path/to/monument/references.json \
  --output .review/monument/reconstruction-r01 \
  --colmap colmap --timeout-seconds 3600 \
  --undistort --max-image-size 4000
```

The worker runs, in order:

1. `feature_extractor` on **only the curated independent images**, with any
   provided masks converted to the filename convention COLMAP expects.
2. `exhaustive_matcher` to find verified image features.
3. `mapper` to estimate camera geometry and sparse points.
4. `model_converter` to inspect registered-image counts from actual models.
5. When `--undistort` is selected: `image_undistorter` on the best sparse
   model and `model_converter` on the **rectified** cameras. This produces
   `undistorted/images/`, `undistorted/text-model/`, and a pinned
   `undistorted-reference-index.json`. Use this for accurate Blender camera
   overlays: COLMAP's common `SIMPLE_RADIAL` model cannot be represented by a
   Blender pinhole camera without rectifying both its pixels and intrinsics.

CPU feature extraction/matching are the default, making the initial pipeline
usable on Macs without a CUDA card. The worker probes each installed COLMAP
binary's help before output creation, selecting modern `Feature*` or legacy
`Sift*` CPU flags, and records its executable path/version in the report.
Unsupported flag sets fail during preflight. Dense reconstruction is
intentionally not run here.
Exhaustive pair matching is quadratic; for much larger photo collections,
later workers should use vocabulary-tree retrieval / hloc rather than blindly
adding thousands of images. See [COLMAP's official CLI documentation](https://colmap.github.io/cli.html).

The output is a fresh immutable-by-convention folder containing staged photos,
optional masks, `staged-reference-index.json`, `reference-audit.json`,
`database.db`, native `sparse/` models when available, converted text models,
per-stage logs and `reconstruction-report.json`. It contains local copies of
third-party reference photos: keep this folder private and do not add it to
GitHub/GLB distribution. Never delete an existing revision to rerun; choose a
new `r02`, `r03`, etc.

## Align actual cameras to an existing statue sculpt

COLMAP's sparse coordinates have unknown scale, rotation and origin. To make
those cameras meaningful against an existing editable sculpt/scan, select at
least three **non-collinear corresponding 3D landmarks** (prefer four or more)
in COLMAP sparse space and in your Blender sculpt's **metre coordinate frame**.
Use manually verified masonry corners, plinth edges or durable surface features,
not invented anatomical points or observations on different restorations.
Replace the illustrative numbers and asset ID in
[`examples/monument-alignment.example.json`](../examples/monument-alignment.example.json).
An anchor pair has `colmap_xyz` and matching `blender_xyz_m`. The fitter
estimates a proper rotation, scale and translation with a residual RMS gate;
it rejects mirrored, degenerate and badly mismatched alignments. A small
residual is necessary but **does not prove** a detailed sculpt is correct.

```bash
kumiki-blender prepare-camera-plan \
  --reconstruction .review/monument/reconstruction-r01 \
  --alignment /path/to/monument/landmarks.json \
  --max-anchor-rms-m 0.05 \
  --output .review/monument/camera-plan-r01

kumiki-blender render-reference-scene \
  --plan .review/monument/camera-plan-r01/camera-plan.json \
  --model /path/to/monument/your-aligned-sculpt.glb \
  --output .review/monument/blender-review-r01 \
  --render-clay --max-cameras 8
```

This creates a **private** editable `reference-review.blend` with registered
perspective cameras, externally linked reference photos for viewport alignment,
and optionally up to 32 transparent neutral-clay renders. Do not pack or
publish third-party photos or the review `.blend`. The model GLB must already
use the anchor target frame; this does not transform the model or export it.
The worker refuses cameras that cannot reproduce the photo projection, such as
distorted images or substantially unequal focal lengths.

To review exact silhouettes, each camera needs an accurate manually checked
white-statue/black-background mask **of the same pixels**. If you ran with
`--undistort`, original masks are geometrically invalid: create new masks on
the rectified images, named `<original-filename>.png` (for example,
`front.jpg.png`), in a private folder. Supply them while building the camera
plan with `--undistorted-masks /path/to/rectified-masks`. The presence of an
original, unrectified COLMAP matching mask is never treated as evidence on a
rectified frame. Nearly all-white matching masks are treated as unscored.

```bash
kumiki-blender silhouette-qa \
  --plan .review/monument/camera-plan-r01/camera-plan.json \
  --renders .review/monument/blender-review-r01/renders \
  --output .review/monument/silhouette-qa-r01 --min-iou 0.80
```

Each score is a mask intersection-over-union (IoU), precision and recall;
mask-only overlays mark mesh-only pixels in red, reference-only in blue and
overlap in gray. Outputs never include online photo pixels. Fewer than three
scored cameras, missing renders, poor masks or low IoU require review. Even
perfect IoU is **not** a photorealism or historical-fidelity certificate:
sculptors still need face, eyes, hair, drapery, and authentic close-up study.

## Measure region-specific facial, hair and drapery likeness

Silhouette IoU can miss a completely wrong face inside a correct head outline.
The `landmark-qa` tool measures *where the sculpt puts identified 3D points*
against manually annotated pixels of the **same points** in registered photos.
Copy [`examples/monument-landmarks.example.json`](../examples/monument-landmarks.example.json)
and replace its illustrative identifiers and numbers with true evidence:

- `required_regions`: e.g. `face`, `hair`, `ear`, `chin`, `drapery`.
- Each `measurement` identifies a registered `image_id`, a `region`, a stable
  `landmark_id`, the fixed sculpt coordinate `sculpt_xyz_m`, and where that
  physical feature occurs in the reference image as `image_xy_px`.
- Repeated observations of a landmark must use the **same** 3D coordinate;
  use the rectified image when a COLMAP `--undistort` run created new camera
  intrinsics. Do not interpolate unknown or occluded points.
- Each region requires at least three distinct landmarks across at least two
  camera viewpoints; at least three registered images must be annotated for
  a provisional threshold candidate. Missing coverage cannot pass silently.

```bash
kumiki-blender landmark-qa \
  --plan .review/monument/camera-plan-r01/camera-plan.json \
  --annotations /path/to/monument/annotated-landmarks.json \
  --output .review/monument/landmark-qa-r01 --max-error-px 8
```

The report lists per-point image projection, pixel error, camera depth,
per-region median/max error, and regions with insufficient viewpoints. It
marks behind-camera landmarks as mismatches rather than fabricating scores.
It is a measurement aid for sculpt correction, not an automatic editor or
guarantee of accuracy; the point must be correctly identified in both places.

Exit codes:

- `0`: intake audit ran, or COLMAP produced a sparse **candidate** with at
  least three registered images. Human review is still required.
- `2`: invalid or changed input, missing software, bad arguments, existing
  output; no output directory is created on preflight failure.
- `3`: Output staging or COLMAP failed to produce a viable sparse candidate.
  Logs (when a stage ran) and a `review-required` manifest remain for diagnosis.
  An interrupted copy can leave partial staged files but cannot be mistaken
  for an approved reconstruction.

## Artistic next steps and honest stopping criteria

1. **Authenticate the subject.** Confirm that references/scan depict the
   actual monument rather than a replica or interpretive restoration.
2. **Evaluate real camera overlap.** Inspect matched views and the sparse
   model; an arbitrary bag of web photos commonly fails here.
3. **Prefer authentic survey/scan geometry.** The existing Sphinx scan intake
   covers GLB replacement but cannot prove sculptural fidelity alone.
4. **Fit a continuous high-resolution sculpt.** When SfM fails, use measured
   frontal/side profiles and calibrated multi-view landmarks to direct actual
   Blender mesh edits. Do not reconstruct a face with arbitrary Gaussian
   protrusions. The active statue PRs already experiment with these tools;
   they should consume this reference dossier when consolidated.
5. **Gate quality independently.** Matched-camera neutral clay renders,
   region-specific anatomical landmarks, silhouettes, and actual scan
   deviations must be inspected before a production GLB is promoted.
6. **Retopologize and bake.** Only after the high-resolution form is accepted,
   bake normals/displacement to reusable medium/low-detail GLB exports.

**Remaining framework work:** real COLMAP evaluation on a licensed multi-angle
statue dataset; robust arbitrary-photo feature retrieval/hloc; general
survey/point-cloud intake beyond Sphinx; semantic/region-aware facial and
drapery sculpt *editing* tools; close-up texture likeness QA; high/low LOD
geometry and texture baking. Camera pose and
silhouette tooling help artists measure shape mismatches, but do not themselves
repair anatomy or recover unknown surface detail.
