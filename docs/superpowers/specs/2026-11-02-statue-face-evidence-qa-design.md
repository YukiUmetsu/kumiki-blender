# Statue Face Evidence and Likeness QA Design

## Intent

Give statue artists and modeling agents objective, region-specific feedback about how a candidate face differs from reference photographs. The system must turn multiple photos into trustworthy *evidence*, not pretend that a landmark detector or a few screenshots can recover a unique 3D surface.

## Goals

- Work with front, three-quarter, profile, and rear photos of the same sculpture; report missing views, occlusion, and weak evidence.
- Accept manually annotated semantic facial landmarks, view labels, head outlines, and sculpted-hair boundaries. Produce optional machine-generated landmark **proposals**, never silently approved measurements.
- Compare corresponding photo and candidate-render annotations in common camera/view coordinates and report region-level normalized displacement and silhouette mismatch, with review images.
- When registered cameras and 3D point identities are supplied, reuse the existing `reconstruction.landmarks` reprojection checks rather than reimplementing camera math.
- Export deterministic JSON reports and PNG overlays with provenance and explicit `measured`/`unmeasured`/`not-visible` statuses. Include concise correction *observations*, not speculative 3D sculpt commands.
- Keep a standalone entry point that can be incorporated into other workflows later without modifying the shared `cli.py`.

## Non-goals

- No mesh generation, morphable-model fitting, automatic sculpting, texture/material alteration, Gaussian splatting, camera solving, or photo downloads.
- No facial similarity score derived from raw RGB difference or a human-recognition embedding; statue lighting and material make those misleading.
- No assertion that inferred MediaPipe depth is ground-truth 3D or that unobserved anatomy is known.
- No copying `face3d` code or bundling Basel Face Model weights; upstream licensing is not suitable for unreviewed commercial integration.
- No uploaded source images committed to Git. Keep private input images and generated image overlays out of the repository.

## Existing interfaces and boundaries

- `reconstruction.references` already provides validated reference manifests, image dimensions, view labels, source hashes, and coverage auditing. Reuse its structs when the caller supplies a `ReferenceSet`; also accept direct local image paths for user uploads without inventing source URLs.
- `reconstruction.landmarks.check_landmark_reprojection` already computes calibrated 3D-to-2D errors. Call it only when a valid camera plan and 3D landmark annotation file are supplied.
- `reconstruction.head_base`, dense pipelines, material code, `cli.py`, and scene builders are owned by other work and remain untouched.
- Implement entirely within `src/kumiki_blender/face_analysis/`, `tests/test_face_analysis_*.py`, and a dedicated guide. Expose `python -m kumiki_blender.face_analysis` as a separate CLI; no `pyproject.toml` entry-point changes.
- Prefer the existing Python 3.11 baseline and Pillow (`qa` extra). MediaPipe is optional and lazy-imported only for detection. Core analysis must work without Blender, PyTorch, CUDA, or a network.

## Inputs and data contract

- A photo set consists of 1–256 local PNG/JPEG photos, each with stable `image_id`, dimensions, SHA-256, and an optional view label. Direct-local inputs are tagged `user-supplied / rights-unverified` until provenance is provided.
- Annotation JSON v1 records a per-image, human-reviewed view label (`front`, `front-three-quarter`, `profile`, `rear-three-quarter`, `rear`, `unknown`), semantic `landmark_id`, pixel `[x,y]`, origin (`manual` or `proposed`), visibility (`visible`, `occluded`, `uncertain`, or `not-visible`), and optional outline polygon. Every coordinate is bounded by the source image; IDs must be unique per image. Without a manual view label or calibrated camera, the view is `unknown`—it is never inferred from an image filename.
- Semantic landmark vocabulary covers face outline, chin/jaw, eyes/brows, nose, lips, forehead, visible ears, hair mass and hairline; an annotation can intentionally contain no facial points for rear views.
- A candidate comparison pairs a photo and render from the same view or camera registration, with matched semantic IDs and explicitly supplied candidate annotations. No raw RGB registration is inferred.
- Results record errors normalized by a face reference scale from the verified crop/outline, not by full photo dimensions. If scale or correspondence is missing, return `unmeasured`, never a numeric score.
- The output contains `photo-audit.json`, `landmark-proposals.json` (only when detection requested), `likeness-qa.json` (only when pairs exist), and generated `overlays/*.png`; all paths are private/local.

## Validation case: 15 user-supplied Christ the Redeemer photos

The supplied 15 frames, named `Christ_the_Redeemer_face1.png` through `face15.png`, vary in dimensions and traverse roughly frontal-to-rear views. They provide valuable silhouette/hair-drape evidence; rear photographs have no visible facial landmarks. The set appears related in capture and must not be reported as 15 statistically independent camera observations. A complete reverse-side/360-degree reconstruction and accurately calibrated views are not established by these images alone.

For the private validation run, load all 15 local files, manually review and record view labels in a private annotation file, produce view/coverage diagnostics and overlays, and refuse to fabricate landmark positions on rear views. Do not check in the photos or their full-resolution derivatives. A likeness score remains `unmeasured` until a matching candidate render and annotation set exists.

## Approaches

1. **Chosen: independent evidence-and-QA module.** Lowest merge-conflict risk; useful immediately for reference preparation and later for independent model QA.
2. Embed all annotation and QA handling in the existing CLI/reconstruction modules. More convenient now but conflicts with active modeling and camera work.
3. Fit a full 3DMM model automatically from the photos. This crosses into modeling, relies on uncertain inference for unusual stone faces, and introduces licensing/dependency concerns.

## Acceptance

- 15-photo run emits truthful viewpoint/coverage status with no false face points on rear images.
- Deterministic manual-annotation report passes measured two-view synthetic fixture with known expected displacement; mismatched or uncalibrated pairs remain `unmeasured`.
- Invalid paths, duplicate IDs, out-of-bounds points, blank/occluded faces, and missing optional detector are tested.
- No edits to existing reconstruction, material, dense, splat, or shared CLI modules; core tests pass without MediaPipe.
