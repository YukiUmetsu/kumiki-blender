# Anatomical head-base fitting for monument studies

## Intent

Stop rebuilding anatomical human faces from primitive procedural forms for every
monument. Use an existing, human-authored or scanned **static head GLB** as the
starting geometry, align it to measured statue facial landmarks, and preserve an
editable Blender revision for artist-directed refinement.

## Existing architecture

Kumiki-Blender already provides a source/license-checked static GLB importer,
immutable `.blend`/GLB/preview exports, a reference-image dossier and COLMAP
camera handoff, and silhouette/landmark reprojection QA. These systems are not
replaced. The missing link is matching a reusable anatomical head to a target
sculpture with an auditable geometric transform and optional non-destructive
local shape changes.

## Goals

- Consume an **explicitly supplied**, locally reviewed, static face/head GLB
  with existing asset provenance and redistribution checks. No model downloads.
- Take manually matched 3D source and statue landmarks (at least four). Fit
  orientation, translation and uniform scale without distorting anatomy.
- Report pre-/post-fit residuals and named landmark regions. Never infer true
  likeness from a low reprojection error.
- Optionally create a restrained, smooth, non-destructive Blender shape key
  from landmark residuals; keep the neutral head and original topology editable.
- Integrate with the existing importer/exporter and provide a new CLI workflow,
  example manifest, tests, and exact handoff to existing registered-camera QA.

## Non-goals

- No new generic procedural face generator or claim of monument accuracy.
- No automatic 2D facial landmark detection or solving unknown 3D anatomy from
  a single picture; the existing camera alignment and surveys supply evidence.
- No redistribution of unlicensed MakeHuman/commercial meshes, no bundled binary
  human head in this PR, and no automatic historical damage/hair/drapery sculpt.
- No edits to active monument PRs; this is a reusable first stage on `main`.

## Approaches

**Selected:** source-neutral existing GLB + manually checked landmark
correspondences + optional Blender shape key. This reuses existing rights checks
and revision output while letting sculptors replace the base and inputs.

- Alternative: hardwire the MakeHuman/MPFB Blender add-on into Kumiki. This
  would add version/runtime coupling and restrict other CC0 or artist-authored
  head bases; a MakeHuman-exported GLB remains a supported input.
- Alternative: make the model directly from FaceBuilder or COLMAP output. This
  depends on proprietary software or sufficiently overlapping photographs and
  does not guarantee sculptural detail. Such results can still be input GLBs.

## Constraints and review criteria

- Core remains Python 3.11+ standard library; Blender remains optional outside
  the build path. Retain Blender 4.2+ and existing source-provenance validation.
- Manifest paths are relative and cannot escape via `..` or symlinks; source
  SHA-256 may be pinned through existing provenance.
- Reject degenerate source/target point sets, mirrored/invalid transforms,
  non-finite numbers, unstable local deformation and missing target mesh names.
- The render/export goes to a new immutable directory. No output on invalid
  input or missing Blender. Generated reports must retain `review-required`
  and `historical_fidelity_verified: false`.
- A rigid fit can reduce landmark residual while failing the silhouette or
  likeness check; artists still inspect front/profile/three-quarter renders.
