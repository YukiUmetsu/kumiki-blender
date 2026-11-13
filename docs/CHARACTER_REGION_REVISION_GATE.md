# Registered source-fidelity gate for character part revisions

Run this **after** a Blender sculpt, geodesic landmark fit, part swap or retopology. Geometry and rig checks alone cannot establish that an eye, beak, wing or muzzle still resembles the original illustration. The gate compares the *original* and *new* part against the **same verified, immutable reference masks**, with the camera, figure scale and frame held fixed.

## Usage

~~~bash
kumiki-character-build audit-region-revision \
  --character .review/characters/pip/source-plan-r01 \
  --region eye.L \
  --after .review/characters/pip/landmark-fit-r01/landmark-fitted-source.obj \
  --max-regression 0.01 \
  --output .review/characters/pip/eye-L-fidelity-r01
~~~

The baseline comes from the region's **source_mesh** in CharacterSpec v2. Alternatively, supply **--before /path/to/previous-source.obj** (or PLY/GLB) when the region has no declared original mesh.

**Coordinate contract:** Pass the canonical Z-up **source-frame** OBJ with a filename ending in *-source.obj or *.source.obj, created by the Blender dual-format exporter. Legacy region.obj, low.obj and other Y-up handoff files are rejected so that their rotated coordinates cannot silently inflate or destroy the registered-view score. PLY and GLB must likewise already use the original Blender world frame.

The optional **--min-iou 0.85** establishes a separate absolute IoU floor in every verified camera. **--max-regression 0.01** rejects a drop greater than one IoU percentage point in *any* verified view, not just the average.

A source-unchanged mesh is reported as **no-change-review-required** rather than improvement. Exit code 1 indicates a registered-view failure; 0 means numerical guards passed but does not certify visual quality.

## Evidence in the immutable output

The output folder contains **report.json**, **baseline/report.json**, **candidate/report.json**, camera overlays and an 800×300 **side-by-side before/after panel for each source-verified view**. Red pixels mean missing mesh area, blue excess mesh area and green overlapping silhouette. The baseline is always left, candidate right; neither image is recentered, cropped or retargeted.

The audit records all source and candidate SHA-256 fingerprints, the CharacterSpec source hash, independent IoU before/after values, the worst camera delta and explicit failed-view reasons. Failed candidates still produce their review reports and overlays. Invalid source files or stale image hashes produce no partial final output.

## Pipeline placement

1. Freeze original art and register the original reference views with CharacterSpec v2.
2. Refine a part with build-anatomy, refine-region, fit-landmark-surface or an approved retopo stage.
3. Run audit-region-revision against that part's source-frame candidate OBJ.
4. Review the side-by-side images and run validate-rig to check posed deformation.
5. Only accept the revision after *both* numerical and artistic/animation review.

A silhouette IoU gate does not infer hidden anatomy, certify 3D depth, compare colors or textures, or approve changes to the rig. It must not compare different poses as if they were the same neutral source pose. Geometry from unsupported/unverified cameras is never used to create an artificial score.

The synthetic tests cover two independent registered views, an edit that worsens just the front while leaving the side unchanged, immutable source files, strict thresholds, unchanged revisions, unmarked legacy OBJ rejection, and atomic output reports.
