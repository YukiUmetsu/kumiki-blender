# Evidence-aware reconstruction implementation plan

## Goal

Keep all source photographs and immutable revisions, but use early segmentation
confidence to prioritize real subject features and give disconnected SfM cohorts
a validated route to a combined camera solution. Never treat a candidate mask
or synthetic pixels as ground truth.

## Approaches considered

- **Selected:** extend the existing per-photo cache, bounded MPS LightGlue
  worker and PyCOLMAP cohort recovery. Reuse COLMAP's original feature indices,
  paired geometric observations and existing camera gates.
- Mask the entire image before SIFT: rejected because bad masks would erase
  faces, paws and other irreplaceable measurements.
- Relax reconstruction/coverage thresholds: rejected because this would
  relabel uncertain camera geometry as verified evidence.

## Global constraints

- Source images, masks and previous model revisions stay immutable.
- Every cache receipt binds content hashes, model hash and settings schema.
- Existing non-statue `reconstruct-photos` remains unchanged by default.
- MPS remains first choice; optional CPU is supported by the existing worker.
- An unsuccessful alignment cannot produce a promoted merged candidate.

## Tasks

### 1. Photo QA before feature extraction

- Test `tests/test_photo_preprocess_cache.py` for corrupted masks, low-quality
  photos, byte-identical reuse and single-photo invalidation.
- Add `reconstruction/photo_early_qa.py`, a per-source cached image QA receipt
  and three-class high/uncertain/background mask. Assess blur, exposure, image
  size, EXIF orientation, near duplicate cues, mask boundaries and possible
  background leakage. Produce a contact sheet and run-level JSON warnings.
- Wire it into `prepare_statue_photo_cache`, without ever altering normalized
  RGB source pixels or trusting candidate silhouettes as hard labels.

### 2. Subject-aware matching

- Test `tests/test_reconstruction_mps_lightglue.py` for mask-based ranking,
  invalid provenance and original keypoint indices; test post-verifier
  contamination rejection on a toy COLMAP database.
- Expose early QA by staged source hash through `reconstruct-photos` and pass
  its verified per-pixel confidence maps to the bounded MPS LightGlue worker.
- Favor high-confidence foreground points under the feature cap, retain a
  budget for uncertain and background regions, and remove pairs whose two
  endpoints are confidently irrelevant/background. Never renumber database
  keypoints or rewrite already matched pairs during resume.

### 3. Disconnected-model alignment and merge

- Test `tests/test_sfm_cohort_recovery.py` for zero common cameras, robust
  inlier/outlier fits, spatial degeneracy, false geometric matches and no
  mutation/promotion on rejection.
- Add `reconstruction/sfm_point_alignment.py`: extract 3D-to-3D
  correspondences through verified COLMAP two-view geometry and original 2D
  point indices; fit a deterministic RANSAC Sim(3), with independent geometric
  validation and traceable match provenance.
- If common cameras exist, retain the established merger. For zero-common
  images, perform a guarded PyCOLMAP merge of aligned disjoint observations,
  then apply the existing baseline drift/registered target gates.

### 4. Integration, docs, verification

- Check CLI stage hashes and resume/revision guarantees; update pipeline
  logging/reporting, `docs/PHOTO_TO_GLB.md`, and focused CI.
- Run the focused tests, the full runnable unittest suite, compilation, and
  inspect the final diff before pushing tested commits to `main`.

## Review focus

1. Mask maps with invalid dimensions or hashes must fail closed.
2. User-added photos must not invalidate unchanged normalized and QA entries.
3. Background matches must not eliminate all evidence for an uncertain part.
4. A high inlier count concentrated on a tiny patch must not pass Sim(3).
5. Two unrelated statues with coincidental matches must never be combined.

## Limitations

Passing synthetic model-alignment tests cannot establish that the Great Sphinx
photos contain sufficiently overlapping, accurate observations. Physical
object-surface validation and camera coverage remain required.
