# Statue Face Evidence and Likeness QA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build isolated, evidence-driven facial landmark and likeness QA for sculpture photos and candidate renders.

**Architecture:** Add a standalone `kumiki_blender.face_analysis` package for photo intake, annotation validation, optional landmark proposals, normalized view comparisons, and reports. Read existing reconstruction artifacts without mutating them; all generated images and reports remain private outputs.

**Tech Stack:** Python 3.11, standard library, optional Pillow (`qa` extra), lazy optional MediaPipe, pytest.

**Spec:** `docs/superpowers/specs/2026-11-02-statue-face-evidence-qa-design.md`

## Global Constraints

- No edits to `src/kumiki_blender/cli.py`, `reconstruction/`, `head_sources/`, materials, Gaussian-splat, model builders, or shared `pyproject.toml`.
- No bundled BFM weights or source copied from `face3d`; MediaPipe optional and never counted as a verified statue measurement.
- No user photographs or full-resolution generated overlays committed to Git.
- Every report distinguishes measurement, proposal, occlusion, absent coverage, and unknown calibration.
- Synthetic fixtures are committed; the user's 15 attached photos are used only in private validation.

## Review Focus

1. A rear view must be valid with zero facial landmarks, and detection must not claim a face merely because an outline exists (Task 2).
2. Out-of-bounds coordinates and inconsistent original image dimensions must fail before producing misleading overlays (Task 1).
3. A pair without common view/camera registration must report `unmeasured` rather than a distance (Task 3).
4. Duplicate semantic IDs and guessed model confidence must be rejected/left unverified (Tasks 1–2).
5. An output directory must not overwrite a previously reviewed evidence revision (Task 4).

---

### Task 1: Private photo intake and semantic annotation schema

**Files:**
- Create: `src/kumiki_blender/face_analysis/__init__.py`
- Create: `src/kumiki_blender/face_analysis/schema.py`
- Create: `src/kumiki_blender/face_analysis/inputs.py`
- Test: `tests/test_face_analysis_inputs.py`

**Interfaces:**
- `load_photo_set(paths: Sequence[Path]) -> tuple[Photo, ...]`; immutable `Photo(image_id, path, width, height, sha256, view, provenance)`.
- `load_annotations(path: Path, photos: Sequence[Photo]) -> AnnotationSet`; per-image manually reviewed view label and `Annotation(image_id, landmark_id, xy_px, origin, visibility)`, plus optional outline points.
- `inspect_coverage(photos: Sequence[Photo], annotations: AnnotationSet | None) -> dict[str, Any]`.

- [ ] **Step 1: Write failing tests** for mixed PNG/JPEG sizes, stable source hashes, duplicate IDs, out-of-bounds coordinates, unsupported visibility/origin, missing-view-as-unknown, and empty rear-view annotations.
- [ ] **Step 2: Run** `pytest tests/test_face_analysis_inputs.py -q` and confirm failures are about missing interfaces.
- [ ] **Step 3: Implement** strict, bounded parsing (1–256 files, 64-pixel minimum, 40-megapixel maximum, no symlink escapes, explicit source/hash/size records) and a fixed semantic landmark vocabulary.
- [ ] **Step 4: Run** the task tests and confirm they pass.
- [ ] **Step 5: Commit** new package and tests with `feat(face-analysis): validate private photos and landmark evidence`.

### Task 2: Optional statue-aware landmark proposals

**Files:**
- Create: `src/kumiki_blender/face_analysis/detectors.py`
- Test: `tests/test_face_analysis_detectors.py`

**Interfaces:**
- `propose_landmarks(photo: Photo, *, detector: str = "none") -> dict[str, Any]`.
- Returns `status` (`proposed`, `no-face`, `unavailable`), detector identity, and optional `Annotation` proposals; never reports calibrated depth or fabricated per-point confidence.

- [ ] **Step 1: Write failing tests** for no detector, missing MediaPipe, detector returning no face on a back-of-head fixture, and a mocked successful proposal marked `origin=proposed`.
- [ ] **Step 2: Run** `pytest tests/test_face_analysis_detectors.py -q` and confirm expected failures.
- [ ] **Step 3: Implement** lazy MediaPipe import; convert available face landmarks to a small stable semantic subset; leave incompatible, hidden, or low-evidence views without a face; do not estimate unseen landmarks.
- [ ] **Step 4: Run** the task tests and confirm no mandatory ML dependencies.
- [ ] **Step 5: Commit** with `feat(face-analysis): add optional unverified landmark proposals`.

### Task 3: Measured view comparison and correction observations

**Files:**
- Create: `src/kumiki_blender/face_analysis/compare.py`
- Test: `tests/test_face_analysis_compare.py`

**Interfaces:**
- `compare_views(reference: AnnotationSet, candidate: AnnotationSet, pairs: Sequence[ViewPair]) -> dict[str, Any]`; `ViewPair(reference_image_id, candidate_image_id, alignment_id, verified)` declares same-view pixel-coordinate alignment. A missing or unverified `alignment_id` cannot produce a measurement.
- `summarize_observations(report: dict[str, Any]) -> list[dict[str, Any]]` with region, status, measured displacement, and projected-direction wording.

- [ ] **Step 1: Write failing tests** for matched synthetic coordinates (known normalized landmark error), chin-vs-eye region displacement, missing scale, conflicting views/camera identifiers, non-visible points, and duplicate IDs.
- [ ] **Step 2: Run** `pytest tests/test_face_analysis_compare.py -q` and confirm expected failures.
- [ ] **Step 3: Implement** per-landmark/region normalized errors using a verified head crop/outline scale; use optional silhouette polygon IoU only when both polygons and alignment are present; never compare raw photograph colors.
- [ ] **Step 4: Run** task tests; ensure the direction of observations is explicitly in rendered-view image coordinates, not a claim of proven 3D deformation.
- [ ] **Step 5: Commit** with `feat(face-analysis): score registered facial evidence by region`.

### Task 4: Standalone CLI, review overlays, and private 15-photo acceptance run

**Files:**
- Create: `src/kumiki_blender/face_analysis/__main__.py`
- Create: `src/kumiki_blender/face_analysis/reports.py`
- Create: `docs/STATUE_FACE_ANALYSIS.md`
- Test: `tests/test_face_analysis_cli.py`

**Interfaces:**
- `python -m kumiki_blender.face_analysis inspect --images-dir DIR --output DIR [--detector none|mediapipe] [--annotations FILE]`.
- `python -m kumiki_blender.face_analysis compare --reference FILE --candidate FILE --pairs FILE --output DIR`.
- `write_analysis_revision(output: Path, audit: dict[str, Any], proposals: dict[str, Any] | None, comparisons: dict[str, Any] | None) -> Path` writes deterministic JSON, optional Pillow overlays, and a manifest without modifying input photos.

- [ ] **Step 1: Write failing CLI tests** for inspect/compare behavior, informative missing-dependency errors, no-output-on-invalid-input, and rejection of an existing output directory.
- [ ] **Step 2: Run** `pytest tests/test_face_analysis_cli.py -q` and confirm expected failures.
- [ ] **Step 3: Implement** the isolated `__main__` entry point, structured reports, overlays for verified/pending points, and user-facing guide with annotation example and status semantics.
- [ ] **Step 4: Run** `pytest tests/test_face_analysis_*.py -q` and the repository's existing applicable tests.
- [ ] **Step 5: Run** private acceptance on `upload/Christ_the_Redeemer_face*.png` (all 15) with a private, manually reviewed view-label JSON; verify real dimensions, view coverage, and that rear views have no fabricated facial landmarks. Without candidate renders, likeness must be `unmeasured`. Keep photos/results untracked and never publish them.
- [ ] **Step 6: Commit** with `feat(face-analysis): add isolated CLI and review artifacts`, then prepare an independent feature PR without modifying active modeling-agent files.
