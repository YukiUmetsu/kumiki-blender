# Advanced Material Finishing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible, guided photo material finishing to private multi-view dense reconstruction outputs.

**Architecture:** A self-contained manifest/algorithm module returns corrected albedo and roughness/height assets. Existing finish command records and hashes those outputs; Blender review consumes validated nodes. A separate UDIM packager rewrites OBJ UV indexing without touching source geometry.

**Tech Stack:** Python 3.11+, Pillow, optional NumPy, Blender Python adapter, unittest.

**Spec:** `docs/superpowers/specs/2026-11-02-advanced-material-finishing-design.md`

## Global Constraints

- Private, opt-in outputs; immutable source OBJ, MTL and images.
- Enforce the existing dense-revision provenance and review-required status.
- Never confuse a generated roughness prior with a measurement or diffuse intensity with height.

## Review Focus

- Mismatched or symlinked map paths are rejected before output directory creation.
- Material mask overlaps and region gaps are reported; no unintended global de-lighting.
- Transparent and black atlas gutters remain untouched.
- Additional roughness and displacement maps are hash-checked before Blender import.
- Shared OBJ `vt` entries are duplicated as needed for independent UDIM tile offsets.

---

### Task 1: Guided surface finishing

**Files:** `src/kumiki_blender/reconstruction/surface_finish.py`, `tests/test_surface_finishing.py`

- [x] Write tests for recipe validation, divided regions, guided de-lighting, cleanup, roughness priors and 16-bit height maps.
- [x] Run the tests and observe missing API failures.
- [x] Implement bounded linear-light operations with strict local file validation.
- [x] Run focused tests and confirm they pass.

### Task 2: Existing revision and Blender integration

**Files:** `src/kumiki_blender/reconstruction/texture_finish.py`, `blender_dense.py`, `cli.py`, `tests/test_surface_finishing.py`

- [x] Add integration tests for CLI recipe, report integrity and tamper detection.
- [x] Observe failures, then connect the opt-in finishing stage to copy and hash PBR artifacts.
- [x] Validate images before adding roughness/height nodes to the private .blend.
- [x] Run focused tests and confirm they pass.

### Task 3: UDIM packaging and operator documentation

**Files:** `src/kumiki_blender/reconstruction/udim.py`, `cli.py`, `docs/DENSE_STATUE_RECONSTRUCTION.md`, `tests/test_surface_finishing.py`

- [x] Add failing tests for multi-material OBJ UV tile remapping and mutation rejection.
- [x] Implement a separate private UDIM OBJ/MTL and image package with hashed provenance.
- [x] Document source limits, manual inputs, Blender/Cycles vs GLB differences and exact CLI examples.
- [x] Run the entire test suite, inspect diff and create a feature PR.
