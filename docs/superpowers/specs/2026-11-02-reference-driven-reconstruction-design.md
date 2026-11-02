# Reference-driven reconstruction foundation

## Intent

Give statue authors a repeatable way to turn a folder of independently sourced monument photographs into auditable reconstruction evidence, rather than adjusting procedural parameters based on visual memory. This is a foundation for the Sphinx, Statue of Liberty, and Christ the Redeemer PRs, not a claim that photogrammetry or an uncalibrated photo can yield a faithful face.

## Goals

- Accept a local, source-attributed, reference-only photo collection without granting redistribution rights or publishing photographs in the repository.
- Validate image identity, dimensions, source and paths; report duplicates, missing viewpoints, and important anatomical regions lacking coverage.
- Produce a reproducible feature-matching/reconstruction handoff through a local COLMAP executable, with optional exclusion masks, isolated outputs, and explicit failure reports.
- Keep the package's core standard-library-only, with Pillow available under the existing `qa` extra. Do not require `bpy`, CUDA, or a downloaded ML model for intake.
- Reuse this evidence in future scan alignment, camera registration, and artist-directed sculpting work. Preserve the existing asset/export pipelines.

## Non-goals

- No automatic scraping, copying third-party meshes, texture extraction, or shipping of online photographs.
- No automatic face sculpt or photo-fidelity claim; sparse SfM feature matches do not verify historical likeness or create a production GLB.
- No duplicate implementation of camera fitting currently under review in statue PRs #3, #6, and #8.

## Approach

Selected: **source-neutral reference dossier + optional external SfM worker**. It keeps photo acquisition/review separate from Blender, rejects invalid inputs before spending GPU/CPU time, and leaves camera/mesh output as evidence for a future sculpting workflow.

Alternative 1: integrate COLMAP and OpenMVS deeply into `bpy` now. Rejected because Python/runtime incompatibilities and archival photo overlap failures create a brittle monolith.

Alternative 2: only implement 2D contact sheets. Rejected because this perpetuates an unregistered photo-versus-render comparison and cannot discover whether 3D reconstruction is even feasible.

## Contracts and failure behavior

- A `ReferenceSet` JSON manifest identifies the intended subject and lists local relative photo paths, HTTPS reference URLs, manually reviewed viewpoint labels, and observed named regions. Masks, when provided, must match photo dimensions; black excludes background in COLMAP.
- Bounds: max 256 images, 50 MiB per image, 128 KiB manifest, IDs are stable hyphenated slugs; relative paths cannot escape the manifest directory through `..` or symlinks.
- Images are verified using Pillow decoding, SHA-256 and perceptual near-duplicate hints. Near-duplicates are warnings and never silently removed.
- `reference-audit` reports unique camera/viewpoint evidence and weak regions, but never says a face is accurate or an SfM run will work.
- `reconstruct-photos` creates a new output directory, stages only listed photos/masks, runs a configured COLMAP executable and records commands, output state, and observed registered images; failures are visible and never labeled approved.
- For fewer than three distinct images, the SfM command fails before creating output. If COLMAP lacks a successful sparse model, output remains review-required with diagnostics.
- Tests exercise real PNG/JPEG fixtures, malicious paths, duplicate detection, masks, optional executable absence, and a controlled stand-in for the external CLI boundary.
