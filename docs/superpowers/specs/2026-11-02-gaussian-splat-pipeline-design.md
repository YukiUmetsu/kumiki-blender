# Mac-first Gaussian-splat pipeline design

## Intent

Make photographed objects, especially historical sculptures, usable as view-dependent Gaussian-splat assets for premium video scenes. The primary authoring machine is an Apple Silicon Mac; a CUDA workstation should use the same commands and asset contract.

## Approaches

- **Selected: external OpenSplat CLI.** The upstream trainer accepts COLMAP cameras and sparse points, trains on Apple Metal/MPS or NVIDIA CUDA depending on its build, and exports Gaussian-splat PLY/SPZ. Keep the AGPL-3.0 program separate from this MIT Python repository; do not vendor its code or install GPU dependencies into Blender's Python.
- **Deferred: MLX trainer plus Nerfstudio CUDA adapter.** This would make separate training, configuration and output contracts necessary, and Nerfstudio's Splatfacto is not Mac MPS-ready.
- **Rejected for this feature: mesh/GLB conversion.** A splat is not a polygonal mesh, and converting it to GLB cannot preserve the same view-dependent appearance.

## User-facing input

1. `--photos /private/object-photos --asset-id statue-id`: a folder of 8–256 local JPEG/PNG/WebP reference photographs of the *same* stationary object, from overlapping viewpoints. This convenient mode has unknown provenance and does not grant distribution rights.
2. `--references /private/references.json`: the existing source-attributed multiview `ReferenceSet` manifest, with view/region coverage audits.
3. `--reconstruction /private/colmap-r01`: an already registered and undistorted Kumiki COLMAP revision, avoiding repeat SfM.

The modes are mutually exclusive. Photo and manifest modes run the existing COLMAP/PyCOLMAP sparse-registration and undistortion stages first. A non-mutating planning command reports input readiness; the build command executes the stages.

## Workflow and artifact contract

1. Read and validate photo input. Check count, decoding, hashes, duplicate evidence, symlinks, path containment and source-specific rights status. Never infer camera position from filename labels or invent invisible surfaces.
2. Require real registered COLMAP camera data (8 or more registered independent photos and at least 60% registration by default). Rectify image/camera pairs using the existing worker; refuse unrectified camera files for splat training.
3. Stage immutable-by-convention private copies into OpenSplat's canonical `dataset/images/` and `dataset/sparse/0/`. Optionally stage *rectified*, size-checked foreground masks with names OpenSplat recognizes. Never reuse original distorted masks as rectified masks.
4. Invoke a user-installed `opensplat` executable with a bounded iteration count and timeout, outputting a PLY by default, with optional SPZ. Capture logs and process outcome. The OpenSplat build determines whether it uses Apple Metal, CUDA, or CPU; do not mislabel the actual accelerator without evidence.
5. Verify output exists and is nonempty. For PLY, validate the bounded header, nonzero vertex count and Gaussian attributes. Record SHA-256, size, input hashes, settings, tool path/version when available, and an explicit `candidate / visual-review-required` status.
6. Keep reference photographs, sparse workspace and splat artifacts private by default. The CLI never publishes a model, grants redistribution rights or certifies monument likeness.

The result folder contains `dataset/`, a splat file, `logs/`, `splat-report.json`, and, for raw photo inputs, a private `sparse-reconstruction/`. It does *not* claim to produce a polygonal GLB or finished video.

## Non-goals for this PR

- Training without a real overlapping multiview photo set.
- Downloading remote images or clearing copyright/license rights.
- Fully automatic foreground segmentation or sculptural editing.
- Implementing a custom Metal/CUDA rasterizer or bundling OpenSplat.
- Cinematic MP4 rendering (a separate viewer/Kumiki-UI integration consumes the splat and camera metadata).

## Verification

- Unit and CLI tests for three inputs, invalid photos, symlinks, duplicates, insufficient cameras, masks and output format validation.
- A fake local trainer verifies actual command execution, status handling, logs and manifests without GPU/network access.
- Existing reconstruction tests and the full Python suite remain green. Real Metal/CUDA visual quality is separately marked unverified until run on a compatible GPU with a rights-cleared dataset.

## Risk boundaries

- Never treat Gaussian-splat PLY as a regular mesh PLY; GLB/mesh validation is a separate contract.
- Never silently fall back to CPU while claiming Metal acceleration.
- Never delete or overwrite an existing revision.
- Never follow unchecked symlinks or pass user input through a shell.
- Never certify photographic likeness using only source count, camera registration or file integrity.
