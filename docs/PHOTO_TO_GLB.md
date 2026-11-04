# Reconstruct one 3D mesh from multiple photographs

`build-photo-model` connects the existing camera-registration, dense-mesh and
Blender GLB-export workers into one command. This is **photogrammetry**, not a
procedural reconstruction or Gaussian splat. It can use a private folder of
8–256 overlapping photographs, including a 30-photo monument reference set.

For mixed internet photos where left/right-side cameras remain disconnected,
see [learned COLMAP matching and camera coverage comparisons](NEURAL_CAMERA_RECONSTRUCTION.md).

## One-command, quality-gated statue reconstruction (Apple Silicon)

`build-statue-glb` runs a fresh COLMAP SIFT-LightGlue/MPS camera solution,
first prepares an incremental, private normalization/foreground-preview cache,
produces an annotated missing-angle report, prepares SAM2.1/MPS mask
candidates, audits cross-view mask consistency, then runs **resolution-level 0**
OpenMVS without geometry-destroying refinement. When measured elevated or side
coverage is weak, it automatically tries to register missing real cameras
before stopping. If all conservative gates pass,
it exports an editable `.blend` and `model.glb` without rendering previews.

```bash
# First use: remember source photos and automatically choose a fresh output.
kumiki-blender build-statue-glb \
  --photos "$HOME/Pictures/sphinx-reconstruction-candidates/warren-2019/photos" \
  --asset-id great-sphinx

# Add real photos to the same folder, then rebuild with just an asset ID:
kumiki-blender build-statue-glb --asset-id great-sphinx
```

### Live CLI progress

The command now displays a live, single-line phase dashboard in an interactive
terminal. It shows the current phase (such as camera matching, SAM2 masks,
OpenMVS dense geometry or Blender export), elapsed time, recent worker activity,
and a bar counting **completed workflow phases**. For example:

```text
KUMIKI great-sphinx [━━━·············] 25% phases · 3/12 Matching photos and solving cameras · 00:27 · Worker running
```

The overall percentage counts completed phases; it **does not estimate time
remaining**. Real per-worker counts are displayed separately when available,
including COLMAP matched-pair counts, generated SAM2 masks (with cache hits),
and completed OpenMVS steps. Long-running native jobs with no measurable count
still show elapsed time and the latest log activity. In non-interactive terminals,
the command prints readable phase lines plus a heartbeat every 15 seconds.

The progress dashboard writes to **stderr**; the final machine-readable JSON
still writes to **stdout**. Detailed worker logs remain in
`<output>/logs/<phase>.log`, including if a phase fails. To disable the dashboard:

```bash
kumiki-blender build-statue-glb --asset-id great-sphinx --no-progress
```

Run from the same project directory each time. The local private registry
`.review/great-sphinx/asset-state.json` remembers the last photo folder and
allocates `.review/great-sphinx/statue-hq-r001`, `r002`, etc. It records the
source even if quality gates stop the first build. You must specify `--photos`
once per asset ID; older runs predating this registry also require that first
invocation. Override it later with `--photos` or an explicit `--output`.
`--resume` without `--output` first tries the most recently saved revision.
If photos or quality settings changed, it automatically allocates the next
unused revision (such as `r002`), preserves the old result, and starts fresh
while reusing the shared per-photo preprocessing cache. An explicit `--output`
with `--resume` always refuses incompatible inputs instead of switching paths.
The defaults for macOS binaries are `/Applications/OpenMVS-2.4.0` and
`/Applications/Blender.app/Contents/MacOS/Blender`; override either if needed.

Prerequisites: COLMAP, Blender, the OpenMVS executables, and the official
`facebookresearch/sam2` Python package with PyTorch. If the SAM2.1 small
checkpoint is not supplied, the command looks in local environment variables,
the shared Kumiki cache, PyTorch's checkpoint cache, Downloads and a local
`checkpoints/` folder. If absent, it downloads the official weights once over
HTTPS from Meta into `~/.cache/kumiki-blender/models/`. Use
`--no-auto-model-download` for fully offline mode or `--sam2-checkpoint` to
choose another installed model. The checkpoint SHA-256 and origin are recorded.

`--front-photo` is optional. The pipeline looks for a registered photo with an
unambiguous front-facing filename hint and marks it **tentative**, never a
verified anatomical orientation. If none can be identified, camera angles are
reported relative to an arbitrary reference and anatomical left/right coverage
is not claimed. Explicit `--front-photo` remains an override.

### Automatic camera recovery from existing real photographs

When the initial camera coverage gate fails and unregistered real photographs
remain, `build-statue-glb` automatically attempts an independent recovery
under `<output>/camera-recovery/`:

1. COLMAP `image_registrator` solves new PnP cameras using existing verified
   2D-to-3D correspondences. Original registered cameras and source photos are
   never overwritten.
2. If some camera poses register but elevated photographs remain missing,
   COLMAP triangulates new **real 3D tracks** and retries registration (at most
   two more rounds). The first successful registration does not stop the search.
3. Isolated, targeted COLMAP cohorts are attempted for still-missing elevated
   source photos, prioritizing **registered-to-missing** verified pair seeds.
   The recovery tries existing geometrically verified pairs in descending
   confidence tiers (25, 18, 12, then 10 inliers), each with a separate
   immutable output and a bounded time budget. These lower counts affect
   seed selection only: they **do not** relax 3D-track, shared-camera,
   similarity-alignment, model-merging, or camera-drift acceptance checks.
   Inspect `camera-recovery-report.json` → `cohort_attempts` for attempted
   thresholds, reasons and trial outcomes.
4. If those geometric attempts fail to recover the elevated candidates, an installed MapAnything model can
   estimate camera/depth hypotheses on an **8-photo MPS batch** emphasizing
   missing elevated views. These hypotheses are kept as private diagnostics;
   they never count as registered cameras without verified photo tracks.
5. When a new COLMAP camera solution passes the real-track, camera-ID and
   baseline stability checks, the system builds a new undistorted/rectified
   camera revision, reruns the coverage audit and can retry once more from that
   larger verified camera set. **Only if the quality gate passes** does the
   workflow use it for SAM2 masking and OpenMVS geometry.

This can recover photographs that had enough visual overlap but no original
camera pose. It cannot guarantee recovery from non-overlapping images and it
never silently substitutes AI-generated intermediate images for real evidence.
An older blocked `--resume` revision with an unsuccessful second recovery
automatically adds a `camera-recovery-confidence/` attempt using the new pair
tiers, without rerunning the original camera matching or overwriting the old
cohort diagnostics.
MapAnything is optional (`pip install -e '.[mapanything]'`); the COLMAP recovery
strategies run without it. It uses Apple MPS and a bounded group rather than
all 57 photos in one neural forward pass. Initial model weights may still need
to be downloaded from the model provider. Disable automatic recovery with
`--no-camera-recovery`, or inspect its standalone result with:

```bash
kumiki-blender recover-registered-cameras \
  --reconstruction .review/great-sphinx/statue-hq-r001/cameras \
  --output .review/great-sphinx/manual-camera-recovery-r01
```

If no guide file is supplied, SAM2 automatically samples conservative blue-sky
negative points and starts with a broad subject prompt. Those mask candidates
are reviewed by cross-view and contamination QA, not treated as hand-labeled
statue parts. The output remains **private and unverified** even after automated
gates pass. Optional `--mask-guides /private/guides.json` supplies head/headdress/torso/
paws/tail boxes and positive/negative points (see
[`examples/statue-mask-guides.example.json`](../examples/statue-mask-guides.example.json)).
Guide keys identify the registered `photo-0001.png`-style rectified images in
the camera revision, not necessarily their original downloaded filenames.
Automatic guidance cannot guarantee correct anatomy; background leakage,
incomplete silhouettes and weak geometric registration still stop the run.

On a blocked run, inspect `statue-pipeline-report.json`, `coverage/photo-advice.md`,
`masks/mask-contact-sheet.jpg`, and the SHA-named mask audit contact sheet.
Additional real photos require a **fresh output revision**: the input hashes
cannot change within a revision. Omit `--output` and the CLI allocates the
next one automatically, even if `--resume` is present. To correct masks, run the existing
`refine-dense-masks` command against the same `cameras/` revision, then
reissue the same `build-statue-glb` invocation with `--resume --masks-dir
/path/to/corrected-revision/masks`. The resumed run reuses the completed
camera solution, verifies the corrected mask revision's camera provenance
and per-image hashes, re-audits the new masks, then continues if its gates pass.

When a reliable front reference exists, the gates require at least two registered
side views on **each** side beyond ±65°. They also require two elevated cameras above
12°, the configured registration fraction (80% by default), bounded mask
foreground, no missing prompted anatomy or explicit background violations,
under 12% cross-view label disagreement, and no obvious blue-sky leakage.
Existing unregistered photos described as `needs-overlap-bridge` are **advice**,
not an automatic failure: 2D–3D track support can be weak even when overlapping
photos exist. The pipeline writes `coverage/cohort-recovery-plan.json` for
unregistered views when the source match graph supports that diagnostic.
Multiple verified neighbors now prioritize camera recovery, not another photo
request. Try camera recovery and intrinsic diagnostics before acquiring
additional images. AI-generated in-between photographs are **not independent
geometric measurements** and must never be silently added as real COLMAP
inputs; they are suitable only for separately labeled creative references.

### Optional AI bridge previews (ComfyUI)

```bash
# Automatically discover a local checkpoint and generate a ComfyUI workflow:
kumiki-blender build-statue-glb --asset-id great-sphinx --ai-bridges

# With a locally installed ComfyUI API workflow, generate preview candidates:
kumiki-blender build-statue-glb --asset-id great-sphinx \
  --ai-bridge-workflow /path/to/comfyui-api-workflow.json
```

The existing `generate-completion-image` adapter can produce up to three
private synthetic reference previews from **verified, real photos**. When
`--ai-bridges` is enabled without a workflow, the command asks your **running
local ComfyUI** for installed checkpoints and creates an API-format
`auto-comfyui-workflow.json` using standard checkpoint, VAE, conditioning and
image-to-image nodes. If ComfyUI is stopped, it also checks the usual
`~/ComfyUI/models/checkpoints` and `~/Documents/ComfyUI/models/checkpoints`
directories (and the optional `COMFYUI_DIR` or `KUMIKI_COMFYUI_DIR` root).
You still need to start ComfyUI to perform inference. It selects a local model
deterministically, preferring
smaller SD 1.5 checkpoints on Apple Silicon. No model is downloaded or
approved automatically. It never invents a checkpoint that is not installed.
A configured workflow needs exactly one `LoadImage` and one `SaveImage`, with
local-only approved ComfyUI nodes and already installed model checkpoints.
ComfyUI must be running at `http://127.0.0.1:8188`; remote hosts and redirects
are rejected. Set `KUMIKI_COMFYUI_BRIDGE_WORKFLOW` to discover a saved workflow
without passing a path (still opt in via `--ai-bridges`).

Candidates, source/anchor hashes and provenance are stored separately under
`<output>/ai-bridge-previews/`. If ComfyUI is unavailable or no compatible
checkpoint is listed, the command writes a **plan only** and a diagnostic,
without claiming an AI model ran. Generated images are
artistic guesses and **cannot** supply verified camera poses, independent
overlapping photo evidence, or accurate unseen geometry. This pipeline never
adds them to COLMAP or the final GLB.
Camera azimuth directions are relative to the **front image**
when known, not the Sphinx's verified anatomical axes. These are deliberately conservative
indicators, not proof of a watertight mesh, a faithful face, or accurate scale.
An unattended model with an incomplete left side is worse than a paused run
with an actionable missing-photo checklist.

The original `build-photo-model` command below remains available for a quick,
unmasked diagnostic GLB and continues to use its unchanged defaults.

### Incremental cache when adding photos

Before matching cameras, every original photo is processed into a persistent
private `<asset-output-parent>/_statue_photo_cache/` directory (override with
`--photo-cache-dir`). It stores:

- A byte-verified, EXIF-corrected RGB `normalized.png` used for COLMAP matching.
- A SAM2.1 subject-mask **candidate** and transparent `isolated-*.png`
  foreground preview, with model hashes and quality warnings. These are not
  approved geometry masks and **are never substituted for original photographic
  evidence during SfM**. The camera-derived rectified masks are generated and
  verified separately downstream.
- `normalized.json` / `foreground-*.json` integrity receipts and a per-run
  `photo-preprocess/photo-preprocess-report.json` index linking originals to
  cached pixels and recording cache hits/misses.

The cache uses **original image SHA-256, normalization version, checkpoint
SHA-256 and SAM2 settings**, not filenames or ordinal `photo-XXXX` identifiers.
When a new photo is added, only the new photo incurs normalization and SAM2
preprocessing; unchanged images are verified and reused, even when adding a
file shifts their staged names. Editing a photo or changing the SAM2 model
invalidates the affected cache keys. Interrupted preprocessing can resume
without recomputing completed, valid entries. The existing new-photo camera
matching still needs to solve correspondences with the updated photo set.

For an existing blocked reconstruction, use the usual `--resume`. It prepares
the new source cache and can run a second recovery pass from the already
verified camera revision without repeating completed original matching.

### Rectified segmentation cache

Each new `build-statue-glb` output revision uses the persistent sibling folder
`_statue_mask_cache/` (override with `--mask-cache-dir`). Automatic SAM2 mask
results for **exactly unchanged rectified pixels** are reused, including the
same model checkpoint, model size, image dimensions, processing resolution
and per-image anatomy guidance. The report records `mask_cache.hits/misses`
and each candidate mask's `cache_hit` and content key. Cache entries contain
only binary masks and small provenance receipts, never the original photos.

Adding a new photo can change estimated cameras and image undistortion. A
changed rectified image correctly results in a cache miss; old masks are **not**
resized into a different camera grid. COLMAP camera solving, new photo-pair
matching and OpenMVS depth maps still need fresh work when the source set
changes. A stopped run on the *same* images can instead use `--resume` to skip
its completed stages. Neither mechanism fabricates missing viewpoints.

To regenerate after adding photos, run
`kumiki-blender build-statue-glb --asset-id great-sphinx`; it chooses the next
unused output automatically. The cache from `r001` is discovered as long as
both outputs share a parent directory.
On an entirely cached re-mask, the official SAM2 model is not loaded at all;
cache misses are recomputed with the installed model and a verified checkpoint
hash. To inspect reuse, read `mask_cache` in `statue-pipeline-report.json`.

## Mac (Apple Silicon)

Install Python with `pip install -e '.[photogrammetry,qa]'` and Blender separately.
Build/install the **OpenMVS command-line utilities**, and ensure these names
resolve on your `PATH`: `InterfaceCOLMAP`, `DensifyPointCloud`,
`ReconstructMesh`, `RefineMesh`, `TextureMesh`. Install `pycolmap` in the Python
environment running `kumiki-blender`. `blender` must also be on your `PATH`
or supplied with `--blender /full/path/to/blender`.

```bash
kumiki-blender plan-photo-model \
  --photos /path/to/sphinx-photos --asset-id great-sphinx \
  --output .review/great-sphinx/photo-r01

# Only start after the plan reports ready-to-reconstruct:
kumiki-blender build-photo-model \
  --photos /path/to/sphinx-photos --asset-id great-sphinx \
  --output .review/great-sphinx/photo-r01
```

For a curated dossier with source URLs, viewpoints and optional object masks,
replace `--photos DIR --asset-id NAME` with `--manifest PATH`. Run `plan` and
`build` with the **same unused output directory**: `plan` never creates it.
Each `build` run consumes that destination; for retries use a new revision,
such as `photo-r02`, to preserve diagnostic evidence.

For a CUDA-enabled computer, select `--engine colmap --backend colmap`, after
installing COLMAP with GPU support. This path does not presently create a
photographic texture atlas: the GLB receives the dense mesh's material.
On Mac, the default `--engine pycolmap --backend openmvs` supports CPU sparse
registration and CPU dense meshing, with OpenMVS texture mapping enabled.
To skip texture transfer, pass `--no-texture`.

## What gets created

On success, the private output folder contains:

| File | Meaning |
| --- | --- |
| `model.glb` | Blender-exported mesh, re-audited as glTF 2.0 with geometry |
| `review/private-*.blend` | Editable Blender file with imported reconstructed geometry |
| `review/renders/` | Axis QA renders and up to eight real registered-camera views (unless `--no-renders`) |
| `sparse/` | Registered cameras, undistorted photos, COLMAP logs and evidence |
| `dense/` | Reconstructed geometry, textures when available, meshing reports |
| `photo-model-report.json` | Stage status, actual registered counts, final GLB SHA-256, blockers |

The GLB is a **private candidate, not an approved likeness**. A passing GLB
format audit does not measure historical facial similarity, correct scale,
surface roughness, right-of-publicity, or photo/texture redistribution rights.
Keep generated meshes and photos outside Git and review source licenses before
publishing texture-derived assets.

## When 30 photos still cannot reconstruct a 3D mesh

Eight distinct image files do **not** mean eight overlapping cameras.
Internet photos may be compressed, cropped, shot from nearly identical angles,
or have tourists/scaffolding hiding key features. A dataset dominated by front
and left views may not recover the Sphinx's right cheek or back at all.

1. Check `photo-model-report.json`: `failed_stage` names the failing stage.
2. For camera failures, read `sparse/reconstruction-report.json` and logs.
   If fewer than eight images register, use more photos **with adjacent,
   overlapping viewpoints**, not just more unrelated images.
3. For missing or patchy mesh surfaces, inspect `dense/dense-report.json` and
   try reviewed masks or more overlapping source views. Existing advanced
   `prepare-dense-masks` and `build-dense-photos` commands remain available.
4. For Blender export failures, inspect `logs/blender-export.log` and
   `review/blender-review-report.json`; no `model.glb` is promoted on failure.
5. Compare the registered-camera renders in `review/renders/` with their
   corresponding undistorted photographs before claiming the statue is faithful.

The pipeline deliberately **never substitutes an illustrative Sphinx** for
unrecovered geometry. A measured scan/survey or artist sculpting can still be
needed when photographs do not contain reconstructible details.

## Alternative camera registration: MapAnything (`--engine mapanything`)

For step-by-step usage, tuning and troubleshooting, see
[MAPANYTHING_PHOTO_REGISTRATION.md](MAPANYTHING_PHOTO_REGISTRATION.md).

When COLMAP registers too few photos (for example 4 of 14 in a Mac smoke run),
`--engine mapanything` predicts a camera pose, intrinsics and depth for every
photo in one feed-forward pass instead of matching features. It is an
alternative *sparse* stage; the dense OpenMVS stage and Blender export are
unchanged.

```bash
pip install -e '.[photogrammetry,mapanything,qa]'   # torch + MapAnything pinned commit
kumiki-blender build-photo-model \
  --photos /path/to/sphinx-photos --asset-id great-sphinx \
  --output .review/great-sphinx/mapanything-r01 \
  --engine mapanything --mapanything-device auto
```

- **Devices.** `--mapanything-device auto` picks CUDA, then Apple MPS, then CPU.
  An explicit device fails if it is unavailable rather than falling back. On
  Mac, unsupported MPS operators fall back to CPU per operator.
  For CUDA, install a CUDA build of torch first, as usual for PyTorch.
- **Model and license.** The default `facebook/map-anything-apache` is
  Apache-2.0. `facebook/map-anything` is CC-BY-NC-4.0 and is recorded as such in
  the report. Do not distribute non-commercial-licensed outputs.
- **First run downloads.** The checkpoint cache is about 9 GB from Hugging Face,
  and the DINOv2 hub code comes from GitHub through torch hub. The first load
  took about 2.5 minutes on the test Mac. Pre-cache both for offline machines.
- **Inputs.** Each photo is scaled so its longest side is 518 px and pasted,
  without cropping, onto a shared 518×518 canvas. MapAnything resizes and crops
  a batch to one aspect ratio, so the canvas keeps the whole photo and its
  pixels and intrinsics stay consistent. Photos with object masks are rejected
  for this engine.
- **What is registered.** The largest group of views whose predicted 3D points
  re-observe another view with agreeing depth: at least 30% of sampled points
  must land on the other view's predicted depth within
  `--mapanything-depth-tolerance` (default 0.30, i.e. 30%; allowed 0.01–1.0).
  Views outside that group are listed in `reconstruction-report.json` as
  `unregistered_filenames`. A registered view means *mutually consistent
  prediction*, not a verified camera.
- **  Choosing the tolerance.** On a 26-photo statue set, 0.05 kept 3 views, none
  of which COLMAP registered; 0.20 kept 7 (3 shared with COLMAP); 0.30 kept 21
  (5 of COLMAP's 6). The 0.30 default was picked while looking at these same 26
  photos and COLMAP's registrations on them, so the 21-of-26 figure is in-sample.
  Confirm on a second photo set against the COLMAP baseline before trusting extra
  views.
- **Outputs.** The same `undistorted/` COLMAP-text revision the dense stage and
  camera plan already read, plus `mapanything/worker-meta.json` (device, AMP
  mode, torch version). Status is still `candidate / visual-review-required`.
  Raw predictions in `mapanything/predictions.npz` are about 4–5 MB per photo;
  keep them under `.review/`.

The report keys `engine`, `model_id`, `model_license`, `covisibility` and
`worker` identify the run. Failures use the same `failed_stage` names as the
COLMAP path, with `mapanything-inference` and `mapanything-output` added.

Known limits: the 518 px canvas caps per-view resolution, so fine relief depends
on the dense stage and on the photos; the depth-agreement gate can miss
legitimately sparse views; and nothing here verifies facial likeness, scale or
photo rights.
