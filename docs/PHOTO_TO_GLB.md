# Reconstruct one 3D mesh from multiple photographs

`build-photo-model` connects the existing camera-registration, dense-mesh and
Blender GLB-export workers into one command. This is **photogrammetry**, not a
procedural reconstruction or Gaussian splat. It can use a private folder of
8–256 overlapping photographs, including a 30-photo monument reference set.

For mixed internet photos where left/right-side cameras remain disconnected,
see [learned COLMAP matching and camera coverage comparisons](NEURAL_CAMERA_RECONSTRUCTION.md).

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
