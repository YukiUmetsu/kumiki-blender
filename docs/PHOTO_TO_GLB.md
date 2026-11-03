# Reconstruct one 3D mesh from multiple photographs

`build-photo-model` connects the existing camera-registration, dense-mesh and
Blender GLB-export workers into one command. This is **photogrammetry**, not a
procedural reconstruction or Gaussian splat. It can use a private folder of
8–256 overlapping photographs, including a 30-photo monument reference set.

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
