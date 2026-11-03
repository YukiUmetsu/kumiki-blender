# MapAnything camera registration: how to use it

`--engine mapanything` is an alternative to COLMAP for the camera-registration
step of [photo-to-GLB](PHOTO_TO_GLB.md). Use it when COLMAP registers too few
photos of an object. It predicts pose, intrinsics and depth for every photo in
one pass, then keeps the largest group of views whose 3D predictions agree with
each other. Dense meshing and Blender export are the same as before.

Read the [PHOTO_TO_GLB guide](PHOTO_TO_GLB.md) first if you have not built a
photo model before. This page covers only what changes with MapAnything.

## 1. Requirements

| Needed for | Requirement |
| --- | --- |
| Camera registration | Python 3.11+, `torch`, `mapanything` (this repo's `mapanything` extra) |
| Full photo-to-GLB | Also OpenMVS (`InterfaceCOLMAP`, `DensifyPointCloud`, `ReconstructMesh`, `RefineMesh`, `TextureMesh` on `PATH`) and Blender |
| First run only | Internet access to `huggingface.co` (checkpoint, about 9 GB) and `github.com` (DINOv2 hub code) |

Install:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[photogrammetry,mapanything,qa]'
```

The extra pins MapAnything to commit `3d10cf7`. Installing it needs git access
to GitHub.

Check the install and see which accelerators are visible:

```bash
python -c "import torch, mapanything; print('mps', torch.backends.mps.is_available(), 'cuda', torch.cuda.is_available())"
```

**NVIDIA/CUDA:** install a CUDA build of torch before the `pip install -e` line,
using the index that matches your driver from pytorch.org. The code path is
written for CUDA, but it has not been run on NVIDIA hardware yet. See
[Limits](#10-limits).

## 2. Choose the photos

- **8–256 photos** of the same object, from **overlapping** viewpoints. Photo
  count alone does not guarantee overlap. Adjacent angles help more than extra
  unrelated shots.
- **No object masks.** Masks are rejected by this engine. Use `--engine pycolmap`
  if you need them.
- Keep the photos in their own folder. The output folder must be outside it.
- Check photo rights. The report sets `missing_photo_rights` when a photo has
  no source URL.

## 3. Preflight (read-only)

Run this first. It checks the photos and the tools, and it does not create the
output folder.

```bash
kumiki-blender plan-photo-model \
  --photos /path/to/photos --asset-id my-object \
  --output .review/my-object/mapanything-r01 \
  --engine mapanything
```

Read `status` in the JSON it prints:

- `ready-to-reconstruct`: all dependencies are present.
- `blocked-missing-dependencies`: `missing_dependencies` lists what to install.
  On a Mac without OpenMVS and Blender, expect OpenMVS and `blender` to be listed.
  Camera registration alone does not need them; the full build does.

## 4. Build

```bash
kumiki-blender build-photo-model \
  --photos /path/to/photos --asset-id my-object \
  --output .review/my-object/mapanything-r01 \
  --engine mapanything \
  --mapanything-device auto \
  --mapanything-depth-tolerance 0.30
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--mapanything-device` | `auto` | `auto` picks CUDA, then Apple MPS, then CPU. An explicit device fails if it is not available. |
| `--mapanything-model` | `facebook/map-anything-apache` | Hugging Face checkpoint. Apache-2.0. The other option, `facebook/map-anything`, is CC-BY-NC-4.0 and is recorded in the report. |
| `--mapanything-depth-tolerance` | `0.30` | How closely a view's depth must match another view's prediction for them to count as consistent. Range 0.01–1.0. See [Tuning](#5-tune-the-depth-tolerance). |
| `--min-registered-images` | `8` | Still applies. Fewer registered views stop the build at `sparse-registration`. |

Each build writes a new output folder and never overwrites one. To retry with
different settings, use a new revision name such as `mapanything-r02`.

**Run time on the test Mac** (Apple MPS, fp32, 26 photos): about 100 s once the
model is cached. The first run downloads about 9 GB; the first model load took
about 2.5 minutes including that download. Run time grows with photo count.

## 5. Tune the depth tolerance

The default 0.30 was chosen from one photo set. Check it on your own photos
against COLMAP:

1. Run the existing COLMAP path to get a baseline (no MapAnything flags):

   ```bash
   kumiki-blender build-photo-model --photos DIR --asset-id NAME \
     --output .review/NAME/colmap-r01 --engine pycolmap
   ```

   On a machine without OpenMVS and Blender this stops at `dense-mesh` or
   `blender-export`. The camera counts are still recorded under `sparse` in the
   report, which is all this comparison needs.

2. Run MapAnything at two or three tolerances, each in a new folder:

   ```bash
   for t in 0.20 0.30; do
     kumiki-blender build-photo-model --photos DIR --asset-id NAME \
       --output .review/NAME/mapanything-t$t --engine mapanything \
       --mapanything-depth-tolerance $t
   done
   ```

3. Compare `registered_images` in each `reconstruction-report.json`. Then check
   whether the views COLMAP registered
   (`undistorted/text-model/images.txt`) fall inside the MapAnything group
   (`registered_filenames`).

On the 26-photo test set: 0.05 kept 3 views (none of COLMAP's 6), 0.20 kept 7
(3 of COLMAP's 6), and 0.30 kept 21 (5 of COLMAP's 6). Higher tolerances keep
more views but agree less with COLMAP. A view outside COLMAP's set is not wrong,
but it has not been confirmed either.

## 6. Read the result

Each revision folder contains:

```
mapanything-r01/
  reconstruction-report.json   # status, counts, covisibility, worker info (start here)
  reference-audit.json         # photo intake audit
  staged-reference-index.json  # photo names, hashes, original sizes
  images/                      # staged copies of the photos
  mapanything-input.json       # exact worker input
  mapanything/
    predictions.npz            # raw per-photo pose, intrinsics, points, depth (about 4-5 MB per photo)
    worker-meta.json           # device used, AMP mode, torch version
  logs/mapanything.log         # worker output, written on every run
  undistorted/
    images/                    # the registered photos, resized to the model frame (no padding)
    sparse/                    # cameras.txt, images.txt, points3D.txt (COLMAP text)
    text-model/                # same files; the camera plan reads this copy
  undistorted-reference-index.json
```

The most useful fields in `reconstruction-report.json`:

| Field | What it tells you |
| --- | --- |
| `status` | `candidate / visual-review-required` means the camera step succeeded. Anything else means it stopped; see `failed_stage`. |
| `registered_images` / `staged_independent_images` | How many photos are in the mutually consistent group, out of the total. |
| `registered_filenames` / `unregistered_filenames` | Which photos were kept and which were not. |
| `covisibility` | `graph_edges` and `largest_component` show how connected the views are. `views_with_valid_pose` counts views whose predicted pose was usable at all. |
| `depth_tolerance` | The tolerance this revision used. |
| `worker` | Device and torch version, so you can tell which accelerator ran the model. |
| `max_focal_asymmetry` | Largest difference between predicted fx and fy. The stored camera uses the mean, so this is the approximation size. |
| `model_license` | `Apache-2.0`, `CC-BY-NC-4.0`, or `unknown-review-required`. |

Even a successful candidate needs visual review. Check the renders the full build
writes under `review/renders/` against the photos, and check the dense mesh
before using it. A passing report does not verify that the poses are correct.

## 7. When it fails

The report names the stage that failed in `failed_stage`:

| `failed_stage` | Likely cause | What to do |
| --- | --- | --- |
| `mapanything-inference` | The worker exited with an error. Common causes: the `mapanything` extra is not installed, the first-run download was blocked, the device was not available, or the machine ran out of memory. | Read `logs/mapanything.log`. Check the install and network, or try `--mapanything-device cpu` to rule out the accelerator. |
| `mapanything-output` | The saved predictions had an unexpected shape. This is a bug, not a photo problem. | Keep the log and report the revision. |
| `insufficient_registered_views` | Fewer than three photos are mutually consistent. | Add photos from adjacent angles, or try a higher `--mapanything-depth-tolerance`. |
| `undistortion_validation` | The camera files could not be written or re-read. | Check disk space and the error text in the report. |
| `sparse-registration` (from `build-photo-model`) | The camera step succeeded, but fewer photos registered than `--min-registered-images` (default 8) or than 25% of the set. | Add overlapping photos, or lower `--min-registered-images` for a deliberate test. |
| `dense-mesh` | OpenMVS did not produce a mesh. | See [PHOTO_TO_GLB.md](PHOTO_TO_GLB.md) for the OpenMVS steps. |
| `blender-export` | Blender did not export a GLB. | See `logs/blender-export.log`. |

## 8. Using only the camera step

There is no command that runs only the MapAnything camera step. `build-photo-model`
always continues into dense meshing and Blender. To get the camera evidence
alone, call the builder from Python. This is the same call the test run used:

```python
from pathlib import Path
from kumiki_blender.reconstruction.splat_input import load_photo_folder
from kumiki_blender.reconstruction.feedforward_sparse import build_feedforward_sparse_reconstruction

photos = load_photo_folder(Path("/path/to/photos"), "my-object")
report = build_feedforward_sparse_reconstruction(
    photos, Path(".review/my-object/camera-r01"),
    model_id="facebook/map-anything-apache",
    device="auto",
    depth_tolerance=0.30,
)
print(report["status"], report["registered_images"], report["unregistered_filenames"])
```

Run it with the same environment as the CLI. The output folder must not exist yet.

## 9. Device notes

- **Apple Silicon (MPS):** the default choice on a Mac. The worker sets
  `PYTORCH_ENABLE_MPS_FALLBACK=1`, so operators that MPS does not support run on
  the CPU. It runs fp32. This is the only device tested end to end.
- **CUDA:** uses bf16 autocast. Not run on NVIDIA hardware yet.
- **CPU:** works, but has not been timed. Expect it to be much slower.

## 10. Limits

- **Resolution.** Each photo is scaled so its longest side is 518 px. Fine
  surface detail therefore comes from the photos only as far as 518 px allows.
- **Gate is a heuristic.** Consistency between predicted depths is evidence that
  the views agree, not that the camera is right. Use visual review.
- **Scale and world frame** come from the model's prediction. They are not
  calibrated to a real-world measurement.
- **No object masks**, and no sparse-only CLI command yet.
- **OpenMVS and Blender steps** after registration were not run end to end on
  this Mac. OpenMVS is not in Homebrew core.
- **Photo rights and likeness** are not checked by this code. The report only
  records what it knows.
