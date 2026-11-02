# Multi-view Gaussian Splat Pipeline (Mac Metal first, CUDA compatible)

Reconstruct a **private, inspectable Gaussian splat** from multiple photographs
of the *same* object taken at different angles. This pathway complements the
[dense mesh reconstruction](DENSE_STATUE_RECONSTRUCTION.md), but does **not**
create an editable polygon mesh, GLB, rigged model, or cinematic MP4. A Gaussian
splat is a camera-dependent point representation that can preserve photographic
appearance when camera registration and image coverage are good.

## Input options

1. `--photos <directory> --asset-id <slug>`: 8–256 independent `.jpg`, `.jpeg`,
   `.png` or `.webp` files from a single object, all kept private. Names may be
   arbitrary; camera angles are estimated from image features, **not** filenames.
2. `--references <manifest.json>`: a curated
   [reference dossier](REFERENCE_RECONSTRUCTION.md) with image/source metadata.
   The photos must still overlap sufficiently for camera registration.
3. `--reconstruction <revision>`: a previously successful
   `reconstruct-photos --undistort` revision with at least 8 registered views.
   This skips camera reconstruction and allows optional `--rectified-masks`.

**For good results:** aim for 30–100 sharp, overlapping views spanning front,
sides, back, and elevated angles. Walk around a stationary object; keep focus,
exposure and lighting consistent, avoid moving backgrounds and highly reflective
surfaces. At least eight distinct files is only an intake minimum, **not** a
quality guarantee. Random internet photos with different seasons, lighting,
crop, telephoto perspective, or insufficient overlap can fail camera alignment.
This is especially important for large statues and human faces.

## Install camera registration

```bash
pip install -e '.[photogrammetry]'
```

The default `--engine pycolmap` runs PyCOLMAP camera reconstruction without
requiring an NVIDIA GPU. To use an already installed `colmap` executable,
select `--engine colmap --colmap /path/to/colmap`. Sparse camera reconstruction
must produce **undistorted images, cameras, and triangulated sparse points** before
Gaussian training can start. Camera registration does not guarantee geometry
or historical accuracy.

## Install OpenSplat (external GPU trainer)

OpenSplat is **not** bundled with Kumiki Blender. It is a separately installed
[AGPL-3.0 project](https://github.com/WebODM/OpenSplat). Check the upstream
build docs and license before using/distributing derivatives or integrating it
into a product. A matching OpenSplat build supports either platform:

**macOS Apple Silicon (MPS/Metal):**

```bash
brew install cmake opencv pytorch
# Install full Xcode and its command-line tools first;
# confirm `xcrun -sdk macosx metal --version` works.
git clone https://github.com/WebODM/OpenSplat.git
cd OpenSplat
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/homebrew/libtorch -DGPU_RUNTIME=MPS
cmake --build build -j 4
./build/opensplat --help
```

`/path/to/homebrew/libtorch` is a placeholder. Find your installed libtorch
CMake prefix following [upstream's macOS instructions](https://github.com/WebODM/OpenSplat#macos).
OpenSplat defaults to MPS when the Metal toolchain is available; otherwise
its build can fall back to **CPU**. Merely running on a Mac does not prove GPU
acceleration. A 24 GB unified-memory Mac may need `--max-image-size 2000`,
fewer photos, or fewer iterations depending on the scene.

**Linux + NVIDIA CUDA:** install a CUDA toolkit and compatible CUDA-enabled
LibTorch, then follow [upstream's CUDA build](https://github.com/WebODM/OpenSplat#cuda).
The Kumiki CLI uses the same OpenSplat executable protocol on both platforms.
It does not install CUDA or choose GPU device at runtime.

## One-command workflow

```bash
# Preview number of inputs and readiness without creating an output directory
kumiki-blender plan-splats \
  --photos /absolute/path/to/statue-angles \
  --asset-id my-statue --output .review/my-statue/splats-r01

# Register overlapping camera poses, rectify photos, train a private splat.
kumiki-blender build-splats \
  --photos /absolute/path/to/statue-angles \
  --asset-id my-statue --output .review/my-statue/splats-r01 \
  --engine pycolmap --opensplat /absolute/path/to/OpenSplat/build/opensplat \
  --max-image-size 2500 --iterations 7000

# Equivalent source-attributed input:
kumiki-blender build-splats --references /path/to/references.json \
  --output .review/my-statue/splats-r02 --opensplat /path/to/opensplat

# Or reuse already registered and undistorted camera evidence:
kumiki-blender build-splats --reconstruction .review/my-statue/reconstruction-r01 \
  --output .review/my-statue/splats-r03 --opensplat /path/to/opensplat
```

To isolate background pixels, add **only reviewed, camera-matched, rectified**
binary/grayscale masks to the existing `--reconstruction` route using
`--rectified-masks <folder>`; see the dense reconstruction mask guide. A mask
file for `photo-0001.jpg` must be named `photo-0001.jpg.mask.png` in that input
folder. It is staged as `dataset/masks/photo-0001.png` for OpenSplat, where
white means keep and black means ignore. Original unrectified masks are not
interchangeable with undistorted images.

## Revision output and verification

```
splats-r01/
  sparse-reconstruction/  # Only when input is --photos/--references
  dataset/images/         # Pinned undistorted images
  dataset/sparse/0/       # COLMAP registered cameras and 3D points
  dataset/masks/          # Only when reviewed masks are supplied
  dataset/dataset-report.json
  splat.ply               # Or splat.spz when --format spz
  splat-report.json       # Actual process exit code, source, SHA-256, status
  logs/opensplat.log
```

- Inputs, image hashes, camera registration and training execution are audited.
  Preflight rejects output directory reuse and tampered registered photos.
- Registered view QA checks the *actual* COLMAP camera-to-photo mapping,
  rectified camera dimensions, finite normalized camera poses, sparse binary
  record counts/lengths and a nonzero translation baseline. The report
  summarizes distinct camera centers and optical-axis separation;
  narrow-angle capture is warned about, not
  automatically rejected (parallel-camera stereo can still be valid).
- Failed registrations or GPU runs produce a **failed/review-required** report.
  Only a verified output artifact is marked `candidate / visual-review-required`.
- `.ply` checks Gaussian attributes, vertex count, data length, SHA-256, and
  samples up to 256 Gaussian records for non-finite values. It also reports
  sparse point counts and COLMAP reprojection evidence when available.
- `.spz` checks NGSP v4 or legacy gzip framing, point count, and (for v4) the
  compressed stream-size table. A dedicated SPZ decoder is still needed to
  verify compressed attributes. Neither check proves photographic likeness.
- Inspect the splat in a Gaussian-splat viewer (for example
  [SuperSplat](https://playcanvas.com/supersplat/editor)) and compare it with
  the source views, including thin geometry, faces and unseen sides.
- These are **private artifacts** with `distribution_allowed: false` until
  image, source, training-tool, and model licenses have been reviewed. Do not
  commit photos, splats or training data to the repository.

For conventional editable 3D models, continue with the separate dense
photo-to-mesh / Blender GLB workflow. This CLI does not yet render splats to
MP4 or integrate them into Kumiki-UI; that requires a dedicated splat-capable
viewer and camera/render pipeline.
