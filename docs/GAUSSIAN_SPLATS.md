# Multi-view Gaussian Splat Pipeline (Mac Metal first, CUDA compatible)

Reconstruct a **private, inspectable Gaussian splat** from multiple photographs
of the *same* object taken at different angles. This pathway complements the
[dense mesh reconstruction](DENSE_STATUE_RECONSTRUCTION.md), but does **not**
create an editable polygon mesh, GLB, or rigged model. It can optionally render
a real Gaussian-splat camera tour to a standalone MP4. A Gaussian
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

## GPU Gaussian-splat video and viewer (Mac or CUDA GPU)

Video rendering uses [PlayCanvas SplatTransform](https://github.com/playcanvas/splat-transform)
to rasterize *actual Gaussian splats* to a deterministic WebP camera-track frame
sequence; FFmpeg encodes H.264 MP4. The rasterizer uses WebGPU and needs a
working GPU adapter, including the appropriate macOS Metal or desktop GPU
driver. This is separate from OpenSplat's MPS/CUDA *training* backend. The
local video output does **not** come from an invented polygon mesh.

```bash
npm install -g @playcanvas/splat-transform
brew install ffmpeg  # macOS (or install FFmpeg from your Linux package manager)
splat-transform --list-gpus

# Check source and installed programs before rendering anything.
kumiki-blender plan-splat-video --splat .review/my-statue/splats-r01 \
  --output .review/my-statue/video-r01

# Auto-frame a trimmed sample of the PLY, add an eased 115° orbit,
# render 6 seconds at 1080p/30fps, and generate an optional local HTML viewer.
kumiki-blender render-splat-video --splat .review/my-statue/splats-r01 \
  --output .review/my-statue/video-r01 --viewer

# Manual framing for a statue or for SPZ (automatic SPZ bounds unavailable).
kumiki-blender render-splat-video --splat /path/to/splat.spz \
  --output .review/my-statue/video-r02 \
  --target 0 1.5 0 --distance 4 --elevation-degrees 16 \
  --start-degrees -75 --orbit-degrees 150 --seconds 8 --fps 30 \
  --width 1920 --height 1080 --shutter 0.5

# Use an authored JSON camera track instead of the generated orbit.
kumiki-blender render-splat-video --splat /path/to/splat.ply \
  --camera-track /path/to/track.json --fps 30 --seconds 6 \
  --output .review/my-statue/video-r03
```

`track.json` follows SplatTransform's schema, including **one pose per frame**:

```json
{"frameRate": 30, "frames": [
  {"position": [2, 1, -2], "target": [0, 0, 0], "up": [0, 1, 0], "fov": 50},
  {"position": [2.1, 1, -1.9], "target": [0, 0, 0], "up": [0, 1, 0], "fov": 50}
]}
```

The example shows the shape only. A six-second 30fps video needs 180 poses,
not two. SplatTransform also accepts SuperSplat editor project directories and
viewer settings with animation tracks; those must render exactly the specified
frame count. Prefer the generated `camera-track.json` as a reusable starting
point. For recorded or hand-authored camera paths, preview a short section to
avoid camera/object intersections, blank frames and inverted orientations.

```
video-r01/
  camera-track.json     # Generated, editable frame-accurate camera poses
  frames/frame.0000.webp ... frame.0179.webp  # GPU-rasterized output
  video.mp4             # Verified H.264 frame count, resolution, FPS, duration
  contact-sheet.jpg     # Six equally spaced review thumbnails
  viewer.html           # Optional, private self-contained viewer
  logs/render.log
  logs/ffmpeg.log
  logs/decode-audit.log
  logs/contact-sheet.log
  video-report.json     # Source SHA-256, commands, failures, output SHA-256
```

- The output directory is never overwritten. An incomplete GPU render leaves
  a failed report and retains its logs; missing/extra frames prevent encoding.
  The final MP4 is decoded to a null sink to catch corrupt bitstreams, and the
  input splat hash is rechecked after rendering. A contact sheet supports, but
  does not replace, watching the complete clip in motion.
- An authored JSON camera track is snapshotted in the output revision so
  subsequent edits cannot silently change a video revision. For SuperSplat
  project directories, the external dependency is recorded but not copied.
- For `.ply`, auto-framing uses a robust *sampled* bounding volume, not a
  verified artistic composition. Override `--target` and `--distance` for
  close-ups, off-center faces, giant statues or foreground-heavy captures.
- For `.spz`, provide a camera track or explicit target and distance.
- `--shutter` enables optional GPU motion blur. Rendering and encoding are
  silent by design; narrations and sound effects belong to ContentGen's
  authoritative audiovisual timeline.
- The MP4 probe checks technical properties, **not** visual quality. Review
  motion at normal playback speed, faces, background floaters, occlusions,
  holes and source agreement from multiple views. A Mac/Metal or CUDA render
  must be run and inspected on the corresponding GPU before claiming success.
- Generated viewer HTML can contain the full splat. Keep all video/viewer files
  private until source and license rights are verified. Don't publish the
  viewer by default.

For conventional editable 3D models, continue with the separate dense
photo-to-mesh / Blender GLB workflow. A shared Kumiki-UI scene/asset binding
and Blender scene compositing are still separate integration work. A splat
MP4 is useful as a pre-rendered shot; it does not make the object editable in
Blender or riggable in Kumiki-UI.
