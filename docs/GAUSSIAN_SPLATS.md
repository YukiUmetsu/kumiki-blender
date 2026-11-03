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

### Mixed JPEG, PNG, and WebP photo compatibility

Camera reconstruction stages each image as a **private, standard RGB PNG**
before passing it to COLMAP or PyCOLMAP. This prevents the FreeImage decoder
from silently skipping readable WebP or unusual JPEG inputs. EXIF orientation
is applied to the staged pixels and corresponding masks; transparent pixels
are composited over white. Original photos remain untouched. The
`sparse-reconstruction/staged-reference-index.json` records original filenames,
source SHA-256, staged SHA-256, dimensions and normalization metadata.

After feature extraction, the pipeline checks the actual COLMAP SQLite
database against the staged photo index. If photos were skipped, it stops with
`feature_extraction_incomplete` and lists the missing images, rather than
attempting an unreliable reconstruction.

**If registration still fails**, diagnose the existing revision without
retraining or modifying its photos:

```bash
python3 scripts/diagnose_splat_registration.py \
  .review/mac-splat-smoke/splat-<failed-revision> \
  --contact-sheet /tmp/splat-registration.jpg
open /tmp/splat-registration.jpg
```

The diagnostic shows imported and missing photos, connected matching groups,
isolated images and the strongest verified pairs. A successful import is
necessary but not sufficient: diverse online photos may still lack the
overlap and parallax needed to recover eight consistent camera poses.

### When all photos import but only a few cameras register

The photo decoder is not the remaining bottleneck if
`feature_import.imported_images == feature_import.staged_images`. For example,
a 26-photo monument set with only four registered cameras has insufficient
verified cross-view geometry even if its reprojection error is low.

First inspect the **failed revision without retraining**:

```bash
bash scripts/mac_splat_test.sh diagnose \
  .review/mac-splat-smoke/splat-<failed-revision>
```

Review the contact sheet and the verified-match graph. If the largest connected
group is still tiny, high-recall SIFT may recover more matches, but cannot invent
overlap or parallax between unrelated photographs. The optional, slower CPU
PyCOLMAP profile increases SIFT features, lowers the detection threshold,
attempts affine/domain-size feature extraction when supported, relaxes the
descriptor ratio slightly and enables guided geometric matching. It retains
the same geometric verification and minimum eight-camera / 60% registration
gate; it does not force weak photos into a reconstruction.

```bash
REGISTRATION_PROFILE=high-recall MAX_IMAGE_SIZE=3200 \
  bash scripts/mac_splat_test.sh run ~/Pictures/statue-photos/sphinx sphinx
```

#### Why seeded recovery can register zero new photos

If `bash scripts/mac_splat_test.sh recover <revision> 0` reports
`No images with matches` and `no-registration-improvement`, first inspect
the existing run **without repeating expensive SfM**:

```bash
bash scripts/mac_splat_test.sh recover-qa \
  .review/mac-splat-smoke/splat-20261103T052755Z-7115 0

PHOTO_SUBJECT_TYPE=statue \
  bash scripts/mac_splat_test.sh photo-advice \
  .review/mac-splat-smoke/splat-20261103T052755Z-7115
```

`recover-qa` writes a new immutable `recovery-match-qa.json` with
per-photo seed-match evidence. The second command writes the labeled
contact sheet, Markdown photo triage, and missing-angle recommendations.

The run
has **not** improved the camera model. The updated
`recovery-report.json` now includes `match_evidence`:

- `raw_match_pairs`, `verified_pairs`, and `strong_verified_pairs`
  distinguish stored feature matches from geometrically verified pairs.
- `seed_to_missing_pairs` and `strong_seed_to_missing_pairs` identify
  whether any verified image pairs bridge registered seed cameras to
  unregistered photographs (threshold: 15 inliers).
- `unregistered_photo_advice` reports each missing camera's seed match
  support and the next action: collect an overlapping bridge shot or
  investigate 2D–3D tracks, parallax, and camera calibration.

A zero `Loading matches` line in COLMAP does **not** prove the entire
database contains zero matches; it may reflect mapper filtering or
disconnected matches. A single matched pair is not sufficient proof of
a registerable PnP camera. Do not rerun recovery unchanged and expect
new information. On no improvement, the Mac wrapper automatically
generates the photo-value and missing-angle report from the original
sparse revision (if it can read the diagnostic inputs).

#### Warren 2019 Sphinx: 26/26 registered, OpenSplat native crash

The independent Warren LeMay 2019 cohort achieved **26/26 registered cameras**,
**29,844 sparse points**, 271 strong verified pairs in one connected component,
and **1.262 px median sparse reprojection error**. This is a major SfM
improvement over the mixed-source 7/26 run, but **not a verified surface scan**.
OpenSplat exited with native signal 11 (`SIGSEGV`), so there is **no finished
splat**. Do not rerun COLMAP to debug the native trainer.

Use the existing staged dataset in a fresh, isolated training revision:

```bash
git pull --ff-only
REV=.review/mac-splat-smoke/splat-20261103T063944Z-55327
DOWNSCALE_FACTOR=4 ITERATIONS=300 bash scripts/mac_splat_test.sh train-existing "$REV"
```

This command validates image and camera SHA-256 hashes, **copies** the COLMAP
dataset (OpenSplat may create a cache next to its input), and invokes
`opensplat <new-copy>/dataset -n 300 -d 4 -o <new-output>/splat.ply`.
It preserves all original photos, registered poses and sparse tracks. On failure,
inspect the new `logs/opensplat.log` and `retry-report.json` for the native
exit signal and an optional LLDB stack-trace command.

If the GPU path still crashes, isolate it with a **diagnostic-only CPU run**:

```bash
TRAIN_CPU=1 DOWNSCALE_FACTOR=8 ITERATIONS=100 \
  bash scripts/mac_splat_test.sh train-existing "$REV"
```

CPU may be much slower. A successful CPU run with a failed MPS run would
implicate the native GPU path, but **would not by itself identify the cause**.
The OpenSplat binary should also have a matching Metal library
(`default.metallib`) in its build directory. No fallback is automatically
claimed to fix SIGSEGV without reviewing the log or native backtrace.

**Warren 2019 follow-up:** Both MPS (`-d 4, -n 300`) and CPU
(`--cpu -d 8 -n 100`) retries exited with `SIGSEGV (-11)`. A GPU-only
failure is therefore unlikely; neither result proves a corrupt COLMAP dataset.
Capture a native stack trace from the failed CPU retry before changing data:

```bash
git pull --ff-only
CPU_RETRY=.review/mac-splat-smoke/splat-retry-20261103T125237Z-90435
bash scripts/mac_splat_test.sh crash-qa "$CPU_RETRY"
```

The `crash-qa` command validates the retry report, copies and SHA-256 checks
its staged dataset, probes the binary version and linked libraries, then runs
LLDB with `-o run -k "thread backtrace all"`. Results are stored in a new
`opensplat-crash-qa-*/lldb.log` and `crash-qa.json` without modifying
the original registered photos or COLMAP model. If LLDB cannot launch the
inferior due to macOS developer-tools permissions, the result reports
`debugger-launch-failed` and exits nonzero; that is **not** a native
backtrace. A debugger run without a crash reports `trace-inconclusive`, and a
crash without stack frames reports `crash-seen / no-backtrace`. Only a run
that captures both the crash and stack frames reports
`trace-captured / inspect-log` and exits successfully. Inspect the first
crashing frame and its caller before changing PyTorch, OpenSplat, or COLMAP.

**M4 Mac crash evidence (2026-11-03):** The CPU retry's LLDB trace showed
`EXC_BAD_ACCESS` in `libprotobuf.36.2.0.dylib` with an immediate caller in
`libprotobuf.29.3.0.dylib`, during descriptor initialization. This is strong
evidence of incompatible native C++ Protobuf runtimes in one process, **not**
evidence of bad sparse cameras or a Metal-only crash. C++ Protobuf does not
guarantee ABI compatibility across versions. The full dependency ownership is
not established by the stack frames alone. Run `otool -L` on the executable
and its dependencies to find which library requires each Protobuf version;
rebuild with one coherent library set. Do **not** manually change dylib
symlinks or overwrite the original OpenSplat binary as a shortcut.

`crash-qa.json` now records `protobuf_libraries_in_trace` and an actionable
`native_diagnosis` when multiple versions appear. If LLDB times out *after*
printing a native crash and frames, the result is
`trace-captured / debugger-timeout` rather than discarding useful evidence.
This preserves the fact that the debugger had to be killed while recognizing
that the stack trace was captured.

#### Visual QA after a successful `train-existing` retry

A successful immutable retry is a candidate with `retry-report.json` and
`dataset/dataset-report.json` (not the usual `splat-report.json`). The same
calibrated camera/photo comparisons and orbit-video preflight now accept this
format, checking the recorded dataset-report hash and trained PLY SHA-256.
For example, after the 26-camera M4 retry completed successfully with the
separate `build-homebrew/opensplat` executable:

```bash
RETRY=.review/mac-splat-smoke/splat-retry-20261103T173341Z-77079
bash scripts/mac_splat_test.sh qa "$RETRY"
```

Inspect the generated `multi-angle-contact-sheet.jpg` and
`visual-qa-report.json`. Training at `-n 100 -d 8` is a smoke test, **not** a
finished or photo-faithful Sphinx. If it renders successfully, repeat training
with the same working executable, more iterations and a smaller downscale
factor, using a new immutable retry revision. Mask preparation from the original
`sparse-reconstruction` is still a separate workflow.

The `photo-advice` report now requests **zero additional photos for
registration** when all images are registered. This does not prove full surface
visibility: the collection visibly emphasizes side/rear tourist angles and
has limited close-up frontal facial detail.

#### Grounded photo-angle and bridge recommendations

The photo advisor now prioritizes **existing photos that might be recovered**
before requesting additional images. Its Markdown/JSON report identifies each
recover-first or needs-overlap-bridge image, the best **registered overlap
partner** (ranked by verified matches touching existing 3D tracks), and
candidate 2D–3D count/spatial coverage. The contact sheet also displays
**unverified filename hints** (front, rear, overhead, side) where available;
these are never treated as camera measurements.

If the reference photo is not a manually verified **registered front view**,
the advisor does **not** issue numeric target azimuth/elevation instructions.
An arbitrary COLMAP reference orientation cannot tell someone where to stand
relative to the actual monument. It instead requests overlapping
**transitions between named photos** and identifies weakly supported frontal,
rear or overhead reference shots. The SVG orbit legend states that no
physical target azimuth is available.

For the current Sphinx photos, `sphinx-face7.jpg` is a recover-first
candidate and `sphinx-face10.jpg` needs a bridge, based on the user's
photo-value report. Their 2D–3D candidate counts are **not PnP inliers**.
Keep the face-front, rear and overhead photos as visual references even
if they do not currently register. For heterogeneous internet photographs,
prefer finding coherent, same-source overlapping sequences or an actual
survey/scan over accumulating isolated viewpoints.

#### Find which photos help and which camera angles are missing

After a reconstruction attempt, run **photo-advice** to get a per-image
recommendation, a labeled contact sheet, and a prioritized capture plan:

```bash
bash scripts/mac_splat_test.sh photo-advice \
  .review/mac-splat-smoke/splat-20261103T053355Z-11041 \
  .review/mac-splat-smoke/cohorts-20261103T053322Z-10742/cohort-01/source-index.json

# If you know which REGISTERED photo depicts the front of the statue:
PHOTO_SUBJECT_TYPE=statue PHOTO_FRONT_REFERENCE=photo-0002.png \
  bash scripts/mac_splat_test.sh photo-advice \
  .review/mac-splat-smoke/splat-20261103T053355Z-11041
```

Use `PHOTO_SUBJECT_TYPE=statue|object|building|landscape` (default:
`object`) to tailor recommended framing and shared landmarks to the
subject. Statue advice prioritizes head/neck/torso continuity; buildings
prioritize facades and corners; landscapes prioritize stable ridges and
terrain features; other objects prioritize distinctive overlapping details.

The new `photo-advice-*/` directory contains:

- **`photo-advice.md`**: table of all original photo names, estimated
  reconstruction value, keypoint/verified-match/PnP evidence, measured angles
  where available, and a specific next action. Registered photos also show
  how many 3D tracks they observe and how many **two-view tracks would lose
  triangulation if that camera were removed** (a measured fragility proxy,
  not a leave-one-out SfM rerun).
- **`photo-advice.json`**: machine-readable statuses and prioritized
  suggested photo azimuth/elevation, framing and reason.
- **`photo-advice.jpg`**: contact sheet labeling each original photo
  with status and its known/unknown angle.
- **`photo-angle-map.svg`**: top-down relative camera-orbit diagram, with
  registered camera viewpoints and proposed intermediate shots. Created only
  when a usable reconstructed point center and camera poses exist.

Statuses distinguish **keep**, **recover-first**, **needs-overlap-bridge**,
**low-current-value**, **keep-separate-model**, **keep-check-overlap** and
**cannot-use**. Low current value means the photo does not currently connect
well to this SfM reconstruction. It is **not** a verdict that the image is
unnecessary, poorly composed, or can safely be deleted. An unregistered
image may contain valuable unique surface detail.

**Angles are only computed for registered cameras in the largest sparse
model**, relative to a registered reference view and the median of its
reconstructed 3D points. By default that view is *not* assumed to be the
statue's front. Set `PHOTO_FRONT_REFERENCE` to a known frontal photo
(staged or original name) to label relative front/three-quarter/profile
angles. This is an estimated coordinate frame, **not** a verified survey
orientation. The analyzer does not invent camera angles for unregistered
photos, combine disconnected models, or infer photo semantics from filenames.

To add known semantic information (or output from a separate vision
classifier), invoke the Python tool with `--annotations tags.json`, where
`tags.json` can contain:

```json
{
  "photo-0013.png": {"view": "left profile", "framing": "head close-up"},
  "photo-0026.png": {"view": "elevated rear", "framing": "full statue"}
}
```

These tags are user/vision-supplied **annotations**, not automatically
verified pose estimates.

For automatic *visual* labels of **unregistered** photos, an optional
vision-capable model installed in local Ollama can classify approximate
front/three-quarter/profile/rear appearance, elevated/low viewpoint,
head-closeup vs head-and-torso vs full-monument framing, and obvious
blur/occlusion. This is a separate, explicitly **unverified** estimate,
not a calibrated camera pose:

```bash
# Use the name of a vision-capable model already installed in local Ollama.
PHOTO_VISION_MODEL=your-local-vision-model \
  bash scripts/mac_splat_test.sh photo-advice \
  .review/mac-splat-smoke/splat-20261103T053355Z-11041
```

Photos are resized and sent **only** to a loopback Ollama endpoint; remote
HTTP(S) vision endpoints are rejected. No cloud upload is performed by this
feature. Each photo's model guess is labeled as a vision estimate and may
be wrong, especially left vs right or when the statue is cropped.
The report only recommends semantic view-type gaps when most images were
classified successfully; numeric angle suggestions still come solely from
estimated COLMAP poses. Without Ollama, geometric photo-value and angle
recommendations work without AI dependencies.

Suggestions favor 10–15 degree bridging steps,
60–80% overlapping subject content, medium shots including shared
head/neck/torso landmarks, and elevated/lower views when accessible. For
the Sphinx, inaccessible rear geometry should instead be sourced from
verified scans or surveys. The Mac `run` command now generates this report
**automatically when registration fails** and text models exist.

#### Try seeded sparse recovery (no new feature extraction)

The Sphinx cohort's model 0 has 4 registered cameras and only 508
registered-image 3D track observations. The best unregistered original
photos have 76 (#17), 32 (#25), and 20 (#20) conservative 2D-to-3D
candidates, with only 31%, 12.5%, and 31% 4×4 image coverage.
This suggests **limited track support**, not a trivially small
triangulation baseline. The candidates have **not** passed PnP RANSAC.

A non-destructive recovery experiment can triangulate more points between
already registered views, then resume the COLMAP mapper from **one**
existing sparse model, using the existing verified-match database:

```bash
# Prefer the original 26-photo reconstruction's 7-camera model:
bash scripts/mac_splat_test.sh recover \
  .review/mac-splat-smoke/splat-20261103T052755Z-7115 0

# Optional controlled comparison on the 14-photo cohort:
bash scripts/mac_splat_test.sh recover \
  .review/mac-splat-smoke/splat-20261103T053355Z-11041 0
```

Outputs are under a new `.review/mac-splat-smoke/recovery-*/` revision.
The original COLMAP database is copied and SHA-256 verified; original
images and sparse models are never modified. The default fixes existing
seed poses during resumed mapping. The script reports point counts
before/after retriangulation, newly registered cameras, any lost seed
cameras, and whether the **same** 8-camera/60% registration gate was met.
It does not train splats or claim that successful registration means
photo-faithful geometry. Recovered models also export `text-models/`
for a follow-up `pnp-qa` run on the new recovery revision:

```bash
bash scripts/mac_splat_test.sh pnp-qa .review/mac-splat-smoke/recovery-<timestamp>
```

If registration does not improve, inspect `recovery-report.json`. The
remaining options are better overlapping source photographs, carefully
reviewed camera intrinsics, or verified 3D scan/survey geometry. Do not
merge disconnected sparse models without a measured alignment.

#### Explain why matched photos still fail camera registration

The 14-photo Sphinx cohort showed median reconstructed triangulation angles
of **20.811° (model 0)** and **39.908° (model 1)**, with 0% of fitted
point tracks below 2°. This rules out a trivially tiny baseline **among
successfully reconstructed tracks** but does not validate camera poses or
explain missing registrations.

A 2D-to-2D verified match does not imply that its registered-image feature
is part of an existing triangulated 3D track. For a missing camera, the
incremental mapper needs sufficient **consistent 2D-to-3D correspondences**
to estimate and refine its pose.

```bash
bash scripts/mac_splat_test.sh pnp-qa \
  .review/mac-splat-smoke/splat-20261103T053355Z-11041 \
  .review/mac-splat-smoke/cohorts-20261103T053322Z-10742/cohort-01/source-index.json
```

This saves a private `pnp-qa-*.json` and prints, for **each independent
sparse model** and each missing photo:

- Verified matches to already registered photos
- How many of those matches touch a 3D track in that model
- Distinct candidate 3D points, ambiguous 2D keypoints, and conservative
  one-to-one 2D-to-3D associations
- 4×4 image-space coverage of the one-to-one candidate correspondences
  (if image dimensions are present in COLMAP's database)
- COLMAP verified two-view geometry configuration counts

**Interpretation:** Very few 2D-to-3D candidates despite many 2D-to-2D
matches indicates that the existing sparse model has too little shared
triangulated geometry. Many well-distributed candidates but a failed
camera registration suggest bad associations, camera calibration, pose
initialization, or PnP verification thresholds. This audit does **not**
run PnP, establish geometric inliers, or manufacture a mesh.

#### Measure reconstructed camera baselines before changing SfM settings

After reviewing the match lines, run the existing read-only registration diagnosis
again on the **14-photo cohort**:

```bash
bash scripts/mac_splat_test.sh diagnose \
  .review/mac-splat-smoke/splat-20261103T053355Z-11041
```

The new `estimated_geometry.models[]` section in `registration-*.json` reports:

- Distinct camera IDs, camera models and estimated focal length divided by
  the largest image dimension (a dimensionless diagnostic)
- Per-model **median maximum triangulation angle** for reconstructed 3D point
  tracks, and percentage of tracks with ray angles below 1°, 2° and 5°
- A separate result for each disconnected COLMAP sparse model; never mix
  coordinate frames across models

If most reconstructed points have ray angles below ~2°, the fitted
cameras provide little reliable depth leverage. If angles look reasonable
but many cameras still fail to register, investigate mapper initialization,
unregistered-photo pose estimation, image calibration, and possible false
matches on repetitive stone layers. These angles are **computed from the
estimated COLMAP poses**: they are not independently measured ground truth,
and good-looking values do not prove the reconstruction is correct.

#### Inspect actual verified image correspondences (no new SfM run)

When photos all have strong match-graph edges but only a few camera poses
register, the next step is **not** increasing SIFT or repeatedly removing
photos. Inspect where the verified feature matches lie in the photos.

```bash
bash scripts/mac_splat_test.sh match-qa \
  .review/mac-splat-smoke/splat-20261103T053355Z-11041 \
  .review/mac-splat-smoke/cohorts-20261103T053322Z-10742/cohort-01/source-index.json
```

This creates a new `.review/mac-splat-smoke/match-qa-...` folder with:

- `match-contact-sheet.jpg`: actual COLMAP verified 2D inlier
  correspondences drawn between image pairs; prioritizes
  registered-to-unregistered photo pairs, plus control examples
- `match-qa.json`: image-space 4×4 grid coverage, inlier counts,
  registration status and, with the optional `source-index.json`, the
  original 26-photo staged IDs

**Cohort photo IDs were renumbered by `--photos` intake.** Do not compare
`photo-0007.png` in the 14-photo cohort with the same name in the 26-photo
revision without using the source index.

The visualization may reveal matches on pyramids, ground, tourists, photo
cutout edges, or other background objects instead of the Sphinx.
High match counts or broad image coverage do not prove useful parallax.
If the matches mostly land on the monument but registration still fails,
investigate camera intrinsics and seed-pair geometry; if they land on the
background, use carefully reviewed foreground masks at **feature extraction**
(not only OpenSplat training). Do not claim successful surface reconstruction
from these diagnostic images.

#### Split disconnected Sphinx photographs into independent registration cohorts

For the 26-photo Sphinx set, the high-recall run registered 7 cameras in one
sparse model and 5 in another, with 14 photos in the largest verified-match
graph component. **Do not concatenate the two models**: their camera coordinate
frames have not been aligned.

Create new, provenance-preserving photo folders from the existing failed
revision, without downloading or retraining:

```bash
bash scripts/mac_splat_test.sh cohorts \
  .review/mac-splat-smoke/splat-20261103T052755Z-7115
```

The command creates a new `.review/mac-splat-smoke/cohorts-...` revision
containing a `cohort-plan.json` and `cohort-01/photos`,
`cohort-02/photos` (if both matching groups still meet the default three-photo
minimum). All staged photo hashes are verified before copying, and the
original photo revision is never modified.

For this particular diagnostic, expect **14** photos in the first connected
component and **5** in the second; five isolated photos and the two-photo component are
excluded. The first group has 7 registered cameras so far and needs at least
**9 of its 14** cameras to register to satisfy both the eight-camera and 60%
gates. The second group is **too small for production splat training**.
The script prints a reproducible re-run command for each eligible cohort.

A match-graph component is **not** a proof of correct camera poses. Background
matches, scale differences and low parallax may still prevent registration.
Inspect the labeled contact sheet before spending time on another run. The
five-photo group can be examined with the sparse-only
`scripts/experiment_sphinx_commons_2010.py --photos-dir <cohort-02/photos>`
command, but it cannot pass the production splat gate without additional
overlapping views.

This creates a **new immutable revision**. Inspect the new
`camera_reconstruction.models`, `feature_import`, `registered_images`
and registration diagnostic. If high-recall still registers only a few images,
curate a coherent overlapping photo subset or obtain new sequential
photographs/verified survey geometry. Do not weaken the 8-view gate merely to
train a misleading splat.

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

### Remove background artifacts without repeating camera registration (macOS)

#### Foreground-first pipeline (recommended for large statues)

The Mac helper now assembles the existing rectified segmentation, COLMAP
track-consistency audit, hash/provenance verification, training and calibrated
visual QA into two explicit stages. Original photographs and camera poses are
preserved. **Mask generation alone is never approval to train:** removing part
of a head, ear or weathered stone will teach OpenSplat an incorrect silhouette.

For the Sphinx, prefer locally prompted **SAM 2.1** to whole-image IS-Net:
the previous IS-Net candidate retained some heads while erasing the body and
paws. SAM 2.1 supports per-photo head/headdress/neck/torso/paws/tail boxes,
positive anatomy landmarks and negative scenery points. The independent
component masks are unioned. It is still a candidate generator, *not* an
automatic guarantee of a complete statue.

On Apple Silicon M4/24 GB, start with **SAM 2.1 small** (tiny for lower
memory). The official implementation supports local MPS/CPU inference; MPS
may be slower or fail on unsupported operations, so use CPU if necessary.
Install the optional PyTorch dependencies and the official package without
CUDA, then download **sam2.1_hiera_small.pt** from the
[Meta SAM 2 checkpoint list](https://github.com/facebookresearch/sam2#download-checkpoints).
No weights are downloaded implicitly by kumiki-blender; their SHA-256 is
recorded in every mask revision.

```bash
python3 -m pip install -e '.[mask-sam2]'
git clone https://github.com/facebookresearch/sam2.git ../sam2
SAM2_BUILD_CUDA=0 python3 -m pip install -e ../sam2

# Create a JSON guide with EXACT rectified names from
# "$SOURCE/sparse-reconstruction/undistorted-reference-index.json".
# Start from examples/sphinx-sam2-guides.example.json, but replace every
# example coordinate with coordinates measured on your actual photos.
SOURCE=.review/mac-splat-smoke/splat-20261103T063944Z-55327
MASK_GUIDES=/absolute/path/to/sphinx-guides.json \
SAM2_CHECKPOINT=/absolute/path/to/sam2.1_hiera_small.pt \
SAM2_SIZE=small SAM2_DEVICE=auto PYTORCH_ENABLE_MPS_FALLBACK=1 \
  bash scripts/mac_splat_test.sh foreground-prep "$SOURCE" sam2
```

Each box is `[left,top,right,bottom]` and point `[x,y]`, normalized to
the rectified image dimensions (0–1). Annotate **all visible** anatomy per
view, not automatically all six parts: a tightly cropped face shot cannot
be expected to show paws or tail. Include a box for each visible part and
one or more `part_points` on its real stone surface. Mark sky, sand,
pyramids, people and scaffolding with negative points. The example numbers
are *not* Sphinx annotations.

**SAM 2.1 quality gate:** for **each registered photo** add a complete
`part_visibility` inventory containing `head`, `headdress`, `neck`, `torso`,
`paws`, and `tail`. Mark each `visible` or `not_visible` after inspecting the
actual frame. `uncertain` is allowed while drafting, but blocks training until
reviewed. A `visible` part must have a matching `component_boxes` entry and
at least one `part_points` landmark that remains foreground. A part genuinely
cropped out or fully occluded should be `not_visible`. This catches the common
failure where a guide labels only the head, so a head-only segmentation might
otherwise pass. The inventory is a **human visual claim**, not independent proof
of complete geometry. It survives mask refinement and hybrid selection, so a
later revision cannot silently drop the requirement.

Use `--sam2-device cpu` through
`SAM2_DEVICE=cpu` if MPS fails; the default bounded 1280-pixel inference
controls peak memory and produces full-size binary mask outputs.

For the older baseline, generate masks with rembg (or guided GrabCut) on the
same **registered, undistorted** photos. Neither stage reruns SfM or starts
OpenSplat:

```bash
SOURCE=.review/mac-splat-smoke/splat-20261103T063944Z-55327
bash scripts/mac_splat_test.sh foreground-prep "$SOURCE" rembg
```

The command prints immutable `foreground-mask-...`, `foreground-audit-...`,
and `foreground-quality-...` revisions. The quality report includes per-view
area, landmark preservation, background seed violations, track outlier
fractions and actionable corrections. It **blocks** training if any visible
part landmark is missing, any mask retains less than 5% or more than 90%
foreground, more than 5% of its foreground lies outside annotated component
boxes, a camera has over 20% majority outliers, there are fewer than 25
compared tracks, or more than 15% shared tracks have inconsistent labels.
SAM 2.1 candidates also fail when the six-part visibility inventory is
incomplete or contradicts the part boxes/landmarks.
These are conservative rejection gates, not proof of correctness.

Inspect **every** photo in
`mask-contact-sheet.jpg`, especially facial edges, headdress, body and tail.
Also inspect the cross-view `mask-consistency-contact-sheet.jpg`: red markers
suggest background leakage; cyan markers suggest missing subject, and yellow
markers indicate ambiguous tracks. These are review cues, not automatic truth.
The audit now writes `mask-correction-recommendations.json` with tentative
positive/negative point prompts, never automatically applied: background
tracks, occluded surfaces and repeated textures may fool feature consensus.
If a colored marker is **confirmed against the actual photograph**, accept
only its `point3d_id` in that specific photo into a **new guide revision**:

```bash
MASKS=.review/mac-splat-smoke/foreground-mask-<timestamp>
AUDIT=.review/mac-splat-smoke/foreground-audit-<timestamp>
kumiki-blender review-mask-suggestions \
  --reconstruction "$SOURCE/sparse-reconstruction" \
  --mask-revision "$MASKS" --audit-revision "$AUDIT" \
  --output .review/mac-splat-smoke/accepted-mask-guides-r02 \
  --accept 'photo-0003.png:12345' 'photo-0011.png:67890'

# Edit the new guide file to add/adjust component boxes and visible-part
# landmarks, then resegment into fresh masks and rerun cross-view QA.
MASK_GUIDES=.review/mac-splat-smoke/accepted-mask-guides-r02/mask-guides.json \
SAM2_CHECKPOINT=/absolute/path/to/sam2.1_hiera_small.pt \
  bash scripts/mac_splat_test.sh foreground-prep "$SOURCE" sam2
```

The IDs above are placeholders; select real IDs from that view's audited
`correction_recommendations`. Cyan prompts become positive points; red prompts
become negative points. Tied/unknown points cannot be accepted. The command
checks source mask, photo and camera-model hashes before saving; it neither
changes prior masks nor trusts suggestions automatically. If a positive point
lies outside annotated component boxes, expand the correct anatomy box by
hand before accepting. **The previous audit cannot approve the new masks.**

If segmentation cuts away genuine statue surfaces, improve per-image SAM 2
boxes/points or use `refine-dense-masks --edits` with include/exclude polygons,
then
rerun `kumiki-blender audit-dense-masks` on the corrected revision.

After that human review, opt into training with an explicit acknowledgement:

```bash
MASKS=.review/mac-splat-smoke/foreground-mask-<timestamp>
AUDIT=.review/mac-splat-smoke/foreground-audit-<timestamp>
MASKS_REVIEWED=1 \
  OPENSPLAT=/absolute/path/to/working/opensplat \
  DOWNSCALE_FACTOR=4 ITERATIONS=3000 \
  bash scripts/mac_splat_test.sh foreground-train "$SOURCE" "$MASKS" "$AUDIT"
```

The Mac training gate checks mask-report hashes, photo provenance, registered
camera source, binary/full-resolution masks, multi-view audit coverage,
COLMAP model integrity, **statue landmarks and quantified consistency**.
It refuses unreviewed or low-quality masks by default. Successful
training automatically generates a paired multi-angle splat contact sheet
(unless `FOREGROUND_AUTO_QA=0`). The `--downscale` setting reaches the
registered-reconstruction OpenSplat path, allowing realistic memory budgets
on Apple Silicon. This remains a **private splat candidate**, not a validated
mesh or a photo-faithful likeness.

#### Individual mask tools and manual alternatives

The Mac smoke helper can reuse the successful PyCOLMAP cameras inside an
existing `splat-...` revision. A first unmasked model often learns floating
sky, mountains, and city textures:

```bash
SOURCE=".review/mac-splat-smoke/splat-20261103T010954Z-91205"  # Your revision
pip install -e '.[mask-ai]'  # Once: semantic foreground segmentation
bash scripts/mac_splat_test.sh mask "$SOURCE" rembg
```

Open `mask-.../mask-contact-sheet.jpg` and review **every** foreground mask
(front, rear, profile, arms). Repair missing stone parts, holes or background
leakage with `refine-dense-masks` and its polygon edits before retraining.
Candidate masks are not automatically approved. For offline guided GrabCut,
install `.[masks]` and use `mask "$SOURCE" grabcut`. See
[Reference Reconstruction](REFERENCE_RECONSTRUCTION.md) for corrections.

When rembg misses a back/side head (for example, `photo-0002.png` and
`photo-0015.png`) but GrabCut recovers it while leaking scenery, compose a
**hybrid review candidate**. All 15 images remain pinned to the same
rectified photos and camera registration; neither original mask revision is
edited:

```bash
REMBG=".review/mac-splat-smoke/mask-<rembg-timestamp>"
GRABCUT=".review/mac-splat-smoke/mask-<grabcut-timestamp>"
bash scripts/mac_splat_test.sh hybrid "$SOURCE" "$REMBG" "$GRABCUT" \
  photo-0002.png photo-0015.png
```

The command prints a new `mask-...` folder, including labeled per-photo
overlays and a `mask-contact-sheet.jpg`. `alternate` is used **only** for the
named photos; rembg remains primary elsewhere. This is not an automatic
approval: manually remove sky, city, platform or other leaked regions using
`refine-dense-masks --edits` (normalized exclusion polygons), and inspect the
new revision before passing its `masks` directory to OpenSplat. A prominent
`approved_for_reconstruction: false` remains in the reports. Original
segmentation hashes and selected origins are recorded in the hybrid report.

```bash
MASKS=".review/mac-splat-smoke/mask-<timestamp>"  # Printed by the mask step
AUDIT=".review/mac-splat-smoke/foreground-audit-<timestamp>"
# Generate AUDIT with: kumiki-blender audit-dense-masks --reconstruction ...
# --rectified-masks "$MASKS/masks" --output "$AUDIT"
MASKS_REVIEWED=1 ITERATIONS=3000 \
  bash scripts/mac_splat_test.sh run-masked "$SOURCE" "$MASKS" "$AUDIT"
# Use the NEW splat revision printed by run-masked for qa and video.
# Optional: favor a chosen source photograph for the video hero angle:
VIDEO_HERO_PHOTO=photo-0001.png bash scripts/mac_splat_test.sh video \
  ".review/mac-splat-smoke/splat-<new-timestamp>"
```

This reuses pinned registered cameras and rectified photographs. Masks reach
OpenSplat as `dataset/masks/<photo-stem>.png`. Older revisions are untouched;
poor masks can remove valid statue geometry, so inspect them before training.

## Revision output and verification

```
splats-r01/
  sparse-reconstruction/  # Only when input is --photos/--references
  dataset/images/         # Pinned undistorted images
  dataset/sparse/0/       # COLMAP registered cameras and 3D points
  dataset/cameras/        # Pinned cameras.txt + images.txt for multi-angle visual QA
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

## Calibrated comparisons against the original multi-angle photographs

Use the registered cameras **from the actual training revision** to render
the same viewpoints as the undistorted source photos. The tool does not guess
the camera angle from filenames or generate plausible-looking substitute
images. It uses the actual SplatTransform GPU rasterizer and Pillow for
per-pixel diagnostics. No FFmpeg is needed for this step.

```bash
pip install Pillow
npm install -g @playcanvas/splat-transform
splat-transform --list-gpus

kumiki-blender plan-splat-view-qa \
  --splat .review/my-statue/splats-r01 \
  --output .review/my-statue/visual-qa-r01 --max-views 12

kumiki-blender compare-splat-views \
  --splat .review/my-statue/splats-r01 \
  --output .review/my-statue/visual-qa-r01 \
  --max-views 12 --max-dimension 1280
```

The pipeline samples poses that span the camera collection instead of always
taking the first twelve filenames. For each chosen photo, it:

1. Checks SHA-256 against the training snapshot and retrieves that photo's
   COLMAP position, rotation and lens intrinsics.
2. Converts the camera to PlayCanvas's 180°-Z-transformed PLY coordinate
   convention. Requests a real GPU WebP render, preserving the registered
   vertical field of view and camera orientation.
3. Accounts for off-center principal points and unequal `fx/fy` via a
   documented pinhole pixel warp, excluding pixels outside the renderer's
   field of view. Applies a matching pinned foreground mask when provided.
4. Saves the reference, matched render, 50/50 overlay, visible difference
   heatmap and labeled comparison JPEG per angle, plus one contact sheet.

```
visual-qa-r01/
  visual-qa-report.json     # Camera, hash, per-view error, worst angles, status
  multi-angle-contact-sheet.jpg
  views/001/gpu-render.webp
  views/001/comparison/reference.png
  views/001/comparison/registered-render.png
  views/001/comparison/overlay.png
  views/001/comparison/difference.png
  views/001/comparison/valid-pixels.png
  views/001/comparison/comparison.jpg
  logs/001.log
```

The report includes pixel MAE, RMSE, PSNR and the fraction of valid compared
pixels for **each** selected view, plus the three worst-error angles. These
are *diagnostics*, not a photographic-likeness score: lighting, background,
and training-view overfit can dominate them, and a good score does not
demonstrate correct facial anatomy or geometry in unseen angles. Inspect the
side/profile, three-quarter and front images directly; use new held-out
photographs for an independent generalization test when available.

This tool requires a full splat revision with recorded cameras and photos,
not only an exported `.ply`. New training revisions pin `cameras.txt` and
`images.txt` inside `dataset/cameras/`; older revisions can use their
original undistorted text model only if both source files still match their
training-time hashes. Outputs remain private; a successful comparison stays
`candidate / human-multi-angle-visual-review-required`. On a GPU or
renderer failure, a persistent failed report records the view and log.

GPU validation must run on your Mac M4/Metal host or another supported
GPU machine. CPU-only CI tests exercise the command, camera math, hashes,
pixel warps and failure reports using a mock rasterizer; they **do not**
demonstrate real GPU output or photo-faithfulness.

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

# Registered photo revisions infer a subject-centric target from COLMAP camera
# rays and limit the auto orbit to observed viewing angles. Standalone PLYs
# still use trimmed Gaussian bounds and the configured 115° arc.
# Render 6 seconds at 1080p/30fps and generate a local HTML viewer.
kumiki-blender render-splat-video --splat .review/my-statue/splats-r01 \
  --output .review/my-statue/video-r01 --hero-photo photo-0001.png --viewer

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

When registered photo cameras are available, `--hero-photo` selects the
starting view, while automatic object centering uses the intersection of
actual camera rays. If the rays are near-parallel or inconsistent, automatic
framing fails safely and requires manual `--target` and `--distance`. On
standalone PLY input, automatic framing still uses Gaussian bounds. Completed
videos are sampled for consecutive near-black frames; a valid but empty MP4 is
flagged for camera-orbit review and kept for inspection.

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
