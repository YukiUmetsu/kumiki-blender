# Reconstructing missing sides of any object

Use this when an otherwise successful COLMAP reconstruction registers one view
orbit but fails to register photos from another side. This is not specific to
the Great Sphinx, statues, or a particular file naming convention.

The existing `reconstruct-photos` run must have a `sparse-reconstruction/`
directory containing `database.db`, original `images/`, `sparse/0`,
`text-models/0/images.txt`, and `staged-reference-index.json`. Keep all those
files together. The recovery tool never changes them.

## 1. Inspect the proposed independent cohort

```bash
SOURCE="/absolute/path/to/previous/camera-revision"

kumiki-blender plan-sfm-cohort-recovery \
  --revision "$SOURCE" --max-anchors 6 --max-trials 5
```

By default, target photos are **all images absent from the selected registered
model**. This applies to animals, architectural sites, vehicles, products,
statues, or anything else reconstructed using the same image pipeline.

If the failed images represent multiple unrelated viewpoints, specify a subset
using a JSON file containing **staged filenames**, not original filenames:

```json
["photo-from-back.png", "photo-from-left.png", "photo-from-top.png"]
```

Use `--targets /path/to/targets.json`. Existing registered images included in
the target list become alignment anchors automatically. Use repeated `--anchor
STAGED_FILENAME` to explicitly favor useful registered bridge views. All anchors
are required to be registered in the baseline; the rest are chosen from verified
pair matches, prioritizing links to the missing side.

The plan reports:

- Verified seed pairs containing at least one missing-side image, ranked using
  inlier distribution and estimated matched-ray separation when a valid
  COLMAP two-view pose/intrinsics are available. Strongly near-zero calibrated
  pairs are skipped by default (`--min-seed-angle 1.0` degrees).
- Registered anchor images and exact COLMAP image IDs.
- A bounded number of candidate initializations.
- Whether enough registered anchors were selected for a potential merge.
- `coverage_gaps`: disconnected missing-view groups, registered matching
  neighbors, existing reconstructed 3D-track support and best bridge-photo
  targets. No image viewpoint labels or object identities are guessed.

The plan does not prove that 3D reconstruction will work. Strong 2D matches can
lie on background terrain or repeat-pattern surfaces, and cannot independently
establish depth, object identity, or camera geometry.

For a compact failure-focused explanation, include the previous recovery report
after the trial completes:

```bash
kumiki-blender diagnose-sfm-coverage \
  --revision "$SOURCE" \
  --prior-recovery /absolute/path/to/old/cohort-recovery-report.json
```

The diagnostic reports which specific old trials yielded only zero, one or two
shared cameras, and the unregistered group each failed trial attempted.
An existing 2D-to-3D track count is candidate support **only**, not PnP inliers.

### Get a human-readable missing-angle shopping list (recommended)

Before hunting for additional photos, run the same diagnostic with a new guide
directory. This uses the **original file names** in `staged-reference-index.json`,
not only the internal `photo-00XX.png` names:

```bash
GUIDE=".review/missing-angle-guide-r01"  # must not already exist
kumiki-blender diagnose-sfm-coverage \
  --revision "$SOURCE" \
  --prior-recovery /absolute/path/to/old/cohort-recovery-report.json \
  --guide-dir "$GUIDE" --format markdown
```

Results:

- Terminal Markdown states **which registered original photo to connect to
  which unregistered original photo**, with other missing photos in that group.
- `$GUIDE/missing-angle-guide.md` is a step-by-step photo acquisition checklist.
- `$GUIDE/missing-camera-angles.jpg` is a labeled side-by-side contact sheet of
  the current registered reference and all missing photos for each gap, when
  Pillow is installed (`pip install -e '.[qa]'`).
- `$GUIDE/missing-angle-guide.json` provides structured data for other tools.

For a single-file JSON output, omit `--guide-dir` and `--format markdown` (the
default remains `json`). Every gap also contains `photo_request` with original
file names and view hints, so other agents can use it without a manual lookup.

**Important:** A name like `object-left-side.jpg` is a *filename clue*, not an
image classification or measured camera position. The tool explicitly marks
unregistered angles as unknown. See the contact sheet to inspect the actual
views. When captions are supplied in the staged manifest, they are labeled as
source annotations rather than measured geometry. A photograph between two
appearances is only a hypothesis until COLMAP verifies common **object-surface**
features and reconstructs coherent poses.

## 2. Recover independent views, then attempt a guarded merge

In the **same Python virtual environment as PyCOLMAP 4.x**, run:

```bash
OUT="/absolute/path/to/new/private/cohort-revision"

kumiki-blender recover-sfm-cohorts \
  --revision "$SOURCE" --output "$OUT" \
  --max-anchors 6 --max-trials 5 --trial-timeout 1200
```

Optional: pass `--targets`, repeat `--anchor`, and set `--colmap /path/to/colmap`.
An example for the existing Sphinx run is `--anchor photo-0033.png`, but the
code has **no object-specific filenames**.

Implementation:

1. Validate original staged images against their SHA-256 manifest and ensure
   the sparse text and binary models have the same image names.
2. Copy the **existing** COLMAP database once. This preserves image IDs,
   features, verified LightGlue/SIFT matches, and original source attribution.
3. Use PyCOLMAP's `IncrementalPipelineOptions.image_names` to restrict
   each independent mapping to missing-side photos plus a small number of
   old registered anchors. Initialization seeds are always pairs containing a
   previously missing photo, so the dominant old orbit does not win by default.
4. Save every trial under `trial-NN/sparse/0` for later comparison, including
   failed alignment candidates.
5. Require at least **three shared registered cameras**, nondegenerate camera
   spatial distribution, consistent IDs, a proper 3D similarity, bounded
   camera-center residuals, bounded viewing-direction disagreements, and at
   least two newly registered target views.
6. Only after those tests, invoke COLMAP's native `model_merger`, which further
   validates shared **2D/3D reprojection observations**. Reject native merge
   errors, lost baseline cameras, lost new cameras, and unexpected movement of
   the original camera positions/directions.

The merger requires **shared registered images in a common database**. Two
disconnected point clouds with no common camera observations cannot safely be
merged merely because they visually resemble the same object.

The result is deliberately named `merged-candidate/`, with report status
`candidate / needs-object-and-camera-visual-review`. This is **not** an
automatic approval for dense OpenMVS, Gaussian splatting, or GLB generation.
If there is no valid alignment, inspect the independent candidate models and
`cohort-recovery-report.json`, then add intermediate overlapping photos or
verified survey/scan correspondences. Do not lower geometry thresholds just to
raise camera counts.

## 3. Add bridge photographs for any object

Use the gap diagnostic to identify two views that still need a connected path:
one registered view and one currently missing view. Find **real** intermediate
photographs showing common, stable features on the object across successive
angles. Repeated screenshots from the same camera position, feature matches on
the background, or generated views cannot supply trustworthy missing depth.

Put the new JPEG/PNG/WebP files in a local folder. Optionally create a JSON
source-attribution manifest (all filenames refer to files in that folder):

```json
{
  "images": [
    {"filename": "bridge-left.jpg", "source_url": "https://example.org/source",
     "license": "reference-only", "view": "oblique"}
  ]
}
```

Stage a **new** combined input directory without changing the original source:

```bash
BRIDGE_PHOTOS="/absolute/path/to/new-real-overlapping-photos"
PREPARED="/absolute/path/to/unused/combined-bridge-input"

kumiki-blender prepare-sfm-bridge-photos \
  --revision "$SOURCE" \
  --photos "$BRIDGE_PHOTOS" \
  --output "$PREPARED"
```

You may supply `--manifest /absolute/path/to/photo-sources.json`. All source
images are checked against the original hashes, the bridge files are decoded,
duplicates are rejected, and the private `bridge-photo-manifest.json` records
the photo/source mapping. This uses the same 8–256 image, format, dimensions,
and 50 MiB limits as `reconstruct-photos`.

Start a **new sparse reconstruction** using the existing Mac acceleration:

```bash
NEW_CAMERA_REVISION="/absolute/path/to/unused/cameras-with-bridges"

kumiki-blender reconstruct-photos \
  --photos "$PREPARED/photos" --asset-id example-object-with-bridges \
  --output "$NEW_CAMERA_REVISION" --engine colmap \
  --matching-preset sift-lightglue --matching-device mps \
  --undistort --timeout-seconds 7200
```

This rebuilds feature extraction/matching and camera registration in a new
revision. It **does not** splice matches from unrelated input revisions or
silently change existing camera IDs; the source remains available for
comparison by content hash. The new bridge files are always private,
regardless of whether their rights metadata is present.

## 4. Validate physical object landmarks on an existing candidate

Camera agreement and native COLMAP reprojection checks are necessary but do
not prove that matches lie on the intended object (rather than the background).
For a particular trial, identify **three or more noncollinear physical features**
that genuinely correspond across the two sparse models. Example landmark names
for an arbitrary object might be `feature-A`, `feature-B`, and `feature-C`.

Create a JSON file containing surveyed or verified 3D coordinates in **both**
independent model coordinate frames:

```json
{
  "landmarks": [
    {"id": "feature-A", "baseline_xyz": [1.0, 2.0, 3.0], "cohort_xyz": [10.0, 20.0, 30.0]},
    {"id": "feature-B", "baseline_xyz": [4.0, 2.0, 3.0], "cohort_xyz": [11.0, 20.0, 30.0]},
    {"id": "feature-C", "baseline_xyz": [1.0, 6.0, 4.0], "cohort_xyz": [10.0, 21.0, 31.0]}
  ]
}
```

These are **illustrative values only**, not valid monument measurements. Use
COLMAP model inspection to identify real corresponding 3D points or verified
survey positions. Then run:

```bash
kumiki-blender inspect-sfm-cohort-alignment \
  --revision "$SOURCE" \
  --candidate "$OUT/trial-01/sparse/0" \
  --landmarks /absolute/path/to/object-landmarks.json
```

Object landmarks must be evaluated against the **specific trial's** camera
frame. They cannot be reused blindly across trial 01, trial 02, etc. The QA
compares physical landmarks against the similarity fitted to shared camera
poses, and fails on inconsistent scale, distribution or residuals. Passing
these mathematical checks still requires visually comparing registered
camera frustums and 3D geometry against the input photos.

## Apple Silicon and performance

Use the already verified LightGlue/MPS feature database from the source run;
this recovery performs **no extraction or rematching**, so it does not download
or request approval for another model. PyCOLMAP incremental mapping and Ceres
bundle adjustment are CPU operations; `ba_use_gpu=False` is deliberate and
compatible with Apple Silicon. Trials are bounded in number and runtime. The
original MPS feature-matching acceleration is preserved in the match database.

If `No module named 'pycolmap'` appears, install the existing optional
photogrammetry dependency **into the active reconstruction venv**:

```bash
python -m pip install 'pycolmap>=4.0.2,<5'
python -c 'import pycolmap; print(pycolmap.__version__)'
```

For upstream behavior and limitations, see
[COLMAP PyCOLMAP options](https://colmap.github.io/pycolmap/pycolmap.html) and
[COLMAP merging FAQ](https://colmap.github.io/faq.html#merge-disconnected-models).

## Inspect matching failures before retrying

`plan-sfm-cohort-recovery` includes `pair_stage_diagnostics` when the database
retains tentative matches. It distinguishes missing match records (attempt
status unknown), recorded zero tentative matches, a missing verification
record, rejected verification, and verified 2D matches. For verified matches
it counts observations on an existing anchor's 3D tracks; these are **candidate**
2D–3D associations, not PnP inliers. Use the recovery report for actual pose
estimation and validation results. A count alone does not identify matches on
the object or establish pose accuracy.

To produce a separate private report and photo-pair visualizations:

```sh
kumiki-blender audit-sfm-matches --revision /path/to/camera-revision \
  --output /path/to/new-match-audit --max-pairs-per-target 2
```

Each visualization shows tentative-only matches in red, geometrically
verified matches in yellow, and verified matches whose anchor observation is
on an existing 3D track in green. All colors require visual review: even a
verified track may belong to the background. The diagrams do not imply that
the target camera registered or that a match is a PnP inlier. The JSON includes
all target/registered pairs, including those without a visualization. A missing
match record does not prove a matcher attempted or skipped the pair; inspect
pair-list progress and logs for that determination.

### Viewpoint robustness experiments

The existing pipeline uses SIFT + LightGlue on MPS or a CPU fallback. COLMAP
also supports affine-shape and domain-size-pooled SIFT; its [FAQ](https://colmap.github.io/faq.html#increase-number-of-matches-sparse-3d-points)
recommends them for difficult matching. Treat alternative extractors as
experiments: match counts alone do not justify adding poses or tracks.
Detector-free matchers such as [LoFTR](https://openaccess.thecvf.com/content/CVPR2021/html/Sun_LoFTR_Detector-Free_Local_Feature_Matching_With_Transformers_CVPR_2021_paper.html)
are another candidate, but introduce different dependencies and still need the
same geometric and object-region validation.

On the local Sphinx data, an isolated CPU affine/DSP-SIFT experiment for
`photo-0041` / `photo-0057` produced zero verified matches; MPS SIFT-LightGlue
on those newly extracted features produced nine tentative and zero verified
matches. The original pair had 23 tentative and zero verified matches. This is
one pair, not a general comparison of the algorithms. An independent retry of
seven selected pairs without subject-confidence filtering likewise recovered
none of the tested previously unverified pairs. Neither experiment changes the
source database or justifies disabling subject filtering. See the private audit
and experiment directories for the images and raw database evidence.

### Preserve established tracks during camera recovery

The automatic PnP recovery loop calls COLMAP `point_triangulator` with
`--clear_points 0`. COLMAP otherwise defaults to deleting the existing points
and observations before retriangulating. Completion can still filter points,
and its candidate is never promoted solely for adding tracks: every new camera
must pass the existing camera, track and baseline-drift checks.

In the Sphinx investigation, the original default dropped both newly registered
cameras to zero 3D observations. The isolated preservation trial retained 36
and 90 observations, passed the unchanged baseline-camera check, and changed
the top-view candidate's conservative one-to-one track support from 21 to 25.
It also reduced some other support counts, so it is not a guarantee of improvement
and does not validate an overhead camera.

### RoMa multiview track experiment

`reconstruction.roma_track_consensus.agree_track_predictions` groups predicted
image positions by an existing 3D track. At least two distinct anchor photos must
agree within four target-image pixels; conflicting predictions are discarded.
Subject labels exclude only confident background. Unknown labels are retained,
so they do not prove a match is on the object. The result is a PnP candidate,
not a verified observation, independent image measurement, or registered pose.
It does not mutate the reconstruction.

A private Sphinx experiment used the official RoMa outdoor model on MPS (420px
coarse, 560px refinement) and six existing registered anchors. With prediction
confidence above 0.05, the previously unregistered `photo-0060.png` had 39
consensus tracks and 37 PnP inliers at COLMAP's 12px threshold. Holding out
anchor predictions showed mixed agreement, including some large residuals.
The original masks include unknown areas and the candidate has not passed the
normal registration and object-surface checks. Five other tested targets did
not provide enough consensus tracks. No camera or mesh was promoted. These are
preliminary findings from one dataset, not accuracy measurements or proof that
RoMa can recover an unobserved surface.

To test agreed RoMa predictions without touching the camera revision:

```sh
kumiki-blender prepare-roma-candidate-database \
  --revision /path/to/camera-revision --predictions /path/to/predictions.json \
  --output /path/to/new-candidate
colmap geometric_verifier --database_path /path/to/new-candidate/database.db --num_threads 1
mkdir /path/to/new-candidate/registered
colmap image_registrator --database_path /path/to/new-candidate/database.db \
  --input_path /path/to/camera-revision/sparse/0 \
  --output_path /path/to/new-candidate/registered
```

The JSON contains `target` and a `predictions` list with `anchor`, integer
`track_id`, `xyz`, target `pixel` and boolean `subject_candidate`. The importer
checks source photo hashes and that each anchor actually observes each track.
It accepts only multiview-consistent candidates and invalidates the affected
pair verification records so COLMAP must reverify them. It adds zero descriptor
placeholders for predicted target points: **the experimental database must not
be used for feature matching or rematching**. Successful registration still
requires pose, reprojection, subject and independent-view validation.

#### Reusable RoMa recovery experiment (MPS or CPU)

`predict-roma-tracks` replaces one-off model notebooks. It reads only the
hash-pinned staged photos and *existing registered 3D points*, samples official
RoMa outdoor A-to-B warps at each real anchor observation, rejects low-confidence
predictions and pixels explicitly labeled background, and writes separate JSON
files for multiple missing rear/overhead targets. Ambiguous/unknown mask regions
are **not** proof of foreground. No candidate changes the sparse source.

Install `romatch` and a compatible PyTorch build in the existing local MPS
reconstruction environment before the first run. RoMa loads its outdoor
checkpoint on first use. The registered image names are normalized
`photo-XXXX.png` filenames from the revision, not original web photo names.

```sh
kumiki-blender predict-roma-tracks \
  --revision /private/sphinx/cameras-r005 \
  --anchor photo-0003.png --anchor photo-0014.png \
  --anchor photo-0020.png \
  --target photo-0060.png --target photo-0057.png \
  --output /private/sphinx/roma-predictions-r01 \
  --device mps --coarse-resolution 420 --refine-resolution 560
```

Choose registered anchors with *actual shared statue surface*, ideally at
different azimuths, and change target names using the source camera audit.
The names in this example are illustrative; verify registration status first.
Inspect `roma-report.json` for each target's `agreed_tracks`. Two RoMa warps
agreeing is a **prediction-consistency check**, not independent geometric proof.
The command requires at least two different registered anchors and rejects
changed input photos and modified subject-confidence rasters.

Long MPS batches now checkpoint **each finished target** with a SHA-256
receipt. If interrupted, repeat the *identical* command and add `--resume`,
keeping the same `--output` directory. The runner rechecks source COLMAP
database, source 2D/3D tracks, staged photo index, and subject-confidence
index, plus every completed target JSON hash and prediction count. It skips
only verified completed targets; it will never overwrite a changed or
unrecorded JSON file. Changes to anchors, targets, order, model settings, or
source data require a **new output directory**. Pre-checkpoint runs lacking
these fingerprints are deliberately not eligible for resume.

```sh
kumiki-blender predict-roma-tracks \
  --revision /private/sphinx/cameras-r005 \
  --anchor photo-0003.png --anchor photo-0014.png --anchor photo-0020.png \
  --target photo-0060.png --target photo-0057.png \
  --output /private/sphinx/roma-predictions-r01 \
  --device mps --coarse-resolution 420 --refine-resolution 560 \
  --resume
```

Resuming makes the expensive experiment durable, **not** its camera poses
more credible. RoMa matches stay unverified until the independent checks below.

For one target with agreement, run the existing isolated importer and COLMAP
experiment (do not rematch its zero-placeholder descriptors):

```sh
kumiki-blender prepare-roma-candidate-database \
  --revision /private/sphinx/cameras-r005 \
  --predictions /private/sphinx/roma-predictions-r01/photo-0060.png.roma.json \
  --output /private/sphinx/roma-candidate-0060-r01
colmap geometric_verifier \
  --database_path /private/sphinx/roma-candidate-0060-r01/database.db --num_threads 1
mkdir /private/sphinx/roma-candidate-0060-r01/registered
colmap image_registrator \
  --database_path /private/sphinx/roma-candidate-0060-r01/database.db \
  --input_path /private/sphinx/cameras-r005/sparse/0 \
  --output_path /private/sphinx/roma-candidate-0060-r01/registered
kumiki-blender audit-roma-camera \
  --revision /private/sphinx/cameras-r005 \
  --candidate /private/sphinx/roma-candidate-0060-r01 \
  --output /private/sphinx/roma-audit-0060-r01 --outlier-px 8
```

The audit distinguishes original measured COLMAP keypoints from RoMa-created
synthetic ones, measures native calibrated reprojection errors, and reports
camera baseline drift. The registration and audit still **cannot certify** a
new view: small reprojection error on a *predicted* keypoint is circular
evidence. Confirm the face/paws/top/rear-left alignment visually and evaluate
independent, actually measured target-to-registered correspondences. Hold out
anchors from matching when possible; do not count them as independent proof.
The report also lists `unused_verified_matches`: original database matches to
baseline 3D tracks that are absent from the candidate's current observations.
Conflicting feature-to-track associations are excluded. Large errors identify
items to inspect, not automatic pose failures: repetitive texture can produce
incorrect matches even after pairwise verification. These observations may
also have influenced an earlier fitting step, so "unused" does not mean
independently held out. Missing source tracks are reported as unavailable.

If exactly two bad observation indices have been inspected and found in
`high_error_observations`, prune only those indices into a **new** model:

```sh
kumiki-blender audit-roma-camera \
  --revision /private/sphinx/cameras-r005 \
  --candidate /private/sphinx/roma-candidate-0060-r01 \
  --output /private/sphinx/roma-audit-0060-pruned-r01 \
  --outlier-px 8 --remove-index 123 --remove-index 456
```

Indices `123` and `456` are placeholders: copy real `point2D_idx` values from
the first audit, never assume there are only two. Removal is rejected unless
each observation exceeds the threshold. Only `pruned-model/` changes; the
original registration and historical 49-view reconstruction remain intact.
Pruning also refuses an observation whose 3D point is duplicated elsewhere in
the same target image, or whose native track has only two members (deleting it
would remove the entire 3D point). Inspect the track instead of forcing the
operation. On the November 5 Sphinx test, `photo-0060.png` observations 855
and 1462 both referred to 3D point 18295; do not automatically delete 1462.

For a previously created **separate sparse COLMAP model** without a RoMa
candidate receipt, `repair-roma-sparse-camera` offers a guarded, explicit
alternative. It is appropriate only after the duplicate target tracks have
been inspected in the native model. The November 5 Sphinx audit found **three**
duplicated target point IDs:

| 3D point | Target indices | Explicit choice | Evidence / limitation |
| --- | --- | --- | --- |
| 18140 | 1353, 1354 | Drop 1354; keep 1353 | Both pixels exactly identical (3.315px error) |
| 18295 | 855, 1462 | Drop 1462; keep 855 | 855 has lower reprojection error (7.536 vs 8.167px) |
| 18304 | 553, 1465 | Drop 1465; keep 553 | Retain earlier existing index provisionally; 1465 is slightly lower-error (2.983 vs 3.342px), so this is **not** an automatic confidence decision |

Also remove high-error index 1463 (11.960px). The 49 baseline poses had
exactly zero measured center/orientation drift before repair. Selecting an
observation to preserve must never be mistaken for independent pose validation.

```sh
RUN=.review/great-sphinx/statue-hq-r005
kumiki-blender repair-roma-sparse-camera \
  --baseline-model "$RUN/dense-masked-fixed-r01/workspace/sparse" \
  --candidate-model "$RUN/roma-dense-r01/workspace/sparse" \
  --target photo-0060.png \
  --remove-index 1462 --remove-index 1463 \
  --remove-duplicate-index 1354 --remove-duplicate-index 1465 \
  --exclude-camera photo-0013.png --exclude-camera photo-0055.png \
  --outlier-px 8 --allow-duplicate-repair \
  --output "$RUN/roma-top-only-r01"
```

This requires exactly the three added cameras named above, zero baseline
pose drift, both high-error removals above 8px, both additional removals
confirmed as duplicates, exactly one surviving target observation per 3D
point, consistent native track links, surviving multi-view tracks, and at
least 30 remaining target observations. `--remove-duplicate-index` is
separate from `--remove-index`: it permits removing an explicitly reviewed
low-residual duplicate, even when it does not have the higher reprojection
error. Both kinds require `--allow-duplicate-repair` for duplicate tracks.
The report records the removed indices and which duplicate indices survived.
It refuses to deregister a shared multi-camera frame. Native PyCOLMAP 4.x
`deregister_frame` is used to exclude the two unverified views; neither the
original 49-camera model nor the 52-camera experiment is edited. The output
contains `sparse/` and `roma-sparse-repair-report.json`, **not** a dense-ready
rectified reconstruction. Evaluate independent object-surface matches and
calibrated view overlays before preparing a separate rectified/masked 50-view
workspace. Do not rerun OpenMVS solely because this gate passes.

### Stage the isolated 50-camera RoMa dense input

The November 5 Sphinx experiment separately produced a 52-camera dense mesh
(`roma-dense-r02`), including two **unverified** views. Its 519,436 vertices and
1,038,245 faces are useful for a visual comparison but **do not** validate the
two extra registrations. The orange top-view overlay projected the **old
49-camera mesh**, not the new SAM2 foreground mask; missing projected paws or
rear geometry in that overlay must not be mistaken for a mask-only defect.

The repaired 50-camera sparse model was taken from the private workspace of
`roma-dense-r01`. Do **not** infer its camera-revision origin from `r01`/`r02`
folder names: the authoritative source is `source_revision` in that dense
run's receipt. Prepare the new input from **that exact rectified revision**:

```sh
RUN=.review/great-sphinx/statue-hq-r005
SOURCE=$(jq -r '.source_revision' "$RUN/roma-dense-r01/dense-report.json")
kumiki-blender stage-roma-dense-candidate \
  --source-reconstruction "$SOURCE" \
  --repair "$RUN/roma-top-only-r01" \
  --output "$RUN/roma-cameras-top-only-r01"
```

The staging command checks all 52 input rectified-image hashes, compares the
original candidate sparse binary hashes against the sparse model recorded by
the repair receipt, rejects mismatched camera poses and intrinsics, and copies
**only the 50 selected registered images** alongside the repaired sparse model.
It regenerates a COLMAP text model and the pinned 50-image source index for
`plan-dense-photos`. It neither writes to its source nor certifies the RoMa
pose. **Do not** substitute a differently rectified image or mask grid; the
staging and dense preflight commands enforce exact calibration and pixel
dimensions regardless of revision names.

```sh
kumiki-blender plan-dense-photos \
  --reconstruction "$RUN/roma-cameras-top-only-r01" \
  --rectified-masks "$RUN/roma-dense-r01/workspace/images" \
  --output "$RUN/roma-dense-top-only-r01" \
  --backend openmvs --resolution-level 2 --max-resolution 1600 \
  --number-views 3 --no-texture
```

The plan is **read-only**. Before running `build-dense-photos`, visually audit
all 50 mask boundaries, especially the upper body, full forelegs and paws,
rear, and tail. The r02 top-view mask included the paws but had missing black
interior regions and some foreground ground; r01 masks may have similar
defects. Produce a new mask correction revision rather than silently widening
the entire Sphinx silhouette or accepting ground. Then build the 50-camera
mesh into a **new** directory with precisely the same preflight settings and
the corrected r01 masks. Compare photographed geometry and holes against both
the 49- and 52-camera experiments before publishing a GLB.

### Do not convert weak mask votes into SAM2 negative points

The November 6 photographic 50-view audit compared **12,551 shared 3D tracks**;
**6,228 (49.62%)** had mixed foreground labels. This is a serious inconsistency
signal, not a statistic about what fraction of the real Sphinx is missing:
ground, wall, shadow, and predicted RoMa tracks can also contribute.

On `photo-0060.png`, the old report found three red possible-background marks
with only **3:2** negative majorities. Inspection of the actual photograph
shows those red markers on the **stone chest**, not on sand. Turning them into
background SAM2 points could **erase genuine statue geometry**. The strongest
cyan mark near the headdress has **6:1** positive votes, but even this is
cross-view mask agreement, not independent surface truth.

The updated `audit-dense-masks` now separates strong cross-view agreement
(at least four *other* camera votes and 80% agreement) from weak majority
review marks, ranks stronger evidence first even under marker budgets, and
explicitly forbids automatic prompt conversion. Original photos and masks
remain untouched. The report also includes `shape_qa` per image: bounded,
non-mutating measurements of **enclosed black regions**, detached foreground,
and boundary contact. The supplied 636×1043 top-view mask contained roughly
**50 significant enclosed black regions occupying 1.94% of its image area**
when sampled at 384px. Some may be real shadow or gaps between limbs; the
measurements **do not authorize hole filling**. Confirm each anatomical region
and ground boundary on the full-resolution photograph before regenerating a
candidate mask.

To regenerate QA after pulling the new branch, use a fresh output path:

```sh
kumiki-blender audit-dense-masks \
  --reconstruction "$RUN/roma-cameras-top-only-r01" \
  --rectified-masks "$RUN/roma-dense-r01/workspace/images" \
  --output "$RUN/roma-top-only-mask-audit-r02"
```

Inspect `images[].shape_qa`, `images[].review_score`, and the new
`evidence_strength` and `automatic_prompt_allowed` fields on each suggestion.
Neither a high consensus score nor a successful sparse camera repair means the
mesh or mask is ready for production. Keep the 49-, 50- and 52-camera revisions
separate until real-photo QA demonstrates a geometric improvement.

**Before attempting an A/B dense mesh:** independent object-feature support,
stable original poses, calibrated overlays, and reviewed masks must pass. Use a
new complete rectified sparse reconstruction revision and new masks, then the
existing `plan-dense-photos` / `build-dense-photos` commands with the same
OpenMVS settings as the 49-view baseline. A raw `pruned-model/` folder does
not satisfy the dense reconstruction's rectified-image provenance contract.
Compare both meshes with `audit-dense-photographs` and Blender calibrated
reviews, prioritizing top, rear-left, face and paws. Run `audit-dense-masks`,
`refine-dense-masks`, and `finish-dense-textures` on *new* revisions after
reviewing foreground leakage, atlas dark patches, white balance and seams.
Do not overwrite `dense-masked-fixed-r01`, and do not publish a GLB until
photographic fidelity and texture quality pass human inspection.

### Repair severely incomplete masks with anatomy-guided SAM2 (review only)

Visual inspection of the November 5 photo/mask pairs for `photo-0007.png`,
`photo-0031.png`, and `photo-0040.png` found severe false-background regions
across faces, headdresses, paws, and body. Some scenery and scaffolding also
remains marked foreground. The high multi-view disagreement is therefore **not**
safe to fix by blindly turning COLMAP disagreement votes into prompts. Do not
mesh the 50-camera revision with these masks as if they were verified.

For an *isolated experimental retry*, use the anatomical sample guides
`examples/sphinx-sam2-retry-3-views.example.json`. They are manually estimated
from the supplied images and need visual confirmation before interpreting a
new candidate. The full 50-view preflight, photo hashes, and original mask
dimensions are checked first. SAM2 only runs on the three selected photos;
all input photographs, original masks, and the 49/50/52-camera reconstructions
stay untouched.

```sh
RUN=".review/great-sphinx/statue-hq-r005"
# Replace with the location of your installed official SAM2.1 hiera-small weights.
SAM2_CHECKPOINT="/path/to/sam2.1_hiera_small.pt"
PYTHONPATH=src python -m kumiki_blender retry-sam2-mask-candidates \
  --reconstruction "$RUN/roma-cameras-top-only-r01" \
  --source-masks "$RUN/roma-dense-r01/workspace/images" \
  --guides examples/sphinx-sam2-retry-3-views.example.json \
  --image photo-0007.png \
  --image photo-0031.png \
  --image photo-0040.png \
  --sam2-checkpoint "$SAM2_CHECKPOINT" \
  --sam2-size small --sam2-device mps --max-side 1280 \
  --output "$RUN/sam2-anatomy-retry-r01"
```

Check `targeted-sam2-report.json` for `original_prompt_qa`,
`prompt_qa.anatomical_parts`,
`pixel_change_qa`, `original_shape`, and `candidate_shape`. Open
`review/photo-XXXX.png.jpg` alongside the original photograph and candidate
`candidates/photo-XXXX.png.mask.png`. Inspect actual face features, complete
forelegs and paws, body/rear, ground, sky, and scaffolding. Point compliance,
shape metrics, and newly included pixels are **diagnostic only**; they do not
demonstrate photo-faithful 3D geometry. This command intentionally emits **no
ready-to-use `masks/` revision**. If acceptable, review all remaining photo
masks and construct a new explicitly approved complete 50-view mask revision
before a separately staged dense A/B run. Never modify source masks in place.

For the local November 5 follow-up, review-only candidates for `photo-0015`,
`photo-0022`, `photo-0027`, `photo-0028`, `photo-0029`, `photo-0038`, and
`photo-0039` are in the private `sam2-priority-retry-r02` through `r04`
revisions. Their corresponding candidate mask sets and comparison reports
are under `mask-photo-candidate-r03` through `r05` and
`mask-photo-comparison-r03` through `r05`. The first four candidates reduced
mixed-label tracks from 5,476 to 4,545; shared segmentation mistakes can also
improve agreement, so this is not a photographic accuracy measurement.
Further trials for `photo-0034` and `photo-0017` are recorded in
`sam2-priority-retry-r05`, with the combined nine-photo candidate in
`mask-photo-candidate-r07` and comparisons through `r07`: 3,814 mixed tracks
remain. A `photo-0026` scaffold retry violated a negative click and is
excluded from `r07`; `mask-photo-candidate-r06` contains that failed mask and
must not be used. These numbers include an experimental camera and are review
hints, not independent ground truth.
The later `sam2-top-paws-retry-r01` and `sam2-profile-retry-r01` revisions
add review candidates for `photo-0060` and `photo-0030`. Their selected masks
are recorded in `mask-photo-candidate-r08` and `r09`; the top photograph's
camera is still unverified and the candidate set remains unapproved.
The final comparison reports 3,643 mixed tracks; the profile candidate also
increased possible background-leak votes by 10, so inspect those locations.
Scaffolding, nearby masonry and the remaining images still need review. In
particular, the long restored forepaws can resemble unrelated walls: the
experimental `mask-photo-candidate-r02` erased possible paw surfaces and was
rejected. Do not use that revision. The 50th camera remains unverified.
