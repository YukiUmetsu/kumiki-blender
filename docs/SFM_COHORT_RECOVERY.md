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

- Verified seed pairs containing at least one missing-side image.
- Registered anchor images and exact COLMAP image IDs.
- A bounded number of candidate initializations.
- Whether enough registered anchors were selected for a potential merge.

The plan does not prove that 3D reconstruction will work. Strong 2D matches can
lie on background terrain or repeat-pattern surfaces, and cannot independently
establish depth, object identity, or camera geometry.

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

## 3. Validate physical object landmarks on an existing candidate

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
