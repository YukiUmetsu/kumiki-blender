# Reconstruct one 3D mesh from multiple photographs

`build-photo-model` connects the existing camera-registration, dense-mesh and
Blender GLB-export workers into one command. This is **photogrammetry**, not a
procedural reconstruction or Gaussian splat. It can use a private folder of
8–256 overlapping photographs, including a 30-photo monument reference set.

For mixed internet photos where left/right-side cameras remain disconnected,
see [learned COLMAP matching and camera coverage comparisons](NEURAL_CAMERA_RECONSTRUCTION.md).

## One-command, quality-gated statue reconstruction (Apple Silicon)

`build-statue-glb` runs a fresh COLMAP SIFT-LightGlue/MPS camera solution,
first prepares an incremental, private normalization/foreground-preview cache
**and a source-photo QA pass before any feature matching**,
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

### Optional multi-part grounded segmentation (Grounding DINO → SAM2)

The default SAM2-only behavior is unchanged. The optional **Grounding DINO**
detector proposes text-conditioned boxes; SAM2 then segments the original,
unmodified pixels. A detected head is **not** evidence of complete statue
anatomy. Earlier single-box grounding reduced the foreground from roughly 26%
to 19% on a 49-camera Sphinx test, including several near-empty body masks.

The guarded workflow now generates **both broad SAM2 and grounded SAM2**
candidates per image and saves both immutable masks and contact sheets.
Grounding can also propose separate visible parts via repeated
`--grounding-part-prompt`; these union as *candidate* segments, not validated
anatomy. Distant detections are withheld. When the grounded candidate loses
more than 30% of the broad foreground, grows over 35%, overlaps poorly, or
retains less than 5% of the image, the pipeline uses the broad candidate and
marks the result **uncertain**. This avoids the known head-only regression;
it does *not* prove that the broad mask excludes background. Ambiguous results
remain blocked by mask QA and must be reviewed against original photographs
and cross-view feature tracks.

```bash
# Keep your already-installed SAM2 and MPS Python environment.
python -m pip install -e '.[mask-grounding]'

# Use a fresh immutable revision; previous r005 masks and camera evidence stay intact.
kumiki-blender build-statue-glb \
  --asset-id great-sphinx \
  --photos "$HOME/Pictures/sphinx-reconstruction-candidates/warren-2019/photos" \
  --subject-detector grounding-dino \
  --grounding-prompt 'great sphinx statue. stone sphinx.' \
  --grounding-part-prompt 'sphinx stone body.' \
  --grounding-part-prompt 'head and headdress of sphinx.' \
  --grounding-part-prompt 'front paws of sphinx.' \
  --grounding-device auto

# If interrupted, repeat those options with --resume.
```

For an **A/B segmentation test without rerunning COLMAP**, reuse an existing
camera revision and write candidates to a new, separate output:

```bash
RUN='.review/great-sphinx/statue-hq-r005'
NEW='.review/great-sphinx/grounded-masks-r002'
kumiki-blender prepare-dense-masks \
  --reconstruction "$RUN/cameras" --output "$NEW" \
  --segmenter sam2 --sam2-size small --sam2-device mps \
  --sam2-checkpoint "$HOME/.cache/kumiki-blender/models/sam2.1_hiera_small.pt" \
  --subject-detector grounding-dino \
  --grounding-prompt 'great sphinx statue. stone sphinx.' \
  --grounding-part-prompt 'sphinx stone body.' \
  --grounding-part-prompt 'head and headdress of sphinx.' \
  --grounding-part-prompt 'front paws of sphinx.' \
  --grounding-device auto \
  --cache-dir '.review/great-sphinx/_statue_mask_cache'

# Compare all three contact sheets: chosen, broad, and grounded candidates.
open "$NEW/mask-contact-sheet.jpg" \
     "$NEW/mask-contact-sheet-broad.jpg" \
     "$NEW/mask-contact-sheet-grounded.jpg"

jq '{processed: (.images|length), uncertain: ([.images[] | select(.candidate_comparison.status | startswith("uncertain"))] | length), cache}' "$NEW/mask-report.json"

kumiki-blender compare-dense-masks \
  --reconstruction "$RUN/cameras" \
  --baseline-masks "$RUN/masks/masks" \
  --candidate-masks "$NEW/masks" \
  --output '.review/great-sphinx/grounded-mask-comparison-r002'
```

Compare the new masks *visually* with the originals; cross-view score changes
alone cannot establish geometry or anatomy accuracy. JSON reports provide
`candidate_comparison` (overlap, area ratio, rejection reasons), along with
`candidate_masks.broad` and `.grounded` byte hashes and preview paths for each
photo. Existing source photos, previous revision masks and sparse cameras are
never overwritten. Object parts are configured via prompts, not hard-coded to
the Sphinx. When the sparse reconstruction remains blocked by insufficient
elevated/side camera coverage, mask experiments can continue independently but
do **not** bypass the camera-coverage gate or prove a GLB is ready.

### Track-guided SAM2 retries (separate immutable revision)

When broad and grounded candidates disagree, `improve-dense-masks` uses
existing *registered, triangulated COLMAP tracks* to suggest extra positive
SAM2 clicks in areas omitted by the selected mask. It checks that a track is
observed from at least four photographs and that several *other* camera views
agree inside both SAM2 candidates and the detector guide. A second, stricter
route can recover a visible chest or torso point which **both** candidates
missed in the target photo, but only if at least four other registered views
independently confirm the same 3D point as statue in at least 80% of its
observations. The low-plane ground classifier vetoes conflicting background
geometry, and held-out foreground/background checks still decide acceptance.
Neither an AI-generated photograph nor an untriangulated detection qualifies.

A separate background
test looks for **negative background tracks** outside both masks, well clear
of their contours and outside Grounding DINO detections in at least four
other registered views. When COLMAP camera rotations and triangulated 3D
points are available, a second route fits a **low, horizontal ground-plane
candidate** from exterior points observed in several photos. It does not
require DINO's part boxes to exclude ground (those boxes can cover almost the
entire frame). A point on the plane may veto a mask that incorrectly includes
it in the current photo if other camera views identify it as exterior. These
become SAM2 negative clicks or independent held-out background checks.

Both foreground and background tracks have *independent, never-prompted*
held-out subsets. A retry must recover foreground holdouts without losing any,
violate no background holdouts, and add no confidently blue-sky foreground.
Local color/appearance segmentation near the measured ground points provides
another conservative veto on ground-colored mask expansions. It cannot label
the full image as sand based on color alone. When a mask already includes
ground, trusted existing foreground clicks can also guide an automatic SAM2
ground-removal retry; only improvement on independent held-out ground points,
with no loss of held-out statue tracks, can approve it. The fitted plane must
also sit below most camera centers and agree with the height of verified statue
tracks; otherwise the system abstains rather than treating a flat torso or
wall as ground.

The ordinary 20% maximum growth remains the fallback. Expansion up to 55%
(or 90% with substantially stronger evidence) is allowed **only** with enough
spatially independent background checks and when the added pixels are near
recovered positive tracks. Otherwise the prior mask is kept and the reason is
recorded. Background stone and Sphinx stone can still look alike: sparse
tracks are *not* verified semantic labels, so the algorithm abstains when
support is ambiguous rather than silently admitting ground.

**Automatic is the default.** The pipeline chooses each candidate when it has
adequate independent evidence and otherwise keeps the original mask or
abstains. No person needs to click through the photos for this stage to run.
The result is still a *candidate*, not certified statue anatomy: normal
automatic mask-consistency and camera-coverage gates may block a weak dataset.
Visual inspection is recommended, and optional overrides are supported below.
Grounded `build-statue-glb` runs invoke this stage automatically;
`--no-track-guided-masks` opts out. Original revisions are never overwritten.

For an independent A/B test without rerunning COLMAP or Grounding DINO:

```bash
RUN='.review/great-sphinx/statue-hq-r005'
BASE='.review/great-sphinx/grounded-masks-r003'
IMPROVED='.review/great-sphinx/track-guided-masks-r005' # must not exist

kumiki-blender improve-dense-masks \
  --reconstruction "$RUN/cameras" \
  --mask-revision "$BASE" \
  --output "$IMPROVED" \
  --sam2-device mps \
  --sam2-checkpoint "$HOME/.cache/kumiki-blender/models/sam2.1_hiera_small.pt" \
  --cache-dir '.review/great-sphinx/_statue_mask_cache'

open "$IMPROVED/mask-contact-sheet.jpg"
jq '{status, summary, changed: [.images[] | select(.track_guided_replaced) |
  {filename, evidence: .track_guidance}]}' "$IMPROVED/mask-report.json"
jq '{viable_track_ids, viable_background_track_ids, geometric_ground_plane,
     rejected, limitations}' "$IMPROVED/track-evidence-report.json"
```

If interrupted, use the same arguments plus `--resume`. A finished revision
cannot be overwritten. Inspect `candidates/` for proposals that failed the
held-out tests and `masks/` for the chosen output. Existing alternatives are
copied and pinned by SHA-256. Without enough trustworthy multi-view overlap,
the stage abstains; it never invents viewpoints or certifies statue anatomy.
The stronger background-evidence algorithm creates a new revision fingerprint.
Identical SAM2 prompts still reuse the existing MPS mask cache; new background
clicks produce distinct content-addressed cache entries. Never resume an
incomplete older algorithm revision using this new version.

#### Optional manual corrections (not needed for automatic mode)

If someone inspects a withheld proposal and prefers it to the automatically
selected mask, `select-track-masks` can promote *only those explicitly named*
photos into another immutable revision. It does not re-run SAM2:

```bash
CHOICES='.review/great-sphinx/track-masks-user-choice-r001' # must not exist
kumiki-blender select-track-masks \
  --reconstruction "$RUN/cameras" \
  --mask-revision "$IMPROVED" \
  --output "$CHOICES" \
  --use-proposal photo-0021.png photo-0022.png photo-0027.png photo-0034.png photo-0053.png
```

For an image with an otherwise useful proposal that includes unwanted ground,
add `--use-proposal photo-0012.png` **and** `--edits /path/to/edits.json`,
where that file contains image-specific normalized include/exclude polygons:

```json
{
  "images": {
    "photo-0012.png": {
      "exclude_polygons": [
        [[0.10, 0.90], [0.20, 0.90], [0.20, 0.95], [0.10, 0.95]]
      ]
    }
  }
}
```

The coordinates are illustrative only; trace the actual unwanted region in
the original rectified photo. Exclusions are applied *after* inclusions.
Both selected proposals and edits retain source hashes and require the usual
mask QA. Users can continue without ever invoking this optional command.

The bundled detector choice is Hugging Face
`IDEA-Research/grounding-dino-tiny` at immutable revision
`a2bb814dd30d776dcf7e30523b00659f4f141c71` (Apache-2.0 model card).
The first enabled run downloads approximately 689 MB of `safetensors` weights
plus configuration/tokenizer files to the Hugging Face model cache. It never
loads the pickle checkpoint or executes remote model code. `--no-auto-model-download`
requires that revision already be cached, or provide
`--grounding-model-dir /path/to/pinned/snapshot`. Model weight SHA-256,
prompt and device are part of the immutable revision settings and cache identity.

On Apple Silicon, `--grounding-device auto` tries MPS first and retries the
detector on CPU if an MPS inference operator fails; `--grounding-device cpu`
forces CPU. The main SAM2 model still uses `--sam2-device` independently.
Grounding detections are cached by exact image content, model weights, main
prompt and every part prompt. Broad and grounded SAM2 candidates have separate
cache keys, so adding one photograph does not reprocess unchanged photos.
A missed detection
falls back to a broad **unverified** SAM2 candidate and explicitly blocks
the mask QA gate until reviewed. Normalized RGB photographs are never cut out.
No automatic majority-vote correction is applied to reconstructed 3D tracks.
Do not judge improvements solely from a lower cross-view disagreement rate;
inspect mask contact sheets and real photographs before continuing to dense 3D.

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

If no guide file or Grounding DINO detector is supplied, SAM2 automatically
samples conservative blue-sky negative points and starts with a broad subject
prompt. When Grounding DINO is enabled, its text-conditioned candidate box
replaces the broad box when it confidently detects the requested subject.
Those mask candidates
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
- When a locally installed, SHA-verified IS-Net checkpoint and the optional
  `rembg` library are available, a separate `auxiliary-foreground-*.png`
  proposal is created on the **original RGB image**. Peripheral disagreements
  can supply additional SAM2 negative clicks; a refinement is accepted only
  when it preserves subject pixels on which both models agreed. Missing or
  invalid IS-Net weights trigger a recorded SAM2-only fallback, never a model
  download. No model votes create new photogrammetric geometry.
- `normalized.json` / `foreground-*.json` integrity receipts and a per-run
  `photo-preprocess/photo-preprocess-report.json` index linking originals to
  cached pixels and recording cache hits/misses.
- SHA-addressed `qa-confidence-*.png` maps with three feature-priority classes:
  **green/high-confidence subject**, **gold/uncertain**, and **blue/background**
  in the colorized `photo-preprocess/photo-qa-contact-sheet.jpg`. The actual
  monochrome maps store 255, 128 and 0 respectively. The report includes
  exposure, blur/detail, resolution, EXIF orientation, possible near duplicates,
  mask boundary, blue-background warnings and independent-model disagreement.
  When both proposals are usable, disagreement pixels stay **uncertain**;
  invalid foreground masks make all pixels uncertain so useful anatomy is not
  silently discarded. Multiple similarly scored but substantially different
  SAM2 silhouettes also mark the **whole source confidence map uncertain**.
  This prevents an ambiguous automatically selected silhouette from being
  treated as verified object support during LightGlue matching.

The source QA contact sheet is generated **before feature extraction** and
does not replace the later rectified multi-view mask audit. Visual inspection
is still required to confirm that the mask did not remove a cheek, nose, paw,
tail, damaged stone or another distinctive part. Source-photo QA is advisory:
low resolution and poor lighting do not automatically delete photographs.
The contact sheet now shows **original / binary SAM2 candidate / confidence
overlay** separately. For closer inspection, find each 1536-pixel-wide
triptych at `photo-preprocess/photo-qa-details/photo-XXXX.jpg`.
The run report includes `qa_hits`, `qa_misses`, `qa_warning_count`, `mask_qa`
(suspect count, original names, specific problems and full-size preview paths),
`auxiliary_model_status`, `refinement_attempts`, and `refinement_accepted`.
`mask_qa.needs_visual_review` stays true even when no automated warning fires:
without ground-truth outlines, the code cannot certify complete anatomy.
Source mask/QA cache identities are versioned for the new multimask ambiguity
checks, while normalized original RGB photos remain cached unchanged.

On a machine with a previously downloaded, verified IS-Net model, install the
optional library with `pip install -e '.[mask-ai]'` to enable offline
model-consensus refinement. The normal SAM2-only path remains supported when
this extra is absent. The model probe checks the pinned local weight hash
**without importing rembg or ONNXRuntime into the SAM2/PyTorch process**.
Each independent proposal runs in a short-lived subprocess, preventing a native
ONNXRuntime/OpenMP crash from aborting the parent MPS worker on Apple Silicon.
The child verifies the same checkpoint hash and never downloads unverified
weights. If the optional runtime crashes or times out, the parent records
`failed / auxiliary-inference`, disables further optional attempts for that
run and finishes with original RGB evidence and SAM2-only candidates. The
failure is also recorded under `auxiliary_model_status` and
`auxiliary_failure_reason` in the preprocessing report. Cached SAM2 fallback
results remain valid across `--resume` and unchanged photo sets. Never set
`KMP_DUPLICATE_LIB_OK=TRUE`: suppressing native runtime safety checks is not
a substitute for process isolation.
If the previous native abort occurred before the *first* photo, `--resume`
also allows the previously unprobed auxiliary model to become available without
discarding the revision. Resuming after any completed photo still requires
the same model identity, to protect the cached image evidence.

If a fresh `build-statue-glb` revision stops during `photo_preprocess`, inspect
`statue-pipeline-report.json` under `failed_worker` for the stage's exit code
and diagnostic. Retry that incomplete revision with `--resume`; completed
per-photo content-hash cache entries are reused. Do not delete the source
photo cache or change the model while resuming a partial run.

With `build-statue-glb`, LightGlue reads these hash-verified confidence maps:
it reserves most feature capacity for likely statue surfaces while keeping a
budget for uncertain regions and background. Obviously incompatible
subject/background correspondences are excluded **without renumbering original
COLMAP SIFT keypoints**. The original normalized RGB photographs and all
uncertain areas remain available to the extractor. Ordinary `reconstruct-photos`
is unchanged unless `--source-qa-report` is supplied (requires `sift-lightglue`
and an MPS or `torch-cpu` worker). On a Mac use MPS by default; for a CPU
fallback:

```bash
kumiki-blender build-statue-glb --asset-id great-sphinx --matching-device torch-cpu
```

This option changes the camera-matching backend and needs a fresh revision;
previous completed COLMAP matching cannot be retroactively reweighted. The
optional PyTorch CPU worker uses the same original feature indices. It still
requires the LightGlue Python dependency and can be substantially slower.

The cache uses **original image SHA-256, normalization version, checkpoint
SHA-256, SAM2 settings, refinement version and independently verified IS-Net
weight identity (or absence)**, not filenames or ordinal `photo-XXXX` identifiers.
When a new photo is added, only the new photo incurs normalization and SAM2
preprocessing; unchanged images are verified and reused, even when adding a
file shifts their staged names. Editing a photo or changing the SAM2 model
invalidates the affected cache keys. Interrupted preprocessing can resume
without recomputing completed, valid entries. The existing new-photo camera
matching still needs to solve correspondences with the updated photo set.

### Independent camera groups with zero, one or two common registered cameras

Previously, the cohort merger could only align groups sharing at least three
registered cameras. It now has a second, conservative path for **zero shared
registered cameras**. A candidate elevated group is retained in its own
immutable trial directory; the program looks for verified COLMAP 2D feature
correspondences connecting triangulated 3D tracks in both models, using the
original feature indices and source-photo confidence maps. Then it:

1. Rejects ambiguous many-to-one tracks or missing subject-confidence evidence.
2. Estimates a 7-degree-of-freedom similarity transform with robust RANSAC;
   requires many non-collinear, well-distributed subject-feature correspondences
   across multiple independent camera pairs.
3. Checks transformed geometry against **both sets of calibrated camera
   observations** in pixels, rejects implausible camera poses, insufficient
   spatial coverage or unstable support.
4. Copies independently reconstructed camera frames and original 2D/3D tracks
   into a common aligned candidate only after these tests pass. Separate tracks
   remain separate; they are not artificially welded into a sculpted surface.
5. Reuses the existing post-merge baseline camera-drift and photo-coverage
   gates before dense geometry. A rejection leaves the independent candidate
   and its `alignment.reasons` and diagnostics intact.

For **one or two shared registered cameras**, the program additionally
attempts `verified-hybrid-3d-points` alignment. Shared image *feature IDs*
can link two independently triangulated object-surface tracks, but only when
each has independent depth observations and the original image coordinates
match. Cross-camera COLMAP inliers must still support spatially distributed
Sim(3) RANSAC, calibrated reprojection, and at least eight verified object
observations in **every newly added camera**. Shared camera center and viewing
direction must also agree after transformation. The accepted candidate links
only those audited new-camera observations into existing baseline 3D tracks;
ambiguous, unverified, or missing tracks are never welded. This path does not
reduce any threshold for zero-shared or three-plus-shared alignment.

The standard COLMAP `model_merger` is still used when there are sufficient
shared registered cameras. It is **not** used to claim a disjoint-group merge
without shared cameras. Old revisions without early subject QA cannot use the
new zero-overlap path automatically, because the object-surface evidence is
missing; rebuild a fresh revision if you want to test the new matching path.
The independent 3-camera/119-point Great Sphinx result has **not** been shown
to satisfy these new correspondence, reprojection or coverage gates. A passing
synthetic unit test is not a validation of the actual Sphinx photographs.

Every rejected disconnected-model alignment now records an `evidence_audit`:
the number of cross-group verified image pairs, 2D observations, features with
only one model's 3D track, features with *both* 3D tracks, ambiguous points,
and subject-verified correspondences. This identifies why visually similar
elevated photos cannot be aligned without weakening geometric validation.
Hybrid rejection reports also include shared-anchor correspondences,
`new_camera_3d_support`, reprojection errors, and specific alignment reasons.
The one-shared-camera Great Sphinx candidate with 30 reconstructed points is
reason to **test** hybrid alignment, not evidence that it will pass the gates.

For a blocked reconstruction **that already has a subject QA index**, use
`--resume` to retry camera recovery without repeating completed matching.
Older revisions without this index require a fresh sparse reconstruction;
`--resume` deliberately refuses to retroactively change their feature weights.

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

If `prepare-dense-masks` fails **after creating a partial mask directory**,
`build-statue-glb --resume` now passes the same `--resume` option to the mask
worker. The worker checks that the sparse reconstruction, model hash, guides,
settings and cache location still match its incomplete receipt before
continuing. Valid completed per-photo masks are reused without rerunning
SAM2; neither the partial revision nor its source photos need deletion.
The SAM2 checkpoint path is distinct from the report-saving helper, avoiding
an earlier `FileNotFoundError` that printed a Python function instead of a
checkpoint filename.

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
