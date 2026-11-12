# Character sheet → Blender part fidelity QA

**New to this feature?** Start with [the short character fidelity workflow](CHARACTER_FIDELITY_REVIEW.md), including `review-masks` and `propose-registration` commands and their uncertainty gates.


The `kumiki_blender.character_fidelity` package compares **explicitly segmented source artwork** with **visible, named Blender geometry**. It is reusable for owls, people, clothing, accessories, and props; Quill is not a hard-coded class. The previous `face_analysis` and photogrammetry QA workflows remain unchanged.

## Install and run

```bash
python -m pip install -e '.[character-qa]'
python -m kumiki_blender.character_fidelity capture \
  --blend /path/to/character.blend \
  --manifest /path/to/character_fidelity.json \
  --output /tmp/character-capture-001 \
  --blender-bin /path/to/blender
python -m kumiki_blender.character_fidelity compare \
  --manifest /path/to/character_fidelity.json \
  --capture /tmp/character-capture-001 \
  --output /tmp/character-audit-001
```

Open `report.json` and `comparisons/<view>/<part>.png` in the audit directory. Outputs must be **new directories**. No input `.blend` is saved or mutated. The Blender script's temporary shader/material and object-visibility changes exist only in its background process.

## Prepare character data

Copy and edit [`examples/character-fidelity.example.json`](../examples/character-fidelity.example.json). It is a template; replace its placeholder file names and object patterns with **real files** before running it.

1. Set the model camera for each front, side, three-quarter, rear, etc. view. The artwork must be in the **same camera/pose** and the render dimensions must match the manifest.
2. For each anatomical (`head`, `eyes`, `wings`, `feet`, `tail`) or costume (`coat`, `lapels`, `waistcoat`, `bow`) component, create a full-size **binary source mask in that view's reference-image pixel coordinates**. `255`/opaque marks the visible part. Source asset layers may provide exact masks; painted-color segmentation is only a rough annotation and must be reviewed. Do not assume an isolated wing PNG is already positioned in the full character view.
3. Map each semantic component to **disjoint Blender object-name globs**. A mesh object cannot belong to two semantic groups; split geometry when costume and anatomy are combined into one object. Objects not mapped to a part are omitted from capture, so map the complete visible character (including occluders) for a valid depth/silhouette comparison.
4. Specify `alignment.source_to_render` as a 2×3 **source-pixel → Blender-render-pixel affine matrix**, and set `alignment.verified=true` only after matching immutable camera/scale/pose landmarks. Never separately center or scale individual parts to get a better registered overlap number. Identity is valid only when already pixel-aligned.
5. If a different pose is unavoidable, list the view in that part's `unmatched_pose_views`. That part gets intrinsic shape and appearance diagnostics only, without misleading registered position/size scores.

`reference_depth` is optional, e.g. `{ "path": "depth/side.npz", "units": "meters", "provenance": "measured" }`. The `.npz` must contain `depth_m`, be in original image coordinates, and use the same camera-space optical-axis convention as Blender. `validated-multiview` is also accepted only if metric scale and registration are actually established. **Do not generate fake source depth from a single painted drawing and treat it as ground truth.**

## What is measured (per view × semantic part)

| Channel | Measurement | What it cannot establish |
|---|---|---|
| Intrinsic **shape** | Isolated silhouette IoU after uniform scale and translation, *aspect preserved*; source vs sculpt width/height ratio | Fine hidden topology and volumetric continuity |
| Registered **shape + placement** | World-scale-projected IoU, bbox ratios, area ratio, centroid XY offset, symmetric contour mean/P95, boundary F1 at 2/5/10 px | Likeness if camera/pose is wrong |
| **Color** | Median source vs unlit Blender material/texture Base Color in RGB and CIE76 ΔE in Lab; 16-bin Lab marginal histogram total-variation distance for color patterns | Painterly highlights, lighting and translucency; ΔE is not ΔE2000 |
| **Detail proxy** | Mask-interior Laplacian magnitude/edge fraction, gradient intensity and 8-bin edge orientation histogram (source art vs material-color render) | Material roughness, full PBR textures, sculpt surface quality |
| **Depth** | Camera-visible intersection depth P10/P50/P90, and camera-space 3D bound range for each group | Ground-truth depth from a single uncalibrated illustration |
| Measured source **depth** (optional) | Same-camera metric-depth MAE/P95 on valid visible correspondence | Completeness of hidden geometry |

Semantic ID render uses a **single visibility-aware render**, not one isolated render per object. This avoids counting surfaces hidden by coat, feathers, or glasses as visible. The albedo pass connects existing Principled Base Color inputs (including linked image textures) into a temporary unlit emission shader; non-Principled materials fall back to their diffuse color. Evaluated mesh geometry drives the metric depth raycast, so supported modifiers affect depth. The output's `capture.json` preserves exact object → part assignments and projected 3D bounding depth ranges for auditing.

**Limitations:** transparent shader cutouts may differ from Workbench ID masks, curve/volume/hair procedural renderers may not fully enter the BVH, and appearance statistics on weakly aligned silhouettes are diagnostic. The metric depth raycast works on mesh geometry (including evaluated meshes) and can be computationally expensive at high resolutions; start at 256–512 pixels for iteration. CIE76 is a color discrepancy signal, not an objective artistic style score. Never optimize for one combined average: a face can fail badly while the silhouette score appears high.

## Interpretation / review priority

1. Inspect each `comparisons` red/blue overlay. Red is **missing in the model**, blue is **extra in the model**, green is overlapping foreground. Confirm landmarks and pose first.
2. Prioritize **anatomy head/face/tufts, wing construction, feet** before costume details. Inspect both **intrinsic shape** and registered bounding ratios/centroid errors.
3. Inspect **costume** separately: coat outline, lapel drape, vest coverage, bow and gold accessories. Color errors are lower-priority while shape is unacceptable.
4. Check the actual Blender side/rear/elevated views and depth range. Source-absent volumes remain an **artist QA task**; do not convert heuristics into a 3D accuracy claim.
5. Regenerate the report after each sculpt revision and preserve previous revisions for regression tracking. Approval always requires independent multi-view artistic inspection.

## Tests

`pytest -q tests/test_character_fidelity.py` covers shape-vs-translation, aspect-ratio shape mismatch, color, calibrated/unverified depth, missing components, duplicate part IDs, source path traversal, pose mismatch, and report output. A small Blender 5.3 smoke fixture was also rendered locally to validate visibility/color/metric depth export; Blender itself is not required for normal unit tests.

## Ungated advanced analysis (optional)

Install without downloading any model or contacting Hugging Face:

```bash
python -m pip install -e '.[character-qa-advanced]'
```

`contour_geometry` adds normalized symmetric contour distances, 95th-percentile error, Hausdorff distance and curvature-distribution mismatch. `perceptual_color` adds CIEDE2000 median-palette difference; pixelwise CIEDE2000 is reported **only for verified matching camera/pose/registration**. They cannot override a failed silhouette or anatomical placement gate. The baseline CIE76 score is retained for backwards compatibility.

An optional per-view `reference_depth_hint` can point to a local NPZ with a 2D float array named `depth_relative` in source pixel coordinates:

```json
"reference_depth_hint": {
  "path": "depth/front_depth_hint.npz",
  "provenance": "estimated",
  "near_is_small": true
}
```

Set `near_is_small=false` for disparity-style predictions where higher means closer. A source hint only generates a **relative Spearman rank correlation**, not metric-depth MAE; it is ignored if registration/pose is unverified. AI-estimated depth of paintings is an uncertain hypothesis, not anatomical ground truth. Blender itself supplies the metric sculpt depth. The model predictions must be produced separately, reviewed, and referenced as local NPZ files.

### Optional SAM 2.1 segmentation (public weights)

For unsegmented sheets, the official SAM 2 repository provides standalone public checkpoints without Hugging Face. Use the explicit downloader (model inference requires the separately installed SAM2 implementation):

```bash
mkdir -p ~/.cache/kumiki-blender/checkpoints
python -m kumiki_blender.character_fidelity.public_models sam2.1-small \
  --directory ~/.cache/kumiki-blender/checkpoints
# Install SAM 2 separately if desired, without HF login:
# python -m pip install 'git+https://github.com/facebookresearch/sam2.git'
```

Download is manual, never triggered during evaluation. Only two verified Meta-hosted URLs (`sam2.1-tiny` and `sam2.1-small`) are accepted. Weights are not bundled, and download success is not guaranteed on networks blocking Meta. **No Hugging Face account, token, access request, or gated model is required by this character-fidelity workflow.** The downloader prepares a checkpoint; it does not automatically segment images. Source-layer masks still take precedence over predicted masks.

The optional Open3D and Trimesh libraries can be used independently for mesh inspection. Neither requires model weights. This feature does not install Depth Anything 3 or MoGe by default: their model runtime/dependencies and license-per-checkpoint suitability must be evaluated separately before enabling them as automated depth providers.


## SAM 2.1 prompted character-part segmentation

This runs upstream Meta SAM2 inference with a **local** official checkpoint. The Python adapter makes no Hugging Face API calls, downloads no weights, and saves proposed, unapproved masks only.

Install dependencies and upstream SAM2 code:

~~~bash
python -m pip install -e '.[character-qa-models]'
python -m pip install 'git+https://github.com/facebookresearch/sam2.git'
mkdir -p "$HOME/.cache/kumiki-blender/checkpoints"
python -m kumiki_blender.character_fidelity.public_models sam2.1-small \
  --directory "$HOME/.cache/kumiki-blender/checkpoints"
~~~

The downloader uses an allowlisted direct Meta URL rather than Hugging Face. With no network, supply an already downloaded official checkpoint file.

Review and edit examples/character-segmentation-prompts.example.json. It supplies initial pixel clicks and boxes for Quill's 614x722 original front artwork, not verified masks. This is prompt-guided segmentation, NOT automatic text-to-object detection.

~~~bash
python -m kumiki_blender.character_fidelity segment \
  --image /path/to/turnarounds/view_front.png \
  --prompts examples/character-segmentation-prompts.example.json \
  --checkpoint "$HOME/.cache/kumiki-blender/checkpoints/sam2.1_hiera_small.pt" \
  --model small --device mps --output /tmp/quill-sam-001
~~~

Use --device cpu on Linux without supported acceleration, cuda when available. Read review.jpg and report.json; manually approve each output mask before providing it as a source annotation. Output masks use the original full canvas dimensions. Transparent background clipping is enabled by default; turn it off only if the artwork requires it. SAM2 predicted IoU is not a verified comparison with the intended source feature. Original isolated PNG alpha layers, where available, remain preferable to generated masks.

## Optional MoGe-2 relative-depth hints using local weights

Install Microsoft's open-source MoGe package separately from https://github.com/microsoft/MoGe. Obtain an explicitly public MoGe-2 checkpoint through a supported public download process. The command below accepts a **trusted local file path**; it does not contact Hugging Face, download a model, or use a registry model identifier.

~~~bash
python -m kumiki_blender.character_fidelity depth-hint \
  --image /path/to/turnarounds/view_front.png \
  --checkpoint /path/to/local-moge-2-vits-normal.pt \
  --device mps --output /tmp/quill-depth-001
~~~

The output includes depth_hint.npz with a 2D depth_relative float array, depth_hint.png and report.json. Copy the NPZ to a manifest-relative file and add a reference_depth_hint object with path, provenance=estimated, near_is_small=true. Use the existing relative rank comparison only after matching camera and pose have been independently verified.

**Single-illustration inferred depth is not measured depth.** Never put it into the reference_depth ground-truth field or compare absolute meters against Blender as if validated. The depth map may be unreliable for stylized feathers and costume layering.

## Validation boundary

The model integration regression suite in tests/test_character_fidelity_models.py uses mock predictors; it verifies mask shape, prompt validation, missing-local-checkpoint handling, and depth-hint NPZ compatibility without downloading weights. Live inference with public weights and Quill's full sheet remains unverified until the actual checkpoint files can be loaded locally. Do not commit model weights or the supplied character artwork.


## User-supplied ungated SAM Base+ and MoGe-2 checkpoints

SAM 2.1 Base+ is now supported as `--model base-plus` with the official
`configs/sam2.1/sam2.1_hiera_b+.yaml`. Reuse the local
`sam2.1_hiera_base_plus.pt` checkpoint. The official Meta public downloader
also accepts `sam2.1-base-plus` when no local file is available.

The MoGe-2 depth adapter now exports an additional optional
`normal_hint.npz` file, containing `normal_camera` in camera coordinates
(H×W×3; invalid regions are NaN), plus `normal_hint.png`. It converts alpha
artwork onto white before inference and uses full precision on CPU/Apple MPS.
**Both normal and depth maps are uncertain AI estimates for painted artwork.**
They cannot certify the sculpt's 3D surface geometry.

To inspect the supplied weights without loading either neural runtime:

```bash
python -m kumiki_blender.character_fidelity.checkpoints \
  /path/to/sam2.1_hiera_base_plus.pt /path/to/moge-2-model.pt
```

Then with separately installed official `sam2` and `moge` runtime packages,
run `character_fidelity segment --model base-plus` or
`character_fidelity depth-hint` with the local checkpoints. These commands
never request Hugging Face access. Keep model weights and character artwork out
of Git and do not approve generated masks without inspecting them.


## Real Quill checkpoint smoke test (2026-11-12)

The supplied SAM 2.1 Base+ and MoGe 2 Small-with-normals checkpoints completed actual PyTorch CPU inference on the original 614x722 Quill front illustration. No Hugging Face login, approval or download was used. The restricted runtime required temporary local compatibility helpers for Hydra/IOPath and utils3d; these helpers are not distributed or a substitute for proper upstream dependencies.

SAM produced six unapproved semantic part proposals. The independently predicted coat region overlapped approximately 38% of the left wing and 31% of the right wing; the new diagnostic reports these conflicts without clipping source masks. MoGe produced estimated depth and camera-space normal maps, which are not metric ground truth for stylized artwork.


## Compare model captures against SAM source-part proposals

\`character_fidelity reconcile\` uses the already-implemented source-to-Blender
metrics with SAM 2.1 masks and optional MoGe 2 depth hints, without changing any
artwork, checkpoints, or .blend file:

\`\`\`bash
python -m kumiki_blender.character_fidelity reconcile \
  --manifest character_fidelity.json --capture /tmp/part-capture-001 \
  --sam /tmp/sam-proposals-001 --depth-hints /tmp/moge-hints-001 \
  --view front --output /tmp/fidelity-reconcile-001
\`\`\`

Optional \`--mapping mapping.json\` is a JSON dictionary of manifest IDs to
SAM proposal IDs. The source-image SHA-256 must match the segmentation report.
SAM proposals remain **unapproved**, even at high predicted IoU. Reported
semantic-mask overlaps are preserved as warnings rather than auto-clipped.
The output contains \`report.json\`, per-part registered (if verified) shape
overlays and \`review.md\` with separate shape, color, placement and mask review
observations. If camera alignment is unverified, no registered placement,
pixelwise color or source metric-depth accuracy is asserted.

### Preserve unassigned occluders

The manifest may set \`"unassigned_behavior": "occlude"\` to retain unlabeled,
artist-visible meshes as neutral occluders during ID capture and as ray-hit
geometry for depth; the previous default is \`"hide"\`. Existing hidden objects
remain hidden. Capture records the list of unassigned occluders.

A new \`<view>/normal_camera.npz\` contains visible mesh normals in Blender
camera coordinates, in addition to the existing metric \`depth_m.npz\`.
These measured normals must **not** be compared numerically with painted-image
MoGe normal estimates until both camera conventions and image registration
are established.

On Quill revision 04 the first front-view real capture succeeded, with
six SAM proposals (head, two wings, coat, waistcoat and bow tie) and MoGe depth
hints. Wing-coat source-mask overlap and unverified alignment mean those
shape diagnostics are provisional. Model weights, character artwork and
captured render assets are not included in the Git repository.
