# Character sheet → Blender part fidelity QA

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
| **Color** | Median source vs unlit Blender material/texture Base Color in RGB and CIE76 ΔE in Lab | Painterly highlights, lighting and translucency; ΔE is not ΔE2000 |
| **Detail proxy** | Mask-interior Laplacian magnitude/edge fraction (source art vs material-color render) | Material roughness, full PBR textures, sculpt surface quality |
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
