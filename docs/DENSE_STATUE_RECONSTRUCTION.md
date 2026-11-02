# Dense statue reconstruction from overlapping photographs

The sparse COLMAP workflow reconstructs **cameras and a sparse point cloud**,
not accurate eyelids, hair, worn stone or folded garments. This optional stage
runs multi-view depth estimation, fusion, surface meshing, refinement and
private texture projection to produce a mesh for artistic inspection.

## Requirements

- Photos must depict the **same actual statue** with strong overlap, consistent
  geometry and sharp views of the front, profiles, back, head and drapery.
  Unrelated web photos and photos of different restorations may not register.
- In busy outdoor scenes, manually verify **binary subject masks for the
  undistorted photographs**. Without masks, real dense MVS reconstructs
  nearby branches, pavement and ground alongside the monument. Geometry
  detail and foreground isolation are separate problems.
- Optional CPU foreground-mask preparation (`pip install 'kumiki-blender[masks]'`)
  uses OpenCV GrabCut on the *registered/rectified* photos. This is a starting
  silhouette, not an autonomous guarantee: trees, matching stone and occluded
  arms can be misclassified. Review each generated red overlay before MVS.
  For harder outdoor photos, the optional `pip install 'kumiki-blender[mask-ai]'`
  enables local IS-Net semantic background inference through rembg. The first
  use may download model weights; verify those weights' usage rights independently
  before commercial work. The expected ONNX SHA-256 is
  `60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a`;
  the pipeline rejects mismatches. No model weights or reference photos are committed.
- Run the existing photo-dossier workflow using the --undistort option first.
  The dense builder requires at least 8 registered independent photos and 60%
  registration by default; these are minimum gates, not enough evidence to
  guarantee realistic faces or unseen surfaces.
- Install [COLMAP](https://colmap.github.io/) for the original sparse cameras.
  Alternatively, install the [official PyCOLMAP wheel](https://pypi.org/project/pycolmap/)
  with `pip install 'kumiki-blender[photogrammetry]'` and add
  `--engine pycolmap` to `reconstruct-photos`.
  The Python engine isolates stages in subprocesses, uses CPU SIFT/matching,
  and preserves the same per-stage logs, timeouts and rectified camera outputs.
  Both engines normalize invalid JPEG EXIF orientation values (such as `0`)
  on **staged copies** by patching the EXIF tag bytes, without re-encoding any
  photograph or touching the immutable originals. The per-image index records
  both original and staged SHA-256 values and the repair flag. This prevents
  fatal COLMAP metadata-parser aborts observed in real monument photo sets.
  For dense CPU reconstruction (including Apple Silicon), install
  [OpenMVS](https://github.com/cdcseacave/openMVS) and put InterfaceCOLMAP,
  DensifyPointCloud, ReconstructMesh, RefineMesh and TextureMesh on PATH.
  Follow the [official OpenMVS guide](https://github.com/cdcseacave/openMVS/blob/develop/docs/wiki/Usage.md).
- The alternative COLMAP dense backend uses GPU PatchMatch and may be
  unavailable on CPU-only / non-CUDA builds; see the
  [official CLI guide](https://colmap.github.io/cli.html).

## Usage

    kumiki-blender reconstruct-photos \
      --manifest /private/statue/reference-set.json \
      --output .review/statue/sparse-r01 --undistort --max-image-size 3200

    # No standalone COLMAP CLI needed; still runs real SfM on CPU.
    kumiki-blender reconstruct-photos \
      --manifest /private/statue/reference-set.json \
      --output .review/statue/sparse-pycolmap-r01 \
      --engine pycolmap --undistort --max-image-size 2000

    # Checks coverage, image hashes, executable paths and output destination.
    # Writes nothing.
    kumiki-blender plan-dense-photos \
      --reconstruction .review/statue/sparse-r01 \
      --output .review/statue/dense-r01

    # For outdoor statues, generate all rectified subject-mask CANDIDATES first.
    # Every photo gets a binary mask and a red-highlight review JPEG.
    # The default center box is only an initial suggestion; inspect EVERY
    # overlay and refine camera-specific hints for thin arms, hair, etc.
    kumiki-blender prepare-dense-masks \
      --reconstruction .review/statue/sparse-r01 \
      --output .review/statue/masks-r01 \
      --guides /private/statue/mask-guides.json

    # Alternative: automatic semantic-first segmentation; requires [mask-ai].
    # It runs a shared local isnet-general-use inference session, with no
    # photo uploads, but still requires manual review of each resulting mask.
    kumiki-blender prepare-dense-masks \
      --reconstruction .review/statue/sparse-r01 \
      --output .review/statue/masks-semantic-r01 --segmenter rembg

    # Correct unwanted ground or a missing thin feature on selected views.
    # Edit normalized polygons in a private JSON file; never modify r01.
    kumiki-blender refine-dense-masks \
      --reconstruction .review/statue/sparse-r01 \
      --mask-revision .review/statue/masks-semantic-r01 \
      --output .review/statue/masks-r02 \
      --edits /private/statue/mask-edits.json \
      --min-island-pixels 24 --max-hole-pixels 24 \
      --edge-inset-pixels 2

    # Optional separate color-aware draft to compare visually against r02.
    # Trim-only never adds foreground; it can still remove thin anatomy.
    kumiki-blender refine-dense-masks \
      --reconstruction .review/statue/sparse-r01 \
      --mask-revision .review/statue/masks-r02 \
      --output .review/statue/masks-r03 \
      --color-band-pixels 4 --color-trim-only --color-max-side 2048 \
      --color-iterations 4

    # Find camera-specific mask mistakes using actual shared COLMAP 3D tracks.
    # Red = possible extra background; cyan = possible missing statue;
    # yellow = ambiguous tie. Review all colors before editing masks.
    # This is a private QA report, NOT an automatic mask approval.
    kumiki-blender audit-dense-masks \
      --reconstruction .review/statue/sparse-r01 \
      --rectified-masks .review/statue/masks-r02/masks \
      --output .review/statue/mask-consistency-r01

    # Compare r02 and optional r03 against identical sparse feature tracks.
    # Positive deltas signal more disagreement, NOT automatic acceptance.
    kumiki-blender compare-dense-masks \
      --reconstruction .review/statue/sparse-r01 \
      --baseline-masks .review/statue/masks-r02/masks \
      --candidate-masks .review/statue/masks-r03/masks \
      --output .review/statue/mask-compare-r02-r03

    # Use only reviewed masks; OpenMVS consumes <rectified-filename>.mask.png.
    kumiki-blender build-dense-photos \
      --reconstruction .review/statue/sparse-r01 \
      --rectified-masks .review/statue/masks-r02/masks \
      --output .review/statue/dense-masked-r01 --backend openmvs

    # CPU-capable OpenMVS meshing + refinement + texturing.
    kumiki-blender build-dense-photos \
      --reconstruction .review/statue/sparse-r01 \
      --output .review/statue/dense-r01 --backend openmvs

    # Only if OpenMVS creates black/dark texture patches: try a separate
    # reconstruction with seam leveling disabled and compare in Blender.
    kumiki-blender build-dense-photos \
      --reconstruction .review/statue/sparse-r01 \
      --rectified-masks .review/statue/masks-r02/masks \
      --output .review/statue/dense-no-seam-r01 --backend openmvs \
      --no-global-seam-leveling --no-local-seam-leveling

    # If the monument has foliage/sky/pedestrians behind it, prepare an
    # exact-size rectified mask for EVERY undistorted registered image:
    # /private/masks/img-0001.jpg.mask.png, etc. Use grayscale 255=keep,
    # 0=exclude. Verify edges manually in the rectified image coordinate
    # frame. OpenMVS reads these alongside the private image snapshot.
    kumiki-blender build-dense-photos \
      --reconstruction .review/statue/sparse-r01 \
      --rectified-masks /private/masks \
      --output .review/statue/dense-subject-only-r01 --backend openmvs

    # On a supported GPU, use COLMAP PatchMatch instead.
    kumiki-blender build-dense-photos \
      --reconstruction .review/statue/sparse-r01 \
      --output .review/statue/dense-gpu-r01 --backend colmap

    # Inspect the PLY in Blender, preserving editable geometry and a
    # PRIVATE review GLB. Include calibrated views of the *same* raw scan
    # from pinned SfM photos, with no sculpt-space survey required.
    kumiki-blender plan-dense-blender-review \
      --dense-revision .review/statue/dense-r01 \
      --output .review/statue/blender-review-r01 --matched-views 4

    # Quick lower-resolution diagnostic only; use the 100% default for
    # final feathers, eyelids, carved lettering and hair-line inspection.
    kumiki-blender render-dense-blender-review \
      --dense-revision .review/statue/dense-r01 \
      --output .review/statue/blender-review-fast-r01 \
      --matched-views 4 --matched-resolution-percent 40
    kumiki-blender render-dense-blender-review \
      --dense-revision .review/statue/dense-r01 \
      --output .review/statue/blender-review-r01 --matched-views 4

    # Preserve OpenMVS's genuine UV coordinates and diffuse atlas in a
    # second PRIVATE .blend/GLB, with all OBJ/MTL/JPG references SHA-checked.
    # Keep this separate from the neutral clay mesh geometry review.
    kumiki-blender render-dense-blender-review \
      --dense-revision .review/statue/dense-r01 \
      --output .review/statue/blender-textured-review-r01 \
      --textured --matched-views 4

    # A separate, non-destructive atlas correction revision: originals and
    # UV faces are unchanged. Black/uncovered texels remain black.
    kumiki-blender finish-dense-textures \
      --dense-revision .review/statue/dense-masked-r01 \
      --output .review/statue/texture-finish-r01 \
      --exposure-ev 0.65 --gamma 1.12 --shadow-lift 0.06 \
      --saturation 1.05 --neutralize-cast --balance-atlases \
      --max-balance-ev 0.75

    # Alternative: opt-in bounded automatic brightness for very dark scans.
    # Keep manual exposure EV at zero initially to avoid double brightening.
    kumiki-blender finish-dense-textures \
      --dense-revision .review/statue/dense-masked-r01 \
      --output .review/statue/texture-auto-r01 \
      --auto-exposure-luma 115 --max-auto-exposure-ev 1.0 \
      --balance-atlases --uv-aware-stats

    # Better when a statue spans several UV atlas pages: estimate brightness
    # directly from shared mesh vertices along page boundaries, not from
    # unrelated page-wide averages. This is an ALTERNATIVE to --balance-atlases.
    kumiki-blender finish-dense-textures \
      --dense-revision .review/statue/dense-masked-r01 \
      --output .review/statue/texture-seams-r01 \
      --balance-uv-seams --max-seam-ev 0.75 --min-seam-anchors 8 \
      --uv-aware-stats

    # Blender imports the same verified OBJ/MTL/UVs with the corrected atlases
    # and packs them into the private .blend and GLB.
    kumiki-blender render-dense-blender-review \
      --dense-revision .review/statue/dense-masked-r01 \
      --texture-finish .review/statue/texture-finish-r01 \
      --output .review/statue/blender-finished-r01 --textured --matched-views 4

    # Guided material finishing: make a local surface recipe (example below)
    # and 8-bit masks/illumination + optional MEASURED 16-bit height maps.
    # A guidance value of 128 means neutral illumination; brighter values
    # remove illumination highlights, darker values correct shadowed pixels.
    kumiki-blender finish-dense-textures \
      --dense-revision .review/statue/dense-masked-r01 \
      --output .review/statue/finish-material-r01 \
      --surface-recipe .review/statue/surface-materials.json \
      --surface-delight-strength 0.7 --surface-max-delight-ev 1.0 \
      --surface-cleanup-radius 3 --uv-aware-stats

    # A completely separate editable UDIM OBJ/MTL and 1001+ texture tiles.
    kumiki-blender package-finished-udim \
      --texture-finish .review/statue/finish-material-r01 \
      --output .review/statue/udim-material-r01

Options include --min-registered-images, --min-registered-fraction,
--min-mesh-faces, --timeout-seconds, --no-refine (OpenMVS) and --no-texture.
For large statues, `--resolution-level 0` (default) preserves rectified
image detail in OpenMVS depth maps, while `--max-resolution 3200` (default)
and `--number-views 5` (default) bound resource use. Lower fidelity for a
quick diagnostic by setting `--resolution-level 1` or `2`. Artificial
OpenMVS tower-mode fill is disabled to avoid inventing monument geometry.
`--refine-decimate 1.0` (default) disables OpenMVS's aggressive automatic
pre-refinement decimation, important for eyes, relief and feathers, at the
cost of CPU/RAM. Lower values may be used deliberately to create an optimized
LOD. For interrupted CPU runs, repeat the same `build-dense-photos` command
with `--resume`: it rechecks the original and workspace photo hashes and
skips existing verified mesh/dense stages, saving hours of repeated depth maps.
You may explicitly add `--no-refine` while resuming if refinement has not
yet begun in the recorded pipeline; this keeps the original high-resolution
surface as the reviewed mesh. Texturing uses OpenMVS OBJ export with actual
UV-compatible texture atlases rather than an assumed .mvs project archive.
The default `--texture-decimate 1.0` retains all source faces, but for very
large scans OpenMVS texture patch selection may be much slower than meshing.
Use `--texture-decimate 0.25` to build a faster UV-textured LOD **without
modifying the high-poly PLY**. An interrupted texture stage can be resumed
with a lower texture decimation factor; the immutable dense/mesh stages are
reused, not recomputed. This does not replace the original high-poly detail
review or prove that the UV LOD captures thin features.
OpenMVS already performs global and local texture seam leveling by default.
Its upstream [2.4.0 black-patch regression](https://github.com/cdcseacave/openMVS/issues/1251)
can make otherwise usable masked textures appear black; first try a newer
fixed build. If necessary, use `--no-global-seam-leveling` and/or
`--no-local-seam-leveling` for a **new private reconstruction revision**.
Both options are recorded in the report and may change during `--resume`
only when the texture stage failed. Disabling leveling may restore colored
surfaces but increase UV seams; compare textured Blender reviews.
Post-hoc exposure correction cannot repair black texture patches created
during projection.
Rectified mask files must be genuine grayscale PNGs (binary 0/255, both
present, exact dimensions), with exact `<photo filename>.mask.png` names.
`prepare-dense-masks` generates those files with an `approved_for_reconstruction:
false` QA report, SHA-256 input/output pins and per-image red overlays under
`review/`; it does *not* silently approve its own segmentation. Use
`--guides` with the schema in
[`examples/statue-mask-guides.example.json`](../examples/statue-mask-guides.example.json):
normalized 0–1 boxes and optional foreground/background seed points can be
specified for each rectified filename, with a common default. A default
center box is available for quick drafts; use `--max-side` and `--iterations`
to control GrabCut CPU cost. The `--segmenter rembg` alternative uses the explicitly
selected `isnet-general-use` model (not a changing rembg default); it does not
accept rectangle/point hints. If semantic inference misses thin features,
rerun with guided GrabCut or use `refine-dense-masks` to remove small detached
foreground islands, fill small enclosed holes and draw per-image polygon
inclusions/exclusions. The format in
[`examples/statue-mask-edits.example.json`](../examples/statue-mask-edits.example.json)
uses normalized (0–1) coordinates in the **rectified** image, with exclusions
applied last. `--min-island-pixels` and `--max-hole-pixels` are pixel-area
thresholds, not radii: keep them low to avoid erasing fingers, hair or carving.
The command validates all source photograph and mask hashes, writes a separate
immutable candidate revision with change counts and QA contact sheet, and
**never auto-approves** its output. Inspect every revised overlay, then point
OpenMVS to the new revision's `masks/` directory. Refine bad images
in a **new output revision**; masks
that include trees or truncate the statue produce contaminated geometry.
`--edge-inset-pixels 1..8` is an optional inward contour erosion to trim
background-colored halos. It may also erase thin hair or fingers: start with
1–2 pixels, inspect contours at full resolution, and restore essential details
using inclusion polygons (applied after the inset).
`--color-band-pixels 1..32` is an **opt-in local GrabCut edge refinement**:
the source photograph supplies color evidence only within that many original
pixels of each current foreground-mask boundary. Definite interiors and
background outside that band are clamped to their original labels. Use
`--color-max-side` (256–4096) to cap CPU/RAM and `--color-iterations` (1–8)
to control refinement work. This can remove background-colored halos at
high-contrast statue edges, but similar-colored walls, deep folds and thin
hair/fingers can still be misclassified. It writes per-view changed-pixel
counts and a **new, unapproved revision**; inspect it and restore detail with
include polygons if necessary. A broad band or low processing resolution may
make boundaries blockier, so start with 2–6 pixels and review at 100% zoom.
`--color-trim-only` is the safer background-removal variant: only pixels
previously labeled foreground can be removed; it cannot extend the mask into
new background regions. This is **not** guaranteed to improve semantic masks;
compare new and old audits before selecting a revision for dense MVS.
The masks are SHA-pinned, copied only into the private MVS workspace, checked
on resume, and remain absent from Git. Do **not** reuse original distorted-image
masks by simply resizing them: regenerate/warp them in the exact rectified
camera grid and inspect the boundary before reconstruction.

`audit-dense-masks` cross-checks each registered COLMAP 3D point track in the
rectified images and reports where its **same physical feature** is foreground
in some masks and background in other views. It reads the genuine binary or
text COLMAP `images` model, pins the source/model/mask SHA-256 hashes, writes
camera-specific three-color marker overlays, a tinted foreground-mask preview,
a worst-first contact sheet, up to three full-resolution 512px detail crops
per camera and per-view disagreement counts. **Red** marks a possible
extra-foreground/background leak (the local mask includes a feature most
other registered views exclude); **cyan** marks a possibly missing statue
region (the local mask excludes a feature most other views include);
**yellow** means a tied track (often just two views). The JSON report includes
bounded normalized review coordinates, track IDs and label votes so a reviewer
can find the candidate corrections at their original image resolution.
These are diagnostic suggestions, never automatic mask edits. Neither consistent labels
nor a small sparse sample guarantee a good silhouette: consistently included
background features and thin untracked anatomy still require manual review.
All previews remain private and the audit never edits masks or automatically
approves reconstruction.
`compare-dense-masks` runs the same audit on a baseline and a revised mask
folder, writes both private reviews, hashes the reports, ranks per-camera
differences and flags increases in mixed 3D tracks or likely background leaks.
Its `consistency_regressed` field is a **review alert**, not a proof of which
mask is better. Keep the baseline when an optional candidate has not been
visually justified.

Loosening thresholds does not improve a weak reconstruction. Each stage logs
its command and stdout/stderr. Failures preserve a diagnostic report instead
of silently promoting an unusable mesh.

## Results and review

### Real outdoor foreground test (private, 2026-11-02)

Reconstructed all **32/32** cameras from the official EPFL sculpted-head
research photographs using CPU PyCOLMAP 4.2.1 and generated masks for all
32 rectified images in two *private* revisions. The unchecked broad-center
GrabCut approach visibly followed nearby building facades in multiple views;
the local `isnet-general-use` semantic model isolated the sculpture and
pedestal substantially better on visual contact-sheet inspection. Some thin
edges and ground/base patches still need review or manual correction. Neither
set was promoted to an approved silhouette or used as evidence of a completed
background-free 3D mesh. No EPFL photos, masks or renders are included in Git
or CI (research use only without written commercial permission).

The 32-photo EPFL Mask dataset was re-run with the official source ZIP SHA-256
verified, CPU PyCOLMAP 4.2.1 registering **32/32** cameras and **7,906** shared
3D tracks. The semantic IS-Net masks produced **468** mixed-label tracks
(41 tied). A 4-pixel unrestricted color-boundary pass increased disagreement to
**504** (47 tied), and the conservative `--color-trim-only` pass yielded
**482** (40 tied). The baseline is therefore preferable on this metric;
no revised candidate was promoted. This is a *triage metric*, not proof that
the unchanged mask is more accurate: inspection still finds pedestal/ground
patches with inconsistent inclusion, and isolated profile edges are hard to
interpret from sparse tracks. Real-data runs also revealed that COLMAP can
contain multiple observations of the same 3D track within one registered
image; the auditor now deduplicates these votes and excludes contradictory
within-camera labels rather than failing or double-counting them.
The new `compare-dense-masks` command independently reproduced the conservative
candidate's **+14 mixed tracks** and **+7 suspected excess-foreground votes**
relative to the semantic baseline, set `consistency_regressed: true`, and
ranked the affected cameras (notably views 012–014) for inspection.

- OpenMVS: workspace/scene_dense_mesh.ply, optionally
  workspace/scene_dense_mesh_refined.ply and textured OBJ/atlas outputs.
  OpenMVS 2.4 mesh/refinement stages emit PLY without necessarily saving
  .mvs archives: validate the actual geometry and atlas files.
- COLMAP: meshed-poisson.ply and optionally a textured/ folder.
- workspace/: an immutable private snapshot of the registered, undistorted
  images/cameras for reproducible relative paths and downstream depth maps.
- dense-report.json: registration fraction, input manifest, method, stage
  statuses, vertex/face counts, PLY SHA-256, texture OBJ/atlas SHA-256
  hashes and review requirements.
- logs/: per-stage logs, including GPU, memory and image failures.

The Blender review worker validates the mesh against the recorded SHA-256 and
face counts before importing PLY geometry. Its separate immutable folder
contains private-dense-review.blend, private-dense-review.glb, a report and
three 320px neutral-studio views from fixed local axes. With
`--matched-views 1..12` it also validates hashes of selected original
rectified photos, creates calibrated Blender perspective cameras in the
**same COLMAP frame as the dense PLY** and renders photo-matched clay views;
the default 100% camera resolution preserves fine-detail inspection
(adjust `--matched-resolution-percent` to 10–100 for drafts).
Each preview identifies the pinned corresponding photograph in the private
report. These cameras do not require survey anchors until the mesh is
transformed into an independent Blender world coordinate system. The GLB is for
**local artistic review only**, never an authorization to redistribute the
original photographic texture or derived geometry. Dense photo projection
atlases cannot simply be attached to the PLY without the matching UVs:
`--textured` imports the actual OpenMVS OBJ/MTL/JPG UV set, validates every
material path and recorded hash, packs the image into a separate
private-textured-review.blend and private-textured-review.glb. The neutral
high-poly PLY remains separate and unmodified. Real photo atlases can still
contain dark patches, empty-color regions and exposure seams; review these
against authentic photos before declaring a texture finished.
`finish-dense-textures` applies bounded exposure EV, gamma, saturation,
optional sampled cast-neutralization and shadow lift to an independent
OBJ/MTL/atlas revision; it protects near-black/uncovered atlas texels and
reports their approximate coverage and highlight-clipping warnings. With
`--balance-atlases`, it additionally matches approximate exposure across
**multiple atlas pages** using valid non-black, non-transparent texels, capped by
`--max-balance-ev` (default 0.75 EV). A single page is left unchanged by
that option. It also writes original/corrected side-by-side
`review/` previews. All original source files, UVs and face
indices remain untouched. `--texture-finish` requires `--textured` and
verifies both original and corrected content hashes before Blender import.
With `--uv-aware-stats`, exposure, optional white-balance neutralization and
atlas-to-atlas balancing use **only texels belonging to actual OBJ UV faces**,
not unrelated colored padding or unused atlas space. The command parses the
pinned OBJ/MTL face materials, produces a bounded UV occupancy mask for each
page and reports `uv_mapped_atlas_fraction`, `uv_mapped_faces`, and the fraction
of *mapped* texels that are dark/transparent. Its separate
`review/*.uv-dark.jpg` preview marks those on-mesh dark/empty patches in red.
This helps identify missing texture data that overall atlas statistics can
hide. It is disabled by default for compatibility; it rejects unsupported
UV tiling, absent face UVs and unused atlas pages instead of guessing.
Diagnostic masks are approximate at 768px resolution and cannot recover
photographic details or correct UV seam geometry.

### Advanced photographic material finishing

Install `pip install '.[finishing]'` for Pillow + NumPy. The advanced surface
recipe is strictly **opt-in**; existing commands and texture exports remain
unchanged without `--surface-recipe`. Use the exact atlas filename(s) reported
by `dense-report.json`. Each recipe has **one entry for every recorded atlas**;
region masks are local grayscale PNGs at the full atlas resolution and may not
overlap. Keep all referenced files next to the JSON or under safe relative
subdirectories. Files and the recipe are hash-pinned before processing.

```json
{
  "schema_version": 1,
  "atlases": {
    "scene_dense_mesh_refined_texture_material_00_map_Kd.png": {
      "regions": [
        {"id": "weathered-stone", "mask": "masks/stone.png", "roughness": 0.88},
        {"id": "oxidized-metal", "mask": "masks/metal.png", "roughness": 0.72}
      ],
      "illumination_map": "guides/lighting.png",
      "damage_mask": "guides/camera-artifacts.png",
      "height_map": "maps/scanned-height-16bit.png",
      "displacement_scale_m": 0.012
    }
  }
}
```

- **Material separation:** masks identify UV-space regions for separate
  roughness priors and de-lighting, but do not split mesh faces into new
  material slots. Unlabeled texels use a matte fallback; black/transparent
  gutters are protected. Manually inspect halos where regions meet.
- **De-lighting:** guided illumination maps should be prepared from calibrated
  reference evidence or artist review, not an unexamined raw photograph.
  128 represents neutral light; brighter means stronger recorded illumination,
  darker means shadow (values below 16 are treated as unknown). The correction
  is bounded in **linear RGB** by `--surface-max-delight-ev` (0–2 EV) and
  blended by `--surface-delight-strength` (0–1). If there is no illumination
  map but strength is nonzero, a region-masked large-blur heuristic estimates
  illumination; `--surface-low-frequency-radius` controls it (16–1024 pixels).
  The fallback can remove real low-frequency pigment/weathering: inspect every
  original/finished atlas comparison. This is NOT a solved intrinsic-image
  decomposition problem.
- **Damage cleanup:** only areas explicitly painted into `damage_mask` are
  filled with nearby *same-material* colors. The mask may cover at most 5% of
  an atlas and `--surface-cleanup-radius` must be 1–16. Do not paint original
  cracks, chips, authentic patina or damaged monument surfaces into this mask;
  it is for camera artifacts, small projection holes and reconstruction faults.
- **Roughness:** region `roughness` is a deliberate 0.02–1 **prior**, not a
  reflectance measurement. An optional per-region `roughness_map` field accepts
  an externally authored full-resolution grayscale PNG that overrides the prior
  inside the region. A full-resolution, grayscale, hash-pinned roughness
  PNG is generated for the GLB's Principled BSDF. For physically estimated
  roughness, capture calibrated cross-polarized/multi-light views and author a
  measured roughness map in a separate material pipeline. Diffuse image
  brightness alone cannot determine roughness.
- **Displacement:** `height_map` is optional but, when present, **must be true
  16-bit grayscale PNG** accompanied by a real-world `displacement_scale_m`
  between 0.00001 and 0.5. No depth/geometry is invented from albedo. The
  copied height image is connected to a Cycles Displacement node in the
  editable private `.blend`; artist-controlled subdivision/displacement settings
  are necessary for silhouette detail, and the source mesh must have a calibrated
  meter scale for `displacement_scale_m` to be physically meaningful.
  **glTF/GLB does not encode true height
  displacement**, so GLB appearance will differ from a high-resolution Cycles
  render unless geometry is intentionally baked.
- **UDIM:** `package-finished-udim` assigns numbered tiles (1001, 1002, ...)
  to original atlas pages, duplicates shared per-material OBJ `vt` indices,
  and emits `udim.obj`, `udim.mtl`, `albedo.<tile>.png`, plus any roughness and
  height tiles, with a hash/provenance report. In Blender, load the generated
  `<UDIM>` image sequence as a **Tiled** image and wire it to Principled BSDF;
  check material slots and UVs before rendering. glTF has no native UDIM;
  Blender's exporter splits UDIM into tiles/materials, so validate the final
  portable GLB separately. This package does not overwrite the original UVs,
  mesh, material revisions or Blender review.

All of these outputs remain `artistic-review-required` and private; none
establishes photo-perfect likeness. Photograph-to-material decomposition is
ambiguous without illumination and reflectance measurements. Compare
re-rendered front, side and three-quarter views under multiple lights and
review baked shadows/seams before shipping an asset.

For multi-page models, `--balance-uv-seams` instead estimates per-page exposure
differences at **shared OBJ mesh vertices along material/atlas boundaries**.
It samples just inside the adjacent UV faces to avoid black atlas gutters,
rejects transparent, near-black and clipped samples, requires at least eight
valid shared vertices per page pair, and rejects inconsistent brightness gaps.
Accepted page constraints are solved across connected atlas pages, centered
around zero to avoid an overall exposure shift, and individually limited by
`--max-seam-ev` (default 0.75 EV). The output report includes the candidate,
sampled and valid seam vertex counts, pairs accepted/rejected and why, median
and scatter of the source brightness gap, approximate remaining EV gap after
adjustment, per-page `uv_seam_balance_ev`, and warnings for unsupported pages.
No page gain is applied where evidence is inadequate. `--balance-atlases` and
`--balance-uv-seams` cannot be combined because they use different page-exposure
estimators. UV seam balancing does **not** handle in-page chart seams, relight
strong cast shadows, resolve different materials or infer disconnected/duplicated
OBJ vertex positions as the same physical vertex. Review the corrected atlas
previews and Blender model against source photographs before using an output.
This can improve dark overall exposure, broad color cast and atlas-page
exposure mismatch, but **cannot invent missing photographs, paint empty atlas
patches or fix intra-page UV chart seams and baked lighting**. Use the
original-versus-corrected Blender views for
artistic approval; the correction remains private by default.
For dark atlases whose desired exposure is not yet known,
`--auto-exposure-luma 115` estimates a **bounded shared exposure offset**
from the median of non-black, non-transparent atlas texels. The option is
disabled by default; its allowed target range is 32–224 on the 0–255
grayscale scale, and `--max-auto-exposure-ev` (default 1 EV) caps correction.
Any additional manual `--exposure-ev` is additive, so review highlight-clipping
warnings before combining them. Different stone colors and deliberately dark
surfaces can fool median brightness: this is image enhancement, not a physical
albedo calibration or proof of historical color.

Open the PLY in Blender and compare front/profile/three-quarter views against
authentic photos, especially eyes, chin, mouth, ears, hair and deep garment
folds. The resulting mesh may be noisy, over-smoothed, missing behind-camera
regions or closed with fictitious geometry. The mesh coordinates are still
in COLMAP's **arbitrary scale and frame**; align them to surveyed anchors
before making claims about metres. Photo coverage and face count are not
historical-accuracy certificates.

**Rights:** Dense results and source photos must remain **private**.
Texture projection directly uses photo pixels, and the report explicitly
sets distribution_allowed to false. Do not upload generated atlases, scenes,
models or photos until their owners' rights and subject identity are reviewed.
The workflow does not grant redistribution permission or publish a GLB.

After rights clearance and sculpt review, use a Blender mesh import/edit step
to bring the high-resolution result into the scene, then optimize it with
the licensed-scan LOD/PBR pipeline from PR #12. Automated artist-quality
retopology, displacement generation and correction of missing facial anatomy
remain separate work.

The Python regression tests use controlled mock executables to verify secure
preflight, subprocess commands, immutable workspace copies, failure
checkpoints and mesh-output validation. A separate CI Blender 4.5 smoke
imports a first-party synthetic PLY, renders 3 views and validates its
private GLB and editable scene.

## Reproducible real-photo statue benchmark (Pix4D eagle)

Pix4D publishes [44 original 5184x3456 JPG views of an outdoor eagle statue](
https://support.pix4d.com/hc/en-us/articles/360000235126) photographed during
two walks at different heights. The original example ZIP is available at
`https://data.pix4d.com/misc/example_datasets/example_eagle.zip` (388 MiB).
This scene challenges wing, feather, talon and surface-relief fidelity; foliage
and ground may create unwanted reconstructed background geometry.

**Rights:** These are **not** CC0 photos. Pix4D allows personal or professional
training use; commercial/promotional use requires a linked
[Courtesy of Pix4D / pix4d.com](https://www.pix4d.com/) notice, with other usage referred to Pix4D.
Keep the photo ZIP, reconstructed surfaces, textures and rendered previews
**out of Git and public CI artifacts**. The example uses private training data
only; a successful dense run does not clear public redistribution rights.

    # Download the example_eagle.zip from the official source above,
    # then unzip it into a private folder OUTSIDE the repository.
    python scripts/prepare_pix4d_eagle.py \
      --unpacked /private/pix4d-eagle --archive /private/example_eagle.zip

    kumiki-blender reference-audit \
      --manifest /private/pix4d-eagle/reference-set.json
    kumiki-blender reconstruct-photos \
      --manifest /private/pix4d-eagle/reference-set.json \
      --output /private/pix4d-eagle-sparse-r01 \
      --engine pycolmap --undistort --max-image-size 1600
    kumiki-blender build-dense-photos \
      --reconstruction /private/pix4d-eagle-sparse-r01 \
      --output /private/pix4d-eagle-dense-r01 --backend openmvs

The helper verifies the exact photo count, optional pinned source ZIP hash,
and every image hash, and records source and rights separately. It assigns
`unknown` rather than guessing view labels from file order. The photo audit
cannot confirm camera overlap; only the SfM registration and photo-matched
visual inspection can do that. On large statues, use a low-resolution
diagnostic run first; repeat at higher image resolution only after checking
foreground geometry against actual statue photographs.

**Measured private CPU SfM benchmark (2026-11-02):** PyCOLMAP 4.0.2
registered **44/44** eagle photographs after downscaling SIFT to 1600px,
triangulating 14,556 sparse points; median *feature reprojection* error was
1.153 pixels. This demonstrates genuine multi-view camera registration, not
mesh quality, full surface coverage, scale accuracy or historic likeness.

The first private 1600px OpenMVS run produced **2,236,655 dense points** and
an **889,038-vertex / 1,777,733-face** unrefined eagle surface. That exposed
two critical defects in the initial pipeline: OpenMVS 2.4 outputs the actual
mesh PLY without the anticipated .mvs archive, and its default refinement
automatically removes much of the input geometry. The code now recognizes
actual stage outputs, resumes from immutable intermediate evidence, records
raw-vs-refined face retention and defaults to preserving input geometry.
These geometry counts are evidence of density, **not** a surface-accuracy
score; the scan still needs private photo-matched inspection.
An experimental run with OpenMVS's automatic refinement retained only
**312,706 faces (17.6%)** of the raw 1,777,733-face surface, justifying the
new explicit preservation setting; this reduction is not, by itself,
evidence that one version is a closer match to photographs.

**Completed private end-to-end MVS/Blender validation (2026-11-02):** The
Eagle pipeline resumed the expensive dense/mesh stages after correcting the
OpenMVS PLY output contract; it completed a 1,777,733-face raw scan, a
UV-compatible OBJ of the same face count and an **8192px JPEG atlas**.
Blender 4.5.15 imported the real PLY and generated 4 calibrated photo views,
an editable .blend and a ~149 MB private clay GLB. It independently imported
the OpenMVS UV OBJ/MTL/JPEG, embedded the atlas, and exported a ~205 MB
private textured GLB. Texture projection took ~16 min at full mesh density;
`--texture-decimate 0.25` yielded a **444,424-face textured LOD** in ~2.5
min without modifying the raw geometry. All private atlas and model files
remain outside Git. The atlas visibly contains very dark patches and the
outdoor mesh still reconstructs ground/branches, so this is **proof of
pipeline execution and camera/UV compatibility, not finished texture or
silhouette quality**. Verified foreground masks and artist cleanup remain
essential before any historic/production claim.

## Independent full-body and facial reference sets (EPFL)

The [EPFL Geometric Computing Laboratory statue archive](https://lgg.epfl.ch/statues.php)
provides original photographs and student-made reconstructed meshes. Two useful
additional stress tests are:

- **Freddie Mercury standing monument:** 37 overlapping full-body photographs,
  exposed arms, fine facial relief, folded clothing and a busy city background.
  [Original photo archive](https://lgg.epfl.ch/old/statues_data/dataset2013/mercury_pics.zip).
- **Outdoor sculpted face (“Mask”):** 32 photographs that expose nose, eyelids,
  lips, chin and the monument base.
  [Original photo archive](https://lgg.epfl.ch/old/statues_data/dataset2013/head_pics.zip).

EPFL permits use **for research**, not commercial distribution without written
permission. Models derived from EPFL data must remain private. The meshes are
useful qualitative comparisons, not calibrated ground-truth scans.

    # Unpack each original photo ZIP outside the repo and keep its original
    # Photos/ or africanhead_photos/ directory intact.
    python scripts/prepare_epfl_statue.py \
      --scene freddie-mercury --unpacked /private/epfl-freddie \
      --archive /private/mercury_pics.zip
    python scripts/prepare_epfl_statue.py \
      --scene sculpted-mask --unpacked /private/epfl-mask \
      --archive /private/head_pics.zip

    kumiki-blender reconstruct-photos \
      --manifest /private/epfl-freddie/reference-set.json \
      --output /private/epfl-freddie-sparse-r01 \
      --engine pycolmap --undistort --max-image-size 1600

As with the eagle scene, these helpers pin each original archive hash, record
the individual photo hashes and mark all camera directions as unverified until
SfM. Compare the private reconstruction to EPFL's views and mesh without
turning the research dataset into a public example asset.

**Measured private facial-camera benchmark (2026-11-02):** All **32/32** EPFL
sculpted-mask photos registered with PyCOLMAP 4.0.2 CPU; 5,834 sparse matched
points and median feature reprojection error 0.909px. All 32 originals had
invalid EXIF orientation `0`: the first direct test failed in native COLMAP,
and the lossless staged metadata repair enabled the successful second run.
These are camera/metadata results, not a validated facial mesh.

**Measured private full-body benchmark (2026-11-02):** All **37/37** Freddie
Mercury statue photos registered with PyCOLMAP CPU at 1600px, with 11,039
sparse matched points and median *feature* reprojection error 0.479px. This
supports camera coverage of a challenging standing monument, not verified
anatomical or mesh fidelity.

## Reusable CC BY sculpture benchmark (Cycladic figurine)

The [University of Wyoming's original Greek Cycladic Figurine photogrammetry
dataset](https://dataverse.arcc.uwyo.edu/dataset.xhtml?persistentId=doi:10.15786/16543554)
contains **72 photographs** of a replica sculpture (229.2 MiB ZIP). It is
explicitly **CC BY 4.0**. Credit *Hutchens, Chad Everett; Koenig, Charles
(2021), Greek Cycladic Figurine photogrammetry dataset, Wyoming Data
Repository, doi:10.15786/16543554*, link the license and identify any changes.
This small tabletop sculpture is a good legally reusable integration fixture,
but **not** a replacement for the busy-background, large-statue benchmarks
above. No dataset photographs or models are included in Git.

    # Download the original ZIP from the official Dataverse page above.
    # Extract privately and run the source-aware intake (72 images required).
    python scripts/prepare_cycladic_figurine.py \
      --unpacked /private/cycladic-figurine \
      --archive /private/cycladic_figurine_photo_dataset.zip

    kumiki-blender reconstruct-photos \
      --manifest /private/cycladic-figurine/reference-set.json \
      --output /private/cycladic-sparse-r01 \
      --engine pycolmap --undistort --max-image-size 1600

The intake verifies the publisher's recorded archive MD5 when supplied,
pins the actual source ZIP SHA-256, individual photo SHA-256 values and
attribution, and deliberately leaves camera directions unknown until SfM.
Generated output stays private pending manual review and a valid attribution
notice. A full OpenMVS/Blender reconstruction on this source remains a future
benchmark; availability of photographs is not a claim that its camera
registration or mesh appearance has already been verified.
