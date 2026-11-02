# Dense statue reconstruction from overlapping photographs

The sparse COLMAP workflow reconstructs **cameras and a sparse point cloud**,
not accurate eyelids, hair, worn stone or folded garments. This optional stage
runs multi-view depth estimation, fusion, surface meshing, refinement and
private texture projection to produce a mesh for artistic inspection.

## Requirements

- Photos must depict the **same actual statue** with strong overlap, consistent
  geometry and sharp views of the front, profiles, back, head and drapery.
  Unrelated web photos and photos of different restorations may not register.
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

    # CPU-capable OpenMVS meshing + refinement + texturing.
    kumiki-blender build-dense-photos \
      --reconstruction .review/statue/sparse-r01 \
      --output .review/statue/dense-r01 --backend openmvs

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
    kumiki-blender render-dense-blender-review \
      --dense-revision .review/statue/dense-r01 \
      --output .review/statue/blender-review-r01 --matched-views 4

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
Loosening thresholds does not improve a weak reconstruction. Each stage logs
its command and stdout/stderr. Failures preserve a diagnostic report instead
of silently promoting an unusable mesh.

## Results and review

- OpenMVS: workspace/scene_dense_mesh.ply, optionally
  workspace/scene_dense_mesh_refined.ply and textured OBJ/atlas outputs.
  OpenMVS 2.4 mesh/refinement stages emit PLY without necessarily saving
  .mvs archives: validate the actual geometry and atlas files.
- COLMAP: meshed-poisson.ply and optionally a textured/ folder.
- workspace/: an immutable private snapshot of the registered, undistorted
  images/cameras for reproducible relative paths and downstream depth maps.
- dense-report.json: registration fraction, input manifest, method, stage
  statuses, vertex/face counts, PLY SHA-256 and review requirements.
- logs/: per-stage logs, including GPU, memory and image failures.

The Blender review worker validates the mesh against the recorded SHA-256 and
face counts before importing PLY geometry. Its separate immutable folder
contains private-dense-review.blend, private-dense-review.glb, a report and
three 320px neutral-studio views from fixed local axes. With
`--matched-views 1..12` it also validates hashes of selected original
rectified photos, creates calibrated Blender perspective cameras in the
**same COLMAP frame as the dense PLY** and renders photo-matched clay views;
each preview identifies the pinned corresponding photograph in the private
report. These cameras do not require survey anchors until the mesh is
transformed into an independent Blender world coordinate system. The GLB is for
**local artistic review only**, never an authorization to redistribute the
original photographic texture or derived geometry. Dense photo projection
atlases are not automatically attached to the PLY without real matching UVs.

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
