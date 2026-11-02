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

Options include --min-registered-images, --min-registered-fraction,
--min-mesh-faces, --timeout-seconds, --no-refine (OpenMVS) and --no-texture.
Loosening thresholds does not improve a weak reconstruction. Each stage logs
its command and stdout/stderr. Failures preserve a diagnostic report instead
of silently promoting an unusable mesh.

## Results and review

- OpenMVS: workspace/scene_dense_mesh.ply, optionally
  workspace/scene_dense_mesh_refined.ply, workspace/scene_textured.mvs
  and texture atlas images.
- COLMAP: meshed-poisson.ply and optionally a textured/ folder.
- workspace/: an immutable private snapshot of the registered, undistorted
  images/cameras for reproducible relative paths and downstream depth maps.
- dense-report.json: registration fraction, input manifest, method, stage
  statuses, vertex/face counts and review requirements.
- logs/: per-stage logs, including GPU, memory and image failures.

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
checkpoints and mesh-output validation. Real multi-view photo quality remains
unverified until the pipeline runs on a rights-cleared overlapping statue
dataset with genuine COLMAP/OpenMVS installations.
