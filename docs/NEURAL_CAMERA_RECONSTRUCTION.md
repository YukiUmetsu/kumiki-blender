# Learned camera matching for difficult multi-view photographs

Use this workflow when traditional SIFT registers many photographs from one
shoot but misses opposite-side views from other photographers, dates, cameras,
or lighting conditions. It changes **sparse camera registration**, not the
artist's mesh or the downstream OpenMVS backend.

## Supported backends

| `--matching-preset` | COLMAP extractor | Matcher | Notes |
| --- | --- | --- | --- |
| `sift` (default) | SIFT | SIFT brute-force | Existing behavior; also works with PyCOLMAP |
| `sift-lightglue` | SIFT | LightGlue | Good first trial when SIFT finds too few cross-view matches |
| `aliked-lightglue` | ALIKED N16ROT | LightGlue | Learned descriptors for difficult viewpoints |
| `loma` | LoMa B | LoMa B | Higher-cost learned features and matcher |
| `loma128` | LoMa B128 | LoMa B128 | Lower-memory LoMa variant |

Learned presets require `--engine colmap` with an ONNX-enabled **COLMAP 4.2+**
executable. They do **not** work with the `--engine pycolmap` wrapper. The
existing `--registration-profile high-recall` is currently PyCOLMAP-only,
so it cannot be combined with these presets. A new immutable reconstruction
revision creates a fresh feature database for each preset.

COLMAP will download/cache some ONNX model weights on first use. It may use
CoreML on macOS when supported by its ONNX runtime, otherwise CPU; neither
Metal acceleration nor successful registration is guaranteed. **Do not assume
that `brew install colmap` enabled ONNX:** inspect the actual CLI first.

```bash
colmap --version
colmap feature_extractor --help | rg 'FeatureExtraction.type|LOMA_B|ALIKED_N16ROT'
colmap exhaustive_matcher --help | rg 'FeatureMatching.type|LOMA_B|SIFT_LIGHTGLUE'
```

If the learned options are missing, use a newer [COLMAP build with ONNX enabled](https://colmap.github.io/install.html)
(`-DONNX_ENABLED=ON`) or continue using `sift`/the high-recall PyCOLMAP path.
Our preflight checks the advertised extractor/matcher before creating output.

## Great Sphinx: compare actual registered views

Use the same original 36-photo folder for the baseline and each experiment.
Existing model revisions are **never modified**. Set paths explicitly on the
Mac where COLMAP, OpenMVS, and the photos are installed:

```bash
PHOTOS="$HOME/Pictures/sphinx-reconstruction-candidates/warren-2019/photos"
# Update PHOTOS to the folder containing all 36 images if it differs.
BASELINE="$HOME/Documents/projects/services/kumiki-blender-photo-model/.review/great-sphinx/mesh-multiview-r02/sparse"
LEARNED="$HOME/Documents/projects/services/kumiki-blender-photo-model/.review/great-sphinx/cameras-lightglue-r01"

kumiki-blender reconstruct-photos \
  --photos "$PHOTOS" \
  --asset-id great-sphinx-multiview \
  --output "$LEARNED" \
  --engine colmap \
  --matching-preset sift-lightglue \
  --undistort \
  --timeout-seconds 7200

kumiki-blender compare-camera-registrations \
  --baseline "$BASELINE" \
  --candidate "$LEARNED"
```

For a LoMa run, select **another unused** output directory and change the
preset to `loma` (or `loma128` for a lower-cost test):

```bash
LOMA="$HOME/Documents/projects/services/kumiki-blender-photo-model/.review/great-sphinx/cameras-loma-r01"
kumiki-blender reconstruct-photos \
  --photos "$PHOTOS" --asset-id great-sphinx-multiview \
  --output "$LOMA" --engine colmap --matching-preset loma \
  --undistort --timeout-seconds 7200
kumiki-blender compare-camera-registrations --baseline "$BASELINE" --candidate "$LOMA"
```

These sparse-only experiments **do not** run OpenMVS, Blender, or create a
GLB. The report lists `newly_registered`, `lost_registrations`, and
`still_unregistered`, matching source photos by SHA-256 even if file names
change. For earlier revisions with no registration-name list, the comparison
reads `text-models/<best-model>/images.txt`. You can also compare a full
photo-model revision directly (it resolves its `sparse/` subdirectory).

Inspect the candidate's `reconstruction-report.json`, its per-model
`registered_filenames`, and the registered camera poses visually. **A larger
registered count alone is not a success criterion**: wrong correspondences,
incorrect poses, or noisy triangulation can make the mesh worse. Inspect
`sparse_feature_evidence` for reprojection errors and track lengths, and
check left/right silhouettes before running a costly mesh.

## Next steps and limitations

Once a candidate visibly improves opposite-side coverage, use that
`reconstruct-photos --undistort` revision as input to `plan-dense-photos`
and `build-dense-photos` (or the registered-camera reuse workflow when
available). Do not reconstruct dense geometry from disconnected models or
assume unobserved surfaces are accurate. Masks and restoration scaffolding
may still degrade matches; manually review them and the model alignment.

MapAnything and VGGT/Ω are separate neural **camera-prediction** systems,
not drop-in feature matcher options. Their camera intrinsics, coordinate
frames, image orientation, metric scale, and checkpoint licenses require
independent validation before safe COLMAP/OpenMVS import. We intentionally
do not present their unverified outputs as a camera solve in this CLI.

References:

- [COLMAP: Feature Extraction and Matching](https://colmap.github.io/features.html)
- [COLMAP: Installation](https://colmap.github.io/install.html)
- [MapAnything research implementation](https://github.com/facebookresearch/map-anything)
- [VGGT-Ω research implementation](https://github.com/facebookresearch/vggt-omega)
