# Christ the Redeemer — real-scan registration gate (HOLD)

The latest Blender clay study has more detailed anatomy than the original
procedural face, but temple hair still looks like flat ribbons and the face
does **not** resemble the actual sculpture. We visually rejected several
hand-authored posterior hair shell prototypes: they created large helmet/cape
shapes and failed the real monument profile silhouette. These rejected
prototypes are **not** included in the production GLB or this PR.

**Different reconstruction method:** obtain a rights-cleared **real monument
scan** (not a replica or generic anatomy donor) and register its 3D facial
landmarks to the existing full figure. This is a rigid+uniform-scale transform,
not bounding-box fitting or more guessed procedural face offsets.

## New reusable registration

- src/kumiki_blender/scan_registration.py: deterministic robust 3D similarity
  registration with non-coplanar landmark checks, RANSAC outlier rejection,
  mirrored-axis detection, anisotropic deformation rejection, full 4×4 matrix,
  per-feature residuals and **HOLD** status.
- scripts/landmarks/align_christ_scan.py: rights-aware local GLB import,
  pinned source SHA-256, explicit authenticity declaration, provenance
  manifest, separate **HOLD** aligned GLB and audit JSON. Uses existing
  asset_import safety checks.
- Tests recover known rotation/translation/scale and reject mirrored,
  malformed, anisotropically scaled, coplanar and outlier-contaminated inputs.
  The CLI also exercises a synthetic GLB export; the fixture is **not** a
  real monument scan.

## Input

Supply a local .glb and an asset_import v1 provenance JSON with **verified
commercial and redistribution permission**. No scraping of Sketchfab models
or repackaging of view-only surveys. Keep third-party meshes out of the
repository unless their license explicitly permits redistribution.

Example **anchor format** (coordinates illustrative, not measurements):

    {
      "schema_version": 1,
      "coordinate_system": "glb-y-up-metres",
      "source_kind": "actual-monument-scan",
      "source_authenticity_review": "documented source and independent visual verification pending",
      "source_sha256": "<actual lowercase SHA-256 of local GLB>",
      "landmarks": [
        {"name": "chin_tip", "source_xyz": [0, 1, 0], "target_xyz": [0, 35, 1]},
        {"name": "nose_tip", "source_xyz": [0, 2, 0], "target_xyz": [0, 36, 2]}
      ]
    }

**Provide at least six independent, non-coplanar corresponding 3D anchors**;
never fabricate anchors by projecting unrelated 2D images. Suggested anchors:
nose tip, nose bridge, chin tip, both cheekbones, brow ridge, both hair roots,
and lower ear-covering hair contours. Each point must be physically located
on the source scan and target figure. The example above is intentionally
incomplete and must not be used as an actual fit.

    python scripts/landmarks/align_christ_scan.py \
      --source /path/to/licensed-real-christ-scan.glb \
      --provenance /path/to/source-provenance.json \
      --anchors /path/to/christ-3d-landmarks.json \
      --output .review/christ/registered-scan-HOLD.glb \
      --audit .review/christ/registered-scan-HOLD.json

## Quality gate before promoting geometry

1. Confirm actual monument provenance and commercial reuse rights.
2. Register at least six 3D landmarks with robust fit (default 6 cm inlier
   threshold and 4.5 cm maximum RMS); inspect rejected points individually.
3. Render front, profile, three-quarter and low-angle against real photos.
4. Verify eye/brow/nose/mouth/chin and sculpted hair silhouette, not merely
   vertex count, registration success or bounding dimensions.
5. Retopologize/crop and integrate the approved scan head into the figure;
   verify PBR stone, neck/robe transitions and hand/arm preservation.
6. Only then promote a production GLB and update its asset manifest.

**Current status: no rights-cleared authentic scan has been imported or
registered; existing production figure unchanged.**


## Prepared target landmark scaffold

The file [CHRIST_REDEEMER_SCAN_TARGET_ANCHORS_HOLD.json](CHRIST_REDEEMER_SCAN_TARGET_ANCHORS_HOLD.json)
contains 17 named *existing generic donor* target anchors in **GLB Y-up
metres**. They are derived from the earlier 2D-photo registration fixture
with the explicit axis conversion (x, -z, y) -> (x, y, z) in GLB space.
All source_xyz entries are intentionally null, so the template **cannot
run or silently masquerade as a scan**. Replace them with actual
3D coordinates selected on the licensed monument scan; inspect every
correspondence against the source model before alignment.
