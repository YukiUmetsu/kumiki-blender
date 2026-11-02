# Statue face evidence and likeness review

This independent tool prepares multi-angle sculpture photographs for human review and compares **registered 2D observations** against candidate renders. It does **not** reconstruct a head, sculpt a mesh, estimate camera poses, or certify 3D accuracy. It is intentionally separate from existing modeling, material, and Gaussian-splat pipelines.

## Install and inspect

From the repository checkout, install the existing QA extra and run:

```bash
python -m pip install -e '.[qa]'
python -m kumiki_blender.face_analysis inspect \
  --images-dir /private/christ-face-photos \
  --annotations /private/christ-face-annotations.json \
  --output /private/christ-face-review-r01
```

`--annotations` is optional; without it, directly supplied images have `unknown` view labels. `--detector mediapipe` can be added for optional machine-generated landmarks. MediaPipe must be installed separately, and its points remain **unverified proposals**. The CLI reports missing or incompatible MediaPipe instead of silently switching algorithms. Back-of-head photos should not receive facial points.

If photographs are already registered in a validated Kumiki reconstruction reference manifest, you can use `--reference-manifest /private/references.json` **instead of** `--images-dir`; the existing loader checks the dossier and the adapter preserves its image IDs. Its detailed camera/view labels are mapped to general front, three-quarter, profile, and rear categories. A dossier's `top` view maps to `unknown`; a manifest is not camera calibration or proof of licensing.

Each output is a new, immutable local directory containing:

| File | Meaning |
| --- | --- |
| `photo-audit.json` | Per-image size/hash/view, counts, human-annotation coverage, unknown camera independence |
| `face-evidence.json` | Validated 2D annotations for downstream comparison, without source photo bytes |
| `contact-sheet.png` | Private thumbnail index in numeric filename order (e.g. `face2` before `face10`) |
| `overlays/*.png` | Private markup only for views with observed outlines, points, or proposals |
| `landmark-proposals.json` | Only if a detector is requested; never evidence of photo-faithful sculpt geometry |
| `manifest.json` | Report status and explicit `three_d_fidelity_verified: false` |

Source photos and contact sheets may be subject to third-party rights; keep them and generated reports outside the Git repository. The tool does not publish or download images. Images are checked for file format, size, dimensions, and duplicate identifiers.

## Annotate real evidence

Use a separate JSON file. Image IDs are normalized filename stems: `Christ_the_Redeemer_face1.png` becomes `christ-the-redeemer-face1`. Coordinates are **pixels of the original full-resolution photo**, with the origin at top left. Every view label and point should be manually reviewed.

```json
{
  "schema_version": 1,
  "images": [
    {
      "image_id": "christ-the-redeemer-face1",
      "view": "front",
      "landmarks": [
        {
          "landmark_id": "chin-tip",
          "xy_px": [440, 762],
          "origin": "manual",
          "visibility": "visible"
        }
      ],
      "outline_px": [[300, 250], [580, 250], [600, 830], [280, 830]]
    },
    {
      "image_id": "christ-the-redeemer-face15",
      "view": "rear",
      "landmarks": []
    }
  ]
}
```

**This snippet is illustrative, not the actual measured coordinates or outline of the supplied statue.** Leave `landmarks: []` when measurements have not been reviewed. Do not mark an invisible or occluded face point as `visible`. Accepted views: `front`, `front-three-quarter`, `profile`, `rear-three-quarter`, `rear`, `unknown`. Fields `origin` and `visibility` distinguish observations from proposed, uncertain, or not-visible points. Outlines need at least three non-collinear points.

The initial Christ the Redeemer example contains 15 related viewpoints, approximately front through rear. It provides valuable 2D shape, hair, and silhouette evidence, but neither independent-camera calibration nor complete 360° coverage is guaranteed.

## Compare candidate renders

First create a candidate annotation bundle by inspecting candidate-render PNG/JPEG files the same way. A matching pair needs the same reviewed view and **explicitly verified, shared pixel-coordinate alignment**, normally after the artist has matched reference/candidate camera and crop. The `verified` flag means a human/workflow has established registration; it is not inferred by the tool.

`pairs.json`:

```json
{
  "schema_version": 1,
  "pairs": [
    {
      "reference_image_id": "christ-the-redeemer-face1",
      "candidate_image_id": "candidate-front-r01",
      "alignment_id": "manual-camera-alignment-r01",
      "verified": true
    }
  ]
}
```

```bash
python -m kumiki_blender.face_analysis compare \
  --reference /private/christ-face-review-r01/face-evidence.json \
  --candidate /private/candidate-review-r01/face-evidence.json \
  --pairs /private/pairs.json \
  --output /private/christ-likeness-review-r01
```

`likeness-qa.json` reports per-feature and per-region **projected-image** displacements normalized by the manually observed head-outline scale. It also reports silhouette IoU when both verified outlines exist. Findings such as “rendered chin is higher” refer only to the aligned image plane; they are not authoritative 3D sculpt directions. A missing view, unknown/unverified alignment, occluded landmark, or missing reference scale is `unmeasured`. A rear-only silhouette can be measured without inventing rear facial landmarks.

To iterate, make a fresh `-r02` output directory. Existing report directories are not overwritten. Keep the original photos, annotations, candidate renders, and comparison results private; commit only the reusable module and synthetic tests.

### When a calibrated camera plan and sculpt-point matches exist

The separate `reproject` command deliberately **reuses** the repository's existing `reconstruction.landmarks.check_landmark_reprojection` implementation; it does not invent another projection formula. Both inputs must follow that module's registered camera-plan and 3D sculpt-landmark schemas. No reprojection can be run merely from the 15 uncalibrated photos.

```bash
python -m kumiki_blender.face_analysis reproject \
  --camera-plan /private/registered-cameras.json \
  --sculpt-landmarks /private/sculpt-to-photo-points.json \
  --output /private/christ-reprojection-r01
```

The revision contains `landmark-qa.json` with measured registered-camera pixel errors, plus `manifest.json`. It still states `three_d_fidelity_verified: false`; calibration and sparse-point error alone cannot prove complete or historically faithful anatomy.

## Why not use face3d / a human reconstruction model directly?

[face3d](https://github.com/yfeng95/face3d) and [3D-FaceReconstruction](https://github.com/Sourodyuti/3D-FaceReconstruction) can be useful **starting points for facial priors** on ordinary photographed humans, subject to the models' licensing and training-data limits. Christ the Redeemer has stylized geometry, a downturned head, deeply carved features, stone surface, and integrated hair mass. An unconstrained 3D morphable-model fit may produce a plausible human face that is *less* faithful to the statue. Here they are not hard dependencies; their future outputs can be compared as candidate renders after manual registration and annotation.
