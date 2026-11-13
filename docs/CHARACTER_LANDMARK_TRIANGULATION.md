# Reconstruct anatomical 3D landmarks from registered character drawings

Use the existing **1600 × 1200 orthographic reference frame** and two or more independently registered views to recover 3D source-guided points. This is intended for accurate eye centers, corners of eyelids, beak roots/tips, muzzle features, wing pivots and other control vertices used by Blender's source-anchored anatomy authoring.

The algorithm is a deterministic weighted least-squares solution using the same camera azimuth/elevation, image center, figure scaling, Z-up and -Y-forward convention as the existing character pipeline. It does not need cloud inference or a new ML model.

## Input

Create a JSON file with schema_version 1, character_id, measured height_m, registered_views and landmarks.

Each registered view is an **existing full-frame 1600 × 1200 PNG/JPG/WebP file** beside the spec, with an explicit registration claim (original-aligned or artist-registered). The source image bytes are hashed, and symlinks/path traversal are refused. An unknown camera or an image of a different size is rejected, not guessed.

Each landmark has:
- A stable semantic ID such as eye.L.inner or beak.tip.
- At least two observations, with the view name and the pixel location in that *unchanged full-frame* image.
- A confidence weight in [0.1, 1], and a maximum accepted reprojection error in pixels.

For example, the synthetic point [0.033, -0.046, 0.22] m projects to these approximate locations when height_m = 0.35 m:

~~~json
{
  "schema_version": 1,
  "character_id": "example",
  "height_m": 0.35,
  "registered_views": {
    "front": {"file": "refs/front.png", "registration": "original-aligned"},
    "right": {"file": "refs/right.png", "registration": "original-aligned"}
  },
  "landmarks": [
    {
      "id": "eye.L.center",
      "max_reprojection_error_px": 3.0,
      "observations": [
        {"view": "front", "xy_px": [894.286, 471.429], "confidence": 1.0},
        {"view": "right", "xy_px": [931.429, 471.429], "confidence": 1.0}
      ]
    }
  ]
}
~~~

Do not use example numbers as real anatomy. Record actual pixel positions on the source sheets and validate that the images were registered correctly.

## Command

~~~bash
kumiki-character-build triangulate-landmarks \
  --spec /path/to/character/registered-landmarks.json \
  --output .review/characters/character/landmarks-r01
~~~

Output **landmarks.json** contains world-space metre coordinates in a directly usable points_m dictionary, source image hashes, the exact camera observations, per-view projected pixels, residual errors and a camera-baseline observability metric. The original images/specification stay untouched. The output folder is immutable.

Copy or programmatically map the resulting points_m into the 3D parts required by plan-anatomy and build-anatomy. For an eye, triangulate its center plus several eyelid corners and intermediate contour points. For a beak, triangulate corresponding stations in multiple source views; inferred hidden surfaces remain explicitly classified as inferred. **This implementation does not yet substitute landmark IDs directly into anatomy JSON fields.**

## Failure cases and limits

An opposing front/back pair can be insufficient to recover depth: the solver detects rank deficiency and rejects it. Two nearly identical camera angles have weak depth observability and also fail. If three views disagree on the same landmark, the solver reports their reprojection error instead of discarding inconvenient evidence or moving the cameras to inflate agreement.

The source view registration statuses are **user/artist claims**, not automatically verified calibration. The result is a triangulated candidate pending actual source and artistic review, especially for hidden surfaces and stylized perspective violations.

The synthetic tests cross-check the solver against the repo's existing Camera.project math, verify front/right and elevated view recovery, reject ambiguous camera baselines and contradictory observations, and reject modified source images.
