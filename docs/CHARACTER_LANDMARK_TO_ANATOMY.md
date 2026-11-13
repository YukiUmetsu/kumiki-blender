# Direct source-landmark-to-anatomy authoring

This connects verified-view 2D landmark triangulation to Blender-native facial anatomy authoring without manually copying XYZ coordinates or modifying the original character sources.

## 1. Triangulate source-aligned drawings

Annotate the same feature in at least two independently registered orthographic views, then run:

~~~bash
kumiki-character-build triangulate-landmarks \
  --spec /path/to/registered-landmarks.json \
  --output .review/character/triangulated-r01
~~~

The immutable landmarks.json contains world-space metre positions, source image SHA-256 hashes, camera-baseline diagnostics and per-view reprojection errors.

## 2. Insert explicit placeholders in an anatomy template

Use the [anatomy recipe format](CHARACTER_ANATOMY_AUTHORING.md) and replace only an eligible three-coordinate world-position field with an object containing one landmark ID.

For instance, an ocular_globe part can specify:

~~~json
{"id": "eye.L", "kind": "ocular_globe",
 "center_m": {"landmark": "eye.L.center"}}
~~~

This excerpt is not a complete anatomy part; radii, eyes' axes, iris/pupil proportions, materials, provenance and attachment fields are still mandatory.

The resolver also supports placeholders in individual swept_tube control_points_m and contour_loft sections_m entries, including intermediate eyelid and beak control vertices.

A missing landmark, mismatched character ID or unsupported registered view is a hard error. No symmetry or hidden surface is ever inferred. The original evidence status is not changed.

## 3. Resolve, build, compare

~~~bash
kumiki-character-build resolve-anatomy-landmarks \
  --spec /path/to/anatomy-template.json \
  --landmarks .review/character/triangulated-r01/landmarks.json \
  --output .review/character/resolved-r01

kumiki-character-build plan-anatomy \
  --blend /path/to/approved-character.blend \
  --spec .review/character/resolved-r01/resolved-anatomy.json \
  --output .review/character/anatomy-plan-r01

kumiki-character-build build-anatomy \
  --plan .review/character/anatomy-plan-r01/anatomy-plan.json \
  --output .review/character/anatomy-r01 \
  --blender /Applications/Blender.app/Contents/MacOS/Blender
~~~

The resolver writes a complete geometry spec and a landmark-provenance JSON receipt, including the original source images' hashes and the source 2D reprojections. The Blender builder then produces an independent editable anatomy.blend plus part-specific OBJ files. Use compare-region to check each part against independently registered source masks and review posed collisions.

## Important limitations

The resolver checks underlying source image integrity at the time it runs. The anatomy planner then pins the resolved anatomy JSON, but it does not yet re-check the entire upstream source-image provenance when Blender runs. Preserve the immutable source revision and re-triangulate if art changes.

This feature resolves authored positional landmarks only. It does not perform image feature detection, independently verify camera registration, sculpt unknown anatomy, automate seam retopology, or guarantee Pip/Quill likeness. The artist must still review shape, surface detail, materials and animation.
