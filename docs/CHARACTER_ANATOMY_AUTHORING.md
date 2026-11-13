# Source-anchored anatomical parts: eye, eyelid, beak and contour authoring

This Blender-first feature produces **real, separate, editable volumetric meshes**. It does not infer anatomical measurements from illustrations or claim to have reconstructed hidden surfaces. Supply approved 3D landmark positions in world metres, material colors, explicit orientation, and reference-evidence provenance.

The three supported authoring primitives are:

- **ocular_globe**: a watertight ellipsoid sclera, 3D iris annulus and 3D pupil, each with a separate mesh, material and UV set.
- **swept_tube**: a smooth, capped, elliptically cross-sectioned 3D curve through source-aligned points for eyelids, eyebrows, lip rims and feather shafts. Each control point can have distinct width/depth.
- **contour_loft**: volumetric, watertight, curvature-smoothed skin through corresponding artist-authored 3D section contours for beaks, muzzles, cheeks, ears and digit forms.

All features retain separate objects. Nothing is mirrored, welded, voxel-unioned, or generated as a flat decal. The input evidence records registered-verified, artist-placed or inferred provenance, but the worker cannot independently certify that the authoring control points correspond to original artwork.

See [the illustrative example](../examples/character-anatomy.example.json). Its coordinates and colors **are not authentic Pip/Quill measurements**. Replace them with your own reviewed values from the original reference sheets.

## Local workflow

Copy and edit the example JSON, then run:

~~~bash
kumiki-character-build plan-anatomy \
  --blend /path/to/approved-character.blend \
  --spec /path/to/approved-parts.json \
  --output .review/character/anatomy-plan-r01

kumiki-character-build build-anatomy \
  --plan .review/character/anatomy-plan-r01/anatomy-plan.json \
  --output .review/character/anatomy-r01 \
  --blender /Applications/Blender.app/Contents/MacOS/Blender
~~~

The planner hashes the original Blender source and anatomy specification, validates geometry bounds and limits the work to 32 parts and approximately 2 million generated vertices. Changed source/spec files invalidate the plan. Outputs are immutable; old source files are not overwritten.

The Blender worker writes **anatomy.blend** and individual **parts/<part-id>.obj** (legacy interchange) and **parts/<part-id>.source.obj** (Z-up source-frame) files, plus **report.json** with per-part manifold/volume data, geometry/material metadata, provenance, source hashes and original-geometry immutability checks. Example object names: ANATOMY.eye.L.sclera, ANATOMY.eye.L.iris, ANATOMY.eye.L.pupil, ANATOMY.upper_lid.L and ANATOMY.beak.

The existing original mesh, UVs, materials, shape keys, transforms and rig pose are checked for change. Parts can be left separate or parented explicitly to an existing Blender bone while preserving their world-space position. Rigid parenting **is not** a substitute for deforming flesh or feathers.

## QA and reference alignment

Exported part OBJ geometry can be passed to the existing CharacterSpec v2 **compare-region** command, provided an equivalent part ID and independently registered full-frame mask are available:

~~~bash
kumiki-character-build compare-region \
  --character .review/character/plan-r01 \
  --region eye.L \
  --mesh .review/character/anatomy-r01/parts/eye.L.source.obj \
  --output .review/character/eye-qa-r01
~~~

Review camera-projected silhouettes and materials against the originals, then test animated poses using validate-rig. Topology integrity is necessary but not enough: these procedural shapes still need artist-directed contour details, eyelid/jaw animation, source-faithful textures and deformation-safe joins.

## Tests

The pure Python tests check authored mesh integrity, source/plan hashes, evidence provenance, boundary conditions and local geometry. The native Blender 4.5 LTS smoke creates a procedural head, five anatomy components, individual UVs/materials and three legacy and three source-frame OBJ part exports. It checks unchanged original geometry/rig and that an eye follows its named head bone. **Synthetic test success is not a likeness score.**
