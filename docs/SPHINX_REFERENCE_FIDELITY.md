# Sphinx: Reference-Driven Geometry Fidelity

The current procedural head is **not yet a convincing reconstruction** of
the surviving Great Sphinx face. Its mask-like cheeks, simplified ears, eye
sockets and lips remain poor matches for close photographs. A denser mesh and
more weathering noise do not cure incorrect underlying anatomy.

## Primary historical geometric evidence

- [ARCE / AERA Sphinx Mapping Digital Database](https://arce.org/project/sphinx-mapping-digital-database/): the 1979–83 mapping triangulated 63 points on the surviving head and headdress from ground survey points.
- [ARCE survey data at Open Context](https://opencontext.org/projects/141e814a-ba2d-4560-879f-80f1afb019e9): the recorded front and side photogrammetric elevations, head sections and profiles were originally drawn at 1:50. They are the preferred shape constraints once traced into machine-readable coordinates and their camera/grid system is verified.
- [Mark Lehner, *Documentation of the Sphinx* (1992)](https://gizapyramids.org/static/pdf%20library/lehner_sphinx_1992.pdf): figure 2 contains the front elevation and face profile with **25 cm contour intervals**; figures 3–6 provide side elevations and profile sections.
- [AERA geology of the Sphinx](https://aeraweb.org/geology-of-the-sphinx/): bedrock layers on the lion body weather differently from the harder head stone. Depth of erosion should vary by geological layer, not be globally distributed as procedural noise.

These are reference links, not meshes or texture assets shipped by this kit.
The archive's 63 surveyed point coordinates and the contours have **not yet
been digitized into this source code**. No claim of archaeological accuracy
is made for the existing generator.

## Repeatable face-to-reference comparison

The original manually annotated JSON records in `examples/` illustrate
point correspondences in a user-provided photographic frontal view and the
**latest procedural Blender frontal QA render**. The photograph itself is
**not redistributed**.
Coordinates are rough visual placements, **not survey measurements** and
must be refined for serious modeling. In particular, photographic perspective,
camera pitch, differing headcloth orientation and shadows limit comparisons.

```bash
PYTHONPATH=src python3 -m kumiki_blender compare-sphinx-landmarks \
  --reference examples/sphinx-frontal-photo-landmarks.json \
  --candidate examples/sphinx-frontal-procedural-landmarks.json
```

The command reads *only JSON annotations*, has no image dependency, and emits
signed horizontal and vertical mismatches for the eyes, ear centers, nose,
mouth, cheek outline and chin. Alignment uses a **single uniform 2D scale and
translation** fitted to crown and two nemes-wing anchors; allowing different
horizontal and vertical scales would hide a vertically compressed face.
Errors are divided by the spread of those three reference anchors so they
remain comparable across different pixel resolutions. They are **dimensionless
image-space errors**, not centimeters and not measured 3D depth.

To compare a new candidate (including a locally reviewed external scan),
mark the same named points on its straight-on view and save its JSON with the
same anchor names; rerun the command. For profile reviews use a separate set
with `"view": "profile-left"` or `"profile-right"` consistently on both
images. Never compare opposite profiles as if mirrored anatomy were identical.
Pick matching, identifiable three-point anchors and include the chin and nasal
profile as independent evaluation landmarks.

The useful new **production path** is to trace the ARCE front/side elevation
and 3D head control points, fit a new original continuous face surface to
those constraints, then sculpt missing small-scale damage under photo review.
Until the historical coordinates are recovered, use this tool to **reject
bad face shapes**; do not declare a new sculpt realistic because it merely
passes manifold checks or has a large vertex count.

## Experimental front-image-constrained geometry

An **opt-in** sculpt pass now takes exactly the two annotated frontal JSON
records above and aligns **vertical** landmarks on the *original* continuous
face mesh. The camera alignment is isotropic (same scaling in X and Y). The
procedural headcloth wing-tip separation establishes the image-to-model
scale; it does **not** assert the original Sphinx is 3.14 meters wide.

The first 2D radial-basis experiment **failed visual QA**: it folded the
face's parametric grid, producing a long nasal spike and distorted mouth
despite manifold/GLB tests passing. It was rejected. The safer revision uses
order-preserving cubic interpolation of chin, mouth, nose, eyes and crown
heights. It does **not** apply suggested horizontal eye/cheek movements, since
uncalibrated camera differences can require impossible crossing of anatomical
feature positions. Both front and back of the face follow the same vertical
map, preserving coherent thickness and a watertight head source.

```bash
PYTHONPATH=src python3 -m kumiki_blender plan-monument --kind sphinx \
  --sphinx-face-reference examples/sphinx-frontal-photo-landmarks.json \
  --sphinx-face-candidate examples/sphinx-frontal-procedural-landmarks.json

blender --background --factory-startup --python-exit-code 1 \
  --python scripts/blender_entry.py -- \
  build-monument --kind sphinx \
  --sphinx-face-reference examples/sphinx-frontal-photo-landmarks.json \
  --sphinx-face-candidate examples/sphinx-frontal-procedural-landmarks.json \
  --output .review/sphinx-photo-fit
```

The same inputs reproduce the same face and their SHA-256 appears in the
scene manifest. The CLI rejects missing records, viewpoint mismatch,
inverted landmark height ordering or landmark shifts larger than 29 cm at
the default study scale. Without these flags the original illustrative
generator is unchanged. Blender CI builds and renders both versions so
**real visual QA decides whether to use the result**.

**Still missing:** true side-profile fitting, 3D survey point coordinates,
an anatomically sculpted eye/ear/lip surface, calibrated camera perspective,
and accurate rock damage. A frontal view cannot determine nose projection,
temple depth or chin undercut. This **experimental fit is not yet a
survey-accurate or photorealistic Sphinx**; do not substitute it for scan
reconstruction.
