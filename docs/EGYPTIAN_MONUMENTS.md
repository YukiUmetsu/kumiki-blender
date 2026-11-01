# Egyptian Monument Geometry Kit

An original, Blender-independent set of **3D masonry and sculptural building
blocks** for Egyptian-themed scenes. It prioritizes reusable parametric solids
and physically distinct materials—not a textured billboard pretending to be
the Great Sphinx.

## APIs

| API | Use |
|---|---|
| `loft_elliptical_sections(sections, axis, radial_segments, subdivisions, cross_section_exponent)` | Close and skin varying circular-to-angular superelliptical cross sections along X, Y or Z for anatomy, stonework, trunks, carved relief and terrain props |
| `rock_height_field(x, y, z, seed, strata_spacing)` | Repeatable world-meter mineral bedding, broad erosion, chipped medium-scale detail and fine grain; shared across different stone types |
| `erode_loft(mesh, axis, radial_segments, strength, seed, strata_spacing)` | Displace an existing closed loft's actual vertices to make worn, stratified rock surfaces without breaking its end caps |
| `rough_bedrock_slab(length, width, height, roughness, seed)` | Topographically varied, watertight limestone base with eroded perimeter and nonplanar top |
| `pyramid_frustum(bottom_width, top_width, height)` | Solid tapering square mass; positive volume and outward faces |
| `pyramid_casing_stone(bottom_width, top_width, course_height, start, end, inset, joint, face_grid, erosion_strength, seed)` | One closed tapered masonry block facing +Y; tessellated worn surfaces have actual geometric relief, while polished casing can remain planar |
| `pyramid_assembly(prefix, base_width, height, courses, block_width, joint, casing_fraction)` | Recessed core, individually named alternating-course blocks, worn vacancies, light lower casing and capstone |
| `sphinx_assembly(prefix, length, width, height, restored_nose)` | Sculpted body/haunches/paws/toes/tail and hood/face/ears; recessed eye sockets, broken nose and mouth are carved directly into the face |
| `sphinx_integral_parts(parts)` | Identify the eight overlapping anatomical volumes for one visible carved lion body |
| `sphinx_head_parts(parts)` | Identify the six closed hood/face/ear/lappet solids for one head sculpture without overlay eye/brow pieces |
| `combine_sculpture_sources(parts)` | Flatten transformed closed body meshes into a world-space surface input for Blender's voxel union |
| `make_monument_study(kind, ...)` | Scene-independent composition for `pyramid`, `sphinx`, or `giza-study` |

Every piece is a separate `MeshInstance` with unique name, material role and
world transform. The mesh kernel rejects impossible dimensions, degenerate
profiles and overlapping brick joints; all shapes can be audited with
`audit_mesh()`. Relief is physically displaced geometry—not just bump shading—
so it changes silhouettes and casts real shadows under raking light. The
Blender adapter applies additional packed limestone albedo, roughness and
micro-normal maps, Cycles raking
sunlight, local metric UVs, edge softening, dedicated camera views, native
source preservation and portable GLB export. Studio ground/lights are excluded
from the exported model.

### Seamless statue sculpt in Blender

The pure-Python recipe deliberately keeps the torso, shoulder, neck, two
forelegs, haunches and tail as **eight separate editable closed solids**. In
Blender, a bounded-resolution voxel remesh produces a **single continuous
limestone skin** from these intersecting body volumes. This removes the
visible overlap seams while retaining real erosion across the statue. A
second, finer-resolution union sculpts the face, hood, ears and lappets into
one coherent head with a quieter, lighter stone material. The face's recessed
eyes, surviving nasal bridge, damaged nose and mouth are geometry-level relief
rather than floating decorative spheres or lines. Head weathering is reduced
relative to the deeply stratified lion body to preserve readable anatomy.
Toe reliefs and the rough bedrock remain separate so fine detail is preserved.

The `source.blend` preserves every original anatomical solid in a hidden
`Sphinx / hidden editable body sources` and `Sphinx / hidden editable head
sources` collections alongside the two visible unified surfaces. The GLB
exports those and the other detail meshes: **367 visible model objects** for
the default 379-part combined source study. The manifest records each union's
voxel size, source count and vertex count. This remains an original
illustrative sculpt, not photogrammetry.

The default procedural sculpt now uses a lower head-to-total-length ratio,
rounded rather than rectangular lion torso/foreleg sections, a broader
shoulder-to-neck transition, and narrower headdress lappets. It is still an
artistic approximation: these changes improve the *silhouette*, but do not
replace a surveyed statue mesh or guarantee authentic facial anatomy.

### Chipped pyramid stone (geometry, not just bump maps)

Exposed courses now have an 8-by-16 front grid per block with stochastic
quarry pits, inward-broken vertical and horizontal arrises, stratified mineral
relief and independently varied limestone color/PBR textures. Outer stone
faces are displaced while the back remains a full watertight wedge. The
default casing fraction has been reduced from 18% to **4%**, producing a
mostly eroded contemporary-style pyramid instead of a broad smooth white
band. `--casing-fraction` can still create a restored or selectively cased
study. AERA's [Sphinx geology overview](https://aeraweb.org/geology-of-the-sphinx/)
motivates the layered bedrock treatment; our individual chips and fracture
locations are procedural, **not mapped damage from an archaeological survey**.

## Examples

```bash
# No Blender or third-party Python packages needed for planning.
PYTHONPATH=src python3 -m kumiki_blender plan-monument --kind pyramid \
  --pyramid-base 36 --pyramid-height 23 --pyramid-courses 18 \
  --casing-fraction 0.2

PYTHONPATH=src python3 -m kumiki_blender plan-monument --kind sphinx \
  --sphinx-length 14

# Blender 4.2+ required to build new immutable artifact directories.
blender --background --factory-startup --python-exit-code 1 \
  --python scripts/blender_entry.py -- \
  build-monument --kind giza-study --output .review/egyptian/r01
```

### Place your own licensed Sphinx scan

If the stylized Sphinx anatomy is not sufficient for a cinematic shot, supply
a **tightly cropped, static GLB photogrammetry or surveyed mesh**. The builder
will replace the visible procedural Sphinx with the GLB while retaining the
procedural pyramid, Cycles daylight, sand-colored stage, six camera views,
scene-native `.blend` and portable exported `.glb`. Imported materials and
texture UVs remain intact. The 21 procedural Sphinx source pieces remain in a
hidden editable authoring collection in `source.blend` for comparison, but do
not appear in the render or exported GLB.

Prepare a JSON record alongside the *local* GLB (example for format only):

```json
{
  "schema_version": 1,
  "asset_id": "my-licensed-sphinx-scan",
  "creator": "Actual scan creator or institution",
  "source_url": "https://example.org/actual-stable-asset-page",
  "license": "CC-BY-4.0",
  "attribution": "Actual creator, title, source, and license as required",
  "redistribution_allowed": true,
  "commercial_use_allowed": true,
  "expected_sha256": "REPLACE_WITH_ACTUAL_64_CHARACTER_LOWERCASE_SHA256"
}
```

Replace the sample values with the asset's verified rights and real digest;
`expected_sha256` can be omitted while evaluating a first-party source. Only
declare redistribution and commercial use allowed when your rights permit both
**exporting the incorporated GLB** and its intended use. An incompatible
license (including noncommercial/no-derivatives clauses), missing permission,
animation/rigging, malformed GLB, or hash mismatch is rejected before scene
construction. This is a declaration and technical preflight, **not automatic
license verification**. No third-party scan is bundled or downloaded.

```bash
blender --background --factory-startup --python-exit-code 1 \
  --python scripts/blender_entry.py -- \
  build-monument --kind giza-study \
  --sphinx-source /path/to/reviewed-sphinx.glb \
  --sphinx-provenance /path/to/verified-rights.json \
  --sphinx-front-axis=-X \
  --output .review/egyptian/scanned-r01
```

The front-axis choices are `-X`, `+X`, `-Y`, and `+Y`, interpreted in **Blender
Z-up world coordinates after glTF import**; the front/face is placed toward
`-X` in the scene. The scanner may export a different axis convention, so
verify a preview first. Scan normalization preserves its proportions: it
rotates about Z, scales all dimensions equally to `--sphinx-length`, aligns
its center with the procedural statue's center, and puts its lowest point at
Z=0.02 m. A loosely cropped landscape scan or an implausible width/height
ratio is rejected: crop the statue in Blender first. The manifest includes
source URL, attribution, SHA-256, material and mesh counts, and the placement
transform/bounds. **This does not make an arbitrary low-quality scan accurate
or photorealistic**; geometry fidelity, source rights and historical quality
still require human review.

All dimensions are meters. Default combined plan uses a **36 m by 23 m
illustrative pyramid**, a **10.5 m-long, roughly 3.6 m-tall Sphinx-shaped study**, 18 stone courses,
and 379 named components. It does **not** imply that the real landmarks have
those measurements. `--restored-nose` is an explicitly optional sculptural
interpretation, not archaeological evidence. Material color and stone erosion
are procedural approximations, not scans of the site.

## Reference and accuracy boundary

- The Egyptian Ministry of Tourism and Antiquities describes the Great
  Pyramid's *original* height as **146.5 m**, its core as local limestone and
  its original higher-quality casing as stone transported from Turah:
  [The Great Pyramid](https://egymonuments.gov.eg/monuments/the-great-pyramid/).
- The ministry describes the [Great Sphinx](https://egymonuments.gov.eg/en/monuments/the-great-sphinx/)
  as carved directly from bedrock, depicting a royal head on a lion's body.
  Our Sphinx geometry therefore uses sculptural, stratified superelliptical
  lofts rather than assembling its main body out of independent "brick" blocks. The separate
  named anatomical parts are an **authoring convenience**, not a claim that
  the historical statue was constructed in pieces.
- The [Ancient Egypt Research Associates Sphinx survey](https://aeraweb.org/projects/sphinx/)
  documents varying bedrock strata, masonry repairs and enclosure conditions.
  Our horizontal weathering bands are an illustrative geometry-level response
  to that evidence, not a recreation of measured layer boundaries or damage.
- The [UNESCO Memphis and its Necropolis listing](https://whc.unesco.org/en/list/86/)
  provides wider archaeological context for the Giza and Dahshur monuments.

### To create convincing documentary-grade replicas later

1. Obtain rights-cleared, surveyed or photogrammetric meshes and site plans,
   including aspect ratios, damaged regions, enclosure/topography, orientation
   and era-specific historical reconstruction choices.
2. Use this kit for **procedural masonry layers, localized repairs, repeated
   decorations, stone material assignments and scene blocking**, then conform
   generated geometry to the authoritative measured surfaces.
3. Replace illustrative limestone PBR with licensed, high-resolution measured
   limestone and bedrock scans; use explicit material provenance.
4. Inspect closeups for anatomy, facial proportions, course alignments,
   erosion strata, lighting, and negative-space fidelity. Passing a mesh audit
   does **not** prove that a 3D scene looks like the real Giza Plateau.

The CI smoke profile verifies render/export **functionality** and makes small
JPEG versions of the actual full-body, side-profile, facial and masonry renders available for visual
review in the separate `visual-qa` job log. It cannot certify anatomy or
cinematic realism. Production-grade acceptance must still examine high-sample
Cycles views and a Three.js import of `model.glb`.
