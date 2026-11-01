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
| `petrie_course_tops_m(last_course=201)` | Historical corrected NE corner course-top elevations for Khufu: 203 observed levels, 201-course present-day cutoff |
| `khufu_petrie_assembly(prefix, last_course=201)` | Full-sized Khufu study with measured course heights and multiple watertight masonry blocks batched per face/course |
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

### Historically scaled Khufu course-height preset

The `--pyramid-preset khufu-petrie` option uses [Petrie's 1883 measurements
as corrected/transcribed by Ronald Birdsall](https://www.ronaldbirdsall.com/gizeh/errata/levels.html).
The source lists 203 historical northeast-corner course-top levels in inches.
The model converts them to meters, retaining their **nonuniform course heights**.
It defaults to **201 modeled courses**: Birdsall notes that the recorded
upper remnants numbered 202 and 203 no longer exist.

The full-sized study has an approximately **230.4 m base** and a historical
**146.5 m original-design apex**, but deliberately ends at the measured top
of course 201 (**137.36 m**), not a fictional capstone. Each block is a
closed masonry wedge with real mortar joints. Four compact, editable mesh
batches per course retain thousands of visible block seams in only **805
Blender mesh objects** including the recessed core. Lower courses get
vertex-level chips, while higher courses use lower-cost geometry and
weathered limestone textures.

```bash
# Plan without Blender
PYTHONPATH=src python3 -m kumiki_blender plan-monument --kind pyramid --pyramid-preset khufu-petrie

# Build an immutable revision with Blender 4.2+
blender --background --factory-startup --python-exit-code 1 \
  --python scripts/blender_entry.py -- \
  build-monument --kind pyramid --pyramid-preset khufu-petrie \
  --output .review/egyptian/khufu-petrie-r01
```

**Measured:** northeast course elevations only. **Illustrative:** each
block's footprint, quarry cuts, fractures, joint widths, condition, core
inset, and nominal continuous pyramid slope. The surviving present-day
stone configuration is not fully surveyed by this data. To avoid fabricating
the site's real relative positions, this preset requires `--kind pyramid`
and cannot be added to the miniature `giza-study` composition.

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
transform/bounds. Its `sphinx_scan.source_geometry_review` compares the
actual post-import width/length and height/length ratios against the
[Egyptian State Information Service's 73.5 m × 19.3 m × 20 m monument
envelope](https://sis.gov.eg/en/egypt/tourism/landmarks/sphinx/). Deviations
over 25% are flagged for review. Passing the ratio screen **does not certify**
the nemes, face, paws, hindquarters or repaired stone. A miniaturized replica
could have perfectly plausible overall dimensions but the wrong anatomy.

### Sphinx source-geometry shortlist

Only a validated capture of the **actual Giza monument**, or a reconstruction
constrained to survey drawings, should replace the illustrative procedural
geometry in a documentary-grade shot. Candidates researched in November
2026; **no third-party model is bundled**:

| Candidate | Role | Source-fidelity and rights caution |
|---|---|---|
| [ARCE Sphinx Project / Open Context](https://opencontext.org/projects/141e814a-ba2d-4560-879f-80f1afb019e9) | Primary 1:50 plans/elevations and repair cross sections; CC BY 4.0 metadata | Archaeological reference drawings, **not** a ready-to-render 3D mesh |
| [Chenzoss Great Sphinx / Sketchfab](https://sketchfab.com/3d-models/the-great-sphinx-of-giza-egypt-f169dbe7974648babe327179091e0ee3) | CC BY, roughly 40k-triangle model for visual assessment | Listing does not prove original monument capture or true facial geometry |
| [Watt Institution / Sketchfab](https://sketchfab.com/3d-models/the-great-sphinx-of-giza-c126509af31142a9a0d116e264e7b629) | **Reject** as historical ground truth | Photogrammetry depicts a *1903 bronze miniature*, not the monument |
| [Scan the World / Zenodo](https://zenodo.org/records/21647117) | Pending rights review | GLB listing exists; commercial redistribution and actual survey origin unverified |
| [TurboSquid Great Sphinx scan](https://www.turbosquid.com/3d-models/great-sphinx-giza-scan-3d-model-1302443) | Optional paid production reference | Payment does not grant the right to redistribute the mesh in an open-source GLB |

Verify the exact asset's current rights, original capture evidence and visual
fit against survey plans before use. A high polygon count or the word “scan”
in a product listing is not evidence of monument authenticity.

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
