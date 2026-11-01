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
| `sphinx_assembly(prefix, length, width, height, restored_nose)` | Sculpted body/haunches/paws/toes/tail, hood, face, eye/brow/mouth relief and headdress stripes |
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

All dimensions are meters. Default combined plan uses a **36 m by 23 m
illustrative pyramid**, a **10.5 m Sphinx-shaped study**, 18 stone courses,
and 393 named components. It does **not** imply that the real landmarks have
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
JPEG versions of the actual Sphinx and masonry renders available for visual
review in the separate `visual-qa` job log. It cannot certify anatomy or
cinematic realism. Production-grade acceptance must still examine high-sample
Cycles views and a Three.js import of `model.glb`.
