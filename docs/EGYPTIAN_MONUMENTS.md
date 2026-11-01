# Egyptian Monument Geometry Kit

An original, Blender-independent set of **3D masonry and sculptural building
blocks** for Egyptian-themed scenes. It prioritizes reusable parametric solids
and physically distinct materials—not a textured billboard pretending to be
the Great Sphinx.

## APIs

| API | Use |
|---|---|
| `loft_elliptical_sections(sections, axis, radial_segments, subdivisions)` | Close and skin varying elliptical cross sections along X, Y or Z for anatomy, stonework, trunks, carved relief and terrain props |
| `pyramid_frustum(bottom_width, top_width, height)` | Solid tapering square mass; positive volume and outward faces |
| `pyramid_casing_stone(bottom_width, top_width, course_height, start, end, inset, joint)` | One closed tapered masonry block facing +Y; rotate it around Z to reuse on all four sides |
| `pyramid_assembly(prefix, base_width, height, courses, block_width, joint, casing_fraction)` | Recessed core, individually named alternating-course blocks, worn vacancies, light lower casing and capstone |
| `sphinx_assembly(prefix, length, width, height, restored_nose)` | Sculpted body/haunches/paws/toes/tail, hood, face, eye/brow/mouth relief and headdress stripes |
| `make_monument_study(kind, ...)` | Scene-independent composition for `pyramid`, `sphinx`, or `giza-study` |

Every piece is a separate `MeshInstance` with unique name, material role and
world transform. The mesh kernel rejects impossible dimensions, degenerate
profiles and overlapping brick joints; all shapes can be audited with
`audit_mesh()`. The Blender adapter applies warm limestone PBR, Cycles raking
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
  Our Sphinx geometry therefore uses smooth sculptural lofts rather than
  assembling its main body out of independent "brick" blocks. The separate
  named anatomical parts are an **authoring convenience**, not a claim that
  the historical statue was constructed in pieces.
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

The stock CI smoke profile verifies render/export **functionality**, not
cinematic realism. Production-grade acceptance must still examine high-sample
Cycles views and a Three.js import of `model.glb`.
