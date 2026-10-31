# Architectural Detail Kit: Authoring Guide

This is a **reusable Python geometry API**, not a collection of rigid premade
buildings. The geometry functions run in ordinary Python, while Blender is
needed only to render/export an asset revision. The examples are originally
generated and are not claimed to reproduce a specific historic landmark.

## Coordinate and object contracts

- All dimensions are **meters**, +Z up, +X horizontally along the facade, and
  the visible front toward **-Y**.
- Functions return `Mesh` or immutable, named `MeshInstance` values. Instances
  carry origin, three right-handed axes and a semantic material role.
- `audit_mesh(mesh)` checks each solid for closed orientation, manifold edges,
  and positive signed volume. Physical mortar/air gaps between solids are okay.
- `world_vertices(instance)` computes real world-space bounds for planning.
- The Blender adapter preserves per-object names and `kumiki_part` categories
  in GLB extras for scripted camera targets and interactive part selection.

## Build architecture without Blender

```python
from kumiki_blender.architecture_details import (
    Opening, coursed_wall, window_assembly, rosette_relief,
    cornice_extrusion, gable_roof, rainwater_system,
)
from kumiki_blender.architecture_geometry import audit_mesh

# A 6 m front wall with a true 2 m by 2.5 m window opening.
wall = coursed_wall(
    width=6, height=5.5, thickness=0.48,
    openings=(Opening(2, 4, 1.4, 3.9),),
    block_width=0.95, course_height=0.48, joint=0.01,
    prefix="masonry-front",
)
window = window_assembly(prefix="window-center", width=2, height=2.5,
                         columns=2, rows=2)
crest = rosette_relief(prefix="ornament-medallion", diameter=0.6, petals=12)
roof = gable_roof(width=6, depth=5, eave_z=5.5, rise=1.7,
                  tile_rows=6, tile_columns=12)
drainage = rainwater_system(width=6, depth=5, eave_z=5.5)

assert audit_mesh(wall[0].mesh).boundary_edges == 0
```

Use `place_mesh()` and `dataclasses.replace()` to position subassemblies in
world space. Windows and doors are centered around their local opening origin;
rosettes lie in local XY and extend outward in local +Z. To place a rosette on
a vertical front facade, orient its axes to world X, world Z, and world -Y.
The ready-to-run composition in `facade.py` shows these placements and the
front/side/back wall rotations.

## Built-in reference study

```bash
PYTHONPATH=src python3 -m kumiki_blender plan-facade --bays 3
blender --background --factory-startup --python-exit-code 1 \
  --python scripts/blender_entry.py -- \
  build-facade --bays 3 --output .review/stone-facade/r01
```

The first command reports semantic part counts/bounds without any Blender
dependency. The second writes an immutable `source.blend`, self-contained
`model.glb`, hero/front/side/detail images and integrity manifest. Choose a new
output directory for every revision. Symmetric façade compositions accept 3,
5 or 7 bays; individual functions support other dimensions.

## Surface realism and preview lighting

- `get_surface("weathered-stone", color="#A9A296", roughness=0.83)` provides
  deterministic embedded albedo, roughness and tangent-normal maps with
  vertical mineral runoff and irregular discoloration.
- Stone block color changes are generated from a **stable SHA-256 of each
  semantic part name**, avoiding Python's nondeterministic hash randomization
  and obvious diagonal stripes.
- `plan_architectural_reflections()` specifies two rectangular sky softboxes;
  `add_architectural_reflections()` adds them to a Blender-only studio collection
  to emphasize specular response from real transmissive glass.
- GLB keeps Principled glass transmission/IOR material values, **but not** the
  studio softboxes. A Three.js scene should provide an appropriate environment
  map, direct lighting and tested support for transmission extensions.
- The surface maps are original, generic procedural patterns; they do not
  replace measured limestone, slate or copper textures and cannot certify
  a reference-accurate landmark.

## Reliability and scaling

The facade adapter reuses Blender mesh datablocks by source geometry and
material, then creates separately named objects for semantic access. Small
bevel and weighted-normal modifiers are preserved in `source.blend` and baked
for portable GLB. The PR's Blender 4.5 smoke job renders the asset and checks
GLB texture embedding, full glTF validation, file integrity and previews.
Smoke previews are technical checks, **not** photorealism approval. Final
landmark work needs dimensioned references, specific ornament profiles,
physically matched materials and higher-sample multi-view image review.
