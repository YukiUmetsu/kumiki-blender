# AssetSpec v2 — one build path for many asset families

The optional **component composer** is the first step toward reusable,
agent-friendly 3D authoring. It compiles a bounded JSON description into
validated `MeshInstance`s, then passes those to a Blender adapter that writes
the same immutable `.blend` / `.glb` / PNG / manifest revisions as existing
Kumiki generators.

It is a **component assembly framework**, not an AI sculptor or a substitute
for photographic reference matching. Models still require visual inspection;
a valid topology or cinematic render preset does not certify photorealism.

## Install

Existing core planning stays dependency-free. For the new schema compiler:

```bash
python3 -m pip install -e '.[composer]'
```

The composer runs from ordinary Python without loading `bpy`. Blender's
embedded Python does **not** need Pydantic: the host compiler writes a
validated, checksum-verified, data-only plan that the Blender worker reads.

## Plan and build

```bash
python3 -m kumiki_blender asset-schema
python3 -m kumiki_blender plan-asset --spec examples/desk-lamp-assembly-v2.json

KUMIKI_RENDER_PROFILE=smoke python3 -m kumiki_blender build-asset \
  --spec examples/desk-lamp-assembly-v2.json --output .review/desk-lamp/r01
```

The last command uses Blender from `PATH` (or specify `--blender /path/to/blender`).
For an explicit two-process workflow, or if Blender is installed on a different
machine, run:

```bash
python3 -m kumiki_blender compile-asset \
  --spec examples/desk-lamp-assembly-v2.json --output /tmp/lamp.plan.json
blender --background --factory-startup --python-exit-code 1 \
  --python scripts/blender_entry.py -- \
  build-asset --plan /tmp/lamp.plan.json --output .review/desk-lamp/r01
```

Choose another output directory for each revision. The output may not already
exist. The quality preset controls the requested studio resolution and sample
count; `KUMIKI_RENDER_PROFILE=smoke` caps render effort without changing mesh
fidelity or GLB export.

## Initial component registry

| `type` | Mandatory parameters | Result |
|---|---|---|
| `solid.box` | `id`, `material`, `size_m` | Closed centered box, optional `origin_m` |
| `solid.lathe` | `id`, `material`, `profile_rz_m` | Capped revolved solid, optional `origin_m` and `radial_segments` |
| `solid.hollow-lathe` | `id`, `material`, `profile_rz_m`, `wall_thickness_m` | Hollow revolved shell with real inner wall and annular open ends |
| `mechanical.rod` | `id`, `material`, `start_m`, `end_m`, `radius_m` | Properly oriented cylinder between two endpoints |
| `detail.radial-studs` | `id`, `material`, `center_m`, `ring_radius_m`, `stud_radius_m`, `stud_height_m`, `count` | Named, shared-mesh repeated fasteners |

Materials are named roles backed by the existing PBR catalog (`preset`, with
optional `color` and `roughness`). Component IDs are unique slugs. Unknown
keys, missing material roles, unsafe numeric values, duplicate JSON keys,
oversize files, and duplicate generated names are rejected. Inputs contain no
arbitrary Python or file paths.

The plan result includes metric bounds, instance counts, semantic part names
and a deterministic canonical recipe digest. Each generated mesh passes the
existing watertightness/volume audit before any Blender work begins.
The compiled interchange independently checks its digest, meshes, bounds,
materials and named transforms inside Blender before it creates an output.

### Agent workflow

1. Query `asset-schema` and the documented component registry.
2. Write an AssetSpec v2 JSON using only known component types and materials.
3. Run `plan-asset`; repair validation errors before launching Blender.
4. Run `build-asset` into a fresh revision. Review hero/front/side/detail PNGs.
5. Add human-reviewed detail/reference passes when the asset must resemble a
   specific object. Do not use a successful build as visual acceptance.

## Extension boundaries

This initial vertical slice exposes four reusable shapes and one generic
CLI/Blender adapter. It deliberately does not claim that a desk lamp assembled
from four primitives is photo-faithful. Next work should add controlled
Geometry Nodes groups, reference alignment, materials with provenance, better
connectors and domain-specific component packs. Optional CAD/scan libraries
such as build123d and Open3D belong in separate workers; they are not installed
in Blender's bundled Python interpreter.
