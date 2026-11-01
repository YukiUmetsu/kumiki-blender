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
python3 -m pip install -e '.[composer,qa]'
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

# A second product family uses reusable rounded corners, mitered trim and fasteners.
KUMIKI_RENDER_PROFILE=smoke python3 -m kumiki_blender build-asset \
  --spec examples/precision-control-enclosure-v2.json \
  --output .review/control-enclosure/r01
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
| `solid.rounded-box` | `id`, `material`, `size_m`, `corner_radius_m` | True welded fillets on every edge and corner, with optional `corner_segments` |
| `solid.lathe` | `id`, `material`, `profile_rz_m` | Capped revolved solid, optional `origin_m` and `radial_segments` |
| `solid.hollow-lathe` | `id`, `material`, `profile_rz_m`, `wall_thickness_m` | Hollow revolved shell with real inner wall and annular open ends |
| `mechanical.rod` | `id`, `material`, `start_m`, `end_m`, `radius_m` | Properly oriented cylinder between two endpoints |
| `detail.radial-studs` | `id`, `material`, `center_m`, `ring_radius_m`, `stud_radius_m`, `stud_height_m`, `count` | Named, shared-mesh repeated fasteners |
| `detail.linear-fasteners` | `id`, `material`, `start_m`, `end_m`, `count`, `head_radius_m`, `head_height_m` | Spaced, instanced button or hexagonal heads along an edge; optional `normal_axis` and `head_style` |
| `detail.panel-frame` | `id`, `material`, `size_m`, `trim_width_m`, `depth_m` | Four independent 45° mitered rails with a real open center; optional `plane` and `origin_m` |

### Hard-surface craftsmanship presets

These three added types reduce the amount of mesh-generation code needed for
product enclosures, furniture, equipment, books, display frames, instruments,
and other hard-surface assets:

- **Rounded box:** `corner_radius_m` is a physical, exported radius, less than
  half the shortest dimension. `corner_segments` ranges from 2–12. It is not
  a render-only bevel modifier; the Blender and glTF geometry agree.
- **Panel frame:** `size_m` is the outer `[width, height]`; `trim_width_m` makes
  a centered clear opening. `plane` chooses `xy` (+Z outward), `xz` (−Y
  outward, e.g. a front panel), or `yz` (+X outward). `origin_m` locates the
  center of the panel at the back of the raised trim.
- **Linear fasteners:** Start and end positions include the first and last
  fastener. Set `normal_axis` to `+x`, `-x`, `+y`, `-y`, `+z` or `-z` to make
  the physical head protrude from the chosen surface. `head_style` selects a
  faceted `hex` head or curved `button` head; both have actual solid geometry.
  Head spacing is validated before Blender runs.

The [precision-control-enclosure example](../examples/precision-control-enclosure-v2.json)
combines fourteen high-level declarations into 62 real, named parts. All reusable
parts are validated and compiled by the same generic `build-asset` command;
no new command or Blender script is needed per product. This is a **detail
library**, not a substitute for survey-based shape reconstruction or physical
reference matching.

For any `solid.lathe`, `solid.hollow-lathe` or `detail.radial-studs`, the optional
`normal_axis` also selects the physical outward direction (`+x`, `-x`, `+y`,
`-y`, `+z`, `-z`; default `+z`). A front-facing dial therefore needs no custom
rotation matrix: its turned knob, hollow bezel and circular calibration marks
can all share the same `-y` normal. The control-enclosure example uses this
to produce three layered dials, each with a sculpted knob, metal bezel and
twelve raised scale markers.

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

## Multi-view visual QA (Pillow + optional OpenImageIO)

Once Blender finishes a revision, create a labeled contact sheet and a JSON
exposure report for artistic review. This runs in ordinary host Python, not
Blender's embedded interpreter:

```bash
python3 -m kumiki_blender visual-qa \
  --revision .review/desk-lamp/r01 \
  --output .review/desk-lamp/r01-inspection
```

For a new revision with **the same asset ID and ordered camera views**,
compare images with a known-good baseline. The command emits a per-view
normalized RMS difference, changed-pixel fraction, and highlighted difference
PNGs for regressions:

```bash
python3 -m kumiki_blender visual-qa \
  --revision .review/desk-lamp/r02 \
  --baseline .review/desk-lamp/r01 \
  --output .review/desk-lamp/r02-inspection \
  --max-rms 0.02 --max-changed 0.05 --pixel-threshold 16
```

Exit statuses: `0` for a generated inspection report or a passing comparison,
`3` for a regression needing review, `2` for invalid inputs. The output is a
new directory and must stay outside the immutable source revisions.

For independent OpenImageIO perceptual comparisons, install its `idiff`
executable and add `--idiff` with `--baseline`. Differences may then require
human review even when the Pillow thresholds pass. A pixel test can detect
render changes, but cannot judge shape craftsmanship, photorealism, historical
fidelity, or whether the composition is aesthetically compelling.

## Extension boundaries

The current registry exposes eight reusable component types and one generic
CLI/Blender adapter. It deliberately does not claim that an assembled desk
lamp or control enclosure is a photo-faithful recreation. Next work should add
controlled Geometry Nodes groups, surveyed reference alignment, materials
with provenance, microdetail maps, calibrated connectors and domain-specific
component packs. Optional CAD/scan libraries such as build123d and Open3D
belong in separate workers; they are not installed in Blender's bundled
Python interpreter.
