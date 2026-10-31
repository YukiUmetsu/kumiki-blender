# Use real PBR material packs in Blender and GLB

This import path accepts **local, reviewed PNG/JPEG texture files** from any
provider. It does not search, scrape, download, license, or redistribute assets
on your behalf. Use it for high-quality scanned architectural textures from a
provider whose rights **actually allow redistribution inside a GLB and `.blend`**.
The `redistribution_allowed` field is your declaration, not a legal assessment.

## Minimal recipe

Place the files beside the JSON (or inside its subdirectories):

```text
stone-pack/
  material-packs.json
  limestone-albedo.jpg
  limestone-roughness.png
  limestone-normal-opengl.png
```

```json
{
  "schema_version": 1,
  "packs": [
    {
      "id": "limestone",
      "creator": "Original texture creator",
      "source_url": "https://example.com/materials/limestone",
      "license": "CC0-1.0",
      "attribution": "Original texture creator / Limestone",
      "redistribution_allowed": true,
      "tile_size_m": 1.2,
      "metallic": 0,
      "normal_strength": 0.7,
      "normal_convention": "opengl",
      "maps": {
        "base_color": "limestone-albedo.jpg",
        "roughness": "limestone-roughness.png",
        "normal": "limestone-normal-opengl.png"
      }
    }
  ],
  "assignments": [{ "target": "wall-stone", "pack": "limestone" }]
}
```

The URL above is an **illustrative placeholder**, not an existing texture.
Replace it with a stable HTTPS source permalink. Additional assignment targets
can use the same pack. To lock each map to its exact downloaded version, add
an optional `sha256` object keyed by map name, with lowercase hex SHA-256 hashes.
Hashes of every map are recorded in the revision manifest even without pins.

Validate all files, hashes, assignment syntax and rights declarations **without Blender**:

```bash
PYTHONPATH=src python3 -m kumiki_blender inspect-texture-packs \
  --input /path/to/stone-pack/material-packs.json
```

## Apply it to a procedural facade

Run from the repo root, substituting your Blender executable as appropriate:

```bash
blender --background --factory-startup --python-exit-code 1 \
  --python scripts/blender_entry.py -- \
  build-facade --bays 3 \
  --texture-packs /path/to/stone-pack/material-packs.json \
  --output .review/facade/limestone-r01
```

Targets are **exact, case-sensitive material roles**:

| Natural materials | Architectural finishes |
| --- | --- |
| `wall-stone`, `wall-stone-light`, `trim-stone`, `foundation` | `window-metal`, `window-glass`, `recess-dark` |
| `ornament-stone`, `ornament-highlight`, `roof-slate`, `roof-slate-light` | `door-timber`, `door-timber-light`, `bronze-metal`, `rain-metal` |

Generated facade meshes use planar **metric UVs**. `tile_size_m` specifies the
real-world repeat length of a texture and should match its scan's measurement.
Unlike a normal-map illusion, trim/roof/masonry shapes remain real 3D geometry.

## Apply it to an imported GLB

First inspect the GLB and copy the exact material name:

```bash
PYTHONPATH=src python3 -m kumiki_blender inspect-asset \
  --input /path/to/building.glb
```

In the recipe, use that source name as the `target` instead of `wall-stone`.
Then run:

```bash
blender --background --factory-startup --python-exit-code 1 \
  --python scripts/blender_entry.py -- \
  import-asset --input /path/to/building.glb \
  --provenance /path/to/building-source.json \
  --texture-packs /path/to/stone-pack/material-packs.json \
  --output .review/imports/stone-r01
```

Imported models **keep existing UVs**. Meshes without UVs get meter-based
planar UVs. Avoid applying a texture pack and `--material-overrides` preset to
the same material; the CLI rejects that conflict.

## Supported formats and safeguards

- Mandatory `base_color`, `roughness`, **OpenGL (+Y) tangent `normal`** map;
  optional `metallic` map. PNG and JPEG only, up to 128 MiB / 8192 pixels per axis. DirectX
  (-Y) normals must be converted before import. Color/normal/roughness maps
  must share the same UV layout and tile consistently.
- glTF-compatible Principled BSDF image nodes; linear color for data maps,
  sRGB for albedo. Textures are packed into the `.blend` and embedded in GLB.
- Paths must stay inside the recipe folder, and symlinks are rejected. Exact
  SHA-256 pins are checked when specified; files are rechecked before Blender
  loads them. Nonexistent materials and duplicate targets fail closed.
- Source creator, URL, SPDX-style license, attribution, texture hashes and
  assignments are recorded in `manifest.json`. CC-BY textures require explicit
  attribution. Textures with **render-only**, **editorial**, or **no raw asset
  redistribution** licenses must **not** be imported into this export workflow.
- Not yet supported: height/displacement parallax, AO channel export, UDIMs,
  EXR/HDR, DirectX normal conversion, automated downloading or validation of
  the truthfulness of the source license.

The project includes `scripts/generate_first_party_texture_pack.py` for
integration testing with our own MIT-licensed procedural images. It verifies
the real Blender import/packing/export path, **not** the quality of any external
scanned texture provider.
