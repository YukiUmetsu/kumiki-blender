# Christ the Redeemer — procedural sculpture study

This generator produces **original modeled geometry** inspired by the Art Deco
Christ the Redeemer in Rio de Janeiro. It is **not** a scan, a licensed derivative
of another 3D model, or a certified museum-quality likeness.

## Reference measurements and QA

| Real-world reference | Generator target |
| --- | --- |
| Statue height excluding pedestal: **30 m** | Statue from z=8 to z=38 |
| Outstretched arms: **28 m** | Approx. 28.35 m across sculpted hands |
| Pedestal height: **8 m** | Octagonal stone pedestal, z=0 to 8 |
| Cladding: pale soapstone mosaic | Warm grey soapstone-inspired PBR material |

Sources for measurements and surface treatment:

- [Paul Landowski's sculpture archive](https://www.paul-landowski.com/en/portfolio/christ-redempteur/) — artist's head and hands studies.
- [Google Arts & Culture](https://artsandculture.google.com/asset/christ-the-redeemer/fAEswWZ70LI6LA?hl=en) — height and span.
- [Conservation paper (ICOM-CC)](https://www.marciabraga.arq.br/site/images/stories/pdf/christ_the_redeemer-icom-cc_2011.pdf) — stone tesserae and surface.
- [Pontifical Catholic University Rio, Redentor project](https://next.dad.puc-rio.br/projeto/projeto-redentor/) — side elevation study.
- [Wikimedia Commons close-up photograph](https://commons.wikimedia.org/wiki/File:Detail,_Christ_the_Redeemer_statue.JPG) — facial reference.

References are for visual inspection only; no external images or third-party
model assets are embedded into the export.

## Build

Blender **4.2+** is required; verified with Blender **4.5.15 LTS** on Linux.

```bash
blender --background --factory-startup --python-exit-code 1 \
  --python scripts/landmarks/christ_redeemer.py -- \
  --verify --output .review/christ-r03.glb \
  --blend .review/christ-r03.blend \
  --metrics .review/christ-r03-metrics.json \
  --views .review/christ-r03/views --size 900 --samples 48
```

The script generates named, editable component meshes for the robe, sleeves,
face, hair locks, fingers, and pedestal. The GLB contains model meshes only,
excluding QA lights/cameras. The accompanying audit checks pedestal contact,
approximate measured dimensions, and outward normals of both sleeves.

For a fast close-up iteration, append `--shot head_front --shot head_profile`
to render just those views. Other choices are `front`, `profile`,
`three_quarter`, and `hand_detail`.

Run the live integration test with:

```bash
KUMIKI_BLENDER_BINARY=/path/to/blender PYTHONPATH=src \
  python3 -m unittest tests.test_christ_redeemer -v
```

## Craft status

The original PR used large primitive spheres for facial features, featureless
torso cylinders, and oversized straight sleeves. The current revision fixes
the physically floating base, sleeve normals and proportions, and replaces
many details with curved tapered meshes. The new front/profile/head/hand
cameras make the gaps visible during review.

**Still needed before a photo-faithful landmark claim:** high-resolution
anatomical sculpting of the eyelids, mouth, nose and jaw; complex cloak
overlap and gravity-aware shoulder drapery; mosaic texture creation with proper
UVs and roughness; reference-aligned human approval from all camera angles.
The current work is a more complete study, not final craftsmanship.
