# Christ the Redeemer — procedural sculpture study

This generator produces **original modeled geometry** inspired by the Art Deco
Christ the Redeemer in Rio de Janeiro. It is **not** a scan, a licensed derivative
of another 3D model, or a certified museum-quality likeness.

## Reference measurements and QA

| Real-world reference | Generator target |
| --- | --- |
| Statue height excluding pedestal: **30 m** | Statue from z=8 to z=38 |
| Surveyed outstretched fingertips: **29.4 m** | Exactly 29.4 m, calibrated from mesh bounds |
| Landowski's sculpted hand length: **3.20 m each** | Approx. 3.19 m each, verified on both fused meshes |
| Pedestal height: **8 m** | Chamfered-square stepped granite pedestal, z=0 to 8 |
| Cladding: pale soapstone mosaic | Embedded original triangular albedo, normal, and roughness maps with metric UVs |
| Pedestal exterior: dark stone ashlar | Original repeating, staggered granite-block albedo map |

Sources for measurements and surface treatment:

- [Paul Landowski's sculpture archive](https://www.paul-landowski.com/en/le-christ-de-rio/) — artist's 3.75 m head and 3.20 m hand measurements and sculpting studies.
- [Google Arts & Culture](https://artsandculture.google.com/asset/christ-the-redeemer/fAEswWZ70LI6LA?hl=en) — height and span.
- [Conservation paper (ICOM-CC)](https://www.marciabraga.arq.br/site/images/stories/pdf/christ_the_redeemer-icom-cc_2011.pdf) — stone tesserae and surface.
- [Sanctuary of Christ the Redeemer](https://santuariocristoredentor.com.br/) — soapstone triangle dimensions and conservation.
- [Pontifical Catholic University Rio, Redentor project](https://next.dad.puc-rio.br/projeto/projeto-redentor/) — side elevation study.
- [Wikimedia Commons close-up photograph](https://commons.wikimedia.org/wiki/File:Detail,_Christ_the_Redeemer_statue.JPG) — facial reference.
- [Pix4D photogrammetric survey](https://www.pix4d.com/blog/mapping-christ/) — a 29.40 m span in the original drawings (29.42 m measured), rather than the frequently rounded 28 m figure.

References are for visual inspection only; no external images or third-party
model assets are embedded into the export.

## Build

Blender **4.2+** is required; verified with Blender **4.5.3 LTS** on Linux.

```bash
blender --background --factory-startup --python-exit-code 1 \
  --python scripts/landmarks/christ_redeemer.py -- \
  --verify --output .review/christ.glb \
  --blend .review/christ.blend \
  --metrics .review/christ-metrics.json \
  --views .review/christ/views --size 900 --samples 48
```

The script generates named, editable component meshes for the robe, weighted
asymmetric mantle, closed draped sleeves, sculpted face, continuous hair mantle,
two voxel-fused anatomical hands, chamfered granite pedestal, and chapel entrance. The GLB
embeds **three original, tileable soapstone textures**: 3 cm-scale triangular
tesserae color, subtle grout/stone normal relief, and high-roughness stone grain.
The chapel exterior uses an additional original staggered-ashlar granite texture.
The GLB contains model meshes only, excluding QA lights and cameras. The live
audit checks pedestal contact, measured dimensions, manifold/outward sleeve
normals, artist-sourced hand reach and closed unions with sufficient vertex
detail, UV coverage, and fully portable PBR texture embedding.

For a fast close-up iteration, append `--shot head_front --shot head_low`
to render just those views. Other choices are `front`, `profile`,
`three_quarter`, `head_profile`, `head_three_quarter`, and `hand_detail`.

For fast anatomy-first experimentation, render the **independent neutral clay
head**, before spending time on the full-scale textured scene:

```bash
blender --background --factory-startup --python-exit-code 1 \
  --python scripts/landmarks/christ_redeemer.py -- \
  --head-study --verify --views .review/christ/head-clay \
  --shot head_front --shot head_profile --shot head_three_quarter \
  --shot head_low --size 850 --samples 24
```

These camera angles expose different faults: straight-on symmetry, nose/forehead
profile, three-quarter facial volume, and a low-angle view closer to the
monument photographs. The clay proof shares the exact same head geometry and
pose as the full build. The generated QA folder is excluded from git.

Run the live integration test with:

```bash
KUMIKI_BLENDER_BINARY=/path/to/blender PYTHONPATH=src \
  python3 -m unittest tests.test_christ_redeemer -v
```

## Craft status

This revision replaces rigid tubular sleeves with more continuous hanging
folds; adds a broad, asymmetrically hemmed forecloth, weighted U-shaped
shoulder shawl, and independently draped stole; and tips the integrated head,
hair, and neck forward toward the characteristic downward gaze. Eyelids, cheek,
nose, chin, jaw, and beard volumes are displaced on the **same dense head
surface**, with continuous upper eyelid contours and finer incised detailing.
The torso's lower robe was widened against photo/mesh silhouettes, and the
hands use high-resolution volumetric unions with the artist-documented 3.20 m
reach, while the fingers stay fused rather than forming separate rods. The
latest facial sculpt uses a dense 38,560-vertex unified head surface. A smooth
cubic cage now removes visible piecewise-linear bands around temples and jaw;
the narrowed lower-face silhouette is framed by a fuller hair curtain, and the
mouth follows a downturned carved contour. The latest reference review removed
unsupported beard ridges, artificial chin grain, and exposed separate ear
primitives, and simplified the nose and closed eyelids. A clay-only head study
and dedicated three-quarter camera make regressions inexpensive to see before
full export. The head and hands can also be examined separately using the
close-up QA cameras, including a low-angle face view based on reference photos.
The soapstone color-map variation was increased to reveal its triangular
tesserae at close viewing distance without exaggerating grout depth.

This remains an original parametric reconstruction, **not an approved
photo-faithful sculpt**. Its anatomical geometry is not a photogrammetric
replica, and the script deliberately does not redistribute a third-party scan.
Photoreal facial likeness, strands/hairline, digits, and textile creases still
require more sculpting against licensed, consistently oriented multi-view images
and human acceptance. The clay QA workflow is deliberately retained to expose
those remaining discrepancies rather than hiding them under materials.
