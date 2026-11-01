# Christ the Redeemer — procedural sculpture study

This generator produces **original modeled geometry** inspired by the Art Deco
Christ the Redeemer in Rio de Janeiro. It is **not** a scan, a licensed derivative
of another 3D model, or a certified museum-quality likeness.

## Reference measurements and QA

| Real-world reference | Generator target |
| --- | --- |
| Statue height excluding pedestal: **30 m** | Statue from z=8 to z=38 |
| Surveyed outstretched fingertips: **29.4 m** | Approx. 29.4 m across sculpted hands |
| Pedestal height: **8 m** | Chamfered-square stepped granite pedestal, z=0 to 8 |
| Cladding: pale soapstone mosaic | Embedded original triangular albedo, normal, and roughness maps with metric UVs |
| Pedestal exterior: dark stone ashlar | Original repeating, staggered granite-block albedo map |

Sources for measurements and surface treatment:

- [Paul Landowski's sculpture archive](https://www.paul-landowski.com/en/portfolio/christ-redempteur/) — artist's head and hands studies.
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
normals, closed hand unions with sufficient vertex detail, UV coverage, and
fully portable PBR texture embedding.

For a fast close-up iteration, append `--shot head_front --shot head_profile`
to render just those views. Other choices are `front`, `profile`,
`three_quarter`, and `hand_detail`.

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
hands use high-resolution volumetric unions to avoid separate rod-like digits.
The head and hands can be examined separately using the close-up QA cameras.

This remains an original parametric reconstruction. Its stylized anatomical
geometry is not a photogrammetric replica, and the script deliberately does not
redistribute a third-party scan. Photo-fidelity at macro face, hair, fingers,
and textile creases would still require a dedicated sculpting and multi-angle
reference-matching pass with human approval.
