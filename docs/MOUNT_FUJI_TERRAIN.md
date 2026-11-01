# Mount Fuji — two distinct 3D interpretations

There are **two intentionally different models**, sharing the same Fuji
subject but not the same geometry or scale:

| Variant | Purpose | Geometry | Footprint |
| --- | --- | --- | --- |
| `--style realistic` (default) | Geographic and documentary | GSI survey DEM, 1:1 vertical scale | Approximately 26 × 26 km |
| `--style iconic` | Read as **Mount Fuji / Japan** instantly, including at small sizes | Art-directed Fuji cone, snow tongues, curved black-and-red torii, two multi-hue sakura trees and volcanic garden | 10.2 × 8.6 arbitrary scene units |

The iconic model deliberately sacrifices realistic dimensions and geography.
It is a miniature symbol rather than a topographic study. All landmarks and
five-petal flowers are **actual exportable mesh**, not camera-facing sprites.

## Iconic spring diorama

Builds without a DEM or network access. Use Blender 4.2+ (4.5 LTS recommended):

```bash
PYTHONPATH=src python -m kumiki_blender plan-fuji --style iconic --detail high
blender --background --factory-startup --python-exit-code 1 \
  --python scripts/blender_entry.py -- \
  build-fuji --style iconic --detail high \
  --output .review/geography/terrain/mount-fuji-iconic-diorama/r01
```

The physically modeled objects include the blue-grey sculptural Fuji flanks,
irregular ivory snow crown and summit depression; lacquered red hashira, nuki,
shimaki, sweeping black kasagi and brass plaque; tapering branching cherry
trunks with pink blossom clouds and many individual five-petal flowers; slate
approach stones and a beveled obsidian display plinth. The floating photo studio
is excluded from GLB. The result is art-directed and **must never be represented
as survey-accurate Fuji**.

Iconic visual references (not copied or distributed as textures):

- [Japan National Tourism Organization: Mount Fuji](https://www.japan.travel/en/fuji-guide/)
- [Japan National Tourism Organization: cherry blossoms](https://www.japan.travel/en/see-and-do/cherry-blossom/)
- [Fushimi Inari Taisha: shrine and torii](https://inari.jp/en/)

### Photo-led color and material pass

The color system is **hand-authored and baked into mesh color attributes** so it
is retained in the portable GLBs, even in offline viewers. No photograph,
stock material or third-party texture is embedded or redistributed.

| Reference (inspiration only) | Observed treatment | Model interpretation |
| --- | --- | --- |
| [JNTO / Fuji-Hakone-Izu National Park, Ochudo](https://www.japan.travel/national-parks/parks/fuji-hakone-izu/see-and-do/ochudo-trek-between-heaven-earth-mtfuji-pilgrimage-trail/) | Dense conifer green below exposed dark volcanic scree | Green lower forest belt, oxidized warm-brown rock seams, cool basalt gullies |
| [JNTO / cherry blossoms](https://www.japan.travel/en/spot/1956/) | Mostly white/blush floral clusters, darker limbs, pink petals below trees | Five subtly different pink/ivory blossom shades, sparse green leaf buds and scattered petals |
| [Fushimi Inari Taisha](https://inari.jp/en/) | Vivid vermilion shrine timber and contrasting dark beam | Warm red lacquer with softened edges, black curved kasagi and metallic details |

Season, forest density and snow coverage are **art direction**, not a satellite
or botanical claim; the realistic variant's elevations remain unchanged.

## Realistic topographic relief

This asset is a geographically measured **relief mesh**, not an idealized cone.
It covers approximately 26 × 26 km and preserves the summit crater, jagged rim,
radial erosional grooves, and the distinct **Hōei craters on the southeast flank**.
The surface uses true elevations and **1:1 vertical scale**; only the albedo
is an artistic seasonal interpretation.

## References / provenance

- Topography: Geospatial Information Authority of Japan (GSI), DEM5A with
  lower-priority DEM5B/DEM5C/DEM10B tile coverage at Web Mercator zoom 13,
  refined at zoom 15 around the summit and the Hōei crater.
  [Official tile index](https://maps.gsi.go.jp/development/ichiran.html),
  [encoding specification](https://maps.gsi.go.jp/development/demtile.html),
  [content terms](https://www.gsi.go.jp/kikakuchousei/kikakuchousei40182.html).
  Attribution: **Elevation data from Geospatial Information Authority of Japan
  (GSI); transformed and visualized by Kumiki Blender.**
- Profile and summit crater: [NASA/JPL Mount Fuji and ASTER terrain](https://www.jpl.nasa.gov/images/pia11166-mt-fuji-japan/)
  (3,776 m summit; ~610 m wide summit crater).
- Hōei flank crater: [Geological Survey of Japan, Fuji 1707 eruption](https://gbank.gsj.jp/volcano/esdb/go.php?volc=Fuji_1707),
  [photo of southeast flank](https://www.web-gis.jp/GM1000/GM_Red1/GM_Red1-147.html).

The photographs are **inspiration/reference only** and are not downloaded or
embedded. The derived elevation dataset is stored in `.review` and is not part
of git or the installable wheel. Refer to the GSI terms when redistributing
geographic derivatives. The GSI tiles are periodically updated, so rerunning
the preparation script may produce slightly different elevation values.

## Generate and inspect

Requires Python 3.10+, Pillow (for the one-time tile preparation), Blender
**4.2+** (4.5 LTS recommended), internet access during terrain preparation.

```bash
python -m pip install Pillow
python scripts/prepare_fuji_dem.py .review/fuji-dem/fuji-heightmap.bin
PYTHONPATH=src python -m kumiki_blender plan-fuji --detail high
blender --background --factory-startup --python-exit-code 1 \
  --python scripts/blender_entry.py -- \
  build-fuji --detail standard --output .review/geography/terrain/mount-fuji-survey-terrain/r01
PYTHONPATH=src python -m kumiki_blender catalog search "Fuji"
```

Detail modes: `preview` = 257² vertices, `standard` = 513², `high` =
1025². The source heightmap contains 1025² samples. The Blender build exports
`source.blend`, `model.glb`, `preview.png`, five alternate views and a
`manifest.json`. Edit the source `.blend` for bespoke shots. The GLB contains
editable physical geometry and color attributes, not an image plane.

## QA gates

1. **Overall form:** compare the southern and northern profiles against photos;
   the slopes and surrounding foothills should be asymmetric and no radial
   ridges should repeat perfectly.
2. **Summit:** confirm a ~610 m-wide circular crater and an irregular rim,
   not a filled pointed cone.
3. **Southeast:** the Hōei eruption depression must appear visibly *below*
   the surrounding slope, with rim structure, rather than a secondary cone.
4. **Scale:** the elevation maximum should be around 3776 m and near
   138.7274°E, 35.3606°N; no artificial vertical exaggeration.
5. **Export:** inspect physical mesh bounds, vertex-color integrity, all
   inspection views and the GLB audit before committing a revision.

Future passes may add rock-scree displacement, forest geometry, texture atlas
and snow-season variants without changing the survey surface's elevations.
