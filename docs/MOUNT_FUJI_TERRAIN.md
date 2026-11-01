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

## r18 — correct the realistic Mount Fuji's mauve volcanic cone

The r16/r17 **actual Blender hero and north-profile renders** revealed a
material error: the barren cone looked purple-pink, with a conspicuous yellow
patch on Hōei's shoulder. These colors were caused by the earlier
blue-magenta charcoal, burgundy ash and over-strong ochre pigment blend—not
by the GSI elevation data or the GLB's texture packing.

| Reference (visual evidence only) | What constrains the new palette |
| --- | --- |
| [NASA Earth Observatory, *Mount Fuji Bare Again After Fleeting Snow* (2024)](https://science.nasa.gov/image-detail/amf-245a0290-660f-4ed7-bccf-77e02871a4d2/) | Broad exposed cone is pale/warm brown-grey, with surrounding deep-green woods and darker volcanic sectors; top-down camera is not registered to the model |
| [Japan National Parks, Fuji-Hakone-Izu](https://www.japan.travel/national-parks/parks/fuji-hakone-izu/explore/) | Dense woods, then upper conifer belt, then unvegetated volcanic desert around the summit |
| [September 16, 2023 Kawaguchiko Fuji photograph](https://fujisanmeisyugura.com/blog/2023/09/89818/) | An oblique snow-free view has charcoal/earth-grey exposed slopes contrasting with green lower forest; time and illumination are particular to that photo |
| [Japan Ministry of the Environment, volcanic color notes](https://kanto.env.go.jp/blog/2020/05/post-870.html) | Black scoria remains prominent in close-ups, while red oxidation appears locally rather than across the whole cone |

Both realistic seasons now use **neutral charcoal basalt and warm-grey volcanic
ash**, with restricted iron-brown scoria and less aggressive Hōei shoulder
ochre; summer remains snow-free and winter retains its independent snow cover.
The material changes are all original procedural colors, not republished photo
textures. The GSI survey elevations, crater patches, UV topology and embedded
GLB color encoding are unchanged. These diverse references are **unregistered
visual comparisons**, not a claim of measured photometric reconstruction;
forest remains a terrain material rather than modeled individual trees.

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

The detailed spring revision replaces radial-looking cherry canopies with
asymmetric main limbs, gravity-influenced secondary forks, modeled trunk bark
fissures and smaller blossom sprays. The smaller tree is given additional
flowering branchlets, and the shrine pathway uses individual irregular slate
flagstones. Close-up botanical references inform the geometry, but no third-party
plant mesh is copied. A dedicated seventh `sakura-botanical` QA camera now
checks branch architecture and petal geometry at closer range.
The high-detail variant uses **8,500 individual five-petal flowers** across
both trees; small opaque rosebud cores sit beneath those sprays instead of
forming an obviously spherical pink canopy.

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

The latest realistic color iteration corrects the earlier uniformly warm-brown
volcanic cone and pale green lower plain. Four photographic observations guide
the **original, vertex-baked albedo**, with no third-party photo pixels reused:

| Source photograph / field observation (inspiration only) | Resulting palette change |
| --- | --- |
| [Japan Ministry of the Environment: *What color is Mount Fuji?*](https://kanto.env.go.jp/blog/2020/05/post-870.html) | Black porous scoria dominates exposed elevations above the fifth station; dark neutral basalt replaces the previous blanket pink-brown hue. The distant blue appearance is atmospheric scattering, not a blue rock pigment. |
| [JNTO: Fujinomiya Trail](https://www.japan.travel/en/fuji-guide/guide-to-the-fujinomiya-trail/) | Sparse reddish volcanic scree is retained around exposed routes rather than blended evenly over the full cone. |
| [NASA Earth Observatory: Mt Fuji from the ISS](https://science.nasa.gov/earth/earth-observatory/mount-fuji-japan-39099/) | Cool white snow against dark rock, with shadowed crater walls and forested lower flanks. |
| [Fujisan Curator: Ochudo tree-line photograph](https://www.fujisancurator.com/tour-hiking/ochudo-uplift-guided-hiking-tour) | Much darker conifer green, pocketed scrub, and an irregular ~2.2–2.4 km transition instead of a pale sage-green stripe. |

Snow remains an intentionally chosen late-winter / early-spring cover pattern;
it cannot represent all seasons at once. The exposed Hōei crater has locally
warmer iron-rich scoria, while the summit basalt stays dark in snow-free faces.
All color work changes only PBR albedo, **not DEM elevations**.

### Separate snow-free late-summer export

Choose `--season summer` on **realistic** builds for the August–September
appearance: absolutely no snow remains on the top crater lip, north-facing
gullies or Hōei walls. The snow-free material reveals charcoal-to-iron scoria,
mineral-ash pigment variation, erosion grooves and the survey-refined rim.
Birch/scrub near the irregular treeline is marginally greener than in winter;
the lower cedar forest and all GSI elevations are unchanged. The default
`--season winter` keeps the original snowy model; **this option does not modify
or replace the winter GLB**.

Reference photos (inspiration only, not reused as textures):

- [Japan Ministry of the Environment: Fuji summer landscape journal](https://kanto.env.go.jp/blog/page_00056.html): late June snow often survives close to July, while exposed late-summer slopes have dark scoria and birch belts around 2,000–2,500 m.
- [Japan Ministry of the Environment: Why Fuji's rock looks black](https://kanto.env.go.jp/blog/2020/05/post-870.html): upper volcanic scoria is dark when viewed close up, rather than blue.
- [Drone photograph of the summit crater's exposed strata](https://note.com/hayakawayukio/n/n4276f012dd71): irregular dark-red/grey rock within the rim; photographic study only.

To create **both** seasons with the same cached high-detail survey data:

```bash
# Winter styling remains available via --season winter (and is the default).
blender --background --factory-startup --python-exit-code 1 \
  --python scripts/blender_entry.py -- build-fuji --detail high \
  --season summer --dem .review/fuji-dem/fuji-heightmap-v2.bin \
  --detail-patch .review/fuji-dem/fuji-summit-z15-patch.bin \
  --detail-patch .review/fuji-dem/fuji-hoei-z15-patch.bin \
  --output .review/geography/terrain/mount-fuji-survey-terrain/r16
```

Season, forest density and snow coverage are **art direction**, not a satellite
or botanical claim; the realistic variant's elevations remain unchanged.

### r13 — updated Blender craft-director QA pass

This pass follows a **reference → highest-risk feature → fixed camera →
comparison → correction** loop. The surveyed geometry is already stronger
than the material impression at normal viewing distance, so the targeted
failures were an overly grassy, uniform forest, under-articulated Hōei ash,
and an inspection framing that made the summit crater too small.

| Evidence | Feature observed | Evidence type / use | Confidence |
| --- | --- | --- | --- |
| [VRT NWS October 2024 snow-free Fuji aerial](https://www.vrt.be/vrtnws/nl/2024/10/30/nog-steeds-geen-sneeuw-op-mount-fuji-in-japan/) | Charcoal/plum scoria, dark pine carpet, iron streaks and a localized pale-ochre Hōei ridge | Photographer's observed image; **reference only** | High for visible color distribution, limited by camera and daylight |
| [Smithsonian Global Volcanism Program: Fujisan](https://volcano.si.edu/volcano.cfm?vn=283030) | Red-oxidized summit rim, 1707 SE crater, summertime exposed volcanic surfaces | Geological photography / description; **reference only** | High for broad geology |
| [GSI DEM tile specifications](https://maps.gsi.go.jp/development/demtile.html) | Survey heights, basin, summit/side-crater location | **Measured**, transformed as documented | High at underlying grid resolution |
| [Yukio Hayakawa summit-crater drone study](https://note.com/hayakawayukio/n/n4276f012dd71) | Steep uneven inner cliff, layered red-grey deposits | Observed drone photography; **reference only** | High for crater structure, uncertain lighting |
| [Ministry of the Environment: scoria color](https://kanto.env.go.jp/blog/2020/05/post-870.html) | Upper mountain is dark basalt/scoria, not a uniform warm cone | Observed field photos; **reference only** | High for close-up stone color |

Corrections in r13/r14: darker and less uniform canopy/crown pigment at 30–70 m
spatial scales; localized ochre SE Hōei shoulder; narrow oxidized radial
rock seams; clearer afternoon sun; tighter hero, Hōei and summit inspection
views. An r13 photo/render sheet exposed a second defect: the Hōei camera was
too elevated, showing a flattened-looking bowl, and the first ochre mask was
centered on **~2,220 m terrain below its own threshold**. In r14, the Hōei
camera was lowered to a raking side view and the ochre accent shifted onto
the measured ~2,660 m southeast shoulder; the deep scoria bowl was darkened.
The r14 macro view also revealed overly continuous red radial seams, so r15
breaks these along spatially irregular outcrops and tiles the existing packed
rock micro-normal six times across the 26 km terrain (lighting only).
All measured elevations are unchanged. The forest remains **color/normal detail on a terrain surface, not
individual 3D trees**. Close-up botanical fidelity would require a separate
view-dependent forest-instancing/LOD system. The pigment variations are
original visual interpretations—not geological material classifications.

An unmodified source photograph can be compared with the actual Blender
render using a non-registered two-panel sheet. Photo pixels are used **only
for local comparison** and are not included in source control or the GLB:

```bash
python scripts/reference_compare.py /path/to/reference.jpg \
  .review/geography/terrain/mount-fuji-survey-terrain/r15/previews/hoei-context.png \
  --output .review/geography/terrain/mount-fuji-survey-terrain/r15/hoei-reference-comparison.png
```

The cameras differ between images; don't interpret an unaligned pixel
comparison as topographic discrepancy. Revisit neutral clay geometry and
registered camera views before changing surveyed elevations.

### r16 summer / r17 winter — portable GLB color correction

The first realistic GLBs carried the entire forest/scoria/snow palette as
`COLOR_0` vertex attributes. Blender rendered that correctly, but some
viewers ignore vertex colors and fall back to a plain white glTF material.
This is an **export-material defect**, not a terrain-shape or lighting defect.

The generator now embeds a full **georeferenced base-color PNG** in every
realistic GLB and connects it directly to the standard glTF
`pbrMetallicRoughness.baseColorTexture` field. The atlas has four texels per
coarse DEM interval (4096² at high detail); the summit/Hōei fine patch colors
overwrite the coarse interpolation at their surveyed positions. Before PNG
encoding, linear pigments are transformed to sRGB to prevent an otherwise
very dark, double-gamma result. Normal detail uses independent tiled UVs.
No external color images, photography or textures are required at runtime.

Inspect **the exported GLB** rather than source.blend alone:

```bash
blender --background \
  .review/geography/terrain/mount-fuji-survey-terrain/r16/source.blend \
  --python-exit-code 1 --python scripts/verify_fuji_glb_color.py -- \
  --root .review/geography/terrain/mount-fuji-survey-terrain/r16

PYTHONPATH=src python -m kumiki_blender audit-revision --root \
  .review/geography/terrain/mount-fuji-survey-terrain/r16
```

The verification script imports the **actual GLB as a new mesh/material**, with
the original source model removed, then renders under the original camera and
light. A preview-resolution prototype had an RGB mean absolute native-vs-GLB
render error below **0.001 on a 0–255 channel scale** after the sRGB fix.
The GLB audit now rejects a realistic Fuji revision missing embedded albedo.
Repeat the build, re-import and audit commands against `r17` with
`--season winter` for the snow-covered version. Both seasonal GLBs carry
their own color atlas instead of depending on viewer vertex-color support.

For closer survey-terrain views, the Blender builder now also generates a
deterministic, 512/1024-pixel **tangent-space rock micro-normal map**. It is
packed into the editable Blender file and embedded directly into the GLB.
The texture modifies surface *lighting*, **never the GSI vertex elevations**;
the original 1:1 geometry stays unchanged. High-detail QA renders use more
samples to retain close-up rock and crater structure.

### Native-resolution crater craftsmanship

The whole-mountain mesh samples the GSI DEM at approximately 25 m intervals.
For true, closer crater relief, optionally prepare two **seam-stitched survey
patches**: one around the summit crater and one around the 1707 Hōei flank
crater. Both measure approximately 2.03 × 2.03 km, keep 321² vertices each
at approximately 6.3 m spacing, and derive their interior heights from the
official **zoom-15 GSI 5 m-class tiles**. A 180 m transition shoulder blends
those GSI samples into the parent DEM. The boundary is **exactly the parent
DEM**, with no gaps, overlapping faces, artificial volcano sculpting, or
vertical exaggeration. The meshes are stitched into **one topological terrain
mesh**, splitting every coarse boundary edge where the finer mesh meets it.

```bash
# After preparing the 1025² whole-Fuji DEM:
python scripts/prepare_fuji_dem.py --crater-patches \
  .review/fuji-dem/fuji-heightmap.bin .review/fuji-dem
blender --background --factory-startup --python-exit-code 1 \
  --python scripts/blender_entry.py -- build-fuji --detail high \
  --dem .review/fuji-dem/fuji-heightmap.bin \
  --detail-patch .review/fuji-dem/fuji-summit-z15-patch.bin \
  --detail-patch .review/fuji-dem/fuji-hoei-z15-patch.bin \
  --output .review/geography/terrain/mount-fuji-survey-terrain/r02
```

The optional patches are tied to a CRC32 fingerprint of the exact parent DEM;
the builder rejects mismatched input, overlapping areas, altered seam values,
and partial survey coverage. They require `--detail high`. Generated patch
data remains a reproducible, excluded-from-git build input in `.review`.

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

The photographs are **inspiration/reference only** and are not redistributed or
embedded in deliverables. The derived elevation dataset is stored in `.review` and is not part
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
1025², or **1,243,585 vertices** with both optional native crater patches.
The source heightmap contains 1025² samples. The Blender build exports
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
