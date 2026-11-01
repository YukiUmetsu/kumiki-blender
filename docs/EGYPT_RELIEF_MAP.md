# Egypt cinematic relief atlas

A first-party procedural 3D illustration based on the supplied angled Egypt map reference. It is designed for educational motion graphics, map flyovers and three-quarter diorama shots. **It is not a surveyed elevation map or a precise geographic boundary dataset.** Landforms and city positions are deliberately exaggerated for readability at editorial scale.

## Reference interpretation

| Reference feature | Editable 3D treatment |
|---|---|
| Deep Mediterranean and Red Sea | Low flat water surfaces in the relief mesh, geospatial mask with glossy water material |
| Organic Sahara sand and Eastern Desert peaks | Vertically exaggerated relief topology, multiscale erosion noise, folded mountain ridges, a raised Sinai massif, westward escarpment terraces, warm rock/sand colors and PBR micro-normals |
| Green Nile ribbon and triangular Delta | Deliberately narrow fertile lowland, six independent blue waterway meshes including Rosetta/Damietta arms and the Suez Canal |
| Qattara Depression | Physical basin and raised rim, not just a printed label |
| Lake Nasser | Irregular submerged ribbon and drowned side channels |
| Lake Qarun and Faiyum | Distinct blue inland lake and small fertile oasis west of Cairo |
| Thin map boundaries | Separate graphite ribbons for Libya, Sudan and eastern Sinai |
| Giza and Luxor landmarks | Three separately selectable Giza pyramids with twelve physical stone courses each; Luxor's battered gateway pylons, paired obelisks, ring-profile columns, capitals, courtyard and guardian avenue |
| Zoomable city districts | Procedural street-spaced masonry buildings with actual roof parapets, rooftop structures and inset facade windows at Cairo, Alexandria, Luxor, Aswan and Sharm El-Sheikh |
| Landmark-specific silhouettes | Cairo's Citadel-inspired dome-and-minarets mosque, Alexandria's tiered Pharos-inspired lighthouse, Aswan's spillway barrage and Sharm resort roof domes |
| Nile agricultural detail and oases | Independently surfaced crop fields on both Nile banks, curved palm fronds at Siwa, Faiyum, the Delta, Luxor and Aswan |
| Embossed type | 3D converted typography for place names, cities, SUDAN and the foreground EGYPT masthead; geographic captions clear the tallest terrain they span without deforming their letterforms |
| Floating table map against charcoal | Beveled solid sand-brown plinth, dark studio stage, three-area film lighting and orthographic cameras |

Map coordinates cover longitude 24°E–38°E, latitude 21.7°N–32.6°N. The main Nile is a deliberately hand-authored path through Aswan, Luxor and Cairo. Simplified coastline shapes include the Gulf of Suez, Gulf of Aqaba, Sinai and the Nile delta. Rendering colors and height relief are authored locally from deterministic noise and masks, without third-party map downloads, image textures, DEMs or runtime packages. **Vertical exaggeration is intentional**: a 16-unit-wide miniature with geographically proportional altitudes appears nearly flat in an oblique hero render. This is an illustrative relief, not a meter-to-Blender-unit elevation scale.

## Usage

Preview the geographic plan without Blender:

```bash
PYTHONPATH=src python3 -m kumiki_blender plan-egypt-map
PYTHONPATH=src python3 -m kumiki_blender plan-egypt-map --detail high
```

Render an immutable revision using Blender 4.2+ (`--detail high` increases only terrain tessellation, from 480×352 to 700×420 cells):

```bash
blender --background --factory-startup --python-exit-code 1 \
  --python scripts/blender_entry.py -- \
  build-egypt-map --detail high --output .review/egypt-map/r01
```

The model is also discoverable in the shared categorized asset catalog (merged from `main`):

```bash
PYTHONPATH=src python3 -m kumiki_blender catalog show egypt-relief-atlas
PYTHONPATH=src python3 -m kumiki_blender catalog search Egypt terrain
blender --background --factory-startup --python-exit-code 1 \
  --python scripts/blender_entry.py -- \
  catalog build egypt-relief-atlas --revision r01
```

The catalog build writes `.review/geography/maps/egypt-relief-atlas/r01/` while the
standalone CLI continues accepting any new `--output` path.

Set `KUMIKI_RENDER_PROFILE=smoke` to limit only preview pixels/samples for QA. For another build choose `r02`; an existing output folder is never overwritten.

Produced files:

- `source.blend` — editable topography, waterway meshes, miniature landmarks, converted type, camera and lights
- `model.glb` — portable geometry with UVs and packed albedo/PBR textures, no lights or studio floor
- `preview.png` — angled hero; `previews/atlas.png` — elevated plan; `previews/delta.png` — Nile fan; `previews/red-sea.png` — Sinai/Red Sea; `previews/{cairo,luxor,alexandria,aswan,sharm}.png` — macro views with individual buildings and monument details; `previews/{qattara,lake-nasser,eastern-peaks}.png` — relief and shoreline inspection views
- `manifest.json` — artifact integrity, geographic extents, part counts and Blender version

Verify with:

```bash
PYTHONPATH=src python3 -m unittest tests.test_egypt_map -v
PYTHONPATH=src python3 -m kumiki_blender audit --model .review/egypt-map/r01/model.glb --require-pbr-textures
PYTHONPATH=src python3 -m kumiki_blender audit-revision --root .review/egypt-map/r01
```

## Customization and motion graphics

- Change main-map tessellation with `EgyptMapSpec(columns=..., rows=...)` in `geography/egypt/terrain.py` or CLI `--detail`.
- Change Nile/canal routes in `RIVER_CHANNELS` and labels in `LABELS`; the same latitude/longitude projection is reused for all map layers.
- `sample_terrain(lon, lat)` provides a deterministic height, geographic region, color and vegetation mask usable for camera targets or overlay timing.
- Animated camera passes can be built using the twelve saved named views, from broad `hero`/`atlas` to city zooms and geographic `qattara`, `lake-nasser` and `eastern-peaks` studies. The map itself is intentionally static; authored motion belongs to a camera track or downstream Kumiki-UI use.
- Urban miniatures are not measured skyline reconstructions. Adjust `district_layout` and the corresponding geometric landmarks in `geography/egypt/geometry.py` / `geography/egypt/blender_landmarks.py` to change density and monuments. All detail meshes are exported as real geometry, not billboard labels.
- All map text is converted into exportable glyph meshes, with rigid elevated captions where mountains would obscure letterforms. The project attempts a locally installed serif font, with Blender's bundled fallback if unavailable; the exported asset does not rely on the font existing on the consumer's computer.

**Visual/factual caveat:** This graphic represents the reference image's visual hierarchy, not a GIS map. To make claims about exact altitudes, coastlines, borders or surveying, feed a separately licensed verified DEM, water polygon dataset and toponyms into a future data-driven source adapter. The supplied reference image is not embedded or redistributed.

## Code organization

| Source | Responsibility |
|---|---|
| `geography/egypt/terrain.py` | Deterministic coordinates, hydrography, relief samples and stitched terrain mesh (no Blender dependency) |
| `geography/egypt/geometry.py` | Reusable 3D mesh sketches, domes, obelisks, pinnate palms and urban block layouts (no Blender dependency) |
| `geography/egypt/blender_support.py` | Waterway ribbons, packed geospatial albedo, materials, type placement and terrain mesh attachment |
| `geography/egypt/blender_landmarks.py` | Giza, Luxor, Cairo, Alexandria, Aswan, Sharm, Nile agriculture and palm-grove assemblies |
| `geography/egypt/blender_scene.py` | Lighting, nine camera shots, scene orchestration, export and revision metadata |
| `assets/geography/maps/egypt-relief-atlas/asset.json` | Discoverable catalog entry with build/plan commands, tags and source links |

The legacy `egypt_map.py`, `egypt_details.py` and `blender_egypt_map.py`
remain thin compatibility modules for existing command-line scripts and callers.
