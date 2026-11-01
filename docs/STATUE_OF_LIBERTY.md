# Statue of Liberty — scan-assisted detail rebuild

## Face checkpoint: CC BY scan, not a finished likeness

The first procedural model has recognizable outlines but substantially misses
the reference photographs' deep eyelids, realistic nose/lips, hair masses,
hands and physically draped fabric. Its sine-wave folds and cylindrical
features cannot be fixed just by subdivision. It is now the legacy prototype.

The first higher-detail checkpoint uses **"Statue Of Liberty Bronze Model" by
jerryfisher**, photogrammetry of a nineteenth-century bronze miniature, published
under **Creative Commons Attribution 4.0 (CC BY 4.0)**:

- Original work and item-level license: <https://sketchfab.com/3d-models/statue-of-liberty-bronze-model-5b9aa58d51624f01b9009b52379bf18a>
- License and obligations: <https://creativecommons.org/licenses/by/4.0/>
- Reproducible binary mirror: `allenai/objaverse` at the exact model ID above
- SHA-256: `e9c789977604980d410e30b95e02208ca6e46d053f44bc0f5c5030877eaa5fe8`
- Exact attribution and known differences: `src/kumiki_blender/assets/landmarks/statues/statue-of-liberty/scan_source.json`

The rebuild reorients and centers the scan, deforms the scan's actual front-face
vertices for localized brow, eye socket, eyelid, nose, lip and chin relief
(rather than applying separate floating primitives), then applies a copper-patina study
material, and stages head/crown geometry for front, both profiles, both three-
quarter angles, low-angle, rear hair and high crown inspection. The source
remains recognizable in the output; creator attribution, license, and a
modifications notice travel in the exported GLB node extras and manifest.
The raw downloaded scan is **not committed** to the source repository.

This scan is **not of the full-size copper monument**, and the miniature's
ocular relief is still too soft for final photo-matched approval. Do not claim
the first face checkpoint passes the final likeness requirement yet.

### Face quality gate (currently **not passed**)

Inspect `previews/face-front.png`, both three-quarter views, both profiles,
`previews/face-low-angle.png`, and the user-supplied face photography at the
same apparent head size. The scan earns credit for having captured, continuous
head/crown geometry and a plausible large-scale profile. It **fails** the
reference match in these specific ways:

| Requirement visible in the photographs | First scan-backed checkpoint |
|---|---|
| Deeply defined upper/lower eyelids, eyeballs and tear ducts | Sockets remain shallow and lack crisp eye anatomy |
| Forelock with separated, swept strands and sculpted temple curls | Hair masses are present but strand detail is largely absent |
| Wide, carved nostrils, sharply defined lips and chin planes | Bridge and mouth exist but their carved transitions are blurred |
| Perforated crown band and individually legible radial rays | Seven rays survive; the crown's window openings are not faithful |
| Surface seams, tool marks and differentiated oxidation | Single study patina lacks survey-grade copper texture |

We also reviewed **jerryfisher's alternate 4-million-triangle scan**
(<https://sketchfab.com/3d-models/statue-of-liberty-6626abe1f3e8469a9d8f4b74d8aa2a71>,
CC BY 4.0). It scans a *different miniature* with more polygons but visibly
noisier, less legible facial relief. Increasing tessellation alone cannot
restore anatomical features absent from a scan, so it was rejected as the
replacement source. Neither source is misrepresented as an NPS survey.

**Next sculpting gate:** landmark-align higher-resolution head references,
retopologize the face and eyelids manually in Blender, add real engraved lip,
nostril and hair geometry, then bake displacement/normal detail and compare
the identical seven camera views. Preserve the source mesh and credits so
every new sculpted change remains auditable. Full robe/tablet/torch/pedestal
reconstruction stays blocked on accepting the facial likeness.

### Build the first face checkpoint

```bash
python3 scripts/fetch_liberty_scan.py --output .review/_sources/liberty-bronze.glb
KUMIKI_RENDER_PROFILE=smoke blender --background --factory-startup --python-exit-code 1 \
  --python scripts/blender_entry.py -- \
  build-liberty-scan-face --source .review/_sources/liberty-bronze.glb \
  --output .review/landmarks/statues/statue-of-liberty/r02
PYTHONPATH=src python3 -m kumiki_blender audit-revision \
  --root .review/landmarks/statues/statue-of-liberty/r02
```

Remove `KUMIKI_RENDER_PROFILE=smoke` for full-resolution facial review. The
revision is immutable and contains an editable `.blend`, GLB, nine views,
manifest and attribution metadata. Artist approval is required before any
integration into the final full-body asset.

## Legacy procedural prototype

`plan-liberty` and `build-liberty` author a 360-degree, individually editable Statue
of Liberty, the torch and the museum-style pedestal. The model is an **original
interpretation** of supplied photography, not a photogrammetry scan, measured
National Park Service model, or verified likeness reconstruction.

## Reference interpretation

The user-supplied 10 photographs and modeling turnarounds guided these forms:

| Photographic feature | Actual generated geometry |
|---|---|
| Long classical face, deep sockets, protruding nasal bridge, cleft between lips, strong chin | High-density deformed head skin; separate nose, lids, eye volumes, brows, mouth relief and chin |
| Diadem with 7 rays, arched 20+ slots and thick trim | Seven swept pointed metal rays, separate raised rims and 27 open-looking dark inset windows with mullions |
| Swept hair and pendant locks by ears | Volumetric scalp and continuous swept forelock, 26 fine comb-direction ridges, temple curl volumes, ears/helixes, back bun and shoulder tresses |
| Diagonal stola and ground-length folds | 128-sector 97-level robe with three-scale relief, diagonal overfold panel, rolled hem, back cape, sleeve and 26 pleat ridges |
| Raised arm and tightly gripping fingers | Five-point tapered anatomy, detailed palm, thumb and four curled torch fingers |
| Tablet engraved `JULY IV MDCCLXXVI` | Tilted physical slab, bead trim, overlapping hand/fingers, converted and raised 3D serif-free Roman inscription |
| Elaborate torch viewed from below | Fluted shaft, stepped ring gallery, radial balusters, gold core and 9 asymmetric flame tongues |
| Classical rusticated pedestal | Separate footing, plinth, masonry core, belt, cornices, four facades, dark window recesses, 24 pilasters and 80 dentils |

The front is **-Y**, with the torch on the viewer's **left (-X)**. All geometry is
in illustrative Blender meters and all exportable mesh parts are manifold.
Texture maps are first-party procedural material maps packed in the GLB. The
patina variation emphasizes sculpted relief; it is not a measured corrosion
scan. The `recess` window surfaces are dark inset geometry, **not real holes**;
pedestal interior architecture is out of scope.

## Run

```bash
PYTHONPATH=src python3 -m kumiki_blender plan-liberty
PYTHONPATH=src python3 -m kumiki_blender plan-liberty --no-pedestal
PYTHONPATH=src python3 -m kumiki_blender catalog show statue-of-liberty
PYTHONPATH=src python3 -m unittest tests.test_liberty -v

KUMIKI_RENDER_PROFILE=smoke blender --background --factory-startup --python-exit-code 1 \
  --python scripts/blender_entry.py -- \
  catalog build statue-of-liberty --revision r01

PYTHONPATH=src python3 -m kumiki_blender audit-revision \
  --root .review/landmarks/statues/statue-of-liberty/r01
```

Choose a new revision output for subsequent builds; the exporter refuses to
overwrite any existing directory. Remove `KUMIKI_RENDER_PROFILE=smoke` for
1280 × 800, 48-sample production render settings. Use `--scale 0.5` for a half-size
asset. Use `--no-pedestal` to ground the statue at Z≈0 with its raised torch.
The latter flags apply to the direct `build-liberty --output ...` command;
catalog builds use the default geometry and automatically choose the
categorized revision path.

The exported package has an editable `.blend`, portable `.glb`, manifest, and
preview images for the 3/4 hero, full front, side, back, face, tablet and torch.
The studio floor, lamps and cameras are excluded from GLB.

## Review criteria and limits

1. Review silhouette in the front/side/back views, not just the hero angle.
2. Inspect the eyelids, brow structure, nose depth, jaw and ears close-up.
3. Check for clipping between robe layers, tablet and hand, crown rays and hair.
4. Inspect the torch from below; verify gold material stays confined to flames.
5. Audit GLB manifold geometry, real mesh part names, asset bounds and UVs.

This is not yet an exact likeness of Frédéric Bartholdi's copper sheet sculpt;
hand-authored procedural anatomy and garments are inherently more stylized
than a detailed scan or retopologized photogrammetry. A production-grade
historical reconstruction would require surveyed dimensions, licensed scans,
manual facial sculpting, copper sheet joints and a proper displacement bake.
