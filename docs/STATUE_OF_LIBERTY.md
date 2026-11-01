# Procedural Statue of Liberty — sculptural asset study

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
