# Statue of Liberty — scan-assisted detail rebuild

## r62 — restored broken shackle, sculpted sandals and chain (**opt-in study**)

The model previously omitted a defining sculptural feature at Liberty's feet.
The National Park Service's [abolition history](https://www.nps.gov/stli/learn/historyculture/abolition.htm)
describes a **broken shackle at her right foot**, a chain passing out of view
under the robe, and its **broken end link reappearing in front of her left
foot**. The [NPS fact sheet](https://www.nps.gov/stli/learn/statue-of-liberty-facts.htm)
also confirms that her **right heel is raised**. Primary photo sources used:

- [HAER 255](https://www.loc.gov/pictures/item/ny1251.color.570024c/),
  Jet Lowe (1984), shows the *installed* right foot, the broad curved shackle
  and the wide copper chain straps beside the drapery.
- [HAER 256](https://www.loc.gov/pictures/item/ny1251.color.570025c/),
  Jet Lowe (1984), is a closer view of the articulated toes and sandal sole.
- [NPS photographic account](https://www.nps.gov/stli/learn/historyculture/abolition.htm)
  gives another legible view of the copper toe and chain construction.

The new `--detailed-feet` flag adds editable, named, closed 3D forms: a raised
right and resting left foot with five tapered toes each and sandal-sole edge;
the open/capped curved shackle at the right foot; two discontinuous lengths of
wide, flattened, slightly inclined interlocking links; and a visibly open
broken link at the left end. The camera plan adds front, oblique, split-link
and shackle close-ups. It defaults **off**, preserving the existing GLB
recipe until the whole statue passes reconstruction review.

This is **illustrative reference-led sculpture**, not a direct 3D scan:
individual toe, loop and placement dimensions are artist-estimated from
uncalibrated photographs. The experimental body/face still fails the
photo-faithful acceptance gate recorded in
[face QA](STATUE_OF_LIBERTY_FACE_QA.md). The broken-chain detail can be
inspected independently; it does **not** certify the monument's likeness.

```bash
KUMIKI_RENDER_PROFILE=smoke blender --background --factory-startup \
  --threads 6 --python-exit-code 1 --python scripts/blender_entry.py -- \
  build-liberty --detailed-feet --detailed-torch \
  --detailed-tablet --detailed-crown --detailed-face \
  --output .review/landmarks/statues/statue-of-liberty/r62-broken-shackle
PYTHONPATH=src python3 -m kumiki_blender audit-revision \
  --root .review/landmarks/statues/statue-of-liberty/r62-broken-shackle
```

## r56 — authentic-foundation and multi-view hold gate

The stricter updated Blender 3D Craft Director workflow now blocks any claim
of photo-faithful face modeling without authenticated geometry, registered
front/profile/three-quarter cameras and actual measured silhouette/landmark
agreement. A checked-in `review.json`-style manifest explicitly holds the
current r53 face, and new `geometry_compare.py`/`reconstruction_gate.py`
scripts independently validate real masks and correspondence annotations
instead of trusting a green render/test manifest. We did **not** fabricate
annotations for the uncalibrated museum/HAER views. These quality tools do not
promote the rejected r54–r55 anatomy. Full rationale and reference ledger:
[facial acceptance gate](STATUE_OF_LIBERTY_FACE_QA.md).

## r54–r55 — measured face QA and reconstruction-method gate

The updated face workflow now uses independent **NPS measured feature ratios**
and **public-domain HAER frontal/profile photographs** for fixed-camera
comparison. Two control-cage and iris/lip experiments did **not** pass
photographic QA and were intentionally rejected. The r53 detailed face
remains the latest opt-in model; no replacement GLB was promoted from the
failed trials. Added a quick three-view neutral-clay Blender renderer,
reproducible side-by-side reference comparison script and a source/rights
ledger with specific geometry defects and next acceptance criteria.

See [facial QA and reference ledger](STATUE_OF_LIBERTY_FACE_QA.md). The next
meaningful step is a genuine anatomical sculpt/retopology or a suitable
authorized real-face scan; another round of additive facial blobs is not an
adequate reconstruction method.

## r53 — continuous relief head and sculpted pinnae (opt-in, **experimental**)

The latest procedural trial replaces the separate nose lozenge, eyeball
solids, brow/eyelid tubes, lip rods and auxiliary chin with a single,
closed high-density copper head skin. Overlapping smooth relief fields form
the orbital recesses, eyelid rims, bridge and alae, nostril grooves, mouth
fissure, philtrum, cheek planes and chin. The two ears are now individually
closed, side-facing pinnae with recessed conchae, curved outer helices and
antihelix shelves, rather than an oval wedge and detached helix wire.

The new `--detailed-face` flag is **off by default**. It is a geometric
continuity experiment, **not** a likeness pass: the eyes remain stylized,
the nose/ear proportions are estimated, and facial landmark positions have
not been surveyed from reference photographs. Front, three-quarter and true
profile close-ups are exported for manual rejection or further sculpting.
The previous face remains unchanged in the ordinary `build-liberty` path.

```bash
KUMIKI_RENDER_PROFILE=smoke blender --background --factory-startup \
  --threads 6 --python-exit-code 1 --python scripts/blender_entry.py -- \
  build-liberty --detailed-torch --detailed-tablet --detailed-crown \
  --detailed-face \
  --output .review/landmarks/statues/statue-of-liberty/r53-sculpted-mask
PYTHONPATH=src python3 -m kumiki_blender audit-revision \
  --root .review/landmarks/statues/statue-of-liberty/r53-sculpted-mask
```

**Acceptance remains blocked** on matching the monument's eyes, cheeks,
mandible, ear/hair overlap and head-to-neck transition. Continuous topology
alone does not establish photographic accuracy.

## r48 — restrained crown copper joinery (opt-in, **ongoing**)

The detailed 25-window diadem and seven folded sheet rays now include one
separately editable closed-mesh hardware part: **50 tiny dome-head fasteners**
on the copper window mullions and **14 near the ray mounting shoulders**.
All 64 are volumetric, watertight bosses embedded into the adjacent copper,
not dark texture dots or flat decal geometry. The initial large-boss render
was rejected because it looked like polka dots; the accepted model uses
smaller, shallow heads with restrained material contrast.

These are **illustrative** fasteners. The National Park Service confirms
the statue's 25 crown windows, seven rays and riveted copper construction,
but their precise sizes and locations here are **not survey measurements**:
[NPS statistics](https://www.nps.gov/stli/learn/historyculture/statue-statistics.htm),
[NPS chronology](https://www.nps.gov/stli/learn/historyculture/liberty-island-a-chronology.htm).

```bash
KUMIKI_RENDER_PROFILE=smoke blender --background --factory-startup \
  --python-exit-code 1 --python scripts/blender_entry.py -- \
  build-liberty --detailed-torch --detailed-tablet --detailed-crown \
  --output .review/landmarks/statues/statue-of-liberty/r48-subtle-fasteners
PYTHONPATH=src python3 -m kumiki_blender audit-revision \
  --root .review/landmarks/statues/statue-of-liberty/r48-subtle-fasteners
```

The procedural figure is still **not** a photographic likeness: the head,
eye/mouth anatomy and neck remain the largest unresolved defects. Hardware
detail is kept behind the pre-existing `--detailed-crown` opt-in flag.

### Rejected face reconstruction paths (r42–r46)

- Local subdivide-and-raycast deformation of the miniature scan toward the
  CC0 donor produced eye/mouth folds and extreme facial roughness from
  sampling discontinuities. It was rejected after front/three-quarter QA.
- The 29.2 MB [Metron E candidate](https://zenodo.org/records/10307962)
  matched its published MD5 `10635d07bf02d176bae177db634c3371`, but 12
  orthographic inspection renders revealed a small decorative figure mounted
  on a base, **not** a survey of the monument's head. No geometry was reused.
- Extending the rear donor crop did not improve the visible temple boundary.
  That approach and the continuous-morph attempts were discarded without
  modifying the production asset or retaining failed face modes.

Until a licensable actual-head scan or manually retopologized reference-based
sculpt is available, photo-faithful facial acceptance is still blocked.

## r41 — temple/neck anatomical wrap (**experimental, not approved**)

The front-only donor patch from r23 had a vertical cheek incision and a hard
rectangular neck slab. A separate `--wrap` review now retains more of the CC0
donor's cheek/temple surface, removes the lower flared donor-neck geometry,
registers its exterior depth against the verified bronze capture, and retains
the original scan neck below the chin instead of carving away its polygons.
This is an **anatomical prototype**, not enabled in `build-liberty`.

```bash
blender --background --factory-startup --threads 6 --python-exit-code 1 \
  --python scripts/blender_liberty_anatomy_review.py -- \
  .review/_sources/liberty-bronze.glb \
  .review/_sources/human-base-meshes.blend \
  .review/landmarks/statues/statue-of-liberty/r41-anatomy-wrap \
  --wrap --export
```

The front and three-quarter clay views show a cleaner neck silhouette without
the r23 donor's floating lower flap. However the face/temple edge is still
geometrically obvious in profile, the upper scan neck has some visible facet
transitions, and the donor's eyes, mouth and brow remain generic rather than
reference-matched. **Visual acceptance remains failed.** An independent
3D-nearest-surface projection and a boundary-skirt prototype were compared
and rejected after they introduced forehead/chin spikes; neither was added to
the production generator. This review remains reproducible and reversible.

**Next:** reference-calibrated facial landmarks and continuous head-to-neck
retopology, including proper ear/hair occlusion. Do not promote the donor
surface based on a front view alone; inspect front, three-quarter, and true
profile at the same orthographic scale.

## Perforated copper crown (opt-in, **ongoing**)

`--detailed-crown` rebuilds the headgear as a continuous annular wall with
25 actual window apertures, 25 integral mullions, top and bottom rolled
copper rims, and seven individually closed tapering flat sheet rays.
This removes the legacy 27 bright mullion rods and 27 nearly opaque dark
window tubes. The model preserves a closed, UV-textured, editable shell with
actual exterior/interior surfaces and walls around each opening, visible in
frontal, oblique, and low-profile QA. The seven rays are not round rods; each
has broad copper faces, a raised central fold, narrow side thickness and an
irregular shoulder taper.

An associated forelock pass replaces the comb-like 26 freestanding strand
cylinders with a continuous, densely tessellated, S-shaped copper hair mass
whose drifting waves are surface relief rather than disconnected wires. The
12 synthetic temple-wave rods are also removed; the larger temple hair
volumes remain. This is a more coherent silhouette but not a photomatch to
Bartholdi's unique hairline or an anatomically resolved hair sculpt.

The monument's 25 crown windows and seven rays are documented by the
[National Park Service](https://www.nps.gov/stli/learn/historyculture/statue-statistics.htm).
Exact window aperture radii and ray cross-sections in this module remain
**illustrative**, not obtained from archival fabrication drawings or a survey.

```bash
KUMIKI_RENDER_PROFILE=smoke blender --background --factory-startup \
  --python-exit-code 1 --python scripts/blender_entry.py -- \
  build-liberty --detailed-torch --detailed-tablet --detailed-crown \
  --output .review/landmarks/statues/statue-of-liberty/r34-crown-hair
PYTHONPATH=src python3 -m kumiki_blender audit-revision \
  --root .review/landmarks/statues/statue-of-liberty/r34-crown-hair
```

Head sculpt, curly forelock, anatomically believable eyelids, jaw/neck and
patina fine detail are still not photomatched. Crown remodeling alone does
not pass the full head-likeness quality gate.

## Keystone tablet and cradling left hand (opt-in, **ongoing**)

The `--detailed-tablet` mode replaces the undersized rectangular slab and
free-standing grip pegs with a tapered keystone silhouette, modeled shallow
edge chamfers, attached wrist and sculpted palm, four fingers curling from the
edge onto the front plane, and a thumb returning onto the visible face.
The Roman-date lines now use serif glyphs aligned with the tablet's tilt and
are converted to real GLB geometry. They remain **dark shallow relief**, not
yet physically incised Boolean lettering.

The National Park Service lists the original tablet at **7.19 m tall, 4.14 m
wide and 0.61 m thick** and describes the keystone symbolism. This procedural
sculpt uses an illustrative, uniformly smaller proportion, approximately
4.14 m x 2.36 m x 0.38 m, to match the existing underscale body and head;
it does **not** claim absolute monument dimensions. References:
[NPS statistics](https://www.nps.gov/stli/learn/historyculture/statue-statistics.htm),
[NPS teaching resource](https://www.nps.gov/common/uploads/teachers/lessonplans/Celebrating-a-symbol-FINAL.pdf).

```bash
KUMIKI_RENDER_PROFILE=smoke blender --background --factory-startup \
  --python-exit-code 1 --python scripts/blender_entry.py -- \
  build-liberty --detailed-torch --detailed-tablet \
  --output .review/landmarks/statues/statue-of-liberty/r31-tablet-grip
PYTHONPATH=src python3 -m kumiki_blender audit-revision \
  --root .review/landmarks/statues/statue-of-liberty/r31-tablet-grip
```

The full statue and three new tablet-only cameras are checked frontally and
in profile. This is an *improved sculptural approximation*, not photographic
acceptance of the fingers, calligraphy, copper seams or face.

## Detailed torch and hand review (opt-in, **ongoing**)

The procedural figure's old gallery/flame could not adequately convey the
cast-metal detail in NPS and visitor photographs. A second, opt-in construction
now uses `--detailed-torch` with `build-liberty` to produce:

- A 96-sided fluted, ringed torch neck, separate lower and upper cornices,
  24 pierced-gallery balusters and 12 radial cast braces.
- A closed, irregular gold flame core and eleven individually editable,
  twisted copper-leaf tongues—volumetric geometry rather than flat planes.
- Calibrated nonuniform scale from the existing handle radius and flame
  height, retaining the existing arm/palm and sculpted fingers.
- A more curved grasp with eight-section tapered fingers, distinct swollen
  joints, and a thumb curling under the shaft.
- Copper and gold PBR materials exported with the GLB. Preview angles
  separately inspect the gallery, flame and hand contact; no reference
  photographs are copied into the model.

```bash
KUMIKI_RENDER_PROFILE=smoke blender --background --factory-startup \
  --python-exit-code 1 --python scripts/blender_entry.py -- \
  build-liberty --detailed-torch \
  --output .review/landmarks/statues/statue-of-liberty/r29-torch
PYTHONPATH=src python3 -m kumiki_blender audit-revision \
  --root .review/landmarks/statues/statue-of-liberty/r29-torch
```

This is still a reference-guided **geometric study**, not a monument survey.
The grip silhouette and flame require photo-matching refinement, especially
at the fingertip contact, undercuts and thin sheet folds. The architectural
torch study is separately opt-in because the scan-assisted face remains
visually unapproved; the base figure's facial quality is not improved by it.

## r23 — donor-face edge fitting and classical-eye study (**not approved**)

Comparison of the r22 front, three-quarter and profile renders found the
rectangular forehead slab and independent long donor neck unacceptable.
`--conform` is a separate, opt-in and reproducible experiment. It:

- Preserves the CC0 donor's real eyelid, nose and lip polygons instead of the
  ray-generated `--patch` heightfield, which caused striped eye/mouth holes.
- Crops rear head, slab forehead and most of the donor neck under a curved
  jawline, while feather-fitting outer facial depth toward the bronze scan.
- Narrows oversized eye openings, adds a restrained brow ridge, and deforms
  the paired eyeballs consistently with their lid openings.
- Leaves a smaller opening in the bronze face so the exposed scan-edge
  triangles around the temples are less conspicuous.

After fetching the two checksum-verified sources as shown below, build:

```bash
blender -b -t 6 --python-exit-code 1 \
  --python scripts/blender_liberty_anatomy_review.py -- \
  .review/_sources/liberty-bronze.glb \
  .review/_sources/human-base-meshes.blend \
  .review/landmarks/statues/statue-of-liberty/r23-conformed \
  --conform --export
```

This exports six neutral-clay comparison views, an editable `.blend`, a GLB,
and JSON provenance/measurement metadata. The forehead now approaches the
captured hairline and the eyes are less doll-like. **The visual gate still
fails**: the jaw/neck connection remains two distinct, non-watertight
surfaces, jagged miniature-scan fragments remain at side angles, and the
sculpt is not a faithful photomatch for the historical mouth, crown or hair.
It must not silently replace the final statue asset.

**Next gate:** continuous manual face/head retopology, with reference-registered
eye, mouth, nose and jaw landmarks and a true seam hidden by scanned hair.
Only then should fine patina, garment and tablet detailing proceed.

## Blender Studio CC0 anatomical face checkpoint (r22, **visual gate failed**)

The newest experiment takes a **different geometric foundation** after repeated
non-improving sculpt attempts on a miniature scan with almost no usable eyelid
or lip topology. It extracts Dan Ulrich's **realistic sculpting head and
separate eye geometry** from Blender Studio's official *Human Base Meshes
v1.2.0* (**CC0 1.0**), scales and reshapes it, and stages it against the
licensed Jerryfisher bronze miniature. A carefully scoped cutter removes
1,665 of the miniature's blurred facial polygons, leaving its forelock,
crown, arm, neck and garment present. The donor has independent, subdivided
eyelid rims, nostrils, lips and jaw—no freestanding decorative tubes.

**This is a prototype, not yet a resemblance match.** The neutral-clay front,
three-quarter and profile pictures demonstrate remaining failures. In
particular, the donor's temple/jaw/neck boundary is visibly disconnected in
profile, the forehead-to-hair edge is unnaturally straight, the eyes are too
generic and open, and the lip separation and proportions lack Bartholdi's
classical character. Thus the new donor has better **feature topology** but
does **not** establish a better all-angle likeness than the r15 scan. It is
kept as a separate face experiment rather than quietly replacing the main
statue asset.

### Reviewed source ledger

| ID | Source | Evidence / rights | Use |
|---|---|---|---|
| F1 | User-supplied front, oblique and profile close-ups | Direct observed photographic reference; reconstruction reference | Gaze, nose, chin, hairline, eyebrow, cheek planes |
| F2 | [Jerryfisher, bronze miniature](https://sketchfab.com/3d-models/statue-of-liberty-bronze-model-5b9aa58d51624f01b9009b52379bf18a) | CC BY 4.0, verified binary | Existing complete statue, clothing and hair mass; **not** a monument survey |
| F3 | [Dan Ulrich / Blender Studio, Human Base Meshes](https://www.blender.org/download/demo-files/) | CC0 1.0, official Blender demo asset, verified bundle and extracted file SHA-256 | Anatomical head, sculptable topology, sclera and iris bases; **not** a statue likeness |
| F4 | [NPS laser scanning documentation](https://www.nps.gov/articles/laser-scan-lady-liberty.htm) | Monument survey described, no reusable mesh authorized | Historical workflow context only; source not imported |
| F5 | [Museum face replica by avredu](https://sketchfab.com/3d-models/statue-of-liberty-face-model-in-the-museum-1a09198686344d279a8220cb7bdb7573) | CC BY-NC-SA; different rights and capture subject | Public reference preview only; **not** imported or redistributed |

The anatomy bundle's exact 49 MB ZIP and 48 MB `.blend` hashes, creator, URL
and CC0 terms are locked in `anatomy_source.json`. Raw donor assets remain in
the local cache rather than the git repository. Both artists' source details
are also embedded in the exported GLB node metadata.

### Generate the inspectable anatomy experiment

```bash
python3 scripts/fetch_liberty_scan.py --output .review/_sources/liberty-bronze.glb
python3 scripts/fetch_liberty_anatomy.py --output .review/_sources/human-base-meshes.blend
blender -b -t 6 --python-exit-code 1 \
  --python scripts/blender_liberty_anatomy_review.py -- \
  .review/_sources/liberty-bronze.glb \
  .review/_sources/human-base-meshes.blend \
  .review/landmarks/statues/statue-of-liberty/r22-anatomy-checkpoint \
  --carve --export
PYTHONPATH=src python3 -m kumiki_blender audit --model \
  .review/landmarks/statues/statue-of-liberty/r22-anatomy-checkpoint/anatomy-checkpoint.glb
```

Use a fresh output directory on each run. The script also accepts `--patch`
to experiment with a continuous scan-to-donor heightfield, but multi-angle
QA rejected that alternative because ray-sampled eyelids and mouth produced
visible sawtooth artifacts. `--quick` renders just front/three-quarter views.
The current r22 build exports 10 named mesh nodes and one portable material,
with zero external GLB references; no claim of watertight facial topology is
made.

**Next structural gate:** build a fully closed, sculpted face/head with
photographic landmark calibration, resolve the eye and lip anatomy in neutral
clay, and weld the side/head/neck seams in true topology. Do not add weathering,
garment detailing or a final patina until the face passes front, three-quarter
and profile reviews. The `.blend` remains an editable starting point.

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
quarter angles, low-angle, rear hair and high crown inspection. The second
detail pass adds **adaptive mesh subdivision only to the captured forelock**,
almond upper/lower eyelid carvings, iris relief, nasal alae and nostril seams,
lip/philtrum shaping and swept hair striations. It also introduces a reusable
tileable `verdigris-copper` PBR material with packed color/normal/roughness
maps. Two macro views now isolate eyes and mouth for close quality inspection.
The source
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
| Deeply defined upper/lower eyelids, eyeballs and tear ducts | Geometric lid creases and eye relief added; not yet a photo-matched sculpt |
| Forelock with separated, swept strands and sculpted temple curls | Fine comb-direction relief added; major individual locks still missing |
| Wide, carved nostrils, sharply defined lips and chin planes | Restrained relief, with face topology preserved because the miniature scan is too noisy |
| Perforated crown band and individually legible radial rays | Seven rays survive; the crown's window openings are not faithful |
| Surface seams, tool marks and differentiated oxidation | Tileable verdigris PBR improved; still missing authentic seams and erosion |

We also reviewed **jerryfisher's alternate 4-million-triangle scan**
(<https://sketchfab.com/3d-models/statue-of-liberty-6626abe1f3e8469a9d8f4b74d8aa2a71>,
CC BY 4.0). It scans a *different miniature* with more polygons but visibly
noisier, less legible facial relief. Increasing tessellation alone cannot
restore anatomical features absent from a scan, so it was rejected as the
replacement source. Neither source is misrepresented as an NPS survey.
We also reviewed Chris Mcardle's downloadable CC BY head-only photogrammetry
(<https://sketchfab.com/3d-models/statue-of-liberty-head-f072ed04dcb14fb4b37e773f407ff603>).
It has substantial reconstruction artifacts and low-resolution facial topology,
so was rejected as a donor. No third-party asset besides the approved
jerryfisher scan enters the export.

The local QA iterations discarded an aggressive displacement attempt (it
produced mechanical grooved hair and slit-like eyelids) and a depth-smoothing
attempt (which created surface defects). The final sculpt uses restrained
relief and the original facial topology; it does **not** pass the definitive
likeness gate.

**Next sculpting gate:** landmark-align higher-resolution head references,
retopologize the face and eyelids manually in Blender, add real engraved lip,
nostril and hair geometry, then bake displacement/normal detail and compare
the identical eleven camera views. Preserve the source mesh and credits so
every new sculpted change remains auditable. Full robe/tablet/torch/pedestal
reconstruction stays blocked on accepting the facial likeness.

### Build the first face checkpoint

```bash
python3 scripts/fetch_liberty_scan.py --output .review/_sources/liberty-bronze.glb
KUMIKI_RENDER_PROFILE=smoke blender --background --factory-startup --python-exit-code 1 \
  --python scripts/blender_entry.py -- \
  build-liberty-scan-face --source .review/_sources/liberty-bronze.glb \
  --output .review/landmarks/statues/statue-of-liberty/r15
PYTHONPATH=src python3 -m kumiki_blender audit-revision \
  --root .review/landmarks/statues/statue-of-liberty/r15
```

Remove `KUMIKI_RENDER_PROFILE=smoke` for full-resolution facial review. The
revision is immutable and contains an editable `.blend`, GLB, eleven views,
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
