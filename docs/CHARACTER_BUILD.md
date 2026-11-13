# Character build: reference images to a fitted, rigged model

`kumiki-character-build` turns a checked set of orthographic character
reference images into 3D data. The data includes a blockout, a silhouette fit
of any mesh, vertex colours, feather or fur cards, quad retopology, a rig, and
part assembly. Each stage measures its result against the approved images.

The reference images come from the `character-modeling-references` skill.
The skill names the views and gives the prompts.
[Character fidelity QA](CHARACTER_FIDELITY_REVIEW.md) stays the detailed
part-by-part review. This tool writes a manifest for it.

## Install

```bash
python -m pip install -e '.[character-build]'
kumiki-character-build --help
```

The `blender-setup`, `retopo`, `rig` and `assemble` commands need Blender 4.2
or newer. The other commands need only NumPy, SciPy, scikit-image and Pillow.

## Pipeline

| Step | Command | Output |
|---|---|---|
| 1. Name the images | `ingest SRC map.json REFS` | Copies with view names: `front.png`, `left.png`, `front_left.png`, ... |
| 2. Check them | `check REFS` | `REFS/check/report.md`, mirror overlays, hull overlays |
| 3. Make 3D data | `handoff REFS --height-m 0.35` | `REFS/handoff/`: aligned views, masks, `blockout.obj`, `spec.json`, generator inputs |
| 4. Fit a mesh | `fit HANDOFF mesh.glb --output fitted.ply` | The mesh, deformed to match every view |
| 5. Colour it | `project HANDOFF fitted.ply --output coloured.ply` | Vertex colours from the views |
| 6. Add cards | `scatter coloured.ply --colour 94,70,58 --output feathers.obj` | Feather, fur or scale cards on one colour region |
| 7. Retopology | `retopo fitted.ply --output retopo/` | `low.obj` (quads, UVs), `low_normal.png`, `low.glb` |
| 8. Joints | `joints HANDOFF fitted.ply --limbs wings --output joints.json` | Editable joint positions |
| 9. Rig | `rig retopo/low.obj joints.json --output rig/ --render-rom` | `rigged.glb`, range-of-motion report and renders |
| Any time | `compare HANDOFF model.glb` | IoU, missing, extra and height error for each view, with overlays |
| Parts | `assemble plan.json --output asm/` | Parts placed with anchor pairs, seams measured |
| Generators | `place HANDOFF a.glb b.glb --output ranked/` | Repaired, solid, oriented candidates ranked by IoU |
| Generators | `solidify mesh.glb --output solid.ply` | A hollow or badly wound mesh rebuilt as one solid |
| Generators | `swap BASE DONOR --region "y>0.05" --output sw/` | One region taken from another generation (`--ellipsoid`, `--align-region`) |
| Texture | `texture HANDOFF retopo/low.obj --output base.png` | UV texture baked from the full-resolution reference views (sharp clothes, buttons, fur painting) |
| Texture | `appearance HANDOFF retopo/low.obj base.png --output app/` | Colour agreement per view (median delta E) with side-by-side images |
| Texture | `conflicts HANDOFF retopo/low.obj --output conflicts/` | Where the views overlap and disagree on the mesh, with pictures and advice |
| Texture | `parts HANDOFF retopo/low.obj --output parts/` | The character's parts, from shape and colour, on the mesh and in each drawing |
| Texture | `texture-parts HANDOFF retopo/low.obj parts/ --output base.png` | Each part baked from its own pixels, with smooth fill and polished seams |
| Texture | `repaint HANDOFF retopo/low.obj base.png --output rp/` | Pictures of the angles between the drawings, repainted from renders by a local image model |
| References | `trim REFS front --output front.png` | A view without the part that its opposite view has no room for |
| Fit | `fit HANDOFF mesh.ply --lattice auto` | Lattice deformation for detailed generator meshes. A finer lattice is kept only if it does no harm |
| Texture | `retopo fitted.ply --colour-from ranked/x.placement.json` | The generator's own texture baked as `low_basecolor.png` |
| Fur | `fur-weights coloured.ply --clothing-top-m 0.2 --eye-band-m 0.21,0.27` | Per-vertex fur weight from the drawings' colours (no fur on clothing or eyes) |
| Fur | `fur retopo/low.obj --basecolor low_basecolor.png --weights w.ply --systems s.json` | Strand fur with Blender's procedural hair assets |
| Blender | `blender-setup HANDOFF --output refs/ --save` | Reference cameras, image empties and blockout in a `.blend` |

Each command writes a new file or folder. It does not overwrite an output.

## Multi-view route (best results so far)

Hunyuan3D-2mv makes one shape from the front, left, back and right views together, so the
face and the tail come from one consistent shape. On Pip it beat every single-image
TRELLIS.2 candidate.

1. **Shape.** `scripts/hunyuan_mv_shape.py` with `handoff/raw/{front,left,back,right}.png`
   (about 30 minutes on an M4: diffusion on MPS in fp16, decoding on MPS in fp32).
   If the character is symmetric and has no view of one side, mirror the other side.
2. **Place.** `place HANDOFF shape.glb --output ranked/`.
3. **Fit through a lattice.** `fit HANDOFF ranked/shape.ply --no-normalise --lattice auto`.
   A coarse lattice changes proportions and keeps the generated detail. A finer
   lattice always raises the IoU. If the views disagree, it does so by bending the
   mesh between them. `auto` fits through 4,4,8 and then 8,8,16. It keeps a stage
   only if the stage adds no folds (`--max-new-folds`). Also, at most 3% of the
   edges may change length by more than 1.5 times (`--max-distortion`). The report
   lists each stage. On the second Pip set, 8,8,16 raised the IoU from 0.88 to 0.89.
   But it turned the tail into a flat slab (8% of the edges). `auto` stopped at 4,4,8.
4. **Retopology.** `retopo fitted.ply --output retopo/ --target-faces 20000`.
5. **Texture.** `texture HANDOFF retopo/low.obj --output base.png --size 4096`.
   One view paints each region of the surface, and the tone is leveled where two
   views meet. See [Conflicts between the views](#conflicts-between-the-views).
6. **Measure.** `compare` for silhouettes, `appearance HANDOFF low.obj base.png` for
   colour (median delta E per view and side-by-side images).
7. **Rig.** `joints` and `rig` as before.

Pip, multi-view route (2026-11-13): silhouette IoU 0.93 to 0.95 in all six views.
Median delta E: 3.8 (front), 4.4 (back, right), 7.1 (left), 8.6 to 9.1 (3/4 views).
The 3/4 drawings disagree most with the others, so no single texture matches them closely.

Quill, multi-view route with the right side mirrored from the left (`texture
--assume-symmetric`): silhouette IoU 0.90 to 0.95. Median delta E: 1.1 (back), 2.1
(front), 2.9 (left), 9.2 (front 3/4). 15-bone wing rig, all vertices heat-weighted.

## Conflicts between the views

AI drawings of one character do not agree, and no mesh matches all of them. A
texture from several drawings shows each disagreement. `conflicts` finds them, and
`texture` corrects what it can:

```bash
kumiki-character-build conflicts HANDOFF retopo/low.obj --output conflicts/
```

It writes `report.json` and `turnaround.png`: the texture, the view that paints each
area, and the conflicts, from eight sides. `owners.png` and `conflicts.png` show the
same in the UV layout.

| Conflict | Colour | Cause | What `texture` does |
| --- | --- | --- | --- |
| Tone seam | blue | Two drawings differ in tone where they meet. | A smooth colour offset makes the two sides meet. It fades with distance from the seam. |
| Misplaced part | red | The mesh is wider or narrower than a drawing, so a part lands on the wrong surface. | The drawing does not paint there. The area takes the colour around it. |
| Contested feature | yellow | Two drawings show different things at the same place. | One drawing paints each patch whole. |
| Unseen | grey | No view sees the area. | The area takes the nearest painted colour. |

The report gives the share of the surface for each conflict. It also lists each pair
of views with the surface that both see (`overlap`) and their tone difference
there. `different` is the share where they show different things. `views_that_face` names cameras
(`az<deg>_el<deg>`) that look at the faults. Use them for a new drawing or for
`repaint --views`. `advice` says what to do in plain sentences.

How the bake decides:

- **Regions, not texels.** The view that sees the neighbourhood of a point best
  paints it (`--region-px`, default 24). On modelled fur the better-facing view
  changes with every bump. A choice for each texel mixes the drawings in streaks.
- **Trust.** A view counts less near its own outline (rim light) and near a place
  where one part hides another (`--occlusion-px`).
- **Objections.** A drawing that sees a point can object to a colour there. It
  objects if it shows something else at that point and nothing like the colour
  within `--slack-px` (default 60). The slack matters. Two drawings put the same eye a little apart, and that is no
  objection. Comparisons count lightness less than hue, because tone is what differs
  most between drawings.
- **Leveling.** `--level-reach-px` sets how far an offset reaches (default 150).
  A surface that its view sees squarely keeps its colour. `--no-level` turns it off.

Results (2048 px, colour difference between the two sides of a seam, in delta E):

| | Seams before | Seams after | Misplaced | Contested | Unseen |
| --- | --- | --- | --- | --- | --- |
| Pip, four drawings and two repaints | 11.4 | 3.0 | 1.6% | 0.9% | 10.8% |
| Quill, three drawings, mirrored | 14.4 | 4.0 | 0.7% | 2.0% | 22.2% |

The cost is a small tone shift against each drawing. On Pip the median error of the
front view is 4.6 delta E with leveling and 4.2 without it. The earlier bake, which
chose a view for each texel, reached 3.5 but showed streaks on the tail.

A feature that two drawings put in two places on the surface shows twice where each
drawing paints alone. Pip's left eye was 60 px further round the head in the side
drawing. No rule on the drawings removes that. A picture that faces the feature
does: see the next section.

## Check every new texture against the original art

After baking or polishing a texture, run
[audit-texture-revision](CHARACTER_TEXTURE_REVISION_GATE.md) with the original
and candidate texture files on the **same** registered OBJ/UV layout.
It checks every requested source view independently, refuses missing aligned
images and retains panels on numerical regressions. A lower mean color error
alone does not override a worse 90th-percentile or color-coverage result.

## Recognise the parts, then bake each part from its own pixels

The rules above compare colours. They do not know what a part is, so some conflicts
stay. Examples are a sliver of vest on the edge of the tail, and a seam that still
shows on close view. `parts` recognises the parts from the shape and from the colours. Then
`texture-parts` bakes each part only from the pixels that show that part.

```bash
kumiki-character-build parts HANDOFF retopo/low.obj --output parts/
kumiki-character-build texture-parts HANDOFF retopo/low.obj parts/ --output base.png --diagnostics d/
```

Give both commands the same sources: `--views`, `--assume-symmetric` and `--extra`.
A side without a source has no colours, and `parts` cannot recognise it.

**Source-locked part masks:** `parts.json` records a SHA-256 signature of the
source handoff configuration, every selected original/reference image, any
supplementary repaint, source-mesh vertices/faces and UV corner coordinates.
`texture-parts` validates the same signature **before** it bakes: a new
character mesh, changed raw drawing, modified repaint, altered UV layout or
different camera selection requires a **new `parts` revision**. It refuses
stale masks rather than silently painting a scarf onto the tail or moving an
eye to the wrong surface. The texture resolution can still change between
part identification and the final bake. A pre-existing folder containing
part labels is never overwritten; use a new output folder for each revision.

How `parts` works:

1. **Shapes, from geometry.** The tail, the wings and the ears meet the body in
   concave creases. The mesh is smoothed, then flooded from its most convex places.
   Two areas stay apart if the border between them bends inward more sharply than
   `--crease-radius-px` (default 60). On Pip this gives two shapes: the tail and
   the body. No colour is used.
2. **Parts, from colour.** A sharp border makes two parts. A different colour does
   not: fur changes from cream to brown without an edge, and a vest ends with one.
   Two touching patches join if the colour step across their border is below
   `--contrast` (default 10 delta E). Patches below `--min-share` join a neighbour.
3. **Materials.** Parts with almost the same mean colour are one material
   (`--material-delta-e`). A strap cuts a vest into two parts, and both are vest.
4. **Shape against colour.** The colours come from a bake. Where the mesh is wider
   than a drawing, the bake puts a part on the wrong shape. A material on less
   than `--bleed-share` of a shape (default 5%), with most of its area on other
   shapes, is such a bleed. Its area goes to the material around it on that shape.
   On Pip this found the scarf on the edge of the tail (1.5% of the tail).
5. **Parts in each drawing.** The parts of the mesh are projected into each drawing.
   A pixel moves to a different part only if it shows a main colour of that part
   and that part is near (`--slack-px`).

Output:

- `parts.json`: parts, materials, bleed, and the agreement between mesh and drawings.
- `uv_labels.png`, `uv_shapes.png` and a mask for each part in each view.
- `turnaround.png`: the first bake, the parts and the shapes from eight sides.
- `semantic.json`: the same masks as a semantic policy for `resolve-conflicts`.

The parts are estimates. Look at `turnaround.png` before you use them. A real part
that crosses a crease on a small area, such as the end of a sleeve, is also taken
for a bleed. Set `--bleed-share 0` if that is wrong for your character.

How `texture-parts` works:

- **One bake for each material.** Each source image is masked to the material. A
  fur texel can take only fur pixels. A texel that no view sees is filled only from
  its own material.
- **Smooth fill.** The tone of an unseen area runs smoothly between the painted
  areas around it. The plain bake copies the nearest colour, which gives flat
  patches with hard borders.
- **Polish.** Each texel near a seam gets a small colour offset. The offsets close
  the colour step across the seam and fade over `--polish-reach-px` (default 12).
  The grain of the texture stays. No offset crosses from one material to another,
  so the edge of a scarf on fur stays sharp. `--no-polish` turns both steps off.
- **UV islands.** Texels on two sides of an island border are neighbours for the
  fill and the polish, so a seam is not cut there.

The report gives the seam contrast: the colour step across a seam, divided by the
step between texels beside it. At 1.0 a seam is no stronger than the grain.

| | Seam contrast, plain parts bake | After polish | Bleed found |
| --- | --- | --- | --- |
| Pip, second set, with three repaints | 3.2 | 1.4 | scarf on the tail |
| Quill, three drawings, mirrored | 4.7 | 1.7 | none |

Median error against the drawings stays about the same (Pip front 4.5 to 4.7 delta
E, Quill front 2.6 to 3.2).

Limits:

- Shapes come from valleys in the surface. A limb that joins the body in a saddle
  (a neck, an arm) is not separated.
- A feature that two drawings put in two places still shows twice. Pip has a second
  eyelid line above the left eye at 45 degrees. A picture from that angle removes it.
- A part that the mesh shapes differently from the drawings is only kept clean. To
  remove the cause, correct the shape. On Pip, a sliver of vest stays on the left
  edge of the tail in the back view. The border of the tail shape lies beside it.

## A view that shows a part the others have no room for

When you replace a view, the old opposite view can still show the old design. Pip's
new back view has a narrow tail. The old front view showed the old tail as two large
shapes beside the head. A front and a back view of one object have mirror outlines,
so `check` fails ("back is the mirror image of front": 0.75).

```bash
kumiki-character-build trim refs/ front --output refs2/front.png
```

`trim` cuts from the view each thick part that lies outside the mirror outline of
the opposite view (`--against` names a different view to trust). A small difference
in the outline stays: `--slack-px` (default 15) and `--min-thickness-px` (default
40) set the limits. On Pip it removed 11% of the front figure. The mirror agreement
rose to 0.84, and the silhouette that the other views cannot explain fell from 16%
to 5%.

The better fix is a new drawing of that view. `trim` cannot change the part of the
drawing that stays: Pip's front view keeps its bare arms, and the new views have sleeves.

## Repaint the angles between the drawings

More drawn angles do not help. Each new AI drawing is one more independent guess. On
Pip the two 3/4 drawings disagreed with the front and side drawings. With them, the
side view error rose from 4.1 to 7.0 delta E. A texture from the front, back and side
drawings only is accurate in those views. But it shows the eyes twice at 45 degrees,
because the drawings place them slightly differently on the shape.

`repaint` takes the in-between angles from the model instead:

1. `texture HANDOFF low.obj --views front,back,left,right --output axis.png`
2. `repaint HANDOFF low.obj axis.png --subject "the cartoon squirrel" --output rp/`
   renders the model from the four 3/4 directions and from 35 degrees above the four
   sides (free cameras are named `az<deg>_el<deg>`). A local FLUX.2 [klein] in ComfyUI
   repaints each render in the style of the nearest drawing. A repaint counts only
   where its outline agrees with the render. The editor often redraws a tail. The
   area around that change is masked out, and the face beside it is kept.
3. `texture HANDOFF low.obj --views front,back,left,right --extra rp/manifest.json --output final.png`

Each drawing keeps the surface that it faces. The repaints take the angles between
the drawings and the tops of the head and tail, which no drawing shows.

On Pip, the repaints from the two front 3/4 directions removed the doubled eyes.
Median error against the drawings: front 4.6, back 5.0, left 5.9, right 6.7 delta E.

Two rules keep the drawings in charge:

- Each repaint gets the tone of its render before the bake. The editor adds its own
  shading, and Pip's repaints were 5 to 10 L* darker than their renders. The
  correction is smooth and local. It applies only near pixels that a drawing painted
  in the render. On a side that no drawing shows, the render is a fill colour.
  For such a side, give `repaint` the owner map of the base texture:
  `texture ... --diagnostics d/` and then `repaint ... --owners d/owners.png`.
  Without it, a fill colour made the repainted side of Pip's face pale.
- Only drawings are witnesses in the objections and in the agreement score. The
  repaints all come from one render, so three of them would outvote one drawing.

A repaint scores 0.8 times a drawing.

A repaint can keep the outline and still draw something else. On Quill, the repaint
from above the front lost the glasses. `texture --screen-extra 0.5` measures that. It
bakes each repaint alone on top of the drawings. If the mean error of a drawn view
rises by more than 0.5 delta E, it drops the repaint.

The mean shows a small change such as lost glasses, which the median can miss. With
`--assume-symmetric`, a repaint of the side without a drawing is judged by its mirror
twin on the drawn side.

The screen is a measure, not a verdict. On Quill it dropped all eight repaints. The
three drawings, mirrored, already look right from every angle. On Pip it kept four
and dropped four.

One of the dropped repaints is the one that removes the second
left eye (+1.0 delta E). A repaint that removes a duplicate must differ from the
drawing that showed the duplicate. Look at `turnaround.png` before you drop it.

Notes:

- Start ComfyUI with `--disable-smart-memory` on a 24 GB Mac. The diffusion model and
  the text encoder together take 15 GB. One 768 px view takes about 15 minutes.
- If the batch stops, run the same `repaint` command again with `--resume`. It keeps
  each finished repaint whose render and settings did not change, and repaints the
  other views.
- Prompt wording matters. With "illustration" the editor drew a flat outlined
  cartoon. Starting from the render at partial denoise (0.7) kept the doubled eye.

## Image-to-3D route

A shape from silhouettes knows only the outlines. Ears, a snout and a tail
volume need an image-to-3D generator. The pipeline then places, scores,
corrects, colours and rigs the generated mesh.

1. **Generate.** [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) runs on
   Apple Silicon through [trellis-silicon](https://github.com/sanchez-kim/trellis-silicon)
   (MPS and Metal). Give it the aligned views with a transparent background
   (`handoff/aligned/<view>.png`), one view per run. A 3/4 view works best:
   on Pip, front-left scored 0.85, right side 0.80 and front 0.69.
2. **Rank.** `place HANDOFF a.glb b.glb ... --output ranked/` repairs each
   candidate (UV seams, face winding, hollow shell) and finds its facing in
   3 degree steps. The side views tell the front from the back. Then it scores
   each candidate against every view. Run several seeds and keep the best.
3. **Correct.** `fit HANDOFF ranked/best.ply --no-normalise --smooth-radius-px 40`.
   A large smoothing radius changes proportions only and keeps the generated
   detail. The default radius (4) crumples a detailed mesh.
4. **Finish.** `project`, `joints`, `retopo` and `rig` as for a blockout.
5. **Optional part swap.** `swap BASE DONOR --region "y>0.05" --region "z>0.04"`
   aligns the donor to the base with a two-way similarity ICP outside the
   region, then cuts and fuses closed solids. It works when a box separates the
   part. It does not work when the part wraps around the body.

6. **Texture.** `retopo --colour-from ranked/<name>.placement.json` bakes the
   generator's own texture onto the low mesh. It is sharper than colours
   projected from the drawings. The bake searches 2 cm from the surface, so it
   fails where the fit moved the surface further.
7. **Fur.** `fur-weights` marks clothing and eyes from the colours projected
   from the drawings. A generator texture often paints clothing in fur colours.
   `fur` then grows strands with Blender's hair node assets (optional combing,
   Cycles render). Each strand takes one colour from its root, so ungroomed
   strands read as a mosaic. On Pip the painted fur in the texture looked
   better. Treat strand fur as a starting point for manual grooming.

TRELLIS.2 exports a hollow, inconsistently wound shell, so `place` and
`solidify` rebuild it as one solid. On a 24 GB Mac the 512 pipeline peaked at
26 GB (memory compression) and took about 5 minutes per mesh.

Setup notes for trellis-silicon:

- Install Eigen (`brew install eigen`) and build `o_voxel` with
  `CPLUS_INCLUDE_PATH=/opt/homebrew/include/eigen3`, or it falls back to a
  slower texture baker.
- TRELLIS.2 loads the RMBG-2.0 background remover at start, which is gated and
  licensed for non-commercial use. Inputs with a transparent background never
  use it. In `TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py`, wrap the
  `rembg_model` load in `try/except OSError` and set it to `None`.

## Views and frame

- View names are camera positions: `front`, `front_left`, `left`, `back_left`,
  `back`, `back_right`, `right`, `front_right`. `three_quarter` is an alias of
  `front_right`, and `three_quarter_back` is an alias of `back_left`.
- Axes are Blender axes. Z is up, the character faces -Y, and its right side is at -X.
- The frame is 1600 x 1200 px. Each view is scaled so the figure is 1000 px
  tall, stands on row 1100 and is centred on column 800. The `REF_CAM_<view>` cameras
  from `blender-setup` render exactly this frame. Diagonal views are shifted to
  the camera frame by the offset that the hull carve measures.
- Blender's numpad 3 "Right" view shows the character's left side.

## What each check means

- **Mirror checks.** In orthographic projection the back outline is the mirror
  image of the front outline. A failure means perspective, a changed pose or a
  redrawn part.
- **3D agreement.** The tool carves a voxel hull from all full-body views. For
  each view it reports the fraction of the silhouette that the hull cannot
  explain. A high value means that no single 3D shape can make all the views.
  The front and the primary side carve exactly. The other views get a
  tolerance of 1% of the height.
- **Direction.** A diagonal view warns only when another diagonal direction
  fits clearly better. Front and back 3/4 silhouettes are often near mirror
  images, so a small difference decides nothing.

## Stages in detail

- **Blockout.** The tool blurs each view's 2D signed distance field by 12 px.
  The blur removes the wobble of fur and feather fringes. It extrudes each field
  along its camera and intersects them with a smooth maximum. The smooth maximum
  (6 px) rounds the edges where views meet. Floating pieces with less than 5% of the body's area are dropped. The
  blur shortens thin tips (ears, tufts) by about 2% of the height. The fit
  restores most of it. Options: `--outline-blur-px`, `--round-px`.
- **Fit.** Vertices outside a reference silhouette move in. Outline vertices
  inside it move out. The targets use the same blurred outline as the
  blockout, so the fit does not groove the surface row by row. The front and
  primary side have double weight. Only the displacement is smoothed. A flip
  guard cancels each move that turns a triangle more than 90 degrees from its
  start. A final local smoothing removes each crease sharper than 90 degrees. The
  report gives the remaining folds (0 on both test characters). For a
  generator mesh, the tool chooses the facing (four quarter turns) and scales
  the mesh to the reference height first.
- **Colours.** A vertex takes colour only from views where it is visible. The
  test uses a depth buffer with a slope-aware tolerance. Each view is weighted
  by (n . d)^4, and 3/4 views count 0.1, so they fill only what the front, side
  and back views see poorly. Otherwise an AI 3/4 view paints its own face
  beside the front view's face (ghost eyes). `--assume-symmetric` fills a missing side view with the
  mirrored opposite side. Do not use it for a one-sided accessory.
- **Relief.** `--relief-mm` moves vertices by the high-pass of the projected
  `*_clay` views. This is stylized relief from shading, not measured geometry.
  Flat clay gives no relief.
- **Retopology.** QuadriFlow refuses a mesh with folded triangles, sliver
  triangles, floating debris or more than about 250k faces. It gives one
  misleading "not manifold" message for all of them, and its result also
  depends on the mesh's absolute size. Large meshes are decimated to 150k faces
  first. The quads are then relaxed and shrink-wrapped onto the high mesh. The worker removes
  debris, merges vertices closer than a tenth of the median edge, tries
  QuadriFlow at several sizes, then at a voxel-remeshed copy. If every try
  fails, a voxel remesh sized to the target face count makes grid-aligned
  quads. The report then warns that they have no edge loops.
- **Joints and rig.** The spine follows the body midline, which can differ
  from x = 0 when an accessory hangs on one side. Legs are found above the
  feet. Arms or wings are the lateral mass. Tail joints sit at centroids of
  distance bands from the tail root, so they follow a fanned or curled tail. Landmark heights come from
  `landmarks.json` (`crown`, `face_bottom`, `shoulder`, `hip`, `knee`, in % of
  height from the top). The rig uses heat weights. Vertices without a weight
  get their nearest bone, and the report gives that fraction. Influences are
  limited to 4 for glTF.

## Results on the two reference sets (2026-11-12)

| | Quill | Pip, first set | Pip, remade with the updated skill |
|---|---|---|---|
| Views | front, back, left, front_left | front, back, left, right, front_left, back_left | the same six views |
| Back mirror | 0.963 | 0.81 FAIL | 0.935 |
| Left / right mirror | | 0.82 FAIL | 0.905 |
| 3D agreement (worst view unexplained) | 3% | 13% FAIL | 5.7% WARN (left drawn slightly larger) |
| Blockout IoU | 0.944 | not built | 0.940 |
| Fitted IoU, folds | 0.955, 0 | | 0.959, 0 |
| Retopology | 11k quads, QuadriFlow | | 11k quads, QuadriFlow |
| Low-poly IoU | 0.952 | | 0.957 |
| Rig | 15 bones, all heat-weighted | | 18 bones with a 3-bone tail, all heat-weighted |
| Worst range-of-motion volume change | 14% (wing into the body) | | 43% (tail bent 45 degrees into the back) |

Remade Pip through TRELLIS.2 used the front-left view, seed 42, the best of
four seeds. Results:

- Placed IoU 0.848, fitted IoU 0.880, height error -0.4%.
- 87% of the surface coloured from the views.
- QuadriFlow retopology with 13k quads, low-poly IoU 0.877.
- An 18-bone rig with every vertex heat-weighted.

All four seeds read the fanned tail as two lobes. The fit narrows the gap.

The first Pip set drew a fanned tail in the front view and a narrow tail in
the back and side views. The remade set used the skill's volume lock and
turnaround strip, and the views agree.

## Limits

- A shape from silhouettes cannot place outline detail in 3D, and a part seen
  only from the front (a wing) becomes a plate. Heads come out boxy with few
  views. For organic detail, generate a mesh from `handoff/gen/` with an
  image-to-3D tool and `fit` it.
- With few views the hull is boxy in diagonal directions. Each extra
  orthographic view (`front_right`, `back_right`, ...) removes a corner.
- The fit cannot satisfy views that disagree. It trades between them. Anchor
  weighting keeps the front and side close.
- Joint estimates are heuristics for biped creatures. Read `joints.json` and
  correct it before `rig`.
- The limits in `checks.py` are start values from one synthetic set and two
  AI-generated sets.

## Tests

```bash
PYTHONPATH=src:tests python -m unittest tests.test_character_build
PYTHONPATH=src:tests python scripts/smoke_character_build_blender.py   # needs Blender
```
