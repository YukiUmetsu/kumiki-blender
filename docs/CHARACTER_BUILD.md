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
| Parts | `split-views HANDOFF --part tail --output split/` | The drawings split into one part and the rest, each a handoff in the character frame |
| Parts | `place-part split/tail tail.glb --output tail.ply` | A part that was modelled alone, placed in the frame of the whole character |
| Parts | `split-eyes eyes/eyes body.ply --from-part --output balls/` | One textured eyeball for each eye that `split-views` took out, the white included |
| References | `redraw-view front.png back.png --change "..." --output front2.png` | One drawing changed to the design of another |
| References | `borrow-view HANDOFF --view front --from back --polygon front:... --output h2/` | A region of one drawing taken from the mirror image of the opposite drawing |
| Parts | `box-part HANDOFF --name arm --polygon back:... --output arm/` | A part split off by boxes or outlines, without the image editor |
| Parts | `detail-part arm/arm --part "the hand" --output arm_detail/` | Larger, sharp drawings of a small part, in the same frame |
| Parts | `cut-part body/low.obj arm/arm --output body_cut.obj` | The body without the faces that a separate part replaces |
| Parts | `mirror-part arm/low.obj --output arm_R.obj` | The same part on the other side of the character |
| Parts | `smooth-part body/low.obj flaps/flaps --output body_s.obj` | A crumpled thin sheet smoothed under the masks of a part |
| Parts | `seat-part body.ply bag/low.obj --metres-per-px 0.00032 --output bag.obj` | A part moved out of the body that it hangs on, as one rigid piece |
| Parts | `join-parts a/low.obj:a.png b/low.obj:b.png --output joined/` | One mesh and one texture from placed, textured parts |
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

## Bake with automatic source QA

Use texture-parts with --qa-baseline approved_previous_texture.png to invoke
the fixed-geometry source registered color gate automatically. Failed candidates
remain editable and receive a failing exit status and a per-camera QA receipt.
The old behavior remains unchanged without --qa-baseline. See
[the texture revision QA guide](CHARACTER_TEXTURE_REVISION_GATE.md).

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
  patches with hard borders. The grain of the area is copied from painted texels
  of the same part, one shift away. Real fur, moved, is still fur. The top of a
  head and the far side of a tail then look like the fur beside them.
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

Use this bake for each part of a character that was modelled in parts too. On
Pip, the plain bake left flat patches on the far side of the tail (24% unseen)
and on the top of the head.

One coat of fur can have two colours. `parts` can make the brown fur and the cream
fur two materials. A side of the head can then lose its pixels. On Pip, one run
made the right side of the head a material of its own, and that side came out
torn.

Raise `--material-delta-e` until the fur is one material. For Pip that is
14, and the default is 8. Look at the `material` of each part in `parts.json`. The vest and the scarf
must stay materials of their own.

**Give the parts that colour cannot find.** A sleeve has the colour of the fur
beside it. A scarf that colour does find gets a ragged border, because the bake
puts its red on the vest below. Mark such parts in the drawings, and give them:

```bash
# The scarf: the red inside a box around the neck. No outline is necessary.
kumiki-character-build box-part HANDOFF --name scarf --box front:304,590,780,767 ... \
  --colour 139,55,60 --colour 100,35,40 --delta-e 20 --min-share 0.1 --output g1/
# The vest, from what is left. --open-px removes tufts of fur on its edge.
kumiki-character-build box-part g1/rest --name vest --colour 84,77,49 --open-px 5 --output g2/
# The sleeves have no colour of their own: outlines.
kumiki-character-build box-part g2/rest --name sleeves --polygon front:... --output g3/
kumiki-character-build parts HANDOFF low.obj --given scarf=g1/scarf --given vest=g2/vest \
  --given sleeves=g3/sleeves --given-grow scarf=6 --material-delta-e 80 --output parts/
```

- With `--colour`, the part is the pixels near those colours inside the boxes (the
  whole drawing, if a view has no box). Gaps close (`--close-px`), and a piece
  smaller than `--min-share` of the largest piece goes: brown fur has some red.
- `--like PART_HANDOFF` takes the same pixels out of another handoff of the same
  drawings.
- On the mesh, each drawing that faces a texel votes: inside the part's pixels, or
  not. A drawing with a hole there, or at a grazing angle, does not vote. The
  border is then smoothed on the surface. A texel without a vote takes the answer
  of the nearest texel that has one.
- In each drawing, a given part is its own pixels and no others.
- A recognised part that loses most of its area to given parts is dissolved.
- `--given-grow scarf=6` counts the scarf 6 pixels wider on the mesh. Use it for
  a part that lies on top: the mesh has the scarf a little wider than it is drawn.
  Give that part first. Do not grow a part with an opening that must stay open.
- `--material-delta-e 80` makes the rest one material. The given parts keep their
  own. Then no run can tear a side of the head.

On Pip, the vest got a straight hem and the scarf a clean edge on the vest.

**Leave out a drawing that disagrees.** Give `parts` and `texture-parts`
`--views front,left,back`. Pip's back-left drawing puts the end of the scarf and
the hem of the vest at other places than the back drawing. With it, the back of
the vest was torn at each angle. Without it, the median error of the back view
went from 6.5 to 5.8 delta E. The front and the side did not get worse. The error
against the drawing that is left out goes up (10.7 to 13.3): that drawing is the
one that disagrees.

The mirror image of the side drawing still paints the other side.

**Keep what the generator modelled.** The generator made the hanging end of
Pip's scarf as a thin fin on the shoulder. Without the scarf end in the back
drawing, the fin took the colour of the sleeve and looked torn. Look at the
bare shape before you cut something from a drawing.

A drawing can show a limb at another depth than the model has it. Pip's side
drawing has the arm 50 pixels behind the arm of the model. The drawing then paints
the vest on the arm of the model, and the arm on the vest. Cut both places from
that drawing: the drawn limb, and the place where the limb of the model is
(`box-part HANDOFF --name limb --box left:...`, then bake from `rest/`). The other
drawings and the fill paint them.

Limits:

- Shapes come from valleys in the surface. A limb that joins the body in a saddle
  (a neck, an arm) is not separated.
- A feature that two drawings put in two places still shows twice. Pip has a second
  eyelid line above the left eye at 45 degrees. A picture from that angle removes it.
- A part that the mesh shapes differently from the drawings is only kept clean. To
  remove the cause, correct the shape. On Pip, a sliver of vest stays on the left
  edge of the tail in the back view. The border of the tail shape lies beside it.

## Model the parts alone

A model made in one piece fuses its parts. The tail, the scarf and the body are one
skin. The tail has the width that the generator guessed, and the colour of one part
lands on the other. If each part is modelled alone, it has its own outline to fit
and its own pixels to take colour from.

1. **Split the drawings.**

   ```bash
   kumiki-character-build split-views HANDOFF --part tail --views left,back,back_left \
     --subject "the cartoon squirrel" --behind "the back, the vest and the scarf" \
     --keep "the scarf with its fringe" --output split/
   ```

   A local image editor (FLUX.2 [klein] in ComfyUI) removes the part from each
   drawing that shows it and paints what was behind it. One view takes about 9
   minutes. The part is where the drawing and the edit differ. No segmentation model
   is necessary. `--keep` names what the editor must not remove with the part. Without
   it, the editor also removed the fringe of Pip's scarf when it removed the bag.

   `split/tail/` holds the drawing's own pixels of the part. `split/rest/` holds the
   drawing, with the edit only where the part was. Both folders are handoffs in the
   frame of the whole character. Their `masks/` are full-frame region masks for a
   CharacterSpec v2. To take out one more part, run `split-views` on `split/rest/`.
2. **Generate each part.** Give the generator `split/tail/raw/*.png` and
   `split/rest/raw/*.png` as separate runs (`scripts/hunyuan_mv_shape.py`). For a
   part, use the view that shows it best as `--front`. For Pip's tail that is the
   back view. The left view is then `--right`.
3. **Place each part.** `place-part split/tail tail.glb --voxel-mm 1 --output tail.ply`
   finds the turn about the vertical axis, the scale and the position at which the
   part covers its masks. Two views along different axes fix all three directions.
   Each direction can have its own scale, at most 15% from the common scale
   (`--stretch`). A part that is generated alone has its own proportions.
4. **Fit, retopology and texture** for each part with its own handoff:
   `fit split/tail tail.ply --no-normalise --lattice auto`, `retopo`, `texture split/tail low.obj`.
5. **Seat a part that hangs on the body.** The body was modelled without the bag, so
   its surface can lie further out than the bag. `seat-part body.ply bag/low.obj
   --metres-per-px 0.00032 --output bag_seated.obj` moves the bag out of the body
   as one rigid piece, until 15% of its surface is still inside. The UVs stay, so
   the texture stays valid. Without `--view`, the bag moves in the level direction,
   of eight around the character, that needs the shortest move. Pip's bag at the
   front of the left hip came out to the front left by 13 mm. Toward the left alone
   it had to move 30 mm.
   The part of the bag that stays inside can come out on the other side of a
   thin body. Pip's bag came through the back of the skin flap. `--flatten-mm 3`
   lifts each vertex that is deeper than 3 mm under the body's surface up to
   that depth. Give the low mesh of the body, as it is in the finished model.
6. **Join.** `join-parts body/low.obj:body.png tail/low.obj:tail.png --output joined/`
   makes one mesh with one texture. Add `:normal.png` to each part for a joined
   normal map. Each part keeps its own region of the texture
   and stays a separate piece of surface. `compare`, `appearance` and `rig` take
   `joined/joined.obj`.

What the editor paints behind a part is a guess. In the finished model the part
hides most of it. Look at `split/rest/raw/` before you generate.

Result on Pip (second set), body, tail and bag modelled alone, against the model in
one piece:

| | One piece | Three parts |
| --- | --- | --- |
| Vest and scarf colours on the tail | yes | no |
| Tail colours on the back and the head | yes | no |
| Silhouette IoU: front, left, back, back-left | 0.89, 0.89, 0.87, 0.82 | 0.88, 0.89, 0.88, 0.83 |
| Median colour error: front, left | 4.6, 5.6 | 4.3, 5.2 |
| Median colour error: back, back-left | 5.2, 6.4 | 6.6, 9.9 |

The colour error of the back views is higher. The tail was generated from three
views, and its outline in the back-left view fits less well (IoU 0.51 for the tail
alone). The colour of a wrong outline lands beside the drawing's.

Limits:

- The generator makes a thin piece badly. Pip's shoulder strap came out as a stub.
- The body under a removed part is the editor's guess.
- A part with one or two views has unseen sides (23% of the tail, 45% of the bag).
- Each part costs about 9 minutes for each edited view and 25 minutes for its shape.

### A small part: a hand, a forearm

A hand is 90 pixels in a drawing of the whole character. The generator makes a
stump from that, and the body's own hand is a ragged end of the arm. Model the
forearm with its hand alone, from larger drawings:

```bash
# 1. Cut the part out of the drawings. Give boxes, or outlines for a part at an angle.
kumiki-character-build box-part eyes2/rest --name arm \
  --polygon back:222,806,160,858,100,892,92,925,125,952,192,900,262,878 \
  --polygon front:... --output armL/
# 2. Enlarge each drawing of the part and have the editor draw it sharp.
kumiki-character-build detail-part armL/arm --part "the forearm and the hand" \
  --subject "a cartoon squirrel" --detail "four short round fingers" --output armL_detail/
# 3. Generate the shape from armL_detail/raw/*.png, then place it. --anchor gives the
#    position in the direction that no view of the part measures (front and back
#    views do not measure the depth).
kumiki-character-build place-part armL_detail arm.glb --anchor body/fitted.ply \
  --tilt-range-deg 40 --output arm.ply
# 4. fit, retopo and texture with armL_detail, as for each part.
# 5. The same part on the other side. It uses the same texture.
kumiki-character-build mirror-part arm/low.obj --mid-x 0.0 --output arm_R/low.obj
# 6. Remove from the body what the parts replace.
kumiki-character-build cut-part body/low.obj arms/arm --join arm/low.obj --join arm_R/low.obj \
  --output body_cut/low.obj
```

`box-part` uses no image editor: the figure's pixels inside the boxes and polygons
are the part. Use it where the editor does not do what you ask. Asked to remove
Pip's hands, it left them in one view and drew new ones in the other. The numbers
are pixels of `HANDOFF/raw/VIEW.png`.

`smooth-part body/low.obj flaps/flaps --rounds 200 --pass-band 0.005 --output body_s.obj`
smooths the surface that lies under the masks of a part. A generator makes a thin
sheet, such as the skin between Pip's arm and body, with a crumpled surface and
a ragged edge. Each round is one step toward the mean of the neighbours and one
step back, so the sheet keeps its size. The faces and the UVs stay. Smooth before
`cut-part`: a later cut then ends on the new part.

`detail-part` gives the redrawn part the colours of the drawing (mean and spread in
Lab). If the outline of the redrawn part is too different (IoU below 0.6), it
keeps the enlarged drawing for that view. `detail.json` has the IoU of each view.

Give `cut-part` the new parts with `--join arm/low.obj` (repeat for each part).
Whole faces go, so the open edge has teeth of the size of a face. A gap can
show between the edge and the part. With `--join`, the edge is smoothed, and each
of its vertices within `--join-px` (default 14) of a part moves onto that part.
On Pip, the flap of skin then reaches the forearm that it hangs from.

`cut-part` removes the faces that lie inside the part's masks along each direction
that the views look. Opposite views (front and back) look along one direction, and
one of them is sufficient for a face. For step 6, make one handoff with both sides
(`box-part` with the outlines of both arms). The UVs of the faces that stay do not change, so the body's
texture stays valid.

`--tilt-range-deg` lets the part turn about the line of sight of its first view. The
generator stands a part as it likes. Pip's forearm, drawn 40 degrees below level,
came back less steep. With the tilt, its overlap with the drawing rose from 0.66
to 0.80.

If the body and the part show a gap, cut a little less than the part. Give
`cut-part` a handoff from outlines that stop short of the joint. The part then
goes into what stays of the body.

Fit the body only to drawings that agree: `fit ... --views front left back`. One
3/4 drawing that disagreed made Pip's body 6% short.

Start the part at a line that hides the joint. Pip's forearm starts inside the
rolled cuff of the sleeve.

## Eyes as separate eyeballs

Two drawings never put an eye at the same place on the head. A texture from both
shows two eyelid lines, and no rule on the drawings removes that. An eyeball that is
its own piece of surface, placed from the front drawing, cannot be painted twice.

**The whole eye, with its white (recommended).** The white of an eye has the colour
of light fur, so colour cannot find it. The image editor can:

```bash
# 1. Take the eyes out of the front drawing. The face gets closed eyelids.
kumiki-character-build split-views split/rest --name eyes --views front \
  --part "two eyes, each with its iris, its pupil and the white of the eye" \
  --behind "closed eyelids, covered with the same fur as the face" --output eyes/
# 2. One surface for each eye, with its texture, on the head mesh.
kumiki-character-build split-eyes eyes/eyes body.ply --from-part \
  --source-handoff split/rest --output balls/
# 3. Take the eye out of each other drawing that shows it. `--within` drops what
#    the editor changed elsewhere (on Pip it repainted the inside of an ear).
kumiki-character-build split-views eyes/rest --name eyes --views left \
  --part "eye, with its iris, its pupil, the white, the eyelid line and the lashes" \
  --behind "only plain fur, with no eyelid and no dark line" \
  --within balls/within --output eyes2/
```

Bake the head from `eyes2/rest`. Add `balls/eye.L.obj:balls/eye.L.png` and
`balls/eye.R.obj:balls/eye.R.png` to `join-parts`.

Step 2 has two shapes:

- `--shape shell` (default). The eye is the piece of the head surface under the eye
  in the drawing. It is raised a little and has a low dome (`--bulge`, in eye
  sizes). Its texture is the drawing's own pixels. A drawn eye is a picture on the face, and the
  shell keeps it that way from each angle.
- `--shape ball`. The eye is an ellipsoid that stands out of the head. On the side
  that the front drawing cannot see, the texture is the white of that eye. On Pip
  the balls looked like discs with blank white sides.

If the editor paints something bad where the part was, add `--no-fill` to step 3.
The rest then has a hole there, and the bake fills it from the texture around it.
For Pip's eye in the side view, the editor painted a dark socket.

On Pip, each eye with its white and its lid line is a separate piece of surface.
The second eyelid line at 45 degrees is gone.

**Iris and pupil only, without the editor.** `split-eyes split/rest body.ply --output eyes/`
finds each pupil (a thick, very dark disc) and the iris around it (the most saturated
colour). It makes one lens for each iris. In the other drawings it removes the
surroundings of the eyes (`--zone`), so that only the front drawing paints the white
and the lids. This is faster. But the white stays on the head, and at 90 degrees the
removed area takes the nearest colour. Closed eyes and eyes without a dark pupil are
not found.

Both write `anatomy.json` with the eyes as `ocular_globe` parts for `plan-anatomy`.

## One drawing in an older design

If you replace some views, the views that stay can show an older design. Pip's front
view had bare arms, and the new side and back views have knit sleeves.

```bash
kumiki-character-build redraw-view refs/front.png refs/back.png \
  --change "both arms wear the cream knit sleeves from image 2" --output refs2/front.png
```

The local image editor changes the drawing by the instruction, with the second
drawing as the example. The result keeps the drawing's own pixels where the editor
changed nothing. The editor works at 768 px, and its copy of the face is softer than
the drawing. Say in `--change` what must stay. Look at the result before you use it.

**A different pose or an old part, without the editor.** A front and a back drawing
show one outline, mirrored. Where they do not, the shape generator gets two answers
and makes both. Pip's old front drawing had the arms a hand's width higher than the
back drawing, the old tail beside the head, and brown feet. The body came out with
doubled, ragged arms.

```bash
kumiki-character-build borrow-view split/rest --view front --from back \
  --polygon front:735,752,800,700,1086,700,1086,1250,805,1250,780,960,722,800 --output rest2/
kumiki-character-build borrow-view rest2 --view front --from front \
  --polygon front:350,752,285,700,0,700,0,1250,280,1250,305,960,363,800 --output rest3/
```

Each pixel of the region takes the pixel of the opposite drawing at the mirrored
place in the character's frame. Background is taken too, so the outline in the
region becomes that of the opposite drawing. Fur, cloth and a hand look much the same
from both sides. A face does not: keep the region off the face.

The second command uses the same drawing as the source. The region then takes the
other side of the drawing: an arm from the other arm. Use it when the opposite
drawing has something in the way on one side. On Pip's back, the end of the scarf
hangs over one arm.

`borrow.json` has `outline_iou_in_region_before`. A value much below 1 says that
the two drawings did not agree in the region. For Pip's arms it was 0.50.

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
