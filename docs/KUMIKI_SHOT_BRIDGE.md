# Render a shot from a Kumiki bake

`plan-shot` and `build-shot` render one shot from three inputs: a **set**, a **cast** and a **per-frame bake**. Kumiki resolves every frame. This package only renders them, so a shot has one clock.

```bash
kumiki-blender plan-shot --spec shot.json
kumiki-blender build-shot --spec shot.json --output renders/shot-r01
```

`build-shot` starts Blender when needed. Set `KUMIKI_BLENDER` or pass `--blender`.

## Actors

An actor is a Character Studio pack (`character.json` + `layers/`) plus a staging profile. Each flat layer becomes an inflated mesh that takes real light and casts real shadows. The art is not redrawn.

```json
{
  "schema_version": 1,
  "height_m": 0.24,
  "parts": { "head": { "layer": -0.03, "inflate": 0.055 } },
  "effectors": { "hand.right": { "part": "wing-right", "joints": 3 } }
}
```

- `layer` is depth in metres. Negative is toward the camera.
- `inflate` is how far the layer domes.
- `ground_parts` names the parts that stand on the ground. Use it when a tail or cape hangs below the feet.
- An effector adds a bone chain that follows the drawn limb, so the limb bends under IK.

`describe-actor` reports each effector's rest position, shoulder and reach.

**Limits:** art is drawn from one view, so keep the camera within about 25 degrees of the actor's facing. Hands overlap held objects. They do not close around them.

## Sets

A set is JSON. Kinds are `box`, `lathe` and `document`. Positions use Kumiki Y-up metres. See `tests/test_shot_integration.py` for a small complete example.

`render-document` makes an exact printed page from data blocks (`text`, `row`, `rule`, `space`). Use it for numbers on paper. Do not use generated images for them.

## Bake

`schema_version`, `fps`, `duration_frames`, `camera`, `objects`, `actors`, `lights`, `contacts`. Each channel has one entry per frame, and a length mismatch stops the build. Actor `parts` entries are Kumiki `CharacterPartState` values, plus optional `yaw` and `pitch` degrees for a 3D turn. Y-up to Z-up conversion happens only in `shot_plan.py`.

## Output

A new bundle with `shot.mp4`, an editable `shot.blend` and `shot-manifest.json`. The manifest records the fingerprint, Blender version, timing and the worst hand-to-target distance per effector. Only frames marked `holding` count, and a distance over 2 mm fails the build before any frame renders.

Frames are written to a work directory, so a retry resumes. The bundle appears only by rename after success, and an existing output is never overwritten.
