# Local Blender facial animation: authored shape keys + offline mouth cues

Character Production v2 supports **offline, source-locked viseme cues** without adding a subscription, model checkpoint, or new Python dependency.

**This is an animation workflow, not a face modeler.** The source Blender file must already contain actual artist-approved facial geometry: a Basis shape and separate viseme shape keys such as viseme_A and viseme_C. It does **not** invent lips, beaks, jaw, or teeth from audio.

## Inputs

- A saved authored-face.blend with a mesh object named Mouth and actual nonempty viseme shape keys.
- Rhubarb-style cue JSON with mouthCues, each with nonoverlapping start, end and value. Supported codes: A–H and X (silence).
- Optional MIT [Rhubarb Lip Sync](https://github.com/DanielSWolf/rhubarb-lip-sync) can generate these JSON cues offline, but is not installed or invoked here. Recognition quality, especially Japanese, must be checked separately.

See [sample Rhubarb cue JSON](../examples/character-mouth-cues.example.json). By default A maps to viseme_A and C to viseme_C; X means neutral (all mapped mouth shapes zero).

## Commands

~~~bash
kumiki-character-build plan-visemes \
  --blend /path/to/authored-face.blend \
  --cues /path/to/mouth-cues.json \
  --mouth-object Mouth \
  --clip hello --fps 24 \
  --output .review/character/viseme-plan-r01

kumiki-character-build animate-visemes \
  --plan .review/character/viseme-plan-r01/viseme-plan.json \
  --output .review/character/face-animation-r01 \
  --blender /Applications/Blender.app/Contents/MacOS/Blender
~~~

The immutable plan records SHA-256 of the source Blender and cue files, object/clip names, exact cues and ramp. It rejects missing/stale inputs, overlapping/invalid cues, only-silence tracks and unsafe parameters.

Output files:
- **visemes.blend:** editable Blender shape-key Action on the named mouth mesh
- **face-animated.glb:** morph-target animation with actual glTF weight channels and the requested clip name
- **report.json:** cue counts, moved vertices per mouth shape, morph target inventory, and animation channels

The worker does not modify source mesh vertex coordinates. It rejects mouth meshes whose shape keys already have active animation, non-neutral initial weights, missing or empty shape keys, or unsupported simultaneous active skeletal+facial Actions rather than overwriting them.

## Limitations and next steps

This is a **single standalone facial clip**. It does not yet combine the clip with skeletal NLA Actions, generate mouth geometry, resolve Japanese phonemes, or test audio/render synchronization. Next: integrate deform-skeleton and shape-key action channels without changing their authored names or key values; then compare a sequence in Kumiki-UI.

## Verification

Focused Python tests check timestamp and source integrity; an actual Blender 4.5 LTS smoke test creates a procedural mouth, keys A/C visemes, exports GLB, and requires morph-weight channels and the requested clip name. Synthetic geometry is not evidence of Pip/Quill likeness or cinematic expressiveness.
