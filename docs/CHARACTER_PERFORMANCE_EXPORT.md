# Combined skeletal and facial character performance

This is a **Blender-native export bridge** for a character that gestures, walks or folds wings while speaking. It produces a **single named GLB animation clip** with actual deform-skeleton animation and facial morph-weight channels. It does not reconstruct facial geometry or generate motion.

## Requirements

- An authored .blend containing one saved character armature, named CharacterRig by default, with a single active **baked deform-bone** Action.
- A facial mesh, named Mouth by default, with approved Basis and facial shape keys, a single active **shape-key** Action, and no NLA tracks.
- Both Actions use the same start/end timeline. Source file contains no CTRL_ authoring bones; bake them using the existing plan-animation-bake/bake-character-animation commands first.
- At least one real skinned body mesh; this pipeline rejects an empty skeleton or animation without actual facial morph motion.

The first iteration deliberately does not import Actions from multiple .blend files, merge NLA stacks, automatically retime actions, or create mouth shapes. Combine the authored body and face Actions in one .blend first, then export them with the following commands.

## Commands

~~~bash
kumiki-character-build plan-performance \
  --blend /path/to/animated-character.blend \
  --clip talk_wave --start 1 --end 72 --fps 24 \
  --armature CharacterRig --mouth-object Mouth \
  --output .review/character/performance-plan-r01

kumiki-character-build export-performance \
  --plan .review/character/performance-plan-r01/performance-plan.json \
  --output .review/character/performance-r01 \
  --blender /Applications/Blender.app/Contents/MacOS/Blender
~~~

The planner hashes and locks the entire source .blend and requires bounded frame ranges. The Blender worker snapshots actual joint and mouth-shape changes across the timeline, checks that both are nontrivial, selects the armature plus skinned meshes and face, and exports an independent result.

**Output:**
- combined.blend: independent editable scene revision
- performance.glb: one named glTF animation containing both skin joint and morph-weight tracks
- report.json: clip, source provenance, sampling/motion diagnostics, skinned meshes and glTF channels

The glTF exporter may produce separate body and face animation entries even when authored simultaneously. The local GLB merger validates target node/path uniqueness and synchronized glTF time accessor ranges, then **merges animation samplers and channels only in the JSON chunk**. It preserves all underlying binary keyframe/texture/geometry buffers byte-for-byte. If the tracks have conflicting node targets or different time origins/durations, it refuses to merge rather than retime them silently. The resulting GLB must contain both real skin-joint and morph-weight animation channels.

## Testing and limitations

A pure-Python suite verifies GLB byte preservation, valid sampler index remapping, duplicate channel rejection, timing checks, source hashes and immutable revisions. A native Blender 4.5 LTS smoke fixture authors a simple armature and a shape-key mouth on one mesh, keyframes both Actions and checks their combined GLB.

These checks establish **functional animation export**, not facial quality or realistic wing mechanics. Final approval still needs a browser playback/render-parity comparison in Kumiki-UI, as well as Pip/Quill source likeness and deformation review.
