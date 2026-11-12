# Character-sheet → Blender fidelity QA: start here

This reusable **character_fidelity** toolkit compares the **actual anatomy, costume, and accessory parts** of a Blender model to a character sheet. It keeps shape, registered position/scale, unlit color, texture/detail proxies, measured Blender depth, and uncertain inferred 2D depth separate. A single similarity percentage cannot approve a sculpt.

## Install

```bash
python -m pip install -e '.[character-qa-advanced]'
kumiki-character-qa --help
```

**No Hugging Face login, access request, or automatic checkpoint download.** `capture`, `compare`, `review-masks`, and `propose-registration` do not need neural models. `segment` and `depth-hint` use *local* checkpoints and separately installed upstream SAM2/MoGe runtimes; both are optional. On Apple Silicon start with `--device mps` where supported. Model runtimes are not installed by the QA extras.

## Workflow

1. **Character sheet / parts:** Start with [`character-fidelity.example.json`](../examples/character-fidelity.example.json). Map Blender mesh name globs to anatomy and costume parts, assign front/side cameras, supply full-canvas reference masks, and set `alignment.verified=false` until reviewed. Give original transparent source layers preference over SAM guesses **only after** placing them in the full turnaround canvas with verified coordinates.
2. **Propose segmentation (optional):** Use `segment --image IMAGE --prompts prompts.json --checkpoint SAM2.pt --model base-plus --device mps --output sam-001`. See [`character-segmentation-prompts.example.json`](../examples/character-segmentation-prompts.example.json). This only proposes masks; it never verifies them.
3. **Resolve mask conflicts:** Run `review-masks --image IMAGE --sam sam-001 --output masks-review-001`. Open `ownership_review.png` (magenta = unresolved, green = manually assigned), `ambiguous_pixels.png`, and `review.md`. Raw masks stay under `raw_masks/`; `masks/` holds mutually exclusive visible-part evidence with unknown pixels removed.
4. **Provide artist evidence if available:** Copy and edit [`character-mask-review.example.json`](../examples/character-mask-review.example.json) and rerun with `--rules rules.json`. Paths are **relative to the rules JSON**, and each mask must match the original image's entire width and height. `source_masks` replace weak SAM proposals; `ownership_masks` assign only ambiguous pixels where exactly one owner is annotated. Contradictory ownership is rejected, not guessed. The output remains **unapproved** until independently reviewed. Cropped source assets must be placed in the actual turnaround coordinate system before use.
5. **Capture model:** Run `capture --blend character.blend --manifest character.json --output capture-001 --blender-bin blender`. Keep `unassigned_behavior: "occlude"` when unmapped geometry should hide visible parts. Real model geometry provides depth and normals.
6. **Propose image registration:** Record **4 or more** independently reviewed anatomical correspondences from the original illustration and Blender's captured view. See [`character-registration-anchors.example.json`](../examples/character-registration-anchors.example.json) for the **format only** (coordinates are placeholders, not real Quill measurements). Run `propose-registration --manifest character.json --capture capture-001 --anchors anchors.json --view front --output registration-001`. Inspect `alignment_overlay.png` and `registration.json` before copying its similarity matrix into the manifest. The proposal **always** says `verified=false`; a low RMSE alone never proves camera or pose match. Only after comparing multiple stable features and pose should an artist set `alignment.verified=true` in their own manifest.
7. **Compare:** Run `reconcile --manifest character.json --capture capture-001 --sam masks-review-001 --view front --output qa-001`. The report identifies incomplete source masks and conflicting ownership, so do not treat their silhouette differences as confirmed sculpt defects. If manually verified masks already exist, `compare --manifest ... --capture ... --output ...` avoids SAM entirely.

Full commands share the prefix `kumiki-character-qa` (or `python -m kumiki_blender.character_fidelity`). Every output is a **new** directory. None of these steps alters the original image, model `.blend`, or checkpoint weights.

## What the outputs mean

| Evidence | Confidence | Allowed decision |
|---|---|---|
| Original full-canvas alpha layer | High for that layer's own silhouette if positioning verified | Prefer for boundaries; may still overlap other layers |
| SAM proposed mask | Low to moderate, unapproved | Review shape and semantic assignment |
| Unresolved magenta overlap | Unknown | Exclude from exclusive masks and fix with source/ownership evidence |
| Registered alignment | Conditional on human camera/pose confirmation | Compare true position, scale and pixelwise correspondence |
| Blender Z-depth / mesh normals | Actual model geometry | Evaluate model's visible 3D structure |
| MoGe relative depth / normals | Estimated from stylized artwork | Compare hypotheses, **not** metric ground truth |

### Interpreting shape numbers

The evaluator separately reports intrinsic silhouette (translation/scale normalized) and **only when verified** registered pixel agreement. Do not use missing painted source pixels to conclude a feature is missing from the mesh. Mask-review `report.json` records ambiguous pixels excluded for every part; source masks with exclusions remain partial evidence. A low score with partial evidence should trigger a segmentation or camera review **before** remodeling the sculpt.

### Packages and entry points

- `src/kumiki_blender/character_fidelity/`: Python QA package; `__main__.py` defines commands.
- `mask_review.py`: conservative semantic ownership with full-canvas source overrides.
- `registration_review.py`: RANSAC similarity fit and human-readable registration overlay.
- `sam2_masks.py` / `moge_depth.py`: optional *offline local checkpoint* inference.
- `blender_capture.py`: read-only Blender ID/albedo/metric depth/normal capture.
- `evaluate.py`, `metrics.py`, `advanced_metrics.py`: part-wise quantitative diagnostics.
- `sam_bridge.py` / `reconcile.py`: comparison report and source provenance.
- [`CHARACTER_FIDELITY_QA.md`](CHARACTER_FIDELITY_QA.md): detailed architecture, schema and limitations.

## Validation

Run `pytest -q tests/test_character_fidelity*.py`. The mask-review and registration unit tests use synthetic fixtures without neural runtimes or Blender; manual verification of the art remains required. This workflow is reusable for humans, animals, fantasy characters, props, and layered costumes. No character artwork or model weights are committed to the repository.
