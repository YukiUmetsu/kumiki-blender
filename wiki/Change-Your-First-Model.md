# Change your first model

[← Home](https://github.com/YukiUmetsu/kumiki-blender/wiki) · [Previous: open it](https://github.com/YukiUmetsu/kumiki-blender/wiki/Open-It-in-Blender)

You will make a **taller blue tower** by changing two settings. No Python coding or manual Blender sculpting required.

## 1. Make a copy of the recipe

The original instructions live in `examples/lattice-observation-tower.json`. A JSON file is just text with named values, such as `"height_m": 120`. Keep the original untouched.

**macOS / Linux**

```bash
cp examples/lattice-observation-tower.json examples/my-first-tower.json
```

**Windows PowerShell**

```powershell
Copy-Item examples/lattice-observation-tower.json examples/my-first-tower.json
```

Open `examples/my-first-tower.json` in a plain-text editor, such as VS Code. Do not use Word or a rich-text editor.

## 2. Change only two values

Find these original lines:

```json
"height_m": 120,
"steel_color": "#C94B38",
```

Change them to:

```json
"height_m": 140,
"steel_color": "#2F7FC1",
```

They appear in different parts of the file, so edit each line in place. `height_m` is height in **meters**; `steel_color` is a six-digit **hex color** (`#RRGGBB`). Keep the quotation marks, commas and surrounding braces unchanged. Save the file.

## 3. Validate the new version

**macOS / Linux**

```bash
python -m kumiki_blender validate --recipe examples/my-first-tower.json
```

**Windows**

```powershell
.\.venv\Scripts\python.exe -m kumiki_blender validate --recipe examples/my-first-tower.json
```

If validation fails, check for missing commas, an incomplete `#RRGGBB` color, or a number outside the accepted range. Change one thing at a time until the recipe validates.

## 4. Build revision 2

### macOS

```bash
KUMIKI_RENDER_PROFILE=smoke /Applications/Blender.app/Contents/MacOS/Blender \
  --background --factory-startup --python-exit-code 1 \
  --python scripts/blender_entry.py -- \
  build --recipe examples/my-first-tower.json \
  --output .review/lattice-observation-tower/r02
```

### Windows PowerShell

```powershell
$blender = 'C:\Program Files\Blender Foundation\Blender 4.5\blender.exe'
$env:KUMIKI_RENDER_PROFILE = 'smoke'
& $blender --background --factory-startup --python-exit-code 1 --python scripts/blender_entry.py -- build --recipe examples/my-first-tower.json --output .review/lattice-observation-tower/r02
```

Update the Blender path to match your installed version.

### Linux

```bash
KUMIKI_RENDER_PROFILE=smoke blender \
  --background --factory-startup --python-exit-code 1 \
  --python scripts/blender_entry.py -- \
  build --recipe examples/my-first-tower.json \
  --output .review/lattice-observation-tower/r02
```

## 5. Compare the two results

Open `r01/preview.png` next to `r02/preview.png`. Then open the new `source.blend` in Blender. Is the overall height different? Is the steel blue?

If the visual difference is hard to judge, compare the `bounds_m` in the two validation outputs. These are coordinates describing the total 3D size of the model.

**Why not reuse `r01`?** The build uses immutable revisions so you can compare changes without accidentally erasing the previous version. If `r02` already exists, use `r03`.

**What you've learned:** a reusable recipe changes geometry and materials; Blender turns that recipe into a real 3D model; a new revision preserves the old result.
