# Make your first 3D model

[← Home](https://github.com/YukiUmetsu/kumiki-blender/wiki) · [Next: open it in Blender](https://github.com/YukiUmetsu/kumiki-blender/wiki/Open-It-in-Blender)

**Goal:** create a red 3D observation tower, an editable Blender file and several pictures. No modeling skills or coding are needed.

Before starting, finish the setup for [Mac](https://github.com/YukiUmetsu/kumiki-blender/wiki/Install-on-macOS), [Windows](https://github.com/YukiUmetsu/kumiki-blender/wiki/Install-on-Windows), or [Linux](https://github.com/YukiUmetsu/kumiki-blender/wiki/Install-on-Linux).

## Step 1. Understand the two actions

- **Validate** checks the recipe (instructions) without making a 3D model. It is fast.
- **Build** opens Blender in the background, creates actual geometry, renders preview pictures and exports a GLB. It can take several minutes.

Blender may not show a graphical window during a build. **That is normal**: it is working in the background.

## Step 2. Validate the tower recipe

Make sure your terminal is inside the `kumiki-blender` repository and your Python environment is ready.

**macOS / Linux**

```bash
python -m kumiki_blender validate --recipe examples/lattice-observation-tower.json
```

**Windows PowerShell**

```powershell
.\.venv\Scripts\python.exe -m kumiki_blender validate --recipe examples/lattice-observation-tower.json
```

You should see a JSON result containing `"asset_id": "lattice-observation-tower"`, `"beams": 277`, and `"decks": 2`. Nothing visual has been created yet.

## Step 3. Build the model

Pick **only one** block for your computer. Each block is one command; the line continuations just make it easier to read.

### macOS

```bash
KUMIKI_RENDER_PROFILE=smoke /Applications/Blender.app/Contents/MacOS/Blender \
  --background --factory-startup --python-exit-code 1 \
  --python scripts/blender_entry.py -- \
  build --recipe examples/lattice-observation-tower.json \
  --output .review/lattice-observation-tower/r01
```

### Windows PowerShell

Update the Blender version/path to match what you installed:

```powershell
$blender = 'C:\Program Files\Blender Foundation\Blender 4.5\blender.exe'
$env:KUMIKI_RENDER_PROFILE = 'smoke'
& $blender --background --factory-startup --python-exit-code 1 --python scripts/blender_entry.py -- build --recipe examples/lattice-observation-tower.json --output .review/lattice-observation-tower/r01
```

### Linux

```bash
KUMIKI_RENDER_PROFILE=smoke blender \
  --background --factory-startup --python-exit-code 1 \
  --python scripts/blender_entry.py -- \
  build --recipe examples/lattice-observation-tower.json \
  --output .review/lattice-observation-tower/r01
```

**Why `smoke`?** This makes faster, lower-resolution preview images so you can check the process before waiting for a high-quality render. It does not remove the model's geometry. It is *not* a final artistic-quality render.

**Why `r01`?** It means **revision 1**. Every build needs a brand-new output folder, so a later attempt will use `r02`, `r03`, and so on. The toolkit intentionally will not overwrite an existing output folder. A failed build may also leave a partial folder: use a new revision when retrying.

## Step 4. Find the results

Inside the repo, open this folder using Finder / File Explorer / your file manager:

` .review/lattice-observation-tower/r01 `

The leading dot on `.review` means the folder may be hidden by your file manager. On a Mac, toggle hidden files in Finder with **Command + Shift + .**; on Windows, turn on **Hidden items** in File Explorer's View menu. Or open the path from your editor.

Expected files:

| File | What it is | What to do with it |
| --- | --- | --- |
| `source.blend` | Editable Blender project | Open in Blender to view or edit |
| `model.glb` | Portable 3D object | Import into a game engine or 3D viewer |
| `preview.png` | Main rendered image | Open as a normal picture |
| `previews/front.png`, `side.png`, `detail.png` | Extra views | Compare shape and details |
| `manifest.json` | Build record | Contains provenance, measurements and file checks |

If these files are missing, **do not assume the build succeeded**. Read the final lines printed by Blender and use [Troubleshooting](https://github.com/YukiUmetsu/kumiki-blender/wiki/Troubleshooting).

## Step 5. Verify the exported result

**macOS / Linux**

```bash
python -m kumiki_blender audit-revision --root .review/lattice-observation-tower/r01
```

**Windows**

```powershell
.\.venv\Scripts\python.exe -m kumiki_blender audit-revision --root .review/lattice-observation-tower/r01
```

An audit checks that the exported files match the build record. It does **not** judge whether the tower looks beautiful or realistic.

## Congratulations: you made an actual 3D object

Next, [Open It in Blender](https://github.com/YukiUmetsu/kumiki-blender/wiki/Open-It-in-Blender) to rotate around the model, explore its parts and learn basic controls.
