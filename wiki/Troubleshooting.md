# Troubleshooting: common beginner problems

[← Home](https://github.com/YukiUmetsu/kumiki-blender/wiki)

Don't worry if something fails the first time. Read the error's **last few lines** and match it with a case below. Commands assume you're inside the `kumiki-blender` project folder.

## Quick checks

**macOS / Linux**

```bash
pwd
python --version
python -m kumiki_blender --help
```

**Windows PowerShell**

```powershell
Get-Location
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m kumiki_blender --help
```

The last command should list actions such as `validate`, `build`, `catalog` and `audit-revision`. If it does, the Python side of the toolkit is working.

## “command not found”, “not recognized”, or “No such file or directory”

**Cause:** your terminal cannot find a program or the path is wrong.

- `git` missing: install [Git](https://git-scm.com/downloads/) and reopen the terminal.
- `python3` or `py` missing: install [Python 3.11+](https://www.python.org/downloads/).
- `blender` missing: use the full executable path from your setup guide or add Blender to your PATH. On Mac, this guide uses `/Applications/Blender.app/Contents/MacOS/Blender`.
- On Windows, your Blender folder might not be `Blender 4.5`. Find the installed version's `blender.exe` and update `$blender`.
- `scripts/blender_entry.py` not found: you're likely in the wrong folder. Use `cd` to enter your cloned `kumiki-blender` directory.

## “No module named kumiki_blender”

**Cause:** Python is running outside the project environment, or the package hasn't been installed.

On Mac/Linux, activate the environment with `source .venv/bin/activate` then run `python -m pip install -e .`. On Windows, run `.\.venv\Scripts\python.exe -m pip install -e .` from the repo folder.

If `pip` fails with an **externally managed environment** message, create and use `.venv` exactly as the installation guide describes. Do not force a system-wide install.

## “Python version is too old”

The project requires **Python 3.11+** for its validation and CLI. `python3 --version` / `py -3 --version` should confirm that. If your environment was created with an older Python, make a **new** virtual environment with an updated Python interpreter.

## “Output directory already exists”

This is deliberate protection against overwriting prior work. Change the end of the output path from `r01` to `r02` (or the next unused revision). **Do not delete the old directory just to retry.** A failed run can leave an incomplete revision folder.

## The build is slow / Blender appears frozen

3D rendering is computationally expensive. For a first test, use the fast preview profile:

- macOS / Linux: prefix the Blender command with `KUMIKI_RENDER_PROFILE=smoke`.
- Windows PowerShell: run `$env:KUMIKI_RENDER_PROFILE = 'smoke'` before the Blender command.

Smoke mode renders smaller, low-sample previews. It is intended for quick checks, not final cinematic quality. Check that the Blender process is still running; give it time if it is using CPU/GPU. Large geometry and higher-quality renders may take much longer.

## Blender closes without producing a model

Look for an error near the end of the terminal output. Keep `--python-exit-code 1` in the command so a Python failure makes the process fail clearly. Verify Blender is **4.2+**, that your recipe validates, and that you used `--python scripts/blender_entry.py --` before the build subcommand.

The `.review/.../r01` folder might exist even though the build failed. Check whether `source.blend`, `model.glb`, `preview.png` and `manifest.json` are all present. Retry with a new output revision.

## I cannot find `.review`

Names beginning with a dot are hidden in some file managers. On Mac Finder, press **Command + Shift + .**. On Windows, enable **Hidden items** in File Explorer. You can also open the project in VS Code or browse to the path directly.

## My object looks gray, black or different from the PNG

- In Blender, switch from **Solid** to **Material Preview** in the Viewport shading controls.
- Imported GLBs usually do **not** include the original preview studio lights or backdrop. Add scene lighting/environment in the viewer.
- Some advanced material effects are Blender-specific and may not reproduce perfectly in a GLB viewer. The project tries to export compatible PBR textures, but visual inspection is still required.
- `KUMIKI_RENDER_PROFILE=smoke` gives less polished preview lighting/noise than a production render.

## My GLB doesn't look like the model I imagined

First inspect it from front, side and three-quarter views. The project checks file structure and some technical properties; it cannot guarantee likeness, correct anatomy or beautiful materials. A generic procedural statue is **not** equivalent to a verified scan. Read [Use Reference Images](https://github.com/YukiUmetsu/kumiki-blender/wiki/Use-Reference-Images).

## Can I build any object just by giving it a photo?

No. Some tools can analyze photo sets or work with an existing anatomical head, but there is no universal one-photo-to-perfect-3D command. Photo reconstruction requires overlapping viewpoints and careful review; some objects need manual sculpting or measured survey geometry.

## Where do I report a real bug?

Open a [GitHub issue](https://github.com/YukiUmetsu/kumiki-blender/issues) and include:

1. Your operating system, Python version and Blender version.
2. The command you ran (remove private paths and tokens if present).
3. The last relevant error lines (as text).
4. Whether `validate` succeeded.
5. Which revision was being built, and which output files appeared.

For a developer, the [technical README](https://github.com/YukiUmetsu/kumiki-blender/blob/main/README.md) and [test instructions](https://github.com/YukiUmetsu/kumiki-blender/tree/main/tests) provide more diagnostics.
