# Install on Windows

[← Home](https://github.com/YukiUmetsu/kumiki-blender/wiki) · [Next: make your first model](https://github.com/YukiUmetsu/kumiki-blender/wiki/Make-Your-First-3D-Model)

These instructions use **PowerShell**, the command window included with Windows 10/11.

## 1. Install three free tools

| Tool | Why | Download |
| --- | --- | --- |
| Blender **4.2 or newer** | Makes 3D models and pictures | [blender.org/download](https://www.blender.org/download/) |
| Python **3.11 or newer** | Validates recipes | [python.org/downloads](https://www.python.org/downloads/) |
| Git for Windows | Downloads the repository | [git-scm.com/downloads/win](https://git-scm.com/downloads/win) |

During Python installation, choose the option to make Python available from the command line if offered. Normal installer defaults for Git are fine.

## 2. Verify the installs

Open **PowerShell**. Paste each command and press Enter:

```powershell
py -3 --version
git --version
```

Python must report 3.11 or later. Blender is often installed at a path like this (change `4.5` if you installed a different version):

```powershell
& "C:\Program Files\Blender Foundation\Blender 4.5\blender.exe" --version
```

If PowerShell says the Blender file does not exist, find `blender.exe` under `C:\Program Files\Blender Foundation\` and use its actual path. **Do not blindly copy a version number that differs from your install.**

## 3. Download the project

```powershell
cd "$HOME\Documents"
git clone https://github.com/YukiUmetsu/kumiki-blender.git
cd kumiki-blender
```

If you use OneDrive and your Documents folder is elsewhere, navigate to the folder where you want the repository first.

## 4. Set up Python

These commands avoid PowerShell's script-execution restrictions: you do **not** have to activate a virtual environment.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

## 5. Check the toolkit

```powershell
.\.venv\Scripts\python.exe -m kumiki_blender validate --recipe examples/lattice-observation-tower.json
```

Success is a block of JSON text with `"asset_id": "lattice-observation-tower"` and `"beams": 277`. The first check does not open Blender.

## Every time you come back

Open PowerShell and go back into the folder where you cloned the project:

```powershell
cd "$HOME\Documents\kumiki-blender"
```

Use `.\.venv\Scripts\python.exe` for subsequent Python commands. If your Documents folder is elsewhere, adjust the path.

Continue with [Make Your First 3D Model](https://github.com/YukiUmetsu/kumiki-blender/wiki/Make-Your-First-3D-Model).
