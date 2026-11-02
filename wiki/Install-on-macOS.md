# Install on macOS

[← Home](https://github.com/YukiUmetsu/kumiki-blender/wiki) · [Next: make your first model](https://github.com/YukiUmetsu/kumiki-blender/wiki/Make-Your-First-3D-Model)

These instructions use a **Mac** and the built-in **Terminal**. You do not need Homebrew.

## 1. Install three free tools

| Tool | Why you need it | Where to get it |
| --- | --- | --- |
| Blender **4.2 or newer** | Creates 3D geometry, editable scenes and pictures | [blender.org/download](https://www.blender.org/download/) |
| Python **3.11 or newer** | Reads and validates the model instructions | [python.org/downloads](https://www.python.org/downloads/) |
| Git | Downloads the source code from GitHub | [git-scm.com/downloads](https://git-scm.com/downloads/) |

After installing Blender, drag it into **Applications** if the installer prompts you. This guide assumes `/Applications/Blender.app` exists. Use a newer supported Blender version if available; CI exercises Blender 4.5 LTS on Linux, not every Mac configuration.

## 2. Verify the installs

Open **Terminal**. Run one command at a time:

```bash
python3 --version
git --version
/Applications/Blender.app/Contents/MacOS/Blender --version
```

The first should say Python 3.11 or newer. The second should show a Git version, and the third a Blender version. If a command says **not found**, finish installing that program before continuing. On a new Mac, running Git may trigger Apple's command-line developer tools installer; follow its prompt.

## 3. Download the project

```bash
cd ~/Documents
git clone https://github.com/YukiUmetsu/kumiki-blender.git
cd kumiki-blender
```

The project folder is now in Documents. If `git clone` says the folder already exists, do not delete it; run `cd ~/Documents/kumiki-blender` instead.

## 4. Create a private Python environment

A **virtual environment** is a folder containing this project's Python packages, separate from the rest of your Mac.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

You may see `(.venv)` at the start of the terminal line. That means the environment is active. If your first command picked an old Python version, reinstall Python 3.11+ and recreate the environment with that version.

## 5. Check that the toolkit works

```bash
python -m kumiki_blender validate --recipe examples/lattice-observation-tower.json
```

Success looks like **JSON text** containing `"asset_id": "lattice-observation-tower"` and numbers such as `"beams": 277`. No Blender window will appear: this first check only validates a recipe.

## Every time you come back

Open Terminal and run:

```bash
cd ~/Documents/kumiki-blender
source .venv/bin/activate
```

You are now ready for [Make Your First 3D Model](https://github.com/YukiUmetsu/kumiki-blender/wiki/Make-Your-First-3D-Model).
