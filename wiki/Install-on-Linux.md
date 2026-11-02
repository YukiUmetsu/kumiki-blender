# Install on Linux

[← Home](https://github.com/YukiUmetsu/kumiki-blender/wiki) · [Next: make your first model](https://github.com/YukiUmetsu/kumiki-blender/wiki/Make-Your-First-3D-Model)

You need **Python 3.11+**, **Git**, and **Blender 4.2+**. Use the package manager for your distribution or the official downloads: [Python](https://www.python.org/downloads/), [Git](https://git-scm.com/downloads/), [Blender](https://www.blender.org/download/). Some distributions package older Blender versions; check before proceeding.

## 1. Check versions

Open a terminal:

```bash
python3 --version
git --version
blender --version
```

If a program is missing or older than required, install the current supported version with your distribution's package manager, or from its official website. On Debian/Ubuntu, Python's virtual-environment package (`python3-venv`) may also be required.

## 2. Download the project

```bash
cd ~/Documents
git clone https://github.com/YukiUmetsu/kumiki-blender.git
cd kumiki-blender
```

If `~/Documents` does not exist on your setup, choose an existing folder such as your home directory instead. If the clone folder already exists, enter the existing one; do not delete it.

## 3. Set up Python and check a recipe

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m kumiki_blender validate --recipe examples/lattice-observation-tower.json
```

You should see JSON with the tower's model ID, bounds and part counts. This first check does not need a graphical window.

## Every time you return

```bash
cd ~/Documents/kumiki-blender
source .venv/bin/activate
```

If you used a different folder, change `cd` accordingly.

Continue with [Make Your First 3D Model](https://github.com/YukiUmetsu/kumiki-blender/wiki/Make-Your-First-3D-Model).
