"""Minimal Blender-side launcher, independent of system-site Python installs."""

from __future__ import annotations

import sys
from pathlib import Path


root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from kumiki_blender.cli import main  # noqa: E402


if __name__ == "__main__":
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    raise SystemExit(main(arguments))
