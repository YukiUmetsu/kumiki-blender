"""Read-only Blender character-sheet geometry and appearance comparison CLI."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .evaluate import evaluate_character
from .schema import load_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m kumiki_blender.character_fidelity")
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture", help="Render per-part IDs, unlit color and metric depth with Blender")
    capture.add_argument("--blend", type=Path, required=True)
    capture.add_argument("--manifest", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--blender-bin", default="blender")
    compare = commands.add_parser("compare", help="Analyze pre-rendered part evidence")
    compare.add_argument("--manifest", type=Path, required=True)
    compare.add_argument("--capture", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "capture":
            load_manifest(args.manifest)
            if not args.blend.is_file() or args.blend.is_symlink():
                raise ValueError("--blend must be a regular Blender file")
            if args.output.exists() or args.output.is_symlink():
                raise ValueError("Capture output must be a new directory")
            command = [args.blender_bin, "-b", str(args.blend.resolve()), "--python",
                       str(Path(__file__).with_name("blender_capture.py")), "--",
                       "--manifest", str(args.manifest.resolve()), "--output", str(args.output.resolve())]
            return subprocess.run(command, check=False).returncode
        result = evaluate_character(args.manifest, args.capture, args.output)
        print(f"{args.output}/report.json: {result['status']}")
        return 0
    except (OSError, ValueError, KeyError, ImportError) as error:
        print(f"character-fidelity: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
