"""Read-only Blender character-sheet geometry and appearance comparison CLI."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

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
    segment = commands.add_parser("segment", help="Propose masks with a local SAM2.1 checkpoint")
    segment.add_argument("--image", type=Path, required=True)
    segment.add_argument("--prompts", type=Path, required=True)
    segment.add_argument("--checkpoint", type=Path, required=True)
    segment.add_argument("--output", type=Path, required=True)
    segment.add_argument("--model", choices=("tiny", "small"), default="small")
    segment.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    segment.add_argument("--no-alpha-clipping", action="store_true")
    hint = commands.add_parser("depth-hint", help="MoGe-2 estimated relative depth, local weights")
    hint.add_argument("--image", type=Path, required=True)
    hint.add_argument("--checkpoint", type=Path, required=True)
    hint.add_argument("--output", type=Path, required=True)
    hint.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    try:
        if args.command == "segment":
            from .sam2_masks import segment_sheet
            result = segment_sheet(args.image, args.prompts, args.checkpoint, args.output,
                                   model=args.model, device=args.device,
                                   clip_to_alpha=not args.no_alpha_clipping)
            print(f"{args.output}/report.json: {result['status']}")
            return 0
        if args.command == "depth-hint":
            from .moge_depth import estimate_depth_hint
            result = estimate_depth_hint(args.image, args.checkpoint, args.output, device=args.device)
            print(f"{args.output}/report.json: {result['status']}")
            return 0
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
        from .evaluate import evaluate_character  # expensive QA dependencies only for compare
        result = evaluate_character(args.manifest, args.capture, args.output)
        print(f"{args.output}/report.json: {result['status']}")
        return 0
    except (OSError, ValueError, KeyError, ImportError) as error:
        print(f"character-fidelity: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
