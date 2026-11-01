"""Compose registered-or-not reference/model pairs for honest visual QA.

The sheet is a labeled side-by-side, *not* an automatic likeness score. Crops
are optional and do not imply that camera perspective has been calibrated.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:
        raise SystemExit("Reference comparison requires Pillow") from error

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("render", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ref-crop", type=int, nargs=4,
                        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    parser.add_argument("--render-crop", type=int, nargs=4,
                        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    parser.add_argument("--size", type=int, default=600)
    args = parser.parse_args()
    if not 256 <= args.size <= 2048:
        parser.error("size must be in [256, 2048]")

    board = Image.new("RGB", (2*args.size+36, args.size+86), (28, 32, 35))
    draw = ImageDraw.Draw(board)
    for i, (path, crop, label) in enumerate(((args.reference, args.ref_crop,
                                              "REFERENCE / source photo"),
                                             (args.render, args.render_crop,
                                              "MODEL / neutral clay"))):
        with Image.open(path) as incoming:
            source = incoming.convert("RGB")
        if crop:
            x0, y0, x1, y1 = crop
            if not 0 <= x0 < x1 <= source.width or not 0 <= y0 < y1 <= source.height:
                parser.error(f"invalid crop {crop} for {path}")
            source = source.crop(crop)
        source.thumbnail((args.size, args.size), Image.Resampling.LANCZOS)
        x = 12+i*(args.size+12)
        y = 54+(args.size-source.height)//2
        board.paste(source, (x+(args.size-source.width)//2, y))
        draw.text((x+8, 18), label, fill=(238, 241, 241))
    draw.text((12, args.size+64), "Visual registration is approximate; compare anatomy, not raw pixels",
              fill=(165, 174, 178))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    board.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
