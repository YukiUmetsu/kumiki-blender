"""Reference versus Blender render QA contact sheet; requires Pillow.

Use side-by-side before camera registration. Only add the blend overlay after
matching camera orientation, framing, scale, and landmark positions.

    python scripts/reference_compare.py reference.png render.png \
        --output .review/face-comparison.png
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def tile(path: Path, size: int) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.contain(source.convert("RGB"), (size, size))
    canvas = Image.new("RGB", (size, size), (30, 35, 43))
    canvas.paste(image, ((size - image.width) // 2,
                         (size - image.height) // 2))
    return canvas


def compare(reference: Path, rendered: Path, output: Path,
            size: int = 750, aligned: bool = False) -> None:
    """Create an honest labeled comparison, with optional registered overlay."""
    left = tile(reference, size)
    right = tile(rendered, size)
    pad, header = 20, 50
    count = 3 if aligned else 2
    canvas = Image.new("RGB", ((size + pad) * count + pad,
                               size + header + 2 * pad), (30, 35, 43))
    labels = ["REFERENCE (actual subject)", "BLENDER RENDER"]
    images = [left, right]
    if aligned:
        # This is meaningful ONLY after independently matching the views.
        images.append(Image.blend(left, right, .5))
        labels.append("REGISTERED OVERLAY (50/50)")
    pen = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for i, (caption, image) in enumerate(zip(labels, images)):
        x = pad + i * (size + pad)
        pen.text((x + 10, pad + 15), caption, fill=(225, 233, 242), font=font)
        canvas.paste(image, (x, pad + header))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("render", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=750)
    parser.add_argument("--aligned", action="store_true")
    args = parser.parse_args()
    if args.size < 64 or args.size > 4096:
        parser.error("--size must be between 64 and 4096 pixels")
    for file in (args.reference, args.render):
        if not file.is_file():
            parser.error(f"Image not found: {file}")
    compare(args.reference, args.render, args.output, args.size, args.aligned)


if __name__ == "__main__":
    main()
