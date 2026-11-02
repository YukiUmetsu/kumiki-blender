"""Make a non-registered reference/render inspection sheet.

This tool never alters a photograph to suggest that unlike camera angles
match. Reference images remain local QA material; do not commit them or infer
redistribution rights from their being publicly viewable.

    python scripts/reference_compare.py photo.jpg preview.png --output sheet.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def comparison_sheet(reference: Path, render: Path, output: Path,
                     *, panel_width: int = 900, panel_height: int = 680) -> Path:
    if min(panel_width, panel_height) < 128:
        raise ValueError("Comparison panels must be at least 128 pixels in each dimension")
    margin, header, footer, gap = 20, 64, 48, 20
    width = panel_width * 2 + margin * 2 + gap
    height = panel_height + header + footer + margin * 2
    canvas = Image.new("RGB", (width, height), (25, 30, 36))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (image_path, label) in enumerate(((reference, "REFERENCE PHOTO"),
                                                 (render, "BLENDER RENDER"))):
        with Image.open(image_path) as im:
            im.load()  # Also catches truncated photos and renders.
            picture = ImageOps.contain(im.convert("RGB"), (panel_width, panel_height),
                                       method=Image.Resampling.LANCZOS)
        left = margin + index * (panel_width + gap)
        top = margin + header
        draw.rectangle((left, top, left + panel_width, top + panel_height),
                       fill=(39, 43, 48))
        canvas.paste(picture, (left + (panel_width - picture.width) // 2,
                               top + (panel_height - picture.height) // 2))
        draw.text((left, margin + 24), label, fill=(235, 238, 243), font=font)
    draw.text((margin, height - footer),
              "Different cameras / projection: compare geological form, colors and detail; NOT pixel alignment.",
              fill=(174, 184, 194), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG")
    return output


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
    parser.add_argument("--size", type=int, default=None,
                        help="Square side-by-side panels (legacy Christ comparison)")
    parser.add_argument("--aligned", action="store_true",
                        help="Only for independently camera-registered photographs")
    parser.add_argument("--panel-width", type=int, default=900)
    parser.add_argument("--panel-height", type=int, default=680)
    args = parser.parse_args()
    if args.aligned or args.size is not None:
        size = args.size or 750
        if size < 64 or size > 4096:
            parser.error("--size must be between 64 and 4096")
        compare(args.reference, args.render, args.output, size, args.aligned)
    else:
        comparison_sheet(args.reference, args.render, args.output,
                         panel_width=args.panel_width,
                         panel_height=args.panel_height)


if __name__ == "__main__":
    main()
