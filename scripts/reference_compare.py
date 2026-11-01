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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("render", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(comparison_sheet(args.reference, args.render, args.output))
