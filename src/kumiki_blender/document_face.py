"""Render a printed document face from data, so numbers on paper are exact.

Runs outside Blender and needs Pillow. Layout comes from data blocks, not from a
document type, so any form, ticket or label can use it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DocumentFaceError(ValueError):
    """Invalid document face request."""


_BLOCKS = {"text", "row", "rule", "space"}


def _colour(value: Any, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    if value is None:
        return fallback
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise DocumentFaceError("Colours must be [r, g, b]")
    return tuple(int(channel) for channel in value)  # type: ignore[return-value]


def render_document_face(spec: dict[str, Any], output: Path, *, root: Path) -> dict[str, Any]:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    if spec.get("schema_version") != 1:
        raise DocumentFaceError("Only document face schema_version 1 is supported")
    width, height = spec.get("width"), spec.get("height")
    if type(width) is not int or type(height) is not int or not 64 <= width <= 8192 or not 64 <= height <= 8192:
        raise DocumentFaceError("width and height must be integers from 64 to 8192")
    margin = spec.get("margin", int(width * .075))
    paper = _colour(spec.get("paper"), (244, 238, 224))
    ink = _colour(spec.get("ink"), (38, 36, 40))
    fonts = spec.get("fonts") or {}

    def font(name: str | None, size: int) -> Any:
        path = fonts.get(name or "body")
        if path is None:
            raise DocumentFaceError(f"Document uses undeclared font {name or 'body'}")
        resolved = Path(path) if Path(path).is_absolute() else root / path
        return ImageFont.truetype(str(resolved), size)

    image = Image.blend(Image.new("RGB", (width, height), paper),
                        Image.effect_noise((width, height), spec.get("grain", 6)).convert("RGB"), .035)
    draw = ImageDraw.Draw(image)
    y = spec.get("top", margin)
    right = width - margin
    for block in spec.get("blocks") or []:
        kind = block.get("kind")
        if kind not in _BLOCKS:
            raise DocumentFaceError(f"Unknown document block {kind}")
        colour = _colour(block.get("color"), ink)
        size = block.get("size", 60)
        if kind == "space":
            y += block.get("height", size)
        elif kind == "rule":
            y += block.get("gap", 30)
            draw.line((margin, y, right, y), fill=colour, width=block.get("weight", 3))
            y += block.get("gap", 30)
        elif kind == "text":
            face = font(block.get("font"), size)
            text = str(block["text"])
            anchor = {"left": margin, "center": (width - draw.textlength(text, font=face)) / 2,
                      "right": right - draw.textlength(text, font=face)}[block.get("align", "left")]
            draw.text((anchor, y), text, font=face, fill=colour)
            y += int(size * block.get("leading", 1.45))
        else:
            label_face = font(block.get("font"), size)
            value_size = block.get("value_size", size)
            value_face = font(block.get("value_font", block.get("font")), value_size)
            value = str(block["value"])
            draw.text((margin, y), str(block["label"]), font=label_face, fill=colour)
            value_width = draw.textlength(value, font=value_face)
            draw.text((right - value_width, y - (value_size - size) * .6), value, font=value_face,
                      fill=_colour(block.get("value_color"), colour))
            y += int(max(size, value_size) * block.get("leading", 1.45))
    if y > height:
        raise DocumentFaceError(f"Document content needs {y} px but the page is {height} px tall")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.filter(ImageFilter.GaussianBlur(spec.get("ink_soften", .6))).save(output)
    return {"output": str(output), "width": width, "height": height, "content_bottom": y}


def load_document_face(path: Path, output: Path) -> dict[str, Any]:
    return render_document_face(json.loads(path.read_text()), output, root=path.resolve().parent)
