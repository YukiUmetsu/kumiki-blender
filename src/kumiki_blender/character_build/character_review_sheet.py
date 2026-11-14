"""Create an honest side-by-side character art review sheet from Blender renders.

Never predicts whether a character is faithful to its source. References need
explicit camera/pose registration for pixel-level scores, which this stage
does not do. Nothing from the user's local reference art enters the repo.
"""
from __future__ import annotations

import json
from pathlib import Path

from .character_review_plan import confirm_review_sources


def make_character_review_sheets(output: str | Path, plan: dict) -> dict:
    try:
        from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
    except ImportError as exc:
        raise RuntimeError(
            "Character contact sheets require: pip install 'kumiki-blender[character-build]'"
        ) from exc
    confirm_review_sources(plan)
    root = Path(output)
    report_path = root / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("status") != "source-visual-review-required" or
            report.get("source_sha256") != plan["source_sha256"]):
        raise ValueError("Refusing to compose an unexpected or stale Blender visual report")
    expected = {
        (scope, view, frame)
        for scope in ("full", *(r["id"] for r in plan["regions"]))
        for view in plan["views"] for frame in plan["frames"]
    }
    found = {
        (item["scope"], item["view"], item["frame"])
        for item in report.get("images", [])
    }
    if len(report.get("images", [])) != len(expected) or found != expected:
        raise ValueError("Blender visual report is missing one or more exact camera/pose captures")
    for item in report["images"]:
        reference = (plan["reference_views"].get(item["view"])
                     if item["scope"] == "full" else
                     next(r["references"].get(item["view"]) for r in plan["regions"]
                          if r["id"] == item["scope"]))
        if item["reference"] != reference:
            raise ValueError("Reference filename/hash changed between rendering and contact sheet")

    width, height, padding, label_height = 408, 328, 14, 28
    columns = 2
    sheet_width = columns * width + padding * (columns + 1)
    output_files = []
    for scope in ["full", *(r["id"] for r in plan["regions"])]:
        for frame in plan["frames"]:
            rows = [next(item for item in report["images"] if
                         (item["scope"], item["view"], item["frame"]) ==
                         (scope, view, frame))
                    for view in plan["views"]]
            sheet_height = padding + len(rows) * (height + label_height + padding)
            sheet = Image.new("RGB", (sheet_width, sheet_height), (18, 22, 28))
            draw = ImageDraw.Draw(sheet)
            for idx, row in enumerate(rows):
                y = padding + idx * (height + label_height + padding)
                render_path = (root / row["render"]).resolve()
                if not render_path.is_relative_to(root.resolve()) or not render_path.is_file():
                    raise ValueError(f"Unexpected Blender render path {row['render']}")
                columns_to_draw = [
                    (row["reference"]["path"] if row["reference"] else None,
                     f"REFERENCE | {row['view']} | not aligned"),
                    (str(render_path), f"BLENDER CLAY | {row['view']} | frame {frame}"),
                ]
                for column, (filepath, label) in enumerate(columns_to_draw):
                    x = padding + column * (width + padding)
                    draw.text((x + 8, y + 6), label, fill=(234, 239, 243))
                    box = (x, y + label_height, x + width, y + label_height + height)
                    draw.rectangle(box, fill=(36, 43, 51))
                    if filepath is None:
                        draw.text((x + 22, y + label_height + height // 2),
                                  "No reference supplied", fill=(159, 168, 176))
                        continue
                    try:
                        with Image.open(filepath) as original:
                            if original.width * original.height > 16_777_216:
                                raise ValueError(f"Source or render exceeds 16 megapixels: {filepath}")
                            original.load()
                            picture = ImageOps.contain(
                                original.convert("RGB"), (width, height),
                                method=Image.Resampling.LANCZOS)
                    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
                        raise ValueError(f"Invalid local review image: {filepath}") from exc
                    sheet.paste(picture, (
                        x + (width - picture.width) // 2,
                        y + label_height + (height - picture.height) // 2,
                    ))
            filename = f"sheet-{scope}-f{frame:05d}.png"
            target = root / filename
            if target.exists():
                raise FileExistsError(f"Immutable review sheet exists: {target}")
            sheet.save(target, "PNG")
            output_files.append(filename)
    confirm_review_sources(plan)
    report["contact_sheets"] = output_files
    report["contact_sheet"] = output_files[0] if output_files else None
    report["limitations"].append(
        "Contact sheets preserve source camera/view labels but cannot register source perspective, poses, or lighting; visual review is mandatory.")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
