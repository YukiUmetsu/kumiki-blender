"""Private, revisioned facial evidence manifests and human review overlays."""

from __future__ import annotations

import json
import html
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Sequence

from .inputs import parse_annotations_data
from .schema import AnnotationSet, Photo


def _read_json(path: Path, *, max_bytes: int = 2 * 1024 * 1024) -> Any:
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError(f"Missing, linked, or oversized JSON: {path.name}")
    try:
        return json.loads(path.read_text(encoding="utf-8"),
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(f"Cannot parse JSON {path.name}: {error}") from error


def _annotation_record(view: Any, photo: Photo) -> dict[str, Any]:
    return {
        "image_id": photo.image_id, "width": photo.width, "height": photo.height,
        "view": view.view if view else photo.view, "sha256": photo.sha256,
        "provenance": photo.provenance,
        "landmarks": [{"landmark_id": item.landmark_id, "xy_px": list(item.xy_px)
                       if item.xy_px is not None else None,
                       "origin": item.origin, "visibility": item.visibility}
                      for item in (view.landmarks if view else ())],
        "outline_px": [list(point) for point in view.outline_px]
                      if view and view.outline_px else None,
    }


def evidence_bundle(photos: Sequence[Photo], annotations: AnnotationSet | None) -> dict[str, Any]:
    """Self-contained *2D annotation* bundle; no input photo bytes are embedded."""
    return {"schema_version": 1, "images": [
        _annotation_record(annotations.images.get(photo.image_id) if annotations else None, photo)
        for photo in photos]}


def load_evidence_bundle(path: Path) -> AnnotationSet:
    """Validate persisted pixel observations using their recorded dimensions."""
    raw = _read_json(path)
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "images"} or raw["schema_version"] != 1:
        raise ValueError("Evidence bundle must contain schema_version=1 and images")
    rows = raw["images"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 256:
        raise ValueError("Evidence bundle needs 1–256 image entries")
    photos = []
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid evidence image entry")
        name, width, height = row.get("image_id"), row.get("width"), row.get("height")
        if (not isinstance(name, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name)
                or len(name) > 96 or type(width) is not int or type(height) is not int
                or width < 64 or height < 64 or width * height > 40_000_000):
            raise ValueError("Invalid evidence image identity/dimensions")
        digest = row.get("sha256", "")
        if digest and (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError("Invalid evidence image hash")
        photos.append(Photo(name, Path(path), width, height, digest))
        normalized.append({key: row[key] for key in ("image_id", "view", "landmarks", "outline_px")
                           if key in row})
    return parse_annotations_data({"schema_version": 1, "images": normalized}, photos)


def load_pairs(path: Path) -> list[dict[str, Any]]:
    raw = _read_json(path)
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "pairs"} or raw["schema_version"] != 1:
        raise ValueError("View-pair manifest needs schema_version=1 and pairs")
    rows = raw["pairs"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 256:
        raise ValueError("View-pair manifest needs 1–256 pairs")
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {
                "reference_image_id", "candidate_image_id", "alignment_id", "verified"}
                or any(not isinstance(row[key], str) or len(row[key]) > 128
                       for key in ("reference_image_id", "candidate_image_id", "alignment_id"))
                or type(row["verified"]) is not bool):
            raise ValueError("Invalid view-pair entry")
    return rows


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _render_overlay(photo: Photo, view: Any, suggestions: list[dict[str, Any]], target: Path) -> None:
    from PIL import Image, ImageDraw

    with Image.open(photo.path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    if view and view.outline_px:
        draw.line(list(view.outline_px) + [view.outline_px[0]], fill="#47d7bd", width=3)
    markers = []
    for point in view.landmarks if view else ():
        if point.xy_px is not None and point.visibility in ("visible", "uncertain"):
            markers.append((point.landmark_id, point.xy_px, point.origin))
    markers.extend((item["landmark_id"], item["xy_px"], "proposed") for item in suggestions)
    for label, (x, y), origin in markers:
        color = "#ffd06c" if origin == "proposed" else "#58f0be"
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=color, width=3)
        draw.text((x + 6, y + 6), label, fill=color, stroke_fill="#101b25", stroke_width=2)
    image.save(target, format="PNG")


def _render_contact_sheet(photos: Sequence[Photo], annotations: AnnotationSet | None,
                          target: Path) -> None:
    """Private thumbnail index for manually reviewing which views were actually supplied."""
    from PIL import Image, ImageDraw, ImageOps

    thumb_width, thumb_height, footer = 235, 200, 44
    gap, margin, columns = 12, 12, min(4, len(photos))
    rows = (len(photos) + columns - 1) // columns
    width = margin * 2 + columns * thumb_width + (columns - 1) * gap
    height = margin * 2 + rows * (thumb_height + footer) + (rows - 1) * gap
    sheet = Image.new("RGB", (width, height), "#1c2530")
    draw = ImageDraw.Draw(sheet)
    for index, photo in enumerate(photos):
        x = margin + (index % columns) * (thumb_width + gap)
        y = margin + (index // columns) * (thumb_height + footer + gap)
        with Image.open(photo.path) as source:
            thumbnail = ImageOps.contain(source.convert("RGB"), (thumb_width, thumb_height))
        sheet.paste(thumbnail, (x + (thumb_width - thumbnail.width) // 2,
                                y + (thumb_height - thumbnail.height) // 2))
        view = annotations.images.get(photo.image_id) if annotations else None
        label = view.view if view else photo.view
        draw.text((x + 2, y + thumb_height + 3), photo.image_id[:34], fill="#f3f6fa")
        draw.text((x + 2, y + thumb_height + 20), label, fill="#b0cfde")
    sheet.save(target, format="PNG")


def _render_registration_svg(pair: dict[str, Any], target: Path) -> None:
    """A private, geometry-only diagnostic; no source image pixels are included."""
    outlines = [pair.get("reference_outline_px"), pair.get("candidate_aligned_outline_px")]
    points = [point for outline in outlines if outline for point in outline]
    for name in ("anchor_reference_px", "anchor_candidate_aligned_px"):
        points.extend(pair.get(name, {}).values())
    for observation in pair.get("heldout", {}).values():
        points.extend((observation["reference_px"], observation["candidate_aligned_px"]))
    if not points:
        return
    xs, ys = [point[0] for point in points], [point[1] for point in points]
    span = max(1.0, max(max(xs)-min(xs), max(ys)-min(ys)))
    padding = max(15.0, span * 0.12)
    left, top = min(xs)-padding, min(ys)-padding
    width, height = max(xs)-min(xs)+2*padding, max(ys)-min(ys)+2*padding
    def fmt(value: float) -> str:
        return f"{value:.3f}"

    def polygon(outline: Any, stroke: str, label: str, dash: str) -> str:
        if not outline:
            return ""
        vertices = " ".join(f"{fmt(x)},{fmt(y)}" for x, y in outline)
        return (f'<polygon points="{vertices}" fill="none" stroke="{stroke}" '
                f'stroke-width="2" stroke-dasharray="{dash}" vector-effect="non-scaling-stroke" '
                f'aria-label="{label}"/>')

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" role="img" '
        'aria-label="UNVERIFIED 2D alignment; not calibrated camera geometry" '
        f'viewBox="{fmt(left)} {fmt(top)} {fmt(width)} {fmt(height)}" '
        'width="800" height="800">',
        '<rect x="' + fmt(left) + '" y="' + fmt(top) + '" width="' + fmt(width)
        + '" height="' + fmt(height) + '" fill="#121e27"/>',
        polygon(outlines[0], "#43dbce", "reference outline", "none"),
        polygon(outlines[1], "#ffb56b", "candidate aligned outline", "6 4"),
    ]
    radius = max(1.6, span * 0.008)
    for name, reference_point in sorted(pair.get("anchor_reference_px", {}).items()):
        candidate_point = pair["anchor_candidate_aligned_px"][name]
        parts.append(f'<circle cx="{fmt(reference_point[0])}" cy="{fmt(reference_point[1])}" '
                     f'r="{fmt(radius)}" fill="#43dbce"/>')
        parts.append(f'<circle cx="{fmt(candidate_point[0])}" cy="{fmt(candidate_point[1])}" '
                     f'r="{fmt(radius)}" stroke="#ffb56b" fill="none" stroke-width="2"/>')
    for name, observation in sorted(pair.get("heldout", {}).items()):
        ref = observation["reference_px"]
        aligned = observation["candidate_aligned_px"]
        parts.append(f'<line x1="{fmt(ref[0])}" y1="{fmt(ref[1])}" '
                     f'x2="{fmt(aligned[0])}" y2="{fmt(aligned[1])}" '
                     'stroke="#f169a7" stroke-width="2" vector-effect="non-scaling-stroke"/>')
        parts.append(f'<text x="{fmt(aligned[0]+radius*2)}" y="{fmt(aligned[1])}" '
                     f'font-size="{fmt(max(6.0, span*0.018))}" fill="#ffe9f4">'
                     f'{html.escape(name)}</text>')
    label_size = max(3.0, span * 0.018)
    parts.append(f'<text x="{fmt(left+padding/3)}" y="{fmt(top+padding/2)}" '
                 f'font-size="{fmt(label_size)}" fill="#f9d38a">UNVERIFIED 2D</text>')
    parts.append(f'<text x="{fmt(left+padding/3)}" y="{fmt(top+padding/2+label_size*1.4)}" '
                 f'font-size="{fmt(label_size*0.7)}" fill="#c9d6dd">'
                 'teal: reference | orange: render | pink: held-out</text>')
    parts.append('</svg>')
    target.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_analysis_revision(output: Path, audit: dict[str, Any],
                            proposals: dict[str, Any] | None,
                            comparisons: dict[str, Any] | None, *,
                            photos: Sequence[Photo] = (),
                            annotations: AnnotationSet | None = None,
                            reprojection: dict[str, Any] | None = None,
                            registration: dict[str, Any] | None = None,
                            workbench_html: str | None = None) -> Path:
    """Create one new immutable revision; never overwrite existing user results."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError(f"Output revision already exists: {output}")
    parent = output.parent.resolve()
    if not parent.is_dir():
        raise ValueError(f"Output parent directory does not exist: {parent}")
    with tempfile.TemporaryDirectory(prefix=".face-analysis-stage-", dir=parent) as temporary:
        staged = Path(temporary) / "revision"
        # Embedded photographic previews must remain inaccessible to other
        # local accounts even after this directory is moved out of the
        # private TemporaryDirectory (normally mode 0700 on POSIX).
        staged.mkdir(mode=0o700 if workbench_html is not None else 0o777)
        files = []
        if audit:
            _write_json(staged / "photo-audit.json", audit)
            files.append("photo-audit.json")
        if photos:
            _write_json(staged / "face-evidence.json", evidence_bundle(photos, annotations))
            files.append("face-evidence.json")
            _render_contact_sheet(photos, annotations, staged / "contact-sheet.png")
            files.append("contact-sheet.png")
        if workbench_html is not None:
            html_path = staged / "annotation-workbench.html"
            html_path.write_text(workbench_html, encoding="utf-8")
            if os.name == "posix":
                html_path.chmod(0o600)
            files.append("annotation-workbench.html")
        if proposals is not None:
            _write_json(staged / "landmark-proposals.json", proposals)
            files.append("landmark-proposals.json")
        if comparisons is not None:
            _write_json(staged / "likeness-qa.json", comparisons)
            files.append("likeness-qa.json")
        if reprojection is not None:
            _write_json(staged / "landmark-qa.json", reprojection)
            files.append("landmark-qa.json")
        if registration is not None:
            _write_json(staged / "registration-proposals.json", registration)
            files.append("registration-proposals.json")
            for pair in registration.get("pairs", []):
                if pair.get("status") != "proposed-unverified":
                    continue
                for field in ("reference_image_id", "candidate_image_id"):
                    image_id = pair.get(field)
                    if (not isinstance(image_id, str) or len(image_id) > 96
                            or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", image_id)):
                        raise ValueError("Unsafe registration image ID")
                overlays = staged / "overlays"
                overlays.mkdir(exist_ok=True)
                stem = f"registration-{pair['reference_image_id']}--{pair['candidate_image_id']}"
                _render_registration_svg(pair, overlays / f"{stem}.svg")
                files.append(f"overlays/{stem}.svg")
        proposal_index = {item["image_id"]: item.get("landmarks", [])
                          for item in proposals.get("images", [])} if proposals else {}
        if photos and (annotations or proposal_index):
            for photo in photos:
                view = annotations.images.get(photo.image_id) if annotations else None
                suggestions = proposal_index.get(photo.image_id, [])
                has_visible_points = view and any(item.xy_px is not None
                                                   and item.visibility in ("visible", "uncertain")
                                                   for item in view.landmarks)
                if not (suggestions or has_visible_points or (view and view.outline_px)):
                    continue
                overlays = staged / "overlays"
                overlays.mkdir(exist_ok=True)
                _render_overlay(photo, view, suggestions,
                                overlays / f"{photo.image_id}.png")
                files.append(f"overlays/{photo.image_id}.png")
        measured = comparisons is not None and any(
            pair.get("status") == "measured" for pair in comparisons.get("pairs", []))
        status = "projected-2d-measurements-only" if measured else "unmeasured"
        _write_json(staged / "manifest.json", {"schema_version": 1,
                                                "likeness_status": status,
                                                "three_d_fidelity_verified": False,
                                                "contains_private_photo_previews": workbench_html is not None,
                                                "files": sorted(files)})
        if output.exists() or output.is_symlink():
            raise ValueError(f"Output revision already exists: {output}")
        os.rename(staged, output)
    return output
