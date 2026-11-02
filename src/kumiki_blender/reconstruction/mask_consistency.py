"""Audit statue masks against real multi-view COLMAP point tracks.

The same triangulated feature should normally be classified consistently in
each rectified photograph where it was observed. This detects disagreements
without assuming that every sparse point belongs to the statue. It is an
inspectable QA signal, never automatic approval of a mask or 3D model.
"""

from __future__ import annotations

import json
import math
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any, BinaryIO

from .dense import (DenseSettings, _check_regular, _json, _sha256,
                    plan_dense_reconstruction)
from .references import ReferenceError


_MAX_IMAGES = 256
_MAX_TRACKED_OBSERVATIONS = 2_000_000
_MAX_MODEL_BYTES = 512 * 1024 * 1024
_IMAGE_HEADER = struct.Struct("<i7di")
_FEATURE = struct.Struct("<ddq")
_COUNT = struct.Struct("<Q")


def _exact(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ReferenceError("COLMAP rectified image model is truncated")
    return data


def _binary_image_tracks(path: Path) -> dict[str, list[tuple[int, float, float]]]:
    entries: dict[str, list[tuple[int, float, float]]] = {}
    image_ids: set[int] = set()
    observed = 0
    with path.open("rb") as stream:
        (count,) = _COUNT.unpack(_exact(stream, _COUNT.size))
        if not 3 <= count <= _MAX_IMAGES:
            raise ReferenceError("Invalid registered image count in COLMAP model")
        for _ in range(count):
            values = _IMAGE_HEADER.unpack(_exact(stream, _IMAGE_HEADER.size))
            image_id = values[0]
            if image_id <= 0 or image_id in image_ids:
                raise ReferenceError("Duplicate image ID in COLMAP model")
            image_ids.add(image_id)
            chars = bytearray()
            for _ in range(256):
                char = _exact(stream, 1)
                if char == b"\0":
                    break
                chars.extend(char)
            else:
                raise ReferenceError("COLMAP image filename exceeds 255 bytes")
            try:
                name = chars.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ReferenceError("COLMAP image name is not UTF-8") from error
            if not name or name in entries:
                raise ReferenceError("Duplicate or empty image name in COLMAP model")
            (feature_count,) = _COUNT.unpack(_exact(stream, _COUNT.size))
            observed += feature_count
            if observed > _MAX_TRACKED_OBSERVATIONS:
                raise ReferenceError("COLMAP model exceeds safe feature observation budget")
            features = []
            for _ in range(feature_count):
                x, y, point_id = _FEATURE.unpack(_exact(stream, _FEATURE.size))
                if point_id < -1:
                    raise ReferenceError("Invalid negative COLMAP point ID")
                if point_id >= 0:
                    if not math.isfinite(x) or not math.isfinite(y):
                        raise ReferenceError("Invalid COLMAP sparse feature coordinates")
                    features.append((point_id, x, y))
            entries[name] = features
        if stream.read(1):
            raise ReferenceError("Unexpected trailing bytes in COLMAP image model")
    return entries


def _text_image_tracks(path: Path) -> dict[str, list[tuple[int, float, float]]]:
    entries: dict[str, list[tuple[int, float, float]]] = {}
    image_ids: set[int] = set()
    observed = 0
    with path.open("r", encoding="utf-8") as stream:
        while True:
            header = stream.readline()
            if not header:
                break
            if not header.strip() or header.startswith("#"):
                continue
            fields = header.split(maxsplit=9)
            if len(fields) != 10:
                raise ReferenceError("Invalid COLMAP text image header")
            try:
                image_id = int(fields[0])
            except ValueError as error:
                raise ReferenceError("Invalid COLMAP image ID") from error
            name = fields[9].strip()
            if image_id <= 0 or image_id in image_ids or not name or name in entries:
                raise ReferenceError("Duplicate or empty image in COLMAP text model")
            image_ids.add(image_id)
            points = stream.readline()
            if not points:
                raise ReferenceError("Missing COLMAP text feature line")
            tokens = points.split()
            if len(tokens) % 3:
                raise ReferenceError("Invalid COLMAP text feature triples")
            observed += len(tokens) // 3
            if observed > _MAX_TRACKED_OBSERVATIONS:
                raise ReferenceError("COLMAP model exceeds safe feature observation budget")
            features = []
            try:
                for index in range(0, len(tokens), 3):
                    point_id = int(tokens[index+2])
                    if point_id < -1:
                        raise ReferenceError("Invalid negative COLMAP point ID")
                    if point_id >= 0:
                        x, y = float(tokens[index]), float(tokens[index+1])
                        if not math.isfinite(x) or not math.isfinite(y):
                            raise ReferenceError("Invalid COLMAP sparse feature coordinates")
                        features.append((point_id, x, y))
            except ValueError as error:
                raise ReferenceError("Invalid COLMAP text feature values") from error
            entries[name] = features
    if not 3 <= len(entries) <= _MAX_IMAGES:
        raise ReferenceError("Invalid registered image count in COLMAP text model")
    return entries


def audit_multiview_masks(reconstruction: str | Path, masks: str | Path,
                          output: str | Path, *, max_overlay_points: int = 200) -> dict[str, Any]:
    """Write hash-pinned cross-view disagreement overlays and a review report."""
    if type(max_overlay_points) is not int or not 1 <= max_overlay_points <= 2000:
        raise ReferenceError("max_overlay_points must be an integer in [1, 2000]")
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:
        raise ReferenceError("Mask consistency QA requires Pillow (install the masks extra)") from error
    mask_root = Path(masks).resolve()
    target = Path(output).resolve()
    if (Path(masks).is_symlink() or not mask_root.is_dir()
            or Path(output).is_symlink() or target.is_relative_to(mask_root)):
        raise ReferenceError("Audit output must be separate from the mask source")
    plan = plan_dense_reconstruction(
        reconstruction, output,
        DenseSettings(min_registered_images=3, min_registered_fraction=0.25,
                      refine=False, texture=False),
        rectified_masks=masks, executables={})
    source = Path(plan["source_revision"])
    model_root = source / "undistorted/sparse"
    model_bin = model_root / "images.bin"
    model_txt = model_root / "images.txt"
    model = model_bin if model_bin.exists() else model_txt
    _check_regular(model, source)
    if model.stat().st_size > _MAX_MODEL_BYTES:
        raise ReferenceError("COLMAP image track model exceeds safe file size")
    tracks_by_image = (_binary_image_tracks(model) if model.suffix == ".bin" else
                       _text_image_tracks(model))
    names = plan["undistorted_images"]
    if set(tracks_by_image) != set(names):
        raise ReferenceError("COLMAP point-track image names differ from rectified photos")
    mask_hashes = {item["filename"]: item["sha256"] for item in plan["rectified_masks"]}
    image_meta = {name: {"filename": name, "source_photo_sha256": None,
                         "mask_sha256": mask_hashes[name],
                         "tracked_observations": 0, "out_of_frame_observations": 0,
                         "compared_observations": 0, "consensus_outliers": 0,
                         "ambiguous_observations": 0, "foreground_observations": 0,
                         "background_observations": 0, "warnings": []} for name in names}
    index = _json(source / "undistorted-reference-index.json")
    for item in index["images"]:
        image_meta[item["filename"]]["source_photo_sha256"] = item["sha256"]
    observed_points: dict[int, list[tuple[str, float, float, bool]]] = defaultdict(list)
    for name in names:
        with Image.open(mask_root / (name + ".mask.png")) as mask:
            decoded = mask.convert("L")
            pixels = decoded.load()
            width, height = mask.size
            seen_ids = set()
            for point_id, x, y in tracks_by_image[name]:
                if point_id in seen_ids:
                    raise ReferenceError("COLMAP repeats one 3D track within an image")
                seen_ids.add(point_id)
                if not 0 <= x < width or not 0 <= y < height:
                    image_meta[name]["out_of_frame_observations"] += 1
                    continue
                foreground = bool(pixels[min(width-1, round(x)),
                                         min(height-1, round(y))])
                observed_points[point_id].append((name, x, y, foreground))
                image_meta[name]["tracked_observations"] += 1
                image_meta[name]["foreground_observations" if foreground
                                 else "background_observations"] += 1
    # Keep red outliers ahead of yellow ties even when the preview budget is
    # saturated. A camera with thousands of ambiguous tracks must not hide
    # the actual majority disagreements that warrant its high review priority.
    red_overlays: dict[str, list[tuple[float, float]]] = {name: [] for name in names}
    yellow_overlays: dict[str, list[tuple[float, float]]] = {name: [] for name in names}
    compared_tracks, mixed_tracks, ambiguous_tracks = 0, 0, 0
    for observations in observed_points.values():
        if len(observations) < 2:
            continue
        compared_tracks += 1
        for name, _x, _y, _flag in observations:
            image_meta[name]["compared_observations"] += 1
        positives = sum(flag for _name, _x, _y, flag in observations)
        negatives = len(observations) - positives
        if not positives or not negatives:
            continue
        mixed_tracks += 1
        tied = positives == negatives
        ambiguous_tracks += tied
        majority = positives > negatives
        for name, x, y, flag in observations:
            if tied:
                image_meta[name]["ambiguous_observations"] += 1
                points = yellow_overlays[name]
            elif flag != majority:
                image_meta[name]["consensus_outliers"] += 1
                points = red_overlays[name]
            else:
                continue
            if len(points) < max_overlay_points:
                points.append((x, y))
    if compared_tracks == 0:
        raise ReferenceError("No shared 3D feature tracks; cross-view mask QA cannot run")
    for name in names:
        item = image_meta[name]
        item["consensus_outlier_fraction"] = round(
            item["consensus_outliers"] / max(1, item["compared_observations"]), 5)
        item["review_priority"] = ("high" if item["consensus_outliers"] else
                                   "medium" if item["ambiguous_observations"] else "normal")
        if item["compared_observations"] < 25:
            item["warnings"].append("Few shared sparse points; cannot infer mask correctness")
        if item["consensus_outliers"]:
            item["warnings"].append("Track labels disagree with multi-view majority; inspect red points")
        if item["ambiguous_observations"]:
            item["warnings"].append("Some tracks have no majority label; inspect yellow points")
        if item["out_of_frame_observations"]:
            item["warnings"].append("Some sparse features fall outside rectified image boundaries")
    target.mkdir(parents=True, exist_ok=False)
    (target / "review").mkdir()
    for name in names:
        with Image.open(source / "undistorted/images" / name) as photo:
            original_width, original_height = photo.size
            preview = photo.convert("RGB")
            preview.thumbnail((1000, 1000))
            draw = ImageDraw.Draw(preview)
            selected_red = red_overlays[name][:max_overlay_points]
            selected_yellow = yellow_overlays[name][:max_overlay_points-len(selected_red)]
            # Paint red last so majority disagreements are visible on top.
            for points, color in ((selected_yellow, (255, 212, 46)),
                                  (selected_red, (250, 58, 40))):
                for x, y in points:
                    cx = round(x * preview.width / original_width)
                    cy = round(y * preview.height / original_height)
                    radius = 5
                    draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius),
                                 outline=color, width=3)
            preview_file = target / "review" / (name + ".jpg")
            preview.save(preview_file, "JPEG", quality=86)
            image_meta[name]["review"] = str(preview_file.relative_to(target))
    columns, cell_w, cell_h = 4, 288, 238
    contact = Image.new("RGB", (columns*cell_w,
                                 ((len(names)+columns-1)//columns)*cell_h),
                        (28, 31, 34))
    label_draw = ImageDraw.Draw(contact)
    ordered = sorted(names, key=lambda name: (0 if image_meta[name]["review_priority"] == "high"
                                               else 1 if image_meta[name]["review_priority"] == "medium"
                                               else 2, name))
    for index, name in enumerate(ordered):
        x, y = (index % columns)*cell_w, (index // columns)*cell_h
        with Image.open(target / image_meta[name]["review"]) as preview:
            preview.thumbnail((cell_w-8, cell_h-34))
            contact.paste(preview, (x+4, y+4))
        label_draw.text((x+6, y+cell_h-26), f"{name[:22]}: {image_meta[name]['review_priority']}",
                        fill=(235, 236, 238))
    contact.save(target / "mask-consistency-contact-sheet.jpg", "JPEG", quality=85)
    report = {
        "schema_version": 1, "source_revision": str(source),
        "rectified_mask_source": str(mask_root),
        "model_file": str(model.relative_to(source)),
        "model_sha256": _sha256(model),
        "max_overlay_points": max_overlay_points,
        "compared_3d_tracks": compared_tracks,
        "mixed_label_tracks": mixed_tracks,
        "ambiguous_label_tracks": ambiguous_tracks,
        "contact_sheet": "mask-consistency-contact-sheet.jpg",
        "images": [image_meta[name] for name in names],
        "approved_for_reconstruction": False, "distribution_allowed": False,
        "status": "candidate / multi-view-mask-review-required",
        "limitations": "Sparse point tracks sample only matched features; agreement cannot prove foreground quality.",
    }
    (target / "mask-consistency-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
