"""Reviewable foreground masks for registered, *rectified* statue photographs.

GrabCut supplies a CPU-only starting point, not proof of subject isolation.
Source photos remain immutable. Every generated mask requires visual review.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .dense import (DenseSettings, _json, _sha256,
                    plan_dense_reconstruction)
from .references import ReferenceError

_DEFAULT_BOX = (0.12, 0.04, 0.88, 0.96)
_ISNET_SHA256 = "60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a"
_SPHINX_PARTS = frozenset({"head", "headdress", "neck", "torso", "paws", "tail"})
_SAM2_CONFIGS = {"tiny": "t", "small": "s", "base_plus": "b+", "large": "l"}


def _normalized_pair(value: Any, label: str) -> tuple[float, float]:
    if (not isinstance(value, list) or len(value) != 2
            or any(type(x) not in (float, int) or not math.isfinite(x)
                   or x < 0 or x > 1 for x in value)):
        raise ReferenceError(f"{label} requires normalized [x,y] values in [0,1]")
    return float(value[0]), float(value[1])


def _box(value: Any) -> tuple[float, float, float, float]:
    if (not isinstance(value, (tuple, list)) or len(value) != 4
            or any(type(x) not in (int, float) or not math.isfinite(x)
                   or not 0 <= x <= 1 for x in value)):
        raise ReferenceError("Subject box needs four normalized [left,top,right,bottom] numbers")
    left, top, right, bottom = (float(x) for x in value)
    if right-left < 0.05 or bottom-top < 0.05:
        raise ReferenceError("Subject box must have positive width and height (at least 5%)")
    return left, top, right, bottom


def _guide(value: Any, *, default_box: tuple[float, ...]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) - {"box", "foreground_points", "background_points",
                                                     "component_boxes", "part_points", "part_visibility"}:
        raise ReferenceError("Mask guide contains unsupported fields")
    box = _box(value.get("box", default_box))
    points: dict[str, list[list[float]]] = {}
    for key in ("foreground_points", "background_points"):
        raw = value.get(key, [])
        if not isinstance(raw, list) or len(raw) > 100:
            raise ReferenceError(f"{key} requires at most 100 normalized points")
        points[key] = [list(_normalized_pair(item, key)) for item in raw]
    components = value.get("component_boxes", {})
    landmarks = value.get("part_points", {})
    visibility = value.get("part_visibility", {})
    if (not isinstance(components, dict) or not isinstance(landmarks, dict)
            or set(components) - _SPHINX_PARTS or set(landmarks) - _SPHINX_PARTS):
        raise ReferenceError("component_boxes and part_points must name Sphinx anatomy")
    if (not isinstance(visibility, dict) or set(visibility) - _SPHINX_PARTS
            or any(value not in ("visible", "not_visible", "uncertain")
                   for value in visibility.values())):
        raise ReferenceError("part_visibility must mark Sphinx anatomy visible/not_visible/uncertain")
    if len(components) > 6:
        raise ReferenceError("Too many component boxes")
    checked_boxes = {name: list(_box(bounds)) for name, bounds in components.items()}
    if checked_boxes and set(landmarks) - set(checked_boxes):
        raise ReferenceError("Every visible part landmark needs a matching component box")
    if checked_boxes:
        for x, y in points["foreground_points"]:
            if not any(left <= x <= right and top <= y <= bottom
                       for left, top, right, bottom in checked_boxes.values()):
                raise ReferenceError("Foreground point lies outside every component box")
    checked_points: dict[str, list[list[float]]] = {}
    for name, raw in landmarks.items():
        if not isinstance(raw, list) or not 1 <= len(raw) <= 20:
            raise ReferenceError(f"part_points for {name} require 1–20 normalized points")
        checked_points[name] = [list(_normalized_pair(point, name)) for point in raw]
        if name in checked_boxes:
            left, top, right, bottom = checked_boxes[name]
            if any(not (left <= x <= right and top <= y <= bottom)
                   for x, y in checked_points[name]):
                raise ReferenceError(f"{name} landmark must lie inside its component box")
    return {"box": list(box), **points,
            "component_boxes": checked_boxes, "part_points": checked_points,
            "part_visibility": dict(visibility)}


def _load_sam2_predictor(checkpoint: Path, size: str, requested_device: str) -> tuple[Any, str]:
    """Lazy, offline SAM 2.1 image predictor; no network downloads or CUDA requirement."""
    try:
        import torch  # type: ignore[import-not-found]
        from sam2.build_sam import build_sam2  # type: ignore[import-not-found]
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore[import-not-found]
    except ImportError as error:
        raise ReferenceError("SAM 2.1 requires torch and the official facebookresearch/sam2 package") from error
    device = ("mps" if torch.backends.mps.is_available() else "cpu") if requested_device == "auto" else requested_device
    if device == "mps" and not torch.backends.mps.is_available():
        raise ReferenceError("PyTorch MPS is unavailable; retry with --sam2-device cpu")
    cfg = f"configs/sam2.1/sam2.1_hiera_{_SAM2_CONFIGS[size]}.yaml"
    try:
        model = build_sam2(cfg, str(checkpoint), device=device, apply_postprocessing=False)
    except (RuntimeError, OSError, ValueError) as error:
        raise ReferenceError(f"Could not load SAM 2.1 {size} checkpoint: {error}") from error
    predictor = SAM2ImagePredictor(model)

    class InferencePredictor:
        def set_image(self, rgb: Any) -> None:
            with torch.inference_mode():
                predictor.set_image(rgb)

        def predict(self, **kwargs: Any) -> Any:
            with torch.inference_mode():
                return predictor.predict(**kwargs)

    return InferencePredictor(), device


def _sam2_mask(photo: Any, guide: dict[str, Any], predictor: Any,
               max_side: int = 1280) -> tuple[Any, dict[str, Any]]:
    """Union separately prompted visible anatomy; never use track votes to paint pixels."""
    import numpy as np  # type: ignore[import-not-found]
    from PIL import Image

    width, height = photo.size
    working = photo.convert("RGB")
    working.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    w, h = working.size
    predictor.set_image(np.asarray(working))
    components = guide.get("component_boxes") or {"subject": guide["box"]}
    merged = np.zeros((h, w), dtype=bool)
    for name, bounds in components.items():
        x0, y0, x1, y1 = bounds
        box = np.array([x0*(w-1), y0*(h-1), x1*(w-1), y1*(h-1)], dtype=np.float32)
        # The local box must not be forced to contain distant anatomy from
        # another component. Negative clicks apply everywhere.
        positives = [point for point in guide["foreground_points"]
                     if x0 <= point[0] <= x1 and y0 <= point[1] <= y1]
        positives += guide.get("part_points", {}).get(name, [])
        if name == "subject":
            positives += [point for part in guide.get("part_points", {}).values() for point in part]
        negatives = guide["background_points"]
        points = positives + negatives
        coords = (np.asarray([[x*(w-1), y*(h-1)] for x, y in points], dtype=np.float32)
                  if points else None)
        labels = (np.asarray([1]*len(positives) + [0]*len(negatives), dtype=np.int32)
                  if points else None)
        try:
            masks, scores, _ = predictor.predict(box=box, point_coords=coords,
                                                  point_labels=labels, multimask_output=True)
        except (RuntimeError, ValueError) as error:
            raise ReferenceError(f"SAM 2.1 failed for {name}; try CPU or refine the guide: {error}") from error
        masks = np.asarray(masks)
        scores = np.asarray(scores).reshape(-1)
        if masks.ndim != 3 or masks.shape[1:] != (h, w) or len(scores) != len(masks):
            raise ReferenceError("SAM 2.1 returned unexpected candidate mask dimensions")
        def grade(index: int) -> tuple[int, float]:
            candidate = masks[index] > 0
            compliance = 0
            if coords is not None:
                for (x, y), label in zip(coords, labels):
                    match = bool(candidate[min(h-1, round(float(y))), min(w-1, round(float(x)))])
                    compliance += int(match == bool(label))
            return compliance, float(scores[index]) if np.isfinite(scores[index]) else -1.0
        best = max(range(len(masks)), key=grade)
        merged |= masks[best] > 0
    binary = Image.fromarray(np.where(merged, 255, 0).astype(np.uint8), mode="L")
    if binary.size != (width, height):
        binary = binary.resize((width, height), Image.Resampling.NEAREST)
    output = np.asarray(binary)
    missing = []
    for part, landmarks in guide.get("part_points", {}).items():
        if any(not output[min(height-1, round(y*(height-1))),
                          min(width-1, round(x*(width-1)))] for x, y in landmarks):
            missing.append(part)
    negative_hits = sum(bool(output[min(height-1, round(y*(height-1))),
                                    min(width-1, round(x*(width-1)))])
                        for x, y in guide["background_points"])
    return output, {"components": list(components), "missing_parts": sorted(missing),
                    "background_seed_violations": negative_hits}


def _load_guides(path: str | Path | None, filenames: list[str],
                 default_box: tuple[float, ...]) -> tuple[dict[str, dict[str, Any]], str | None]:
    if path is None:
        return {name: _guide({}, default_box=default_box) for name in filenames}, None
    source = Path(path).resolve()
    if Path(path).is_symlink():
        raise ReferenceError("Mask guides cannot be a symlink")
    document = _json(source)
    if set(document) - {"default", "images"}:
        raise ReferenceError("Mask guide JSON may contain only default and images")
    image_guides = document.get("images", {})
    if not isinstance(image_guides, dict) or set(image_guides) - set(filenames):
        raise ReferenceError("Mask guide filenames must exactly identify registered rectified photos")
    default = _guide(document.get("default", {}), default_box=default_box)
    guides = {}
    for name in filenames:
        specific = image_guides.get(name, {})
        if not isinstance(specific, dict):
            raise ReferenceError("Per-photo mask guide must be an object")
        guides[name] = _guide({**default, **specific}, default_box=default_box)
    return guides, _sha256(source)


def _grabcut(photo: Any, guide: dict[str, Any], max_side: int,
             iterations: int) -> Any:
    """Run GrabCut at bounded resolution, then nearest-neighbor upsample binary labels."""
    import cv2  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]

    width, height = photo.size
    scale = min(1.0, max_side / max(width, height))
    small_w, small_h = max(8, round(width*scale)), max(8, round(height*scale))
    rgb = np.asarray(photo.convert("RGB"))
    small = cv2.resize(rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)
    left, top, right, bottom = guide["box"]
    x0, y0 = round(left*small_w), round(top*small_h)
    x1, y1 = round(right*small_w), round(bottom*small_h)
    x0, y0 = min(small_w-2, max(1, x0)), min(small_h-2, max(1, y0))
    x1, y1 = max(x0+1, min(small_w-1, x1)), max(y0+1, min(small_h-1, y1))
    mask = np.zeros((small_h, small_w), dtype=np.uint8)
    bg_model, fg_model = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), mask,
                    (x0, y0, x1-x0, y1-y0), bg_model, fg_model,
                    iterations, cv2.GC_INIT_WITH_RECT)
        if guide["foreground_points"] or guide["background_points"]:
            radius = max(2, round(min(small_w, small_h)*0.012))
            for key, label in (("foreground_points", cv2.GC_FGD),
                               ("background_points", cv2.GC_BGD)):
                for x, y in guide[key]:
                    cv2.circle(mask, (min(small_w-1, round(x*(small_w-1))),
                                      min(small_h-1, round(y*(small_h-1)))),
                               radius, label, thickness=-1)
            cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), mask, None,
                        bg_model, fg_model, iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error as error:
        raise ReferenceError("GrabCut could not segment this photograph; review its box/points") from error
    binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                      255, 0).astype(np.uint8)
    if not binary.any() or np.all(binary == 255):
        raise ReferenceError("GrabCut produced an empty or entirely foreground mask")
    if (small_w, small_h) != (width, height):
        binary = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
    return binary


def _rembg_mask(photo: Any, session: Any) -> Any:
    """Optional semantic foreground candidate, with an explicitly selected model."""
    import numpy as np  # type: ignore[import-not-found]
    from rembg import remove  # type: ignore[import-not-found]

    result = remove(photo.convert("RGB"), session=session, only_mask=True)
    if not hasattr(result, "convert"):
        raise ReferenceError("Background model did not return a Pillow mask")
    alpha = np.asarray(result.convert("L"))
    if alpha.shape != (photo.height, photo.width):
        raise ReferenceError("Background model returned a mask in the wrong image dimensions")
    binary = np.where(alpha >= 128, 255, 0).astype(np.uint8)
    if not binary.any() or np.all(binary == 255):
        raise ReferenceError("Background model produced an empty or entirely foreground mask")
    return binary


def _verified_model_sha(session: Any) -> str:
    """Pin the published rembg IS-Net ONNX, not just a changeable model name."""
    try:
        path = Path(session.model_dir()) / "isnet-general-use.onnx"
    except (AttributeError, TypeError) as error:
        raise ReferenceError("Rembg session cannot identify its model weights") from error
    if path.is_symlink() or not path.is_file() or _sha256(path) != _ISNET_SHA256:
        raise ReferenceError("IS-Net background model weights differ from the recorded SHA-256")
    return _ISNET_SHA256


def generate_rectified_masks(reconstruction: str | Path, output: str | Path, *,
                             guides: str | Path | None = None,
                             center_box: tuple[float, ...] = _DEFAULT_BOX,
                             max_side: int = 1280,
                             iterations: int = 5,
                             segmenter: str = "grabcut",
                             sam2_checkpoint: str | Path | None = None,
                             sam2_size: str = "small",
                             sam2_device: str = "auto") -> dict[str, Any]:
    """Make immutable mask candidates, overlays and QA metrics for every photo."""
    if type(max_side) is not int or not 256 <= max_side <= 4096:
        raise ReferenceError("Mask max_side must be 256–4096")
    if type(iterations) is not int or not 1 <= iterations <= 12:
        raise ReferenceError("GrabCut iterations must be 1–12")
    if segmenter not in ("grabcut", "rembg", "sam2"):
        raise ReferenceError("Mask segmenter must be grabcut, rembg or sam2")
    if segmenter == "rembg" and guides is not None:
        raise ReferenceError("Rembg uses semantic inference; use grabcut for manual per-photo box/point guides")
    if sam2_size not in _SAM2_CONFIGS or sam2_device not in ("auto", "cpu", "mps"):
        raise ReferenceError("SAM 2.1 size must be tiny/small/base_plus/large and device auto/cpu/mps")
    checkpoint = None
    if segmenter == "sam2":
        if sam2_checkpoint is None:
            raise ReferenceError("SAM 2.1 requires a local --sam2-checkpoint")
        checkpoint = Path(sam2_checkpoint)
        if (checkpoint.is_symlink() or not checkpoint.is_file()
                or not 0 < checkpoint.stat().st_size < 2_000_000_000):
            raise ReferenceError("SAM 2.1 checkpoint must be a nonempty regular local file")
        checkpoint = checkpoint.resolve()
    default_box = _box(center_box)
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:
        raise ReferenceError("Mask generation requires Pillow (install the masks extra)") from error
    session = None
    model_sha = None
    predictor = None
    device = None
    if segmenter == "sam2":
        assert checkpoint is not None
        model_sha = _sha256(checkpoint)
        predictor, device = _load_sam2_predictor(checkpoint, sam2_size, sam2_device)
    elif segmenter == "rembg":
        try:
            from rembg import new_session  # type: ignore[import-not-found]
        except ImportError as error:
            raise ReferenceError("Semantic masking requires pip install -e '.[mask-ai]'") from error
        # The library may download weights at first invocation. Never select
        # rembg's default (which can change to a differently licensed model).
        session = new_session("isnet-general-use")
        model_sha = _verified_model_sha(session)
    else:
        try:
            import cv2  # noqa: F401  # type: ignore[import-not-found]
        except ImportError as error:
            raise ReferenceError("GrabCut masking requires pip install -e '.[masks]'") from error
    plan = plan_dense_reconstruction(reconstruction, output,
                                     DenseSettings(min_registered_images=3,
                                                   min_registered_fraction=0.25,
                                                   refine=False, texture=False),
                                     executables={})
    source = Path(plan["source_revision"])
    index = _json(source / "undistorted-reference-index.json")
    images = {item["filename"]: item for item in index["images"]}
    guide_map, guide_sha = _load_guides(guides, plan["undistorted_images"], default_box)
    target = Path(plan["output"])
    target.mkdir(parents=True, exist_ok=False)
    mask_dir, preview_dir = target / "masks", target / "review"
    mask_dir.mkdir()
    preview_dir.mkdir()
    report: dict[str, Any] = {
        "schema_version": 1, "source_revision": str(source),
        "guide_sha256": guide_sha, "max_side": max_side, "iterations": iterations,
        "segmenter": segmenter,
        "segmentation_model": ("isnet-general-use" if segmenter == "rembg" else
                               f"sam2.1_hiera_{sam2_size}" if segmenter == "sam2" else None),
        "segmentation_model_sha256": model_sha,
        "segmentation_model_checkpoint": str(checkpoint) if checkpoint else None,
        "segmentation_device": device,
        "requires_anatomy_inventory": segmenter == "sam2",
        "status": "incomplete / manual-mask-review-required", "images": [],
        "approved_for_reconstruction": False,
    }

    def checkpoint() -> None:
        (target / "mask-report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                 encoding="utf-8")

    checkpoint()
    for filename in plan["undistorted_images"]:
        image_path = source / "undistorted/images" / filename
        try:
            with Image.open(image_path) as original:
                original.load()
                expected = images[filename]["dimensions"]
                if list(original.size) != expected:
                    raise ReferenceError(f"Rectified photo size changed: {filename}")
                details: dict[str, Any] = {}
                if segmenter == "rembg":
                    binary = _rembg_mask(original, session)
                elif segmenter == "sam2":
                    binary, details = _sam2_mask(original, guide_map[filename], predictor, max_side)
                else:
                    binary = _grabcut(original, guide_map[filename], max_side, iterations)
                if not binary.any() or (binary == 255).all():
                    raise ReferenceError(f"{segmenter} returned an empty or entirely foreground mask")
                mask = Image.fromarray(binary, mode="L")
                occupancy = round(sum(mask.histogram()[128:])/ (mask.width*mask.height), 5)
                warnings = []
                if segmenter == "grabcut" and guides is None:
                    warnings.append("Unreviewed center-box guess; add per-photo guides for busy backgrounds")
                if segmenter == "sam2" and not guide_map[filename]["part_points"]:
                    warnings.append("No visible-anatomy landmarks: mark head/torso/paws/tail where visible")
                if segmenter == "sam2" and guides is None:
                    warnings.append("SAM 2.1 guessed one central object box; annotate separate anatomy components")
                for part in details.get("missing_parts", []):
                    warnings.append(f"Missing {part} landmark: correct the box, prompts, or include polygon")
                if details.get("background_seed_violations", 0):
                    warnings.append("Foreground overlaps explicit background points; correct mask")
                if occupancy < 0.05 or occupancy > 0.90:
                    warnings.append("Suspicious foreground fraction; check seed box and background")
                elif occupancy > 0.45:
                    warnings.append("Large segmented region; verify that nearby architecture/foliage is excluded")
                edges = (mask.crop((0, 0, mask.width, 1)),
                         mask.crop((0, mask.height-1, mask.width, mask.height)),
                         mask.crop((0, 0, 1, mask.height)),
                         mask.crop((mask.width-1, 0, mask.width, mask.height)))
                if any(edge.getbbox() is not None for edge in edges):
                    warnings.append("Foreground touches photo border; inspect cropped statue")
                mask_path = mask_dir / (filename + ".mask.png")
                mask.save(mask_path, format="PNG", optimize=True)
                original.thumbnail((900, 900))
                base = original.convert("RGB")
                cover = Image.new("RGB", base.size, (238, 58, 35))
                visible = mask.resize(base.size, Image.Resampling.NEAREST).point(
                    lambda intensity: round(intensity * 0.28))
                overlay = Image.composite(cover, base, visible)
                preview = preview_dir / (filename + ".jpg")
                overlay.save(preview, "JPEG", quality=86)
                report["images"].append({
                    "filename": filename, "source_photo_sha256": images[filename]["sha256"],
                    "mask": str(mask_path.relative_to(target)), "mask_sha256": _sha256(mask_path),
                    "preview": str(preview.relative_to(target)),
                    "foreground_fraction": occupancy,
                    "guide": guide_map[filename], "warnings": warnings,
                    "missing_parts": details.get("missing_parts", []),
                    "background_seed_violations": details.get("background_seed_violations", 0),
                    "parts_verified": sorted(guide_map[filename]["part_points"]),
                    "segmented_components": details.get("components", []),
                })
        except Exception as error:
            report.update(failed_image=filename, error=str(error))
            checkpoint()
            raise
        checkpoint()
    columns, cell_width, cell_height = 4, 288, 238
    rows = (len(report["images"]) + columns - 1) // columns
    contact = Image.new("RGB", (columns * cell_width, rows * cell_height), (28, 31, 34))
    draw = ImageDraw.Draw(contact)
    for index, item in enumerate(report["images"]):
        x = index % columns * cell_width
        y = index // columns * cell_height
        with Image.open(target / item["preview"]) as preview:
            preview.thumbnail((cell_width-8, cell_height-30))
            contact.paste(preview, (x+4, y+4))
        draw.text((x+6, y+cell_height-23), item["filename"][:42], fill=(234, 236, 238))
    contact.save(target / "mask-contact-sheet.jpg", "JPEG", quality=85)
    report["contact_sheet"] = "mask-contact-sheet.jpg"
    report["status"] = "candidate / inspect-every-mask-before-dense-reconstruction"
    checkpoint()
    return report
