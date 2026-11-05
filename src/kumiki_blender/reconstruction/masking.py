"""Reviewable foreground masks for registered, *rectified* statue photographs.

GrabCut supplies a CPU-only starting point, not proof of subject isolation.
Source photos remain immutable. Every generated mask requires visual review.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .dense import (DenseSettings, _json, _sha256,
                    plan_dense_reconstruction)
from .references import ReferenceError

_DEFAULT_BOX = (0.12, 0.04, 0.88, 0.96)
_ISNET_SHA256 = "60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a"
# Legacy monument-specific audit utility retains this vocabulary. General
# segmentation guides deliberately accept any bounded object-part labels.
_SPHINX_PARTS = frozenset({"head", "headdress", "neck", "torso", "paws", "tail"})
_PART_NAME = re.compile(r"[a-z][a-z0-9_-]{0,31}\Z")
_SAM2_CONFIGS = {"tiny": "t", "small": "s", "base_plus": "b+", "large": "l"}
_MASK_CACHE_SCHEMA = "rectified-mask-v1"


def _cached_mask(cache_root: Path | None, key: str, dimensions: list[int]):
    """Only reuse an exact-content mask with independently verified pixels."""
    if cache_root is None:
        return None
    from PIL import Image

    folder = cache_root / key[:2] / key
    mask_file = folder / "mask.png"
    metadata = folder / "receipt.json"
    if any(path.is_symlink() for path in (folder.parent, folder, mask_file, metadata)):
        raise ReferenceError("Mask cache contains a symbolic link; refusing unsafe reuse")
    if not mask_file.is_file() or not metadata.is_file():
        return None
    try:
        record = json.loads(metadata.read_text(encoding="utf-8"))
        if record.get("schema") != _MASK_CACHE_SCHEMA or record.get("key") != key:
            return None
        if record.get("dimensions") != dimensions or record.get("sha256") != _sha256(mask_file):
            return None
        with Image.open(mask_file) as mask:
            mask.load()
            if mask.mode != "L" or list(mask.size) != dimensions:
                return None
            histogram = mask.histogram()
            if sum(histogram[1:255]) or histogram[0] == 0 or histogram[255] == 0:
                return None
        details = record.get("details")
        if not isinstance(details, dict):
            return None
        return mask_file, details
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _save_mask_cache(cache_root: Path | None, key: str, mask_file: Path,
                     dimensions: list[int], details: dict) -> None:
    if cache_root is None:
        return
    folder = cache_root / key[:2] / key
    if folder.parent.is_symlink() or folder.is_symlink():
        raise ReferenceError("Mask cache entry cannot be a symbolic link")
    folder.mkdir(parents=True, exist_ok=True)
    for filename in ("mask.png", "receipt.json"):
        if (folder / filename).is_symlink():
            raise ReferenceError("Mask cache entry contains a symbolic link")
    # Atomic single-file replacements; the receipt is committed last. A crash
    # between writes is treated as a cache miss, never as approved evidence.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=folder, prefix=".mask-", suffix=".png",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            with mask_file.open("rb") as source:
                shutil.copyfileobj(source, stream)
        os.replace(temporary, folder / "mask.png")
        temporary = None
        receipt = {"schema": _MASK_CACHE_SCHEMA, "key": key,
                   "dimensions": dimensions, "sha256": _sha256(mask_file),
                   "details": details}
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=folder,
                                         prefix=".receipt-", suffix=".json",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(receipt, stream, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, folder / "receipt.json")
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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
            or any(not isinstance(name, str) or not _PART_NAME.fullmatch(name)
                   for name in set(components) | set(landmarks))):
        raise ReferenceError("component_boxes and part_points require safe object-part names")
    if (not isinstance(visibility, dict) or any(
            not isinstance(name, str) or not _PART_NAME.fullmatch(name) for name in visibility)
            or any(value not in ("visible", "not_visible", "uncertain")
                   for value in visibility.values())):
        raise ReferenceError("part_visibility must mark object parts visible/not_visible/uncertain")
    if len(components) > 9 or len(landmarks) > 9 or len(visibility) > 9:
        raise ReferenceError("Too many named object parts")
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


def _auto_sam2_guide(photo: Any, default_box: tuple[float, ...]) -> dict[str, Any]:
    """Conservative automatic blue-sky negative prompts, not anatomy claims.

    An ambiguous view receives one broad subject box. The result is only a
    candidate, checked against multi-view tracks and blue-sky pixels later.
    """
    from PIL import Image

    tiny = photo.convert("RGB")
    tiny.thumbnail((256, 256), Image.Resampling.BILINEAR)
    negatives = []
    for y in (0.05, 0.13, 0.21, 0.29, 0.37, 0.45):
        for x in (0.04, 0.20, 0.36, 0.64, 0.80, 0.96):
            r, g, b = tiny.getpixel((round(x * (tiny.width - 1)), round(y * (tiny.height - 1))))
            if b > 120 and b > 1.15 * r and b > 1.06 * g:
                negatives.append([x, y])
    return _guide({"background_points": negatives[:24]}, default_box=default_box)


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
    maximum_candidate_disagreement = 0.0
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
        best_compliance, best_score = grade(best)
        # SAM2's mask-score ranking is not evidence that another plausible
        # silhouette is wrong. If equally prompt-compliant candidates disagree
        # strongly, mark the image uncertain in early QA instead of pushing
        # potentially background-based features into the "subject" class.
        for index in range(len(masks)):
            compliance, score = grade(index)
            if (index != best and compliance == best_compliance and
                    score >= best_score - 0.12):
                disagreement = float(np.mean((masks[best] > 0) != (masks[index] > 0)))
                maximum_candidate_disagreement = max(maximum_candidate_disagreement,
                                                      disagreement)
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
                    "background_seed_violations": negative_hits,
                    "sam2_candidate_disagreement_fraction": round(maximum_candidate_disagreement, 6)}


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
                             sam2_device: str = "auto",
                             cache_dir: str | Path | None = None,
                             subject_detector: str = "none",
                             grounding_prompt: str | None = None,
                             grounding_part_prompts: tuple[str, ...] | list[str] | None = None,
                             grounding_model_dir: str | Path | None = None,
                             grounding_device: str = "auto",
                             resume: bool = False) -> dict[str, Any]:
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
    if subject_detector not in ("none", "grounding-dino") or grounding_device not in ("auto", "cpu", "mps"):
        raise ReferenceError("Unsupported subject detector or Grounding DINO device")
    if subject_detector != "none" and (segmenter != "sam2" or not grounding_prompt):
        raise ReferenceError("Grounding DINO needs SAM2 and --grounding-prompt")
    from .subject_grounding import validate_part_prompts
    part_prompts = validate_part_prompts(grounding_part_prompts)
    if part_prompts and subject_detector != "grounding-dino":
        raise ReferenceError("Part prompts require --subject-detector grounding-dino")
    cache_root = Path(cache_dir).resolve() if cache_dir is not None else None
    if cache_root is not None and (Path(cache_dir).is_symlink() or
                                   (cache_root.exists() and not cache_root.is_dir())):
        raise ReferenceError("Mask cache must be a real directory, not a symlink or file")
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
    grounding_folder = None
    grounding_id = None
    grounding_state: dict[str, Any] = {}
    if subject_detector == "grounding-dino":
        from .subject_grounding import grounding_identity, resolve_grounding_model
        grounding_folder = resolve_grounding_model(
            Path(grounding_model_dir) if grounding_model_dir is not None else None)
        grounding_id = grounding_identity(grounding_folder)
    if segmenter == "sam2":
        assert checkpoint is not None
        model_sha = _sha256(checkpoint)
        # The checkpoint hash is enough to validate cache keys. Load the
        # multi-GB model only when at least one image needs fresh inference.
        device = sam2_device
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
    if Path(output).is_symlink():
        raise ReferenceError("Mask output must not be a symbolic link")
    target = Path(output).resolve()
    existing = target.is_dir()
    # Dense preflight normally rejects existing outputs. For an interrupted
    # masks revision, check the same immutable sparse evidence against a
    # hypothetical unused child path; the actual target is separately audited.
    preflight_output = target / "__resume_preflight_only__" if resume and existing else target
    plan = plan_dense_reconstruction(reconstruction, preflight_output,
                                     DenseSettings(min_registered_images=3,
                                                   min_registered_fraction=0.25,
                                                   refine=False, texture=False),
                                     executables={})
    source = Path(plan["source_revision"])
    index = _json(source / "undistorted-reference-index.json")
    images = {item["filename"]: item for item in index["images"]}
    guide_map, guide_sha = _load_guides(guides, plan["undistorted_images"], default_box)
    target = Path(output).resolve()
    if cache_root is not None and (cache_root == target or cache_root.is_relative_to(target)
                                   or cache_root == source or cache_root.is_relative_to(source)):
        raise ReferenceError("Shared mask cache must be outside source and output revisions")
    if resume:
        if not existing or (target / "mask-report.json").is_symlink():
            raise ReferenceError("Mask resume requires an existing partial mask-report.json")
        prior = _json(target / "mask-report.json")
        expected_model = ("isnet-general-use" if segmenter == "rembg" else
                          f"sam2.1_hiera_{sam2_size}" if segmenter == "sam2" else None)
        if (not isinstance(prior.get("cache"), dict) or
                prior.get("status") != "incomplete / manual-mask-review-required" or
                prior.get("source_revision") != str(source) or
                prior.get("guide_sha256") != guide_sha or
                prior.get("segmenter") != segmenter or
                prior.get("segmentation_model") != expected_model or
                prior.get("segmentation_model_sha256") != model_sha or
                prior.get("max_side") != max_side or prior.get("iterations") != iterations or
                prior.get("subject_detector", "none") != subject_detector or
                prior.get("grounding_identity") != grounding_id or
                prior.get("grounding_prompt") != grounding_prompt or
                prior.get("grounding_part_prompts", []) != list(part_prompts) or
                prior.get("candidate_ensemble_schema") != (
                    "grounded-sam2-candidate-ensemble-v1" if grounding_id else None) or
                prior.get("cache", {}).get("directory") !=
                (str(cache_root) if cache_root else None)):
            raise ReferenceError("Cannot resume masks with changed photos, model, or settings")
    else:
        target.mkdir(parents=True, exist_ok=False)
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)
    mask_dir, preview_dir = target / "masks", target / "review"
    if mask_dir.is_symlink() or preview_dir.is_symlink():
        raise ReferenceError("Mask output folders cannot be symbolic links")
    mask_dir.mkdir(exist_ok=resume)
    preview_dir.mkdir(exist_ok=resume)
    alternatives_dir = target / "alternatives"
    if grounding_id is not None:
        if alternatives_dir.is_symlink():
            raise ReferenceError("Alternative mask folder cannot be a symbolic link")
        alternatives_dir.mkdir(exist_ok=resume)
    report: dict[str, Any] = {
        "schema_version": 1, "source_revision": str(source),
        "guide_sha256": guide_sha, "max_side": max_side, "iterations": iterations,
        "segmenter": segmenter,
        "segmentation_model": ("isnet-general-use" if segmenter == "rembg" else
                               f"sam2.1_hiera_{sam2_size}" if segmenter == "sam2" else None),
        "segmentation_model_sha256": model_sha,
        "segmentation_model_checkpoint": str(checkpoint) if checkpoint else None,
        "segmentation_device": device,
        "subject_detector": subject_detector,
        "grounding_identity": grounding_id,
        "grounding_prompt": grounding_prompt,
        "grounding_part_prompts": list(part_prompts),
        "candidate_ensemble_schema": ("grounded-sam2-candidate-ensemble-v1" if grounding_id else None),
        "automatic_guidance": segmenter == "sam2" and guides is None,
        "requires_anatomy_inventory": segmenter == "sam2",
        "cache": {"directory": str(cache_root) if cache_root else None,
                  "hits": 0, "misses": 0},
        "status": "incomplete / manual-mask-review-required", "images": [],
        "approved_for_reconstruction": False,
    }

    def save_progress_report() -> None:
        (target / "mask-report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                 encoding="utf-8")

    def infer_cached_mask(original: Any, guide: dict[str, Any], source_sha: str,
                          expected: list[int], *, grounded: bool
                          ) -> tuple[Any, dict[str, Any], str, bool]:
        nonlocal predictor, device
        from PIL import Image

        fingerprint_input = {
            "schema": _MASK_CACHE_SCHEMA,
            "source_sha256": source_sha, "dimensions": expected, "segmenter": segmenter,
            "model_sha256": model_sha, "model_size": sam2_size if segmenter == "sam2" else None,
            "guide": guide if segmenter != "rembg" else None,
            "max_side": max_side, "iterations": iterations if segmenter == "grabcut" else None,
        }
        if grounded and grounding_id is not None:
            fingerprint_input.update(grounding_identity=grounding_id,
                                     grounding_prompt=grounding_prompt,
                                     grounding_part_prompts=list(part_prompts))
        fingerprint = hashlib.sha256(json.dumps(fingerprint_input, sort_keys=True).encode()).hexdigest()
        cached = _cached_mask(cache_root, fingerprint, expected)
        if cached is not None:
            with Image.open(cached[0]) as image:
                mask = image.copy()
            report["cache"]["hits"] += 1
            return mask, cached[1], fingerprint, True
        if segmenter == "rembg":
            binary, details = _rembg_mask(original, session), {}
        elif segmenter == "sam2":
            if predictor is None:
                assert checkpoint is not None
                predictor, device = _load_sam2_predictor(checkpoint, sam2_size, sam2_device)
                report["segmentation_device"] = device
            binary, details = _sam2_mask(original, guide, predictor, max_side)
        else:
            binary, details = _grabcut(original, guide, max_side, iterations), {}
        if not binary.any() or (binary == 255).all():
            raise ReferenceError(f"{segmenter} returned an empty or entirely foreground mask")
        mask = Image.fromarray(binary, mode="L")
        report["cache"]["misses"] += 1
        if cache_root is not None:
            # Cache each candidate independently, never overwrite prior revisions.
            with tempfile.NamedTemporaryFile(dir=target, suffix=".png", delete=False) as stream:
                temporary = Path(stream.name)
            try:
                mask.save(temporary, format="PNG", optimize=True)
                _save_mask_cache(cache_root, fingerprint, temporary, expected, details)
            finally:
                temporary.unlink(missing_ok=True)
        return mask, details, fingerprint, False

    save_progress_report()
    for filename in plan["undistorted_images"]:
        image_path = source / "undistorted/images" / filename
        try:
            with Image.open(image_path) as original:
                original.load()
                expected = images[filename]["dimensions"]
                if list(original.size) != expected:
                    raise ReferenceError(f"Rectified photo size changed: {filename}")
                detection = None
                if segmenter == "sam2" and guides is None and grounding_folder is not None:
                    from .subject_grounding import grounded_guide
                    assert grounding_id is not None and grounding_prompt is not None
                    active_guide, detection = grounded_guide(
                        original, image_sha256=images[filename]["sha256"],
                        cache_root=cache_root or target.parent / "_grounding_detection_cache",
                        model_dir=grounding_folder, model_identity=grounding_id,
                        prompt=grounding_prompt, device=grounding_device,
                        state=grounding_state, default_box=default_box,
                        part_prompts=part_prompts)
                else:
                    active_guide = (_auto_sam2_guide(original, default_box)
                                    if segmenter == "sam2" and guides is None
                                    else guide_map[filename])
                comparison = None
                alternate_receipts = {}
                detector_guide = active_guide if detection is not None else None
                if grounding_id is not None and guides is None:
                    from .mask_candidate_ensemble import select_mask_candidate

                    broad_guide = _auto_sam2_guide(original, default_box)
                    broad, broad_details, broad_key, broad_hit = infer_cached_mask(
                        original, broad_guide, images[filename]["sha256"], expected,
                        grounded=False)
                    if detection is not None and detection["status"].startswith("candidate"):
                        try:
                            grounded_mask, grounded_details, grounded_key, grounded_hit = infer_cached_mask(
                                original, active_guide, images[filename]["sha256"], expected,
                                grounded=True)
                        except ReferenceError as error:
                            detection["status"] = "uncertain / grounded-sam2-inference-failed"
                            detection["failure_reason"] = str(error)[:200]
                            grounded_mask, grounded_details = broad, broad_details
                            grounded_key, grounded_hit = broad_key, broad_hit
                    else:
                        grounded_mask, grounded_details = broad, broad_details
                        grounded_key, grounded_hit = broad_key, broad_hit
                    choice, comparison = select_mask_candidate(broad, grounded_mask, detection)
                    mask, details, fingerprint, selected_hit = (
                        (grounded_mask, grounded_details, grounded_key, grounded_hit)
                        if choice == "grounded" else
                        (broad, broad_details, broad_key, broad_hit))
                    if choice == "broad":
                        active_guide = broad_guide
                    alternate_paths = {}
                    for label, candidate in (("broad", broad), ("grounded", grounded_mask)):
                        path = alternatives_dir / f"{filename}.{label}.png"
                        candidate.save(path, format="PNG", optimize=True)
                        alternate_paths[label] = str(path.relative_to(target))
                        small = original.convert("RGB")
                        small.thumbnail((900, 900))
                        red = Image.new("RGB", small.size, (238, 58, 35))
                        alpha = candidate.resize(small.size, Image.Resampling.NEAREST).point(
                            lambda intensity: round(intensity * 0.28))
                        candidate_preview = Image.composite(red, small, alpha)
                        preview_path = preview_dir / f"{filename}.{label}.jpg"
                        candidate_preview.save(preview_path, "JPEG", quality=86)
                        alternate_receipts[label] = {"path": alternate_paths[label],
                            "sha256": _sha256(path),
                            "preview": str(preview_path.relative_to(target)),
                            "cache_key": broad_key if label == "broad" else grounded_key,
                            "cache_hit": broad_hit if label == "broad" else grounded_hit}
                else:
                    mask, details, fingerprint, selected_hit = infer_cached_mask(
                        original, active_guide, images[filename]["sha256"], expected,
                        grounded=False)
                occupancy = round(sum(mask.histogram()[128:])/ (mask.width*mask.height), 5)
                warnings = []
                if segmenter == "grabcut" and guides is None:
                    warnings.append("Unreviewed center-box guess; add per-photo guides for busy backgrounds")
                if segmenter == "sam2" and not active_guide["part_points"]:
                    warnings.append("No visible-anatomy landmarks: mark head/torso/paws/tail where visible")
                if segmenter == "sam2" and guides is None:
                    warnings.append("Automatic subject prompts are candidates, not verified anatomy")
                if detection is not None and detection["status"].startswith("uncertain"):
                    warnings.append("Text-guided SAM2 could not produce a reliable subject mask; retained broad SAM2 candidate")
                if comparison is not None and comparison["status"].startswith("uncertain"):
                    warnings.append("Broad and grounded SAM2 candidates disagree; inspect both before dense reconstruction")
                if details.get("sam2_candidate_disagreement_fraction", 0) > 0.10:
                    warnings.append("SAM2 mask candidates disagree; inspect anatomy and boundaries")
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
                    "cache_hit": selected_hit, "cache_key": fingerprint if cache_root else None,
                    "preview": str(preview.relative_to(target)),
                    "foreground_fraction": occupancy,
                    "guide": active_guide, "warnings": warnings,
                    "detector_guide": detector_guide,
                    "guide_source": ("grounding-dino / candidate" if detection and
                                     detection["status"].startswith("candidate") else
                                     "grounding-dino / no-detection" if detection else
                                     "auto-sky-negative-prompts / tentative"
                                     if segmenter == "sam2" and guides is None else
                                     "user-guides" if guides is not None else "default-box"),
                    "subject_detection": detection,
                    "candidate_comparison": comparison,
                    "candidate_masks": alternate_receipts,
                    "missing_parts": details.get("missing_parts", []),
                    "background_seed_violations": details.get("background_seed_violations", 0),
                    "parts_verified": sorted(active_guide["part_points"]),
                    "segmented_components": details.get("components", []),
                })
        except Exception as error:
            report.update(failed_image=filename, error=str(error))
            save_progress_report()
            raise
        save_progress_report()
    def contact_sheet(label: str | None = None) -> str:
        columns, cell_width, cell_height = 4, 288, 238
        rows = (len(report["images"]) + columns - 1) // columns
        contact = Image.new("RGB", (columns * cell_width, rows * cell_height), (28, 31, 34))
        draw = ImageDraw.Draw(contact)
        for index, item in enumerate(report["images"]):
            x = index % columns * cell_width
            y = index // columns * cell_height
            path = (item["candidate_masks"][label]["preview"] if label
                    else item["preview"])
            with Image.open(target / path) as preview:
                preview.thumbnail((cell_width-8, cell_height-30))
                contact.paste(preview, (x+4, y+4))
            draw.text((x+6, y+cell_height-23), item["filename"][:42], fill=(234, 236, 238))
        name = f"mask-contact-sheet-{label}.jpg" if label else "mask-contact-sheet.jpg"
        contact.save(target / name, "JPEG", quality=85)
        return name

    contact_sheet()
    report["contact_sheet"] = "mask-contact-sheet.jpg"
    if grounding_id is not None and guides is None:
        report["alternative_contact_sheets"] = {
            label: contact_sheet(label) for label in ("broad", "grounded")}
    report["status"] = "candidate / inspect-every-mask-before-dense-reconstruction"
    save_progress_report()
    return report
