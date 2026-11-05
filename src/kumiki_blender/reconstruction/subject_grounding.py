"""Reviewable text-grounded subject proposals for SAM2, never geometric evidence.

Grounding DINO provides candidate boxes, not verified statue silhouettes. Every
proposal is cached against exact image bytes, a pinned model and the text prompt.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from .references import ReferenceError


MODEL_ID = "IDEA-Research/grounding-dino-tiny"
MODEL_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
SCHEMA = "grounded-subject-proposals-v2"
SCORE_THRESHOLD = 0.28


def validate_part_prompts(part_prompts: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    """Object-agnostic prompts for distinct visible parts; no anatomy inferred."""
    if part_prompts is None:
        return ()
    if not isinstance(part_prompts, (tuple, list)) or len(part_prompts) > 8:
        raise ReferenceError("Grounding DINO accepts at most eight part prompts")
    if any(not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 160
           for prompt in part_prompts):
        raise ReferenceError("Each part prompt needs 1–160 characters")
    return tuple(prompt.strip() for prompt in part_prompts)


def resolve_grounding_model(explicit: Path | None = None, *,
                            allow_download: bool = True) -> Path:
    """Resolve immutable official weights before a costly camera pipeline runs."""
    if importlib.util.find_spec("transformers") is None or importlib.util.find_spec("torch") is None:
        raise ReferenceError("Grounding DINO requires pip install -e '.[mask-grounding]'")
    if explicit is not None:
        source = explicit.expanduser()
        if source.is_symlink() or not source.is_dir():
            raise ReferenceError("Grounding DINO model must be an existing directory")
        if not (source / "model.safetensors").is_file():
            raise ReferenceError("Grounding DINO directory needs model.safetensors")
        return source.resolve()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ReferenceError("Install pip extras '[mask-grounding]' for Grounding DINO") from error
    try:
        folder = Path(snapshot_download(
            repo_id=MODEL_ID, revision=MODEL_REVISION,
            allow_patterns=["*.json", "*.txt", "model.safetensors"],
            local_files_only=not allow_download))
    except (OSError, ValueError, RuntimeError) as error:
        raise ReferenceError("Grounding DINO weights unavailable; install '[mask-grounding]' "
                             "and allow model download, or provide --grounding-model-dir") from error
    if not (folder / "model.safetensors").is_file():
        raise ReferenceError("Grounding DINO snapshot is missing safetensors weights")
    return folder.resolve()


def grounding_identity(folder: Path) -> str:
    """Content-address actual local weights, not a mutable Hugging Face branch."""
    weights = folder / "model.safetensors"
    if not weights.is_file():
        raise ReferenceError("Grounding DINO safetensors weights not found")
    identity = hashlib.sha256()
    for name in ("model.safetensors", "config.json", "preprocessor_config.json",
                 "tokenizer.json", "tokenizer_config.json", "vocab.txt",
                 "special_tokens_map.json", "added_tokens.json"):
        path = folder / name
        if not path.is_file():
            continue
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").digest()
        identity.update(name.encode("utf-8") + b"\0" + digest)
    return f"{MODEL_ID}@{MODEL_REVISION}:{identity.hexdigest()}"


class GroundingDetector:
    """Lazily load a local pinned detector; MPS operator failures retry on CPU."""

    def __init__(self, folder: Path, device: str = "auto") -> None:
        if device not in ("auto", "mps", "cpu"):
            raise ReferenceError("Grounding DINO device must be auto, mps, or cpu")
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as error:
            raise ReferenceError("Install pip extras '[mask-grounding]' for Grounding DINO") from error
        self.torch = torch
        self.device = ("mps" if torch.backends.mps.is_available() else "cpu") if device == "auto" else device
        if self.device == "mps" and not torch.backends.mps.is_available():
            if device != "auto":
                raise ReferenceError("Grounding DINO MPS unavailable; use --grounding-device cpu")
            self.device = "cpu"
        self.requested_device = device
        try:
            self.processor = AutoProcessor.from_pretrained(
                str(folder), local_files_only=True, trust_remote_code=False)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                str(folder), local_files_only=True, trust_remote_code=False,
                use_safetensors=True)
            try:
                self.model = model.to(self.device).eval()
            except (RuntimeError, NotImplementedError):
                if self.device != "mps" or device != "auto":
                    raise
                self.device = "cpu"
                self.model = model.to("cpu").eval()
        except (OSError, ValueError, RuntimeError) as error:
            raise ReferenceError(f"Could not load Grounding DINO model: {error}") from error

    def detect(self, photo: Any, prompt: str) -> list[dict[str, Any]]:
        def run() -> Any:
            inputs = self.processor(images=photo.convert("RGB"), text=prompt,
                                    return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                predictions = self.model(**inputs)
            return self.processor.post_process_grounded_object_detection(
                predictions, inputs.input_ids, threshold=SCORE_THRESHOLD,
                text_threshold=0.22, target_sizes=[(photo.height, photo.width)])[0]

        try:
            result = run()
        except (RuntimeError, NotImplementedError) as error:
            if self.device != "mps" or self.requested_device != "auto":
                raise ReferenceError(f"Grounding DINO inference failed: {error}") from error
            self.model = self.model.to("cpu")
            self.device = "cpu"
            try:
                result = run()
            except (RuntimeError, NotImplementedError) as fallback_error:
                raise ReferenceError(f"Grounding DINO CPU fallback failed: {fallback_error}") from fallback_error
        output = []
        for box, score in zip(result["boxes"], result["scores"]):
            coordinates = box.tolist() if hasattr(box, "tolist") else list(box)
            confidence = float(score.item() if hasattr(score, "item") else score)
            if len(coordinates) != 4 or not math.isfinite(confidence):
                continue
            x0, y0, x1, y1 = (float(n) for n in coordinates)
            normalized = [max(0.0, min(1.0, x0 / photo.width)),
                          max(0.0, min(1.0, y0 / photo.height)),
                          max(0.0, min(1.0, x1 / photo.width)),
                          max(0.0, min(1.0, y1 / photo.height))]
            if (_valid_box(normalized) and SCORE_THRESHOLD <= confidence <= 1.0):
                output.append({"box": [round(v, 6) for v in normalized],
                               "score": round(confidence, 6)})
        return sorted(output, key=lambda item: item["score"], reverse=True)[:12]


def _valid_box(box: Any) -> bool:
    return (isinstance(box, list) and len(box) == 4
            and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in box)
            and box[2] - box[0] >= 0.05 and box[3] - box[1] >= 0.05)


def _select_box(detections: list[dict[str, Any]]) -> list[float] | None:
    """Favor a well-supported large subject, but never merge distant detections."""
    if not detections:
        return None
    ranked = sorted(detections, key=lambda d: (d["score"] * (0.6 + 0.4 *
                    math.sqrt((d["box"][2] - d["box"][0]) *
                              (d["box"][3] - d["box"][1])))), reverse=True)
    selected = list(ranked[0]["box"])
    for detection in ranked[1:]:
        box = detection["box"]
        overlap = (max(0.0, min(selected[2], box[2]) - max(selected[0], box[0])) *
                   max(0.0, min(selected[3], box[3]) - max(selected[1], box[1])))
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        if overlap / max(1e-9, box_area) < 0.45:
            continue
        expanded = [min(selected[0], box[0]), min(selected[1], box[1]),
                    max(selected[2], box[2]), max(selected[3], box[3])]
        if (expanded[2] - expanded[0]) * (expanded[3] - expanded[1]) <= 0.85:
            selected = expanded
    margin = 0.025
    return [round(max(0.0, selected[0] - margin), 6),
            round(max(0.0, selected[1] - margin), 6),
            round(min(1.0, selected[2] + margin), 6),
            round(min(1.0, selected[3] + margin), 6)]


def _near_subject(part_box: list[float], subject_box: list[float]) -> bool:
    """Suppress unrelated remote detections, without claiming shared identity."""
    horizontal_gap = max(0.0, subject_box[0] - part_box[2],
                         part_box[0] - subject_box[2])
    vertical_gap = max(0.0, subject_box[1] - part_box[3],
                       part_box[1] - subject_box[3])
    return horizontal_gap <= 0.15 and vertical_gap <= 0.15


def grounded_guide(photo: Any, *, image_sha256: str, cache_root: Path,
                   model_dir: Path, model_identity: str, prompt: str,
                   device: str, state: dict[str, Any],
                   default_box: tuple[float, ...],
                   part_prompts: tuple[str, ...] | list[str] | None = None
                   ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate a reviewable SAM2 prompt, with a hashed per-photo detector cache."""
    from .masking import _auto_sam2_guide, _guide

    if not prompt.strip() or len(prompt) > 400:
        raise ReferenceError("Grounding DINO prompt needs 1–400 characters")
    if len(image_sha256) != 64 or any(c not in "0123456789abcdef" for c in image_sha256):
        raise ReferenceError("Invalid grounding source image hash")
    parts = validate_part_prompts(part_prompts)
    key = hashlib.sha256(json.dumps([SCHEMA, image_sha256, list(photo.size),
                                    model_identity, prompt, parts, device, SCORE_THRESHOLD],
                                   sort_keys=True).encode()).hexdigest()
    root = cache_root / "subject-grounding" / key[:2] / key
    receipt = root / "detections.json"
    if any(path.is_symlink() for path in (cache_root, root.parent.parent, root.parent, root, receipt)):
        raise ReferenceError("Subject detection cache cannot contain symlinks")
    detections = None
    part_detections = None
    if receipt.is_file():
        try:
            existing = json.loads(receipt.read_text(encoding="utf-8"))
            proposed = existing.get("detections") if isinstance(existing, dict) else None
            proposed_parts = existing.get("part_detections") if isinstance(existing, dict) else None
            if (isinstance(existing, dict) and existing.get("key") == key
                    and existing.get("schema") == SCHEMA
                    and isinstance(proposed, list) and len(proposed) <= 12
                    and all(isinstance(x, dict) and _valid_box(x.get("box"))
                            and type(x.get("score")) in (float, int)
                            and SCORE_THRESHOLD <= x["score"] <= 1 for x in proposed)
                    and isinstance(proposed_parts, list) and len(proposed_parts) == len(parts)
                    and all(isinstance(entry, dict) and entry.get("prompt") == part and
                            isinstance(entry.get("detections"), list) and
                            len(entry["detections"]) <= 12 and
                            all(isinstance(item, dict) and _valid_box(item.get("box")) and
                                type(item.get("score")) in (float, int) and
                                SCORE_THRESHOLD <= item["score"] <= 1
                                for item in entry["detections"])
                            for part, entry in zip(parts, proposed_parts))):
                detections = proposed
                part_detections = proposed_parts
        except (OSError, ValueError, TypeError):
            pass
    cache_hit = detections is not None
    if detections is None:
        if "detector" not in state:
            state["detector"] = GroundingDetector(model_dir, device)
        detections = state["detector"].detect(photo, prompt)
        part_detections = [{"prompt": part, "detections": state["detector"].detect(photo, part)}
                           for part in parts]
        root.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=root,
                                             prefix=".detection-", suffix=".json",
                                             delete=False) as stream:
                temporary = Path(stream.name)
                json.dump({"schema": SCHEMA, "key": key, "detections": detections,
                           "part_detections": part_detections}, stream)
                stream.write("\n")
            os.replace(temporary, receipt)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    selected = _select_box(detections)
    base = _auto_sam2_guide(photo, default_box)
    if selected is None:
        return base, {"status": "uncertain / no-confident-detection", "detections": detections,
                      "part_detections": part_detections,
                      "cache_hit": cache_hit, "model_identity": model_identity}
    # Grounding proposals guide SAM2; no automatic foreground point is assumed
    # to lie on an irregular or partially occluded object.
    component_boxes = {"subject": selected}
    accepted_parts = []
    rejected_parts = []
    for index, row in enumerate(part_detections or []):
        box = _select_box(row["detections"])
        if box is None or not _near_subject(box, selected):
            rejected_parts.append(row["prompt"])
            continue
        component_boxes[f"part_{index + 1:02d}"] = box
        accepted_parts.append(row["prompt"])
    guide = _guide({"box": selected, "background_points": base["background_points"],
                    "component_boxes": component_boxes},
                   default_box=default_box)
    return guide, {"status": "candidate / text-grounded-box", "detections": detections,
                   "selected_box": selected, "cache_hit": cache_hit,
                   "part_detections": part_detections,
                   "accepted_part_prompts": accepted_parts,
                   "unresolved_part_prompts": rejected_parts,
                   "model_identity": model_identity}
