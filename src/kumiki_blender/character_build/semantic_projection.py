"""Opt-in semantic ownership constraints for character reference texture projection.

A labeled UV texel may only sample the corresponding anatomical/costume region
from a source camera. All masks are authored evidence; we never infer labels
from color or declare generated views verified.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SemanticProjection:
    labels: np.ndarray
    part_names: tuple[str, ...]
    source_masks: dict[str, dict[int, np.ndarray]]

    def allowed(self, view: str, use: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Reject cross-part source claims; x/y are source-image pixel coordinates."""
        allowed = use.copy()
        for index in range(len(self.part_names)):
            selected = allowed & (self.labels == index)
            if not selected.any():
                continue
            mask = self.source_masks.get(view, {}).get(index)
            if mask is None:
                allowed[selected] = False
            else:
                ys, xs = y[selected], x[selected]
                valid = (xs >= 0) & (ys >= 0) & (xs < mask.shape[1]) & (ys < mask.shape[0])
                selected_ids = np.flatnonzero(selected)
                accepted = np.zeros(len(selected_ids), dtype=bool)
                accepted[valid] = mask[ys[valid], xs[valid]]
                allowed[selected_ids] = accepted
        # Unlabeled UV surfaces may not borrow source pixels already claimed
        # by a labeled body part. Otherwise scalp/scarf colors can still leak
        # onto unclassified geometry even when labeled parts are protected.
        unlabelled = np.flatnonzero(allowed & (self.labels < 0))
        for mask in self.source_masks.get(view, {}).values():
            if not len(unlabelled):
                break
            xs, ys = x[unlabelled], y[unlabelled]
            valid = (xs >= 0) & (ys >= 0) & (xs < mask.shape[1]) & (ys < mask.shape[0])
            conflict = np.zeros(len(unlabelled), dtype=bool)
            conflict[valid] = mask[ys[valid], xs[valid]]
            allowed[unlabelled[conflict]] = False
            unlabelled = unlabelled[~conflict]
        return allowed


def _safe_file(root: Path, value: str, *, suffix: str = '.png') -> Path:
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError('Semantic mask must be a relative local filename')
    relative = Path(value)
    if relative.is_absolute() or any(part in ('..', '.') for part in relative.parts):
        raise ValueError(f'Unsafe semantic mask path: {value}')
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()) or path.suffix.lower() != suffix:
        raise ValueError(f'Unsafe semantic mask type or path: {value}')
    current = root
    for segment in relative.parts:
        current = current / segment
        if current.is_symlink():
            raise ValueError(f'Semantic mask cannot be a symlink: {value}')
    if not path.is_file():
        raise FileNotFoundError(f'Missing semantic mask: {path}')
    return path


def _read_mask(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as img:
        if shape is not None and img.size != (shape[1], shape[0]):
            raise ValueError(f'Mask dimensions disagree with source image: {path}')
        alpha = np.asarray(img.getchannel('A')) if 'A' in img.getbands() else None
        if alpha is not None and int(alpha.min()) != int(alpha.max()):
            return alpha >= 128
        return np.asarray(img.convert('L')) >= 128


def load_semantic_projection(
    definition: dict,
    root: Path,
    handoff: Path,
    size: int,
    rows: np.ndarray,
    cols: np.ndarray,
) -> SemanticProjection:
    """Load per-part UV and raw-image masks in the coordinates used by the baker.

    Example: {"parts": [{"id": "tail", "uv_mask": "masks/tail-uv.png",
              "view_masks": {"front": "masks/tail-front.png"}}]}.
    UV masks may have a different size from the bake; they are nearest-resized.
    Reference masks MUST have the exact dimensions of the corresponding raw image.
    """
    if not isinstance(definition, dict) or set(definition) != {'parts'}:
        raise ValueError('semantic must contain only a parts array')
    parts = definition['parts']
    if not isinstance(parts, list) or not parts or len(parts) > 256:
        raise ValueError('semantic.parts must contain 1 to 256 entries')
    root = root.resolve()
    labels = np.full(len(rows), -1, dtype=np.int16)
    names: list[str] = []
    sources: dict[str, dict[int, np.ndarray]] = {}
    for index, part in enumerate(parts):
        if not isinstance(part, dict) or set(part) != {'id', 'uv_mask', 'view_masks'}:
            raise ValueError('Every semantic part needs id, uv_mask and view_masks')
        name = part['id']
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('Semantic IDs must be nonempty and unique')
        names.append(name)
        uv_path = _safe_file(root, part['uv_mask'])
        with Image.open(uv_path) as im:
            mask = im.convert('L').resize((size, size), Image.Resampling.NEAREST)
            covered = np.asarray(mask) >= 128
        owned = covered[rows, cols]
        if np.any(owned & (labels >= 0)):
            raise ValueError(f'Semantic UV masks overlap around {name}; ownership must be disjoint')
        if not owned.any():
            raise ValueError(f'Semantic UV mask selects no sampled texels: {name}')
        labels[owned] = index
        view_masks = part['view_masks']
        if not isinstance(view_masks, dict) or not view_masks:
            raise ValueError(f'Semantic part {name} needs at least one source view mask')
        for view, file in view_masks.items():
            source = handoff / 'raw' / f'{view}.png'
            if not source.is_file():
                raise ValueError(f'{name}: no raw handoff source for {view}')
            with Image.open(source) as ref:
                shape = (ref.height, ref.width)
            source_mask = _read_mask(_safe_file(root, file), shape)
            if not source_mask.any():
                raise ValueError(f'{name}: empty source mask for {view}')
            previous = sources.setdefault(view, {})
            if any(np.any(source_mask & other) for other in previous.values()):
                raise ValueError(f'{name}: source masks overlap in {view}; visible pixels need unique ownership')
            previous[index] = source_mask
    return SemanticProjection(labels, tuple(names), sources)
