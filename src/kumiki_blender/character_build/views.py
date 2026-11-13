"""Orthographic character reference views in one shared, aligned frame.

Every full-body view is scaled so the figure is ``FIGURE_PX`` tall, stands on
``BASELINE`` and has its bounding box centred on ``CX``. One aligned pixel is
``height_m / FIGURE_PX`` metres. Blender cameras made by ``blender_setup``
render exactly this frame.

Axes are Blender axes: Z up, the character faces -Y, its right side is -X.
A view is named by the camera position around the character.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

CANVAS_W, CANVAS_H, FIGURE_PX, BASELINE, CX = 1600, 1200, 1000, 1100, 800
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
CLAY_PREFIXES = ("body_", "joint_")  # the prompts ask for grey clay for these
FACE_PREFIXES = ("expr_", "mouth_", "eyes_")
PART_PREFIXES = ("head_", "part_", "foot_", "hand_", "spread_", "joint_")

# Camera azimuth in degrees. 0 is the front. Positive turns toward the character's left (+X).
AZIMUTH = {"front": 0, "front_left": 45, "left": 90, "back_left": 135, "back": 180,
           "back_right": -135, "right": -90, "front_right": -45}
ALIASES = {"three_quarter": "front_right", "three_quarter_back": "back_left"}
AXIS_VIEWS = ("front", "back", "right", "left")
DIAGONAL_VIEWS = ("front_right", "front_left", "back_right", "back_left")


class ViewError(ValueError):
    """Invalid reference folder, image or view name."""


@dataclass(frozen=True)
class Camera:
    """An orthographic camera around the character."""

    name: str
    azimuth: float

    @property
    def toward(self) -> np.ndarray:
        """Unit vector from the character to the camera (horizontal)."""
        a = math.radians(self.azimuth)
        return np.array([math.sin(a), -math.cos(a), 0.0])

    @property
    def right(self) -> np.ndarray:
        """Unit screen-right vector in world space."""
        d = self.toward
        return np.array([-d[1], d[0], 0.0])

    def project(self, points: np.ndarray, metres_per_px: float, scale: float = 1.0
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """World points to (column, row, depth). Depth grows toward the camera.

        ``scale`` shrinks the aligned frame, e.g. 0.25 for a quarter-size grid.
        """
        u = points @ self.right
        col = (u / metres_per_px + CX) * scale
        row = (BASELINE - points[:, 2] / metres_per_px) * scale
        return col, row, points @ self.toward


def camera(name: str) -> Camera:
    name = ALIASES.get(name, name)
    if name not in AZIMUTH:
        raise ViewError(f"Unknown orthographic view {name!r}")
    return Camera(name, AZIMUTH[name])


def mirror_of(name: str) -> str:
    target = (AZIMUTH[name] + 360) % 360 - 180
    return next(n for n, a in AZIMUTH.items() if a == target or a == target + 360)


@dataclass
class View:
    name: str
    path: Path
    raw: Image.Image            # RGBA, background removed
    raw_mask: np.ndarray        # bool, original pixels
    aligned: Image.Image        # RGBA in the aligned frame
    mask: np.ndarray            # bool, aligned frame
    height_px: int
    notes: list[str] = field(default_factory=list)
    measure: dict = field(default_factory=dict)


def image_paths(folder: Path) -> dict[str, Path]:
    """Canonical view name -> path. Aliases such as three_quarter map to their direction."""
    paths: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES or path.name.startswith("."):
            continue
        stem = path.stem.lower()
        name = ALIASES.get(stem, ALIASES.get(stem.removesuffix("_clay"), None))
        if name is not None and stem.endswith("_clay"):
            name += "_clay"
        name = name or stem
        if name in paths:
            raise ViewError(f"Two images give view {name!r}: {paths[name].name} and {path.name}")
        paths[name] = path
    return paths


def figure_mask(image: Image.Image, *, tolerance: int = 24, bottom_open: bool = False
                ) -> tuple[np.ndarray, float]:
    """Foreground from alpha, or from the difference to the border colour.

    Returns the mask and the fraction of border pixels that differ from the
    border median (a background that is not flat). ``bottom_open`` excludes the
    bottom border, for a head close-up that continues past the image edge.
    """
    pixels = np.asarray(image.convert("RGBA")).astype(np.int16)
    if (pixels[..., 3] < 250).mean() > 0.01:
        return pixels[..., 3] > 127, 0.0
    rgb = pixels[..., :3]
    edge = max(2, int(0.02 * min(rgb.shape[:2])))
    sides = [rgb[:edge], rgb[:, :edge], rgb[:, -edge:]] + ([] if bottom_open else [rgb[-edge:]])
    border = np.concatenate([side.reshape(-1, 3) for side in sides])
    background = np.median(border, axis=0)
    noise = float((np.abs(border - background).max(axis=1) > tolerance).mean())
    mask = np.abs(rgb - background).max(axis=2) > tolerance
    # Anti-aliased, soft or shadowed edges leave specks. One opening removes them.
    return ndimage.binary_opening(mask, iterations=1), noise


def clean_figure(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Keep the largest region, fill enclosed holes. Return the mask and the count of extra regions.

    Only regions of 1% or more of the largest region count as extra objects.
    shortcut: hole filling also closes a real gap, such as an arm on the hip. The A-pose avoids it.
    """
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask, 0
    sizes = ndimage.sum_labels(mask, labels, range(1, count + 1))
    largest = int(np.argmax(sizes)) + 1
    extras = int((sizes >= 0.01 * sizes.max()).sum()) - 1
    return ndimage.binary_fill_holes(labels == largest), extras


def load(path: Path, name: str | None = None) -> View:
    """Read one image. Keep the largest figure. Make a copy in the aligned frame."""
    name = name or path.stem.lower()
    try:
        raw = Image.open(path).convert("RGBA")
    except OSError as error:
        raise ViewError(f"{path.name}: cannot read image ({error})") from error
    bottom_open = name.startswith(("head_",) + FACE_PREFIXES)
    mask, noise = figure_mask(raw, bottom_open=bottom_open)
    if not mask.any():
        raise ViewError(f"{path.name}: no figure found on the background")
    mask, extras = clean_figure(mask)
    rows, cols = mask.shape
    ys, xs = np.nonzero(mask)
    top, bottom, left, right = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    notes = []
    if extras:
        notes.append(f"{extras} extra object(s): text, shadow, prop or a second figure")
    if top <= 1 or left <= 1 or right >= cols - 1 or (bottom >= rows - 1 and not bottom_open):
        notes.append("the figure touches the image edge (cropped)")
    if noise > 0.05:
        notes.append(f"the background is not flat ({noise:.0%} of the border differs)")
    cut = raw.copy()
    cut.putalpha(Image.fromarray(mask.astype(np.uint8) * 255))
    scale = FIGURE_PX / (bottom - top)
    width = max(1, round((right - left) * scale))
    box = (left, top, right, bottom)
    x0 = CX - width // 2
    aligned = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
    figure = cut.crop(box).resize((width, FIGURE_PX), Image.LANCZOS)
    aligned.paste(figure, (x0, BASELINE - FIGURE_PX), figure)
    aligned_mask = Image.new("L", (CANVAS_W, CANVAS_H), 0)
    aligned_mask.paste(Image.fromarray(mask.astype(np.uint8) * 255).crop(box).resize((width, FIGURE_PX), Image.NEAREST),
                       (x0, BASELINE - FIGURE_PX))
    if width > CANVAS_W:
        notes.append("the figure is wider than the aligned frame and is clipped")
    measure = {"image_size": [cols, rows], "top_pct": round(100 * top / rows, 1),
               "bottom_pct": round(100 * bottom / rows, 1),
               "centre_pct": round(100 * (left + right) / 2 / cols, 1),
               "width_to_height": round((right - left) / (bottom - top), 3)}
    return View(name, path, cut, mask, aligned, np.asarray(aligned_mask) > 127, bottom - top, notes, measure)


def load_folder(folder: Path, names: set[str] | None = None) -> tuple[dict[str, View], dict[str, str]]:
    """Load the images of a reference folder. Return the views and the load errors by name."""
    views, errors = {}, {}
    for name, path in image_paths(folder).items():
        if name.startswith("mat_") or (names is not None and name not in names):
            continue
        try:
            views[name] = load(path, name)
        except ViewError as error:
            errors[name] = str(error)
    return views, errors


def downsample(mask: np.ndarray, factor: int) -> np.ndarray:
    """Aligned mask to a coarse grid. A cell is inside when half or more of it is inside."""
    h, w = mask.shape[0] // factor, mask.shape[1] // factor
    blocks = mask[:h * factor, :w * factor].reshape(h, factor, w, factor)
    return blocks.mean(axis=(1, 3)) >= 0.5


def ingest(source: Path, mapping_file: Path, destination: Path) -> dict[str, str]:
    """Copy generator output with arbitrary names into canonical view names.

    The mapping JSON is {"<source file name>": "<view name>", ...}. The source
    folder and the original files stay unchanged.
    """
    mapping = json.loads(mapping_file.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or not mapping:
        raise ViewError("The mapping must be a non-empty JSON object of file name to view name")
    if destination.exists() and any(destination.iterdir()):
        raise ViewError(f"{destination} must be a new or empty folder")
    destination.mkdir(parents=True, exist_ok=True)
    used: dict[str, str] = {}
    for file_name, view in mapping.items():
        src = source / file_name
        if Path(file_name).name != file_name or not src.is_file():
            raise ViewError(f"Missing source file {file_name!r}")
        view = ALIASES.get(view, view)
        if not view or not all(c.isalnum() or c == "_" for c in view):
            raise ViewError(f"Invalid view name {view!r}")
        if view in used:
            raise ViewError(f"{file_name} and {used[view]} both map to {view!r}")
        used[view] = file_name
        shutil.copy2(src, destination / f"{view}{src.suffix.lower()}")
    return used
