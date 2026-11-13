"""Turn checked reference images into 3D data, and compare a model with them.

``build_handoff`` writes ``<folder>/handoff/``:

- ``aligned/<view>.png``: RGBA views in the camera frame. Diagonal views are
  shifted by the offset that the hull carve measured, so every image lines up
  with its ``REF_CAM_<view>`` camera.
- ``masks/<view>.png``: binary silhouettes in the same frame.
- ``gen/``: square images of the full-body views with one shared scale, for
  image-to-3D generators. ``gen/white/`` has white backgrounds.
- ``parts/``: square images of each close-up and part image.
- ``blockout.obj``: smooth blockout in metres (signed-distance intersection of the views).
- ``spec.json``: height, bands, landmarks, cameras and hull agreement.
- ``fidelity_manifest.json``: a ``character_fidelity`` manifest for the cameras.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .checks import colour_map, iou
from .hull import carve, sdf_blockout
from .mesh import rasterize, read_mesh, surface_samples, write_obj
from .views import (AXIS_VIEWS, AZIMUTH, BASELINE, CANVAS_H, CANVAS_W, CX, FIGURE_PX, PART_PREFIXES, camera,
                    figure_mask, load_folder)

SCHEMA = 1


def _shifted(image: Image.Image, columns: int) -> Image.Image:
    out = Image.new(image.mode, image.size, 0)
    out.paste(image, (columns, 0))
    return out


def square_pack(items: dict[str, Image.Image], folder: Path, size: int = 1024, margin: float = 0.06) -> None:
    """RGBA and white-background squares. All items share one crop, so one scale."""
    boxes = [image.getchannel("A").getbbox() for image in items.values()]
    left, top = min(b[0] for b in boxes), min(b[1] for b in boxes)
    right, bottom = max(b[2] for b in boxes), max(b[3] for b in boxes)
    side = max(right - left, bottom - top) * (1 + 2 * margin)
    cx, cy = (left + right) / 2, (top + bottom) / 2
    box = tuple(round(v) for v in (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2))
    (folder / "white").mkdir(parents=True, exist_ok=True)
    for name, image in items.items():
        square = image.crop(box).resize((size, size), Image.LANCZOS)
        square.save(folder / f"{name}.png")
        white = Image.new("RGB", square.size, (255, 255, 255))
        white.paste(square, mask=square.getchannel("A"))
        white.save(folder / "white" / f"{name}.png")


def camera_spec(name: str, metres_per_px: float, height_m: float) -> dict:
    cam = camera(name)
    distance = 3 * height_m
    centre_z = (BASELINE - CANVAS_H / 2) * metres_per_px
    return {"location": [round(float(v), 6) for v in (cam.toward * distance)[:2]] + [round(centre_z, 6)],
            "rotation_deg": [90.0, 0.0, float(cam.azimuth)], "ortho_scale": round(CANVAS_W * metres_per_px, 6)}


def build_handoff(folder: Path, height_m: float, *, factor: int = 5, tolerance_px: float = 10.0,
                  round_px: float = 6.0, outline_blur_px: float = 12.0) -> dict:
    if not (0.01 <= height_m <= 100):
        raise ValueError("height_m must be between 0.01 and 100 metres")
    views, errors = load_folder(folder)
    full = {n: v for n, v in views.items() if n in AZIMUTH}
    if "front" not in full or not ({"right", "left"} & set(full)):
        raise ValueError("The handoff needs front.png and right.png or left.png")
    out = folder / "handoff"
    if out.exists():
        raise FileExistsError(f"{out} exists. Move it away: each handoff is a new revision.")
    for sub in ("aligned", "masks", "gen"):
        (out / sub).mkdir(parents=True)
    px = height_m / FIGURE_PX
    hull = carve(full, factor, tolerance_px)
    shifts_px = {n: s * factor for n, s in hull.shifts.items()}
    for name, view in full.items():
        image, mask = view.aligned, Image.fromarray(view.mask.astype(np.uint8) * 255)
        if name in shifts_px:  # move the image onto the camera frame
            image, mask = _shifted(image, -shifts_px[name]), _shifted(mask, -shifts_px[name])
        image.save(out / "aligned" / f"{name}.png")
        mask.save(out / "masks" / f"{name}.png")
        clay = views.get(f"{name}_clay")
        if clay is not None:  # same frame as its colour view
            _shifted(clay.aligned, -shifts_px.get(name, 0)).save(out / "aligned" / f"{name}_clay.png")
    blockout = sdf_blockout(full, hull, px, tolerance_px=tolerance_px, round_px=round_px,
                            outline_blur_px=outline_blur_px)
    write_obj(blockout, out / "blockout.obj", "Visual hull of the character reference views (metres, Y up)")

    front = full["front"].mask
    side = full.get("right", full.get("left")).mask
    bands = []
    for pct in range(0, 101, 5):
        row = min(BASELINE - 1, BASELINE - FIGURE_PX + pct * FIGURE_PX // 100)
        bands.append({"from_top_pct": pct, "z_m": round((BASELINE - row) * px, 5),
                      "width_m": round(float(front[row].sum()) * px, 5),
                      "depth_m": round(float(side[row].sum()) * px, 5)})
    spec = {
        "schema_version": SCHEMA, "height_m": height_m, "metres_per_aligned_px": px,
        "frame": {"canvas": [CANVAS_W, CANVAS_H], "figure_px": FIGURE_PX, "baseline_px": BASELINE, "centre_px": CX},
        "axes": "Blender: Z up, the character faces -Y, its right side is -X, soles at Z = 0",
        "views": sorted(full), "load_errors": errors,
        "blockout": {"file": "blockout.obj", "voxel_m": round(factor * px, 6), "vertices": len(blockout.vertices),
                     "faces": len(blockout.faces), "carve_tolerance_px": tolerance_px,
                     "round_px": round_px, "outline_blur_px": outline_blur_px,
                     "unexplained_silhouette": {n: round(v, 4) for n, v in hull.missing.items()},
                     "diagonal_shift_px": shifts_px},
        "bounding_box_m": {"width": round(float(front.any(0).sum()) * px, 5),
                           "depth": round(float(side.any(0).sum()) * px, 5), "height": height_m},
        "bands": bands,
        "cameras": {n: {**camera_spec(n, px, height_m), "image": f"aligned/{n}.png" if n in full else None}
                    for n in AZIMUTH},
        "references": {n: {"image": f"aligned/{n}.png", "size": round(CANVAS_W * px, 6),
                           "location": [round(float(v), 6) for v in (-1.5 * height_m * camera(n).toward)[:2]]
                           + [round((BASELINE - CANVAS_H / 2) * px, 6)],
                           "rotation_deg": [90.0, 0.0, float(camera(n).azimuth)]}
                       for n in AXIS_VIEWS if n in full},
    }
    marks_file = folder / "landmarks.json"
    if marks_file.is_file():
        marks = json.loads(marks_file.read_text(encoding="utf-8"))
        keys = sorted({k for v in marks.values() for k in v})
        spec["landmarks_z_m"] = {k: round(height_m * (1 - float(np.mean([v[k] for v in marks.values() if k in v])) / 100), 5)
                                 for k in keys}
        lf = marks.get("front", {})
        if "crown" in lf and "face_bottom" in lf and lf["face_bottom"] > lf["crown"]:
            spec["heads_tall"] = round((100 - lf["crown"]) / (lf["face_bottom"] - lf["crown"]), 3)
    (out / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

    square_pack({n: Image.open(out / "aligned" / f"{n}.png") for n in full}, out / "gen")
    for name, view in sorted(views.items()):
        if name.startswith(PART_PREFIXES) and not name.endswith("_clay"):
            square_pack({name: view.raw}, out / "parts")
    _write_fidelity_manifest(out, sorted(full))
    return spec


def _write_fidelity_manifest(out: Path, names: list[str]) -> None:
    """A whole-figure character_fidelity manifest. Alignment is identity but stays unverified."""
    manifest = {
        "schema_version": 1, "unassigned_behavior": "occlude",
        "views": [{"id": n, "camera": f"REF_CAM_{n}", "width": CANVAS_W, "height": CANVAS_H,
                   "reference_image": f"aligned/{n}.png",
                   "alignment": {"verified": False, "source_to_render": [[1, 0, 0], [0, 1, 0]]}} for n in names],
        "parts": [{"id": "figure", "category": "anatomy", "objects": ["*"],
                   "reference_masks": {n: f"masks/{n}.png" for n in names}}],
    }
    # character_fidelity resolves assets inside the manifest folder only, so it sits at the handoff root.
    (out / "fidelity_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- compare

def compare(handoff: Path, source: Path, *, output: Path | None = None, scale: float = 0.5,
            limits=(0.92, 0.85)) -> dict:
    """Compare a model with the reference silhouettes of a handoff.

    ``source`` is a mesh file (.obj, .ply, .glb), rendered with the NumPy
    rasterizer, or a folder of renders made by ``blender_setup.py --render``.
    Writes ``output``: by default ``<source>-compare/`` (mesh) or
    ``<folder>/compare/`` (renders). The folder must be new or empty.
    """
    spec = json.loads((handoff / "spec.json").read_text(encoding="utf-8"))
    px = spec["metres_per_aligned_px"]
    names = [n for n in spec["views"] if (handoff / "masks" / f"{n}.png").is_file()]
    if source.is_dir():
        out = source / "compare"
        models: dict[str, np.ndarray] = {}
        for name in names:
            path = source / f"{name}.png"
            if path.is_file():
                image = Image.open(path)
                if image.size != (CANVAS_W, CANVAS_H):
                    raise ValueError(f"{path.name}: renders must be {CANVAS_W}x{CANVAS_H}; use the REF_CAM cameras")
                models[name] = figure_mask(image)[0]
        frame = 1.0
    else:
        mesh = read_mesh(source)
        out = source.with_name(source.stem + "-compare")
        samples = surface_samples(mesh, 0.5 * px / scale)
        models = {n: rasterize(mesh, camera(n), px, scale, samples)[0] for n in names}
        frame = scale
    out = output or out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty. Each compare is a new revision.")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, model in models.items():
        reference = np.asarray(Image.open(handoff / "masks" / f"{name}.png").resize(
            model.shape[::-1], Image.BOX)) >= 128
        value = iou(reference, model)
        missing = float((reference & ~model).sum() / max(1, reference.sum()))
        extra = float((model & ~reference).sum() / max(1, model.sum()))
        rows_used = np.nonzero(model.any(axis=1))[0]
        height_error = (rows_used.max() + 1 - rows_used.min()) / (FIGURE_PX * frame) - 1 if len(rows_used) else -1.0
        status = "PASS" if value >= limits[0] else "WARN" if value >= limits[1] else "FAIL"
        rows.append({"view": name, "status": status, "iou": round(value, 4), "missing": round(missing, 4),
                     "extra": round(extra, 4), "height_error": round(float(height_error), 4)})
        colour_map([(reference & ~model, (220, 40, 40)), (model & ~reference, (40, 90, 220)),
                    (reference & model, (175, 175, 175))], reference.shape).save(out / f"{name}.png")
    if not rows:
        raise ValueError("No model silhouette matches a reference view")
    report = {"pass": all(r["status"] != "FAIL" for r in rows), "mean_iou": round(float(np.mean([r["iou"] for r in rows])), 4),
              "views": rows, "legend": "red: in the reference, not in the model. blue: in the model, not in the reference."}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def reference_masks(handoff: Path, scale: float) -> dict[str, np.ndarray]:
    """Handoff silhouettes at a grid scale (camera frame)."""
    spec = json.loads((handoff / "spec.json").read_text(encoding="utf-8"))
    size = (int(CANVAS_W * scale), int(CANVAS_H * scale))
    return {n: np.asarray(Image.open(handoff / "masks" / f"{n}.png").resize(size, Image.BOX)) >= 128
            for n in spec["views"] if (handoff / "masks" / f"{n}.png").is_file()}
