"""Check a reference folder: each image alone, the views against each other, and in 3D.

Results are PASS, WARN or FAIL. Only FAIL makes the folder unusable. The
limits are start values from synthetic tests and two AI-generated sets; tune
them per project with ``limits``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .hull import carve, silhouette
from .views import (AXIS_VIEWS, AZIMUTH, BASELINE, CANVAS_W, CLAY_PREFIXES, CX, DIAGONAL_VIEWS, FACE_PREFIXES,
                    FIGURE_PX, View, downsample, load_folder, mirror_of)

LIMITS = {
    "mirror_iou": 0.88, "clay_iou": 0.92, "body_outside": 0.05, "landmark_spread": 2.0,
    "min_height_px": 700, "front_symmetry": 0.80, "clay_chroma": 10.0, "face_iou": 0.85,
    "new_colour": 0.15, "carve_pass": 0.03, "carve_fail": 0.07,
}


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max(1, (a | b).sum()))


def lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB uint8 (N, 3) to CIE Lab (D65)."""
    c = rgb.astype(float) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], axis=-1)


def figure_lab(view: View, limit: int = 20000) -> np.ndarray:
    pixels = np.asarray(view.raw)[view.raw_mask][:, :3]
    return lab(pixels[::max(1, len(pixels) // limit)])


def palette(colours: np.ndarray, k: int = 8, seed: int = 0) -> np.ndarray:
    """k-means colour centres. Deterministic for a fixed seed."""
    rng = np.random.default_rng(seed)
    centres = colours[rng.choice(len(colours), size=min(k, len(colours)), replace=False)]
    for _ in range(12):
        nearest = np.argmin(((colours[:, None] - centres[None]) ** 2).sum(-1), axis=1)
        centres = np.array([colours[nearest == n].mean(0) if (nearest == n).any() else centres[n]
                            for n in range(len(centres))])
    return centres


def best_shift(inside: np.ndarray, outer: np.ndarray, shifts: range) -> tuple[float, int]:
    """Smallest fraction of ``inside`` outside ``outer`` over horizontal shifts of ``outer``."""
    total = max(1, inside.sum())
    return min(((inside & ~np.roll(outer, s, axis=1)).sum() / total, s) for s in shifts)


def colour_map(pairs, shape) -> Image.Image:
    out = np.full(tuple(shape) + (3,), 245, dtype=np.uint8)
    for mask, colour in pairs:
        out[mask] = colour
    return Image.fromarray(out)


def onion(a: np.ndarray, b: np.ndarray, path: Path) -> None:
    """Grey: both. Red: only the first. Blue: only the second."""
    colour_map([(a & ~b, (220, 50, 50)), (b & ~a, (50, 90, 220)), (a & b, (170, 170, 170))], a.shape).save(path)


def guide_overlay(image: Image.Image) -> Image.Image:
    out = Image.new("RGB", image.size, (217, 217, 217))
    out.paste(image, mask=image.getchannel("A"))
    draw = ImageDraw.Draw(out)
    for step in range(0, 101, 5):
        y = BASELINE - FIGURE_PX + step * FIGURE_PX // 100
        major = step % 10 == 0
        draw.line([(0, y), (CANVAS_W, y)], fill=(255, 0, 0) if major else (0, 140, 255), width=1)
        if major:
            draw.text((6, y + 2), f"{step}%", fill=(255, 0, 0))
    return out


def check_folder(folder: Path, limits: dict | None = None) -> dict:
    """Run every check. Write ``folder/check/`` and return the report."""
    limit = {**LIMITS, **(limits or {})}
    views, errors = load_folder(folder)
    if "front" not in views and "front" not in errors:
        raise FileNotFoundError("The folder needs front.png. It is the approved anchor view.")
    out = folder / "check"
    for sub in ("aligned", "overlay", "onion", "hull"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    def add(check: str, status: str, value, limit_text: str = "") -> None:
        rows.append({"check": check, "status": status, "value": value, "limit": limit_text})

    for name, error in errors.items():
        add(f"{name}: figure found", "FAIL", error)
    for name, view in views.items():
        view.aligned.save(out / "aligned" / f"{name}.png")
        guide_overlay(view.aligned).save(out / "overlay" / f"{name}.png")
        add(f"{name}: one clean figure", "FAIL" if view.notes else "PASS", "; ".join(view.notes) or "ok")
        if name in AZIMUTH and view.height_px < limit["min_height_px"]:
            add(f"{name}: resolution", "WARN", f"figure is {view.height_px} px tall", f">= {limit['min_height_px']} px")
    if "front" not in views:
        return _write(out, rows, {})

    front = views["front"].mask
    value = iou(front, front[:, ::-1])
    add("front faces the viewer (left-right symmetry)", "PASS" if value >= limit["front_symmetry"] else "WARN",
        round(value, 3), f">= {limit['front_symmetry']} (a one-sided accessory also lowers it)")

    for name in ("front", "right", "front_right", "front_left"):
        other = mirror_of(name)
        if name in views and other in views:
            mirrored = views[other].mask[:, ::-1]
            if name in DIAGONAL_VIEWS:  # diagonal views have no shared horizontal centre
                value = max(iou(views[name].mask, np.roll(mirrored, s, axis=1)) for s in range(-80, 81, 4))
            else:
                value = iou(views[name].mask, mirrored)
            onion(views[name].mask, mirrored, out / "onion" / f"{name}__{other}_mirrored.png")
            add(f"{other} is the mirror image of {name}", "PASS" if value >= limit["mirror_iou"] else "FAIL",
                round(value, 3), f">= {limit['mirror_iou']}")

    for name, view in views.items():
        base = name.removesuffix("_clay")
        if name.endswith("_clay") and base in views:
            value = iou(view.mask, views[base].mask)
            onion(views[base].mask, view.mask, out / "onion" / f"{base}__{name}.png")
            add(f"{name} keeps the silhouette of {base}", "PASS" if value >= limit["clay_iou"] else "FAIL",
                round(value, 3), f">= {limit['clay_iou']}")
        if name.endswith("_clay") or name.startswith(CLAY_PREFIXES):
            chroma = float(np.median(np.hypot(*figure_lab(view)[:, 1:].T)))
            add(f"{name} is grey clay (no colour)", "PASS" if chroma <= limit["clay_chroma"] else "FAIL",
                round(chroma, 1), f"median chroma <= {limit['clay_chroma']}")

    for body, dressed in (("body_front", "front"), ("body_right", "right"), ("body_left", "left")):
        if body in views and dressed in views:
            value, _ = best_shift(views[body].mask, views[dressed].mask, range(-150, 151, 10))
            add(f"{body} fits inside {dressed}", "PASS" if value <= limit["body_outside"] else "FAIL",
                round(value, 3), f"<= {limit['body_outside']} outside")

    for name, view in views.items():
        if not name.startswith(FACE_PREFIXES):
            continue
        side = next((s for s in ("right", "left") if name.endswith(f"_{s}")), "front")
        head_name = f"head_{side}"
        if head_name not in views:
            continue
        head = views[head_name].raw_mask
        shape = np.asarray(Image.fromarray(view.raw_mask).resize(head.shape[::-1], Image.NEAREST))
        value = iou(head, shape)
        add(f"{name} keeps the framing of {head_name}", "PASS" if value >= limit["face_iou"] else "FAIL",
            round(value, 3), f">= {limit['face_iou']}")

    centres = palette(figure_lab(views["front"]))
    for name, view in views.items():
        if name == "front" or name.endswith("_clay") or name.startswith(CLAY_PREFIXES + ("look_",)):
            continue
        distance = np.sqrt(((figure_lab(view)[:, None] - centres[None]) ** 2).sum(-1)).min(axis=1)
        value = float((distance > 20).mean())
        add(f"{name} uses the colours of front", "PASS" if value <= limit["new_colour"] else "WARN",
            round(value, 3), f"<= {limit['new_colour']} new colour (a back pattern also adds colour)")

    full_body = {n: v for n, v in views.items() if n in AZIMUTH}
    hull_report = {}
    if "front" in full_body and {"right", "left"} & set(full_body) and len(full_body) >= 3:
        hull = carve(full_body)
        hull_report = {"views": hull.used, "diagonal_shift_voxels": hull.shifts,
                       "missing": {n: round(v, 4) for n, v in hull.missing.items()}}
        for name, value in hull.missing.items():
            status = "PASS" if value <= limit["carve_pass"] else "WARN" if value <= limit["carve_fail"] else "FAIL"
            add(f"{name} agrees with the 3D shape of all views", status, round(value, 3),
                f"<= {limit['carve_pass']} of the silhouette unexplained (FAIL > {limit['carve_fail']})")
        _save_hull_views(hull, full_body, out / "hull")
        for name in DIAGONAL_VIEWS:
            if name in full_body:
                add(f"{name}: direction", *_direction_check(name, full_body))

    marks_file = folder / "landmarks.json"
    if marks_file.is_file():
        marks = json.loads(marks_file.read_text(encoding="utf-8"))
        for key in sorted({k for v in marks.values() for k in v}):
            values = [v[key] for v in marks.values() if key in v]
            if len(values) > 1:
                spread = max(values) - min(values)
                add(f"landmark {key} is at the same height in {len(values)} views",
                    "PASS" if spread <= limit["landmark_spread"] else "FAIL", round(spread, 1),
                    f"<= {limit['landmark_spread']} % of height")
    return _write(out, rows, {n: v.measure for n, v in views.items()}, hull_report)


def _direction_check(name: str, views: dict[str, View], margin: float = 0.02) -> tuple[str, str, str]:
    """Warn when a diagonal view clearly fits another diagonal direction better than its name.

    The axis-only hull is projected in the four diagonal directions. Front and
    back 3/4 silhouettes of a figure are often near mirror images, so a small
    difference decides nothing. Only a lead of ``margin`` IoU or more warns.
    """
    hull = carve({n: v for n, v in views.items() if n in AXIS_VIEWS})
    width, gcx = CANVAS_W // hull.factor, CX / hull.factor
    rows, ys, xs = hull.occupied.shape
    x = np.arange(xs) + hull.x_start + 0.5 - gcx
    y = np.arange(ys) + hull.y_start + 0.5 - gcx
    figure = downsample(views[name].mask, hull.factor)
    scores = {}
    for candidate in DIAGONAL_VIEWS:
        projection = silhouette(hull.occupied, candidate, x, y, width, gcx)
        scores[candidate] = max(iou(figure, np.roll(projection, s, axis=1)) for s in range(-40, 41))
    best = max(scores, key=scores.get)
    text = ", ".join(f"{k} {v:.3f}" for k, v in sorted(scores.items(), key=lambda kv: -kv[1]))
    if scores[best] - scores[name] >= margin:
        return "WARN", f"fits {best} better than {name} ({text})", f"lead >= {margin}: check the file name"
    return "PASS", text, f"no other direction leads by >= {margin}"


def _save_hull_views(hull, views: dict[str, View], folder: Path) -> None:
    width, gcx = CANVAS_W // hull.factor, CX / hull.factor
    rows, ys, xs = hull.occupied.shape
    x = np.arange(xs) + hull.x_start + 0.5 - gcx
    y = np.arange(ys) + hull.y_start + 0.5 - gcx
    for name in hull.used:
        projection = np.roll(silhouette(hull.occupied, name, x, y, width, gcx), hull.shifts.get(name, 0), axis=1)
        figure = downsample(views[name].mask, hull.factor)
        colour_map([(projection, (225, 225, 225)), (figure & projection, (150, 150, 150)),
                    (figure & ~projection, (220, 40, 40))], figure.shape) \
            .resize((width * 3, figure.shape[0] * 3), Image.NEAREST).save(folder / f"{name}.png")


def _write(out: Path, rows: list, measures: dict, hull: dict | None = None) -> dict:
    failed = any(r["status"] == "FAIL" for r in rows)
    report = {"pass": not failed, "checks": rows, "views": measures, "hull": hull or {}}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["| Result | Check | Value | Limit |", "|---|---|---|---|"]
    lines += [f"| {r['status']} | {r['check']} | {r['value']} | {r['limit']} |" for r in rows]
    lines.append(f"\nOverall: {'FAIL' if failed else 'PASS'}. Files are in {out}")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
