"""Reviewable similarity-only camera alignment from named landmark correspondences.

Never asserts verification from a low residual alone. This provides an evidence-
rich candidate transform for an artist to inspect before enabling spatial QA.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .schema import asset, load_manifest


def _point(xy: Any, bounds: tuple[int, int], name: str) -> list[float]:
    if (not isinstance(xy, list) or len(xy) != 2 or
            any(type(x) not in (int, float) or not math.isfinite(x) for x in xy) or
            not 0 <= xy[0] < bounds[0] or not 0 <= xy[1] < bounds[1]):
        raise ValueError(f"{name} must be finite within image bounds")
    return [float(xy[0]), float(xy[1])]


def propose_registration(manifest: str | Path, capture: str | Path,
                         anchors: str | Path, output: str | Path, *,
                         view_id: str = "front", inlier_tolerance_px: float = 3) -> dict[str, Any]:
    """Estimate a uniform-scale 2D similarity and write a visual review overlay.

    This cannot prove matching pose, matching camera projection or anatomical
    correspondence; it deliberately writes verified:false in all cases.
    """
    manifest, capture, anchors, output = map(Path, (manifest, capture, anchors, output))
    if output.is_symlink() or output.exists():
        raise ValueError("Registration output must be a new directory")
    if (not isinstance(inlier_tolerance_px, (int, float)) or isinstance(inlier_tolerance_px, bool)
            or not math.isfinite(inlier_tolerance_px) or not 0.5 <= inlier_tolerance_px <= 25):
        raise ValueError("Inlier tolerance must be 0.5 to 25 pixels")
    manifest_data = load_manifest(manifest)
    view = next((v for v in manifest_data["views"] if v["id"] == view_id), None)
    if view is None:
        raise ValueError(f"Unknown view: {view_id}")
    source = asset(manifest.parent, view["reference_image"])
    metadata = json.loads(asset(capture, "capture.json").read_text(encoding="utf-8"))
    capture_view = metadata.get("views", {}).get(view_id)
    if not isinstance(capture_view, dict):
        raise ValueError("Blender capture does not contain requested view")
    render = asset(capture, capture_view["albedo"])
    with Image.open(source) as im:
        src = im.convert("RGB"); src_size = src.size
    with Image.open(render) as im:
        model = im.convert("RGB"); dest_size = model.size
    if dest_size != (view["width"], view["height"]):
        raise ValueError("Blender render dimensions do not match manifest")
    if anchors.is_symlink() or not anchors.is_file():
        raise ValueError("Anchor JSON must be a regular file")
    data = json.loads(anchors.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1 or data.get("view") != view_id:
        raise ValueError("Invalid or mismatched registration anchor JSON")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if data.get("source_image_sha256") not in (None, digest):
        raise ValueError("Registration anchors source SHA-256 mismatch")
    points = data.get("landmarks")
    if not isinstance(points, list) or not 4 <= len(points) <= 80:
        raise ValueError("Provide 4 to 80 named landmark correspondences")
    seen = set()
    left, right = [], []
    for row in points:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip():
            raise ValueError("Each landmark needs a nonempty ID")
        if row["id"] in seen:
            raise ValueError("Duplicate landmark correspondence")
        seen.add(row["id"])
        left.append(_point(row.get("source_xy"), src_size, "source_xy"))
        right.append(_point(row.get("render_xy"), dest_size, "render_xy"))
    ref = np.array(left, dtype=np.float64)
    dst = np.array(right, dtype=np.float64)
    if np.linalg.matrix_rank(ref - ref.mean(axis=0)) < 2:
        raise ValueError("Source landmarks must span a two-dimensional area")
    cv2.setRNGSeed(0)
    affine, inlier_labels = cv2.estimateAffinePartial2D(
        ref, dst, method=cv2.RANSAC, ransacReprojThreshold=float(inlier_tolerance_px),
        maxIters=2000, confidence=0.999, refineIters=15)
    if affine is None or inlier_labels is None:
        raise ValueError("Could not fit a nondegenerate similarity transform")
    projected = ref @ affine[:, :2].T + affine[:, 2]
    residuals = np.linalg.norm(projected - dst, axis=1)
    inliers = inlier_labels.reshape(-1).astype(bool)
    determinant = float(np.linalg.det(affine[:, :2]))
    scale = math.sqrt(max(0, determinant))
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError("Registration has invalid scale")
    count = int(inliers.sum())
    rmse = float(np.sqrt(np.mean(residuals[inliers] ** 2))) if count else float("inf")
    p95 = float(np.percentile(residuals[inliers], 95)) if count else float("inf")
    accepted = count >= max(3, math.ceil(len(points)*0.75)) and rmse <= inlier_tolerance_px
    # A small fit error cannot validate a stylized shape or the true 3D camera.
    status = "reviewable-candidate / not-verified" if accepted else "rejected-candidate / review-required"
    candidate = {"schema_version": 1, "status": status, "view": view_id,
        "source_image_sha256": digest, "capture_file": str(render),
        "alignment_candidate": {"verified": False,
                                "source_to_render": np.round(affine, 8).tolist()},
        "uniform_scale": round(scale, 7), "anchor_count": len(points),
        "inlier_count": count, "inlier_ratio": round(count / len(points), 5),
        "inlier_rmse_px": round(rmse, 4), "inlier_p95_px": round(p95, 4),
        "all_anchor_residuals_px": [round(float(x), 3) for x in residuals],
        "landmarks": [{"id": p["id"], "source_xy": list(ref[i]),
                       "render_xy": list(dst[i]), "projected_xy": np.round(projected[i], 3).tolist(),
                       "inlier": bool(inliers[i])} for i, p in enumerate(points)],
        "validation": {"camera_matched": False, "pose_matched": False,
            "manual_review_required": True},
        "limitations": ["Landmark fits do not prove camera or pose equivalence.",
            "Never set alignment.verified=true without reviewing the projected overlay and anatomical correspondences.",
            "No perspective/shear deformation is permitted to conceal sculpt errors."]}
    output.mkdir(parents=True, exist_ok=False)
    (output / "registration.json").write_text(json.dumps(candidate, indent=2) + "\n", encoding="utf-8")
    warped = cv2.warpAffine(np.asarray(src), affine, dest_size,
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(245,245,245))
    blended = (np.asarray(model).astype(np.float32)*.5 + warped.astype(np.float32)*.5).astype(np.uint8)
    image = Image.fromarray(blended)
    draw = ImageDraw.Draw(image)
    for i, row in enumerate(points):
        px, py = projected[i]; tx, ty = dst[i]
        color = "#00b96b" if inliers[i] else "#e04458"
        draw.line(((float(px),float(py)), (float(tx),float(ty))), fill=color, width=2)
        draw.ellipse((px-3,py-3,px+3,py+3), outline=color, width=2)
        draw.text((tx+4,ty+4), row["id"], fill=color, stroke_width=1, stroke_fill="white")
    image.save(output / "alignment_overlay.png")
    (output / "review.md").write_text(
        "# Registration proposal (not verified)\n\n"
        f"{count}/{len(points)} inliers; RMSE {rmse:.2f}px; P95 {p95:.2f}px.\n\n"
        "Review the overlay and verify that reference and render have matching camera, pose, "
        "scale and feature identity. The candidate JSON is intentionally verified=false.\n", encoding="utf-8")
    return candidate
