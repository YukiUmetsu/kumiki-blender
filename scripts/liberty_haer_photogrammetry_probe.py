"""Attempt *image-derived* 3D registration of archival Liberty photographs.

This deliberately does not generate a statue or modify the production mesh.
Only photographs of the installed New York monument are accepted.  The output
is a sparse, unscaled reconstruction and a diagnostic report; even a successful
solve is not automatically accurate enough to drive face sculpting.

Optional dependency: pycolmap 4.x (CPU SIFT and incremental SfM).

Example:
    PYTHONPATH=/tmp/liberty-photogrammetry python3 scripts/liberty_haer_photogrammetry_probe.py \
      --photos .review/liberty-haer --output .review/liberty-r65-photogrammetry
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from pathlib import Path

from blender_liberty_reference_studio import ARCHIVED_VIEWS, DETAIL_VIEWS


def summarize_reconstructions(reconstructions: dict) -> dict:
    """Report *observed* registrations without inventing metric accuracy."""
    candidates = []
    for reconstruction_id, reconstruction in reconstructions.items():
        observations = [point.error for point in reconstruction.points3D.values()]
        candidates.append({
            "reconstruction_id": int(reconstruction_id),
            "registered_images": reconstruction.num_reg_images(),
            "image_names": sorted(
                reconstruction.images[image_id].name
                for image_id in reconstruction.reg_image_ids()
            ),
            "triangulated_points": reconstruction.num_points3D(),
            "median_reprojection_error_pixels": (
                round(statistics.median(observations), 3) if observations else None
            ),
        })
    candidates.sort(key=lambda c: (c["registered_images"], c["triangulated_points"]),
                    reverse=True)
    best = candidates[0] if candidates else None
    # This gate is intentionally strict about useful coverage, yet never
    # promotes a sparse cloud to a photorealistic surface reconstruction.
    return {
        "source": "Library of Congress HAER NY-138, installed statue, 1984-85",
        "approach": "SIFT feature matching + incremental structure-from-motion",
        "scale": "UNKNOWN: no calibrated scene control",
        "candidates": candidates,
        "candidate_for_manual_registration_review": bool(
            best and best["registered_images"] >= 5
            and best["triangulated_points"] >= 300
        ),
        "model_release": "HOLD: sparse, unscaled points are not sculptable facial geometry",
        "limitations": [
            "Scaffold occludes much of the head in the archival photographs",
            "Exposure, lighting, age and focus vary between captures",
            "Some restoration close-ups do not overlap the whole-head photos",
            "No known camera intrinsics, pose, or original negative calibration",
            "Profile overlap cannot establish the hidden frontal anatomy",
        ],
    }


def summarize_image_matches(database_path: Path) -> dict:
    """Expose whether apparent strong matches are planar/panoramic degeneracy."""
    pair_base = 2147483647  # COLMAP's stable pair-id packing convention.
    with sqlite3.connect(database_path) as db:
        names = dict(db.execute("SELECT image_id, name FROM images"))
        rows = db.execute(
            "SELECT pair_id, rows, config FROM two_view_geometries "
            "ORDER BY rows DESC LIMIT 10"
        ).fetchall()
    configurations = {
        0: "undefined", 1: "degenerate", 2: "calibrated",
        3: "uncalibrated", 4: "planar", 5: "panoramic",
        6: "planar_or_panoramic", 7: "watermark", 8: "multiple",
        9: "calibrated_rig",
    }
    return {
        "strongest_pairs": [
            {"first": names[pair_id // pair_base],
             "second": names[pair_id % pair_base],
             "inlier_matches": inliers,
             "geometric_configuration": configurations.get(config, f"unknown_{config}")}
            for pair_id, inliers, config in rows
        ],
        "warning": "Planar/panoramic matches do not prove triangulatable depth",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-details", action="store_true",
                        help="Add 5 close-up images, which may have no matches")
    args = parser.parse_args()

    try:
        import pycolmap
    except ImportError as error:
        parser.error("Install optional pycolmap 4.x to attempt SfM")
        raise error

    views = ARCHIVED_VIEWS + DETAIL_VIEWS if args.include_details else ARCHIVED_VIEWS
    photo_root = args.photos.resolve()
    image_names = [view.filename for view in views]
    missing = [name for name in image_names if not (photo_root / name).is_file()]
    if missing:
        parser.error(f"Missing verified installed-monument HAER photos: {missing}")
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Choose a fresh output directory; existing evidence is immutable")
    out.mkdir(parents=True, exist_ok=True)

    db = out / "images.db"
    features = pycolmap.FeatureExtractionOptions()
    features.max_image_size = 2048
    features.num_threads = 4
    pycolmap.extract_features(db, photo_root, image_names=image_names,
                             camera_mode=pycolmap.CameraMode.PER_IMAGE,
                             extraction_options=features, device=pycolmap.Device.cpu)
    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = 4
    pycolmap.match_exhaustive(db, matching_options=matching, device=pycolmap.Device.cpu)
    sparse_dir = out / "sparse"
    sparse_dir.mkdir()
    reconstructions = pycolmap.incremental_mapping(db, photo_root, sparse_dir)
    result = summarize_reconstructions(reconstructions)
    result["two_view_matching"] = summarize_image_matches(db)
    result["input_photos"] = image_names
    result["pycolmap_version"] = pycolmap.__version__
    (out / "registration-report.json").write_text(json.dumps(result, indent=2)+"\n")
    for recon_id, reconstruction in reconstructions.items():
        reconstruction.export_PLY(str(out / f"reconstruction-{recon_id}.ply"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
