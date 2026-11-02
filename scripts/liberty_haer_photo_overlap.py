"""Exploratory feature overlap in six *unregistered* actual-head HAER images.

The photos are noncontemporaneous, have unknown intrinsics, contain scaffolding,
and are often extreme profiles. An epipolar inlier count is NOT camera
calibration, reconstruction, or anatomical accuracy. Do not feed this report
to the photographic likeness acceptance gate.

Optional dependency: ``opencv-python-headless>=4.8`` (SIFT is built in).
Run with ``python scripts/liberty_haer_photo_overlap.py --reference-dir ...``.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

from blender_liberty_reference_studio import ARCHIVED_VIEWS


# Deliberately broad, hand-inspected *pixel* rectangles around the statue's
# head/face, excluding most background/scaffolding. Not photogrammetric masks;
# the crown and visible scaffolding can remain inside some regions.
HEAD_ROIS = {
    236: (150, 360, 620, 825),
    237: (90, 180, 540, 835),
    238: (200, 410, 640, 910),
    239: (280, 175, 930, 710),
    240: (185, 240, 690, 930),
    241: (210, 180, 690, 660),
}


def classify_overlap(good: int, planar: int, epipolar: int) -> str:
    """Conservative *study* flags; no result authorizes a 3D release."""
    if good < 20 or epipolar < 15:
        return "weak_correspondence"
    if planar >= 0.75 * good:
        return "homography_dominated"
    return "unverified_candidate"


def analyze(directory: Path) -> dict:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("OpenCV with SIFT is needed for this optional study") from exc

    descriptors = {}
    detector = cv2.SIFT_create(nfeatures=3000)
    for photo in ARCHIVED_VIEWS:
        path = directory / photo.filename
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"cannot decode authentic archive image: {path}")
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        view = clahe.apply(image)
        key = int(photo.survey_id.rsplit("-", 1)[-1])
        x1, y1, x2, y2 = HEAD_ROIS[key]
        if x2 > image.shape[1] or y2 > image.shape[0]:
            raise ValueError(f"unexpected dimensions for HAER {key}: {image.shape}")
        mask = np.zeros(image.shape, dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        keypoints, values = detector.detectAndCompute(view, mask)
        descriptors[key] = (keypoints, values)

    matcher = cv2.BFMatcher()
    pairs = []
    for (left, (kp1, d1)), (right, (kp2, d2)) in combinations(
        descriptors.items(), 2
    ):
        good = []
        if d1 is not None and d2 is not None:
            for neighbors in matcher.knnMatch(d1, d2, k=2):
                if len(neighbors) == 2 and neighbors[0].distance < .72 * neighbors[1].distance:
                    good.append(neighbors[0])
        # Avoid eight-point degeneracy; robust epipolar estimates need a
        # genuinely distributed, well-corresponding, non-planar track set.
        count_f, count_h = 0, 0
        if len(good) >= 16:
            points1 = np.float32([kp1[match.queryIdx].pt for match in good])
            points2 = np.float32([kp2[match.trainIdx].pt for match in good])
            _matrix, inliers = cv2.findFundamentalMat(
                points1, points2, cv2.FM_RANSAC, 1.5, .995, 5000
            )
            # Some eight-point edge cases can return invalid/uninitialized
            # values, so only masks that explicitly equal 1 count as inliers.
            count_f = int(np.count_nonzero(inliers == 1)) if inliers is not None else 0
            _homography, h_inliers = cv2.findHomography(
                points1, points2, cv2.RANSAC, 3.0
            )
            count_h = int(np.count_nonzero(h_inliers == 1)) if h_inliers is not None else 0
        pairs.append({
            "pair": f"HAER-{left} / HAER-{right}",
            "ratio_pass_matches": len(good),
            "fundamental_ransac_inliers": count_f,
            "homography_ransac_inliers": count_h,
            "study_flag": classify_overlap(len(good), count_h, count_f),
        })
    return {
        "status": "UNREGISTERED / EXPERIMENTAL / NOT LIKENESS EVIDENCE",
        "source": "Library of Congress HAER NY-138, actual installed head, 1984",
        "opencv_version": cv2.__version__,
        "roi_method": "approximate head rectangles, manually estimated; scaffolding remains",
        "camera_calibrated": False,
        "pose_recovered": False,
        "reconstruction_accepted": False,
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Optional JSON report, never a release gate")
    args = parser.parse_args()
    report = analyze(args.reference_dir)
    print(json.dumps(report, indent=2))
    if args.output:
        if args.output.exists():
            parser.error("output exists; use a new checkpoint rather than overwriting")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
