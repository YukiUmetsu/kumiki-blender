"""Compare calibrated binary silhouette masks and named anatomical landmarks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kumiki_blender.reconstruction_quality import (  # noqa: E402
    ReconstructionEvidenceError, compare_registered,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference-mask", "render-mask", "reference-landmarks", "render-landmarks"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--camera-evidence", required=True)
    parser.add_argument("--min-iou", type=float, required=True)
    parser.add_argument("--max-landmark-error", type=float, required=True)
    parser.add_argument("--baseline-render-mask", type=Path)
    parser.add_argument("--baseline-render-landmarks", type=Path)
    parser.add_argument("--require-nonregression", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = compare_registered(
            reference_mask=args.reference_mask, render_mask=args.render_mask,
            reference_landmarks=args.reference_landmarks,
            render_landmarks=args.render_landmarks,
            registration=args.registration, camera_evidence=args.camera_evidence,
            min_iou=args.min_iou, max_landmark_error=args.max_landmark_error,
            baseline_render_mask=args.baseline_render_mask,
            baseline_render_landmarks=args.baseline_render_landmarks,
            require_nonregression=args.require_nonregression,
        )
    except (ReconstructionEvidenceError, ValueError, OSError, ImportError) as error:
        result = {"status": "HOLD", "reasons": [str(error)]}
    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(payload+"\n", encoding="utf-8")
    print(payload)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
