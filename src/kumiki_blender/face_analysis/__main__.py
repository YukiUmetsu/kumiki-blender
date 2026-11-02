"""Standalone photo evidence/QA CLI that never changes the shared model CLI."""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .compare import ViewPair, compare_views, summarize_observations
from .detectors import propose_landmarks
from .inputs import (inspect_coverage, load_annotations, load_photo_set,
                     photos_from_reference_set)
from .registration import load_registration_pairs, propose_registrations
from .reports import load_evidence_bundle, load_pairs, write_analysis_revision


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m kumiki_blender.face_analysis",
                                     description="Independent statue facial evidence and projected-view QA")
    subcommands = parser.add_subparsers(dest="command", required=True)
    inspect = subcommands.add_parser("inspect", help="Audit private multi-view reference photos")
    input_group = inspect.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--images-dir", type=Path)
    input_group.add_argument("--reference-manifest", type=Path)
    inspect.add_argument("--annotations", type=Path)
    inspect.add_argument("--detector", choices=("none", "mediapipe"), default="none")
    inspect.add_argument("--output", required=True, type=Path)
    compare = subcommands.add_parser("compare", help="Compare verified same-view photo and model evidence")
    compare.add_argument("--reference", required=True, type=Path)
    compare.add_argument("--candidate", required=True, type=Path)
    compare.add_argument("--pairs", required=True, type=Path)
    compare.add_argument("--output", required=True, type=Path)
    register = subcommands.add_parser("register", help="Suggest unverified 2D similarity alignment")
    register.add_argument("--reference", required=True, type=Path)
    register.add_argument("--candidate", required=True, type=Path)
    register.add_argument("--anchors", required=True, type=Path)
    register.add_argument("--output", required=True, type=Path)
    reproject = subcommands.add_parser("reproject", help="Reuse calibrated 3D-to-2D landmark checks")
    reproject.add_argument("--camera-plan", required=True, type=Path)
    reproject.add_argument("--sculpt-landmarks", required=True, type=Path)
    reproject.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            raise ValueError(f"Output revision already exists: {args.output}")
        if args.command == "inspect":
            if args.reference_manifest:
                from ..reconstruction.references import load_reference_set
                photos = photos_from_reference_set(load_reference_set(args.reference_manifest))
            else:
                if not args.images_dir.is_dir() or args.images_dir.is_symlink():
                    raise ValueError("images-dir must be an existing regular directory")

                def natural_key(path: Path) -> tuple[tuple[int, int | str], ...]:
                    return tuple((1, int(chunk)) if chunk.isdigit() else (0, chunk)
                                 for chunk in re.split(r"(\d+)", path.name.lower()))

                images = sorted((path for path in args.images_dir.iterdir()
                                 if path.suffix.lower() in (".png", ".jpg", ".jpeg")),
                                key=natural_key)
                photos = load_photo_set(images)
            annotations = load_annotations(args.annotations, photos) if args.annotations else None
            view_map = annotations.images if annotations else {}
            reviewed_photos = [replace(photo, view=view_map[photo.image_id].view)
                               if photo.image_id in view_map else photo for photo in photos]
            audit = inspect_coverage(reviewed_photos, annotations)
            audit["images"] = [{"image_id": photo.image_id, "width": photo.width,
                                "height": photo.height, "sha256": photo.sha256,
                                "view": photo.view, "provenance": photo.provenance}
                               for photo in reviewed_photos]
            proposals = None
            if args.detector != "none":
                proposals = {"schema_version": 1, "images": [
                    {"image_id": photo.image_id, **propose_landmarks(photo, detector=args.detector)}
                    for photo in reviewed_photos],
                    "limitations": "All automatic points are unverified 2D proposals, never sculpt ground truth."}
            write_analysis_revision(args.output, audit, proposals, None,
                                    photos=reviewed_photos, annotations=annotations)
        elif args.command == "compare":
            reference = load_evidence_bundle(args.reference)
            candidate = load_evidence_bundle(args.candidate)
            pairs = [ViewPair(**row) for row in load_pairs(args.pairs)]
            result = compare_views(reference, candidate, pairs)
            result["observations"] = summarize_observations(result)
            write_analysis_revision(args.output, {}, None, result)
        elif args.command == "register":
            reference = load_evidence_bundle(args.reference)
            candidate = load_evidence_bundle(args.candidate)
            anchors = load_registration_pairs(args.anchors)
            report = propose_registrations(reference, candidate, anchors)
            write_analysis_revision(args.output, {}, None, None, registration=report)
        else:
            from ..reconstruction.landmarks import check_landmark_reprojection
            with tempfile.TemporaryDirectory(prefix=".face-reprojection-") as temporary:
                reprojection = check_landmark_reprojection(args.camera_plan,
                                                           args.sculpt_landmarks,
                                                           Path(temporary) / "registered-qa")
            write_analysis_revision(args.output, {}, None, None, reprojection=reprojection)
    except (OSError, ValueError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
