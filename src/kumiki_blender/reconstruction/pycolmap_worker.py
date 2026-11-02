"""Subprocess-isolated CPU PyCOLMAP adapter for the sparse reconstruction worker.

This small CLI uses the same file layout as COLMAP's executable, so existing
timeouts, immutable revision staging and camera/undistortion validation apply.
No third-party assets or images are shipped by the adapter.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    values = sys.argv[1:] if argv is None else argv
    if values == ["--version"]:
        import pycolmap
        print(f"PyCOLMAP {pycolmap.__version__} CPU")
        return 0
    if len(values) != 2 and "--help" in values:
        return 2
    if len(values) == 2 and values[1] == "--help":
        if values[0] == "feature_extractor":
            print("--FeatureExtraction.use_gpu --max_image_size")
        elif values[0] == "exhaustive_matcher":
            print("--FeatureMatching.use_gpu")
        return 0
    if not values or values[0] not in {
        "feature_extractor", "exhaustive_matcher", "mapper",
        "model_converter", "image_undistorter",
    }:
        raise ValueError("Unrecognized PyCOLMAP stage")
    stage = values[0]
    parser = argparse.ArgumentParser(prog=f"pycolmap {stage}")
    parser.add_argument("--database_path", type=Path)
    parser.add_argument("--image_path", type=Path)
    parser.add_argument("--output_path", type=Path)
    parser.add_argument("--input_path", type=Path)
    parser.add_argument("--output_type")
    parser.add_argument("--ImageReader.mask_path", dest="mask_path", type=Path)
    parser.add_argument("--FeatureExtraction.use_gpu", dest="feature_gpu")
    parser.add_argument("--FeatureMatching.use_gpu", dest="matching_gpu")
    parser.add_argument("--max_image_size", type=int, default=3200)
    args = parser.parse_args(values[1:])
    import pycolmap

    # All steps deliberately use the CPU: this path also works on machines
    # without CUDA, while native COLMAP remains available for GPU acceleration.
    if stage == "feature_extractor":
        if args.feature_gpu != "0":
            raise ValueError("PyCOLMAP worker is CPU-only")
        reader = pycolmap.ImageReaderOptions()
        if args.mask_path:
            reader.mask_path = str(args.mask_path)
        options = pycolmap.FeatureExtractionOptions()
        options.max_image_size = args.max_image_size
        pycolmap.extract_features(args.database_path, args.image_path,
                                  reader_options=reader, extraction_options=options,
                                  device=pycolmap.Device.cpu)
    elif stage == "exhaustive_matcher":
        if args.matching_gpu != "0":
            raise ValueError("PyCOLMAP worker is CPU-only")
        pycolmap.match_exhaustive(args.database_path, device=pycolmap.Device.cpu)
    elif stage == "mapper":
        maps = pycolmap.incremental_mapping(args.database_path, args.image_path,
                                            args.output_path,
                                            options={"min_model_size": 3})
        for index, model in maps.items():
            target = args.output_path / str(index)
            target.mkdir(parents=True, exist_ok=True)
            model.write(target)
        if not maps:
            raise RuntimeError("PyCOLMAP registered no image models")
    elif stage == "model_converter":
        if args.output_type != "TXT":
            raise ValueError("Only TXT model conversion is supported")
        pycolmap.Reconstruction(args.input_path).write_text(args.output_path)
    elif stage == "image_undistorter":
        if args.output_type != "COLMAP":
            raise ValueError("Only COLMAP undistortion output is supported")
        options = pycolmap.UndistortCameraOptions()
        options.max_image_size = args.max_image_size
        pycolmap.undistort_images(args.output_path, args.input_path, args.image_path,
                                  undistort_options=options)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"PyCOLMAP worker failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
