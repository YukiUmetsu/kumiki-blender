"""Stage an isolated, explicitly experimental dense input from repaired RoMa cameras.

The already-rectified 52-view camera revision supplies pinned pixels. A
separately repaired 50-view sparse model supplies geometry. They must come
from *the same* rectification, proven by the original candidate model hashes.
Neither inputs nor previous dense reconstructions are changed.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from .cameras import _read_intrinsics, _read_poses
from .dense import _IMAGE_NAME, _SHA256, _check_regular, _json, _sha256
from .references import ReferenceError


def _registered(model: Any) -> dict[str, Any]:
    ids = list(model.reg_image_ids())
    result = {model.images[i].name: model.images[i] for i in ids}
    if len(result) != len(ids):
        raise ReferenceError("COLMAP registered image names are not unique")
    return result


def stage_repaired_roma_dense_revision(
    source_reconstruction: Path, repaired_revision: Path, output: Path, *,
    colmap: str = "colmap", model_loader: Callable[[Path], Any] | None = None,
    converter: Callable[[list[str]], int] | None = None,
) -> dict[str, Any]:
    """Write a hash-checked 50-view dense input; never mark a RoMa pose verified.

    The source is the exact 52-camera rectified revision used by the sparse
    repair (not a later differently-rectified 52-camera experiment). Its
    undistorted image hashes and camera parameters stay pinned.
    """
    src_requested, repair_requested, out_requested = (
        Path(source_reconstruction), Path(repaired_revision), Path(output))
    src, repair, out = (src_requested.resolve(), repair_requested.resolve(),
                        out_requested.resolve())
    if (any(path.is_symlink() for path in (src_requested, repair_requested, out_requested))
            or not src.is_dir() or not repair.is_dir() or src == repair
            or out.exists() or any(out.is_relative_to(folder) or folder.is_relative_to(out)
                                for folder in (src, repair))):
        raise ReferenceError("Require distinct real source/repair directories and an unused isolated output")
    source_report = _json(src / "reconstruction-report.json")
    source_index = _json(src / "undistorted-reference-index.json")
    repair_report = _json(repair / "roma-sparse-repair-report.json")
    before_gate, after_gate = (repair_report.get("before_camera_gate"),
                               repair_report.get("after_camera_gate"))
    excluded = repair_report.get("excluded_unverified_cameras")
    target_name = repair_report.get("target")
    if (not str(source_report.get("status", "")).startswith("candidate") or
            not isinstance(before_gate, dict) or not before_gate.get("accepted") or
            not isinstance(after_gate, dict) or not after_gate.get("accepted") or
            after_gate.get("new_filenames") != [target_name] or
            after_gate.get("reasons") or
            repair_report.get("status") != "isolated-camera-candidate / independent-qa-required" or
            type(target_name) is not str or not target_name or
            not isinstance(excluded, list) or not excluded or
            len(excluded) != len(set(excluded)) or
            any(type(name) is not str or not name for name in excluded) or
            target_name in excluded or
            before_gate.get("new_filenames") != sorted([target_name, *excluded]) or
            not isinstance(source_index.get("images"), list)):
        raise ReferenceError("Source or RoMa repair receipt lacks reviewed isolated-camera evidence")
    entries = source_index["images"]
    if any(not isinstance(item, dict) or
           not isinstance(item.get("filename"), str) or
           not _IMAGE_NAME.fullmatch(item["filename"]) or
           not isinstance(item.get("sha256"), str) or
           not _SHA256.fullmatch(item["sha256"])
           for item in entries):
        raise ReferenceError("Invalid rectified image path or checksum in source index")
    image_entries = {item["filename"]: item for item in entries}
    if (len(image_entries) != len(entries) or
            source_report.get("registered_images") != len(entries) or
            source_report.get("undistorted_images") != len(entries) or
            not isinstance(source_report.get("staged_independent_images"), int) or
            source_report["staged_independent_images"] < len(entries)):
        raise ReferenceError("Source rectified-image index does not match its registered cameras")

    source_sparse = src / "undistorted/sparse"
    repaired_sparse = repair / "sparse"
    if any(folder.is_symlink() or not folder.is_dir() for folder in (source_sparse, repaired_sparse)):
        raise ReferenceError("Missing real rectified sparse camera models")
    expected_hashes = repair_report.get("source_sha256", {}).get("candidate")
    if not isinstance(expected_hashes, dict):
        raise ReferenceError("RoMa repair receipt has no source model hashes")
    for filename in ("cameras.bin", "images.bin", "points3D.bin"):
        original = source_sparse / filename
        repaired = repaired_sparse / filename
        _check_regular(original, src)
        _check_regular(repaired, repair)
        if _sha256(original) != expected_hashes.get(filename):
            raise ReferenceError(
                "The rectified source model differs from the one actually repaired; "
                "use the source_revision recorded in the dense experiment's dense-report.json")
    for model_dir in (source_sparse, repaired_sparse):
        if any(path.is_symlink() or not path.is_file()
               for path in model_dir.iterdir()):
            raise ReferenceError("Unsafe or unexpected sparse model entries")
    if model_loader is None:
        import pycolmap
        model_loader = pycolmap.Reconstruction
    source_model, repaired_model = model_loader(source_sparse), model_loader(repaired_sparse)
    source_names, repaired_names = _registered(source_model), _registered(repaired_model)
    if (set(source_names) != set(image_entries) or
            set(repaired_names) != set(source_names) - set(excluded) or
            target_name not in repaired_names or
            len(repaired_names) != repair_report.get("registered_images")):
        raise ReferenceError("Repaired cameras disagree with source roster or approval receipt")
    earlier_unregistered = source_report.get("unregistered_filenames", [])
    if not isinstance(earlier_unregistered, list) or not all(
            isinstance(name, str) for name in earlier_unregistered):
        raise ReferenceError("Invalid source unregistered-camera evidence")

    # Check that the repaired image calibration still corresponds to these
    # rectified pixels. Poses must not move when removing observations/views.
    import numpy as np
    from PIL import Image

    for name in sorted(source_names):
        original = source_names[name]
        base_camera = source_model.cameras[original.camera_id]
        image_path = src / "undistorted/images" / name
        _check_regular(image_path, src)
        entry = image_entries[name]
        if (_sha256(image_path) != entry.get("sha256") or
                entry.get("dimensions") != [int(base_camera.width), int(base_camera.height)]):
            raise ReferenceError(f"Rectified image or image index does not match camera: {name}")
        with Image.open(image_path) as image:
            if image.size != (int(base_camera.width), int(base_camera.height)):
                raise ReferenceError(f"Rectified image pixels do not match camera: {name}")
            image.verify()
        if name in repaired_names:
            selected = repaired_names[name]
            camera = repaired_model.cameras[selected.camera_id]
            if (int(original.camera_id) != int(selected.camera_id) or
                    (int(camera.width), int(camera.height)) !=
                    (int(base_camera.width), int(base_camera.height)) or
                    not np.allclose(np.asarray(camera.params), np.asarray(base_camera.params),
                                    atol=1e-9, rtol=0) or
                    not np.allclose(np.asarray(original.projection_center()),
                                    np.asarray(selected.projection_center()), atol=1e-9, rtol=0) or
                    not np.allclose(np.asarray(original.viewing_direction()),
                                    np.asarray(selected.viewing_direction()), atol=1e-9, rtol=0)):
                raise ReferenceError(f"Camera calibration/pose changed during sparse repair: {name}")

    if converter is None:
        executable = shutil.which(colmap)
        if not executable:
            raise ReferenceError(f"Missing COLMAP model converter: {colmap}")

        def converter(args: list[str]) -> int:
            try:
                with (out / "model-converter.log").open("w", encoding="utf-8") as log:
                    return subprocess.run([executable, *args], check=False,
                                          stdout=log, stderr=subprocess.STDOUT,
                                          timeout=180).returncode
            except (OSError, subprocess.TimeoutExpired):
                return 127

    # All input validation completed before writing any output.
    (out / "undistorted/images").mkdir(parents=True, exist_ok=False)
    for name in sorted(repaired_names):
        photo = src / "undistorted/images" / name
        destination = out / "undistorted/images" / name
        shutil.copyfile(photo, destination)
        if _sha256(destination) != image_entries[name]["sha256"]:
            raise ReferenceError(f"Rectified photo changed while staging: {name}")
    shutil.copytree(repaired_sparse, out / "undistorted/sparse")
    staged_index = src / "staged-reference-index.json"
    if staged_index.exists() or staged_index.is_symlink():
        _check_regular(staged_index, src)
        shutil.copyfile(staged_index, out / "staged-reference-index.json")
        if _sha256(out / "staged-reference-index.json") != _sha256(staged_index):
            raise ReferenceError("Staged source reference index changed during copy")
    (out / "undistorted/text-model").mkdir()
    code = converter(["model_converter", "--input_path", str(out / "undistorted/sparse"),
                      "--output_path", str(out / "undistorted/text-model"),
                      "--output_type", "TXT"])
    if code:
        raise ReferenceError(f"Isolated COLMAP text model converter failed: {code}")
    text_model = out / "undistorted/text-model"
    _check_regular(text_model / "cameras.txt", out)
    _check_regular(text_model / "images.txt", out)
    intrinsics = _read_intrinsics(text_model / "cameras.txt")
    poses = _read_poses(text_model / "images.txt")
    if (set(name for _, _, name, _, _ in poses) != set(repaired_names) or
            any(image_entries[name]["dimensions"] != [intrinsics[cam]["width"],
                                                     intrinsics[cam]["height"]]
                for _, cam, name, _, _ in poses)):
        raise ReferenceError("Converted rectified camera model does not match pinned photos")

    original_names = set(image_entries)
    kept_entries = [entry for entry in entries if entry["filename"] in repaired_names]
    (out / "undistorted-reference-index.json").write_text(
        json.dumps({"images": kept_entries}, indent=2) + "\n", encoding="utf-8")
    rebuilt_report = {**source_report,
                      "status": "candidate / visual-review-required",
                      "registered_images": len(repaired_names),
                      "registered_filenames": sorted(repaired_names),
                      "unregistered_filenames": sorted(set(earlier_unregistered) |
                                                       (original_names - set(repaired_names))),
                      "undistorted_images": len(repaired_names),
                      "models": [{"model": "0", "registered_images": len(repaired_names),
                                  "registered_filenames": sorted(repaired_names)}],
                      "source_sparse_model": "0",
                      "undistorted_model": "undistorted/text-model",
                      "distribution_allowed": False,
                      "roma_experimental_camera_validation": "independent-pose-and-mask-qa-pending",
                      "roma_repair_source": str(repair),
                      "roma_source_revision": str(src)}
    (out / "reconstruction-report.json").write_text(
        json.dumps(rebuilt_report, indent=2) + "\n", encoding="utf-8")
    result = {"status": "isolated-experimental-dense-input / independent-qa-required",
              "source_revision": str(src), "repair_revision": str(repair),
              "registered_images": len(repaired_names), "target": target_name,
              "excluded_unverified_cameras": sorted(excluded),
              "rectified_images_sha256": {name: image_entries[name]["sha256"]
                                          for name in sorted(repaired_names)},
              "sparse_sha256": {name: _sha256(out / "undistorted/sparse" / name)
                                for name in ("cameras.bin", "images.bin", "points3D.bin")},
              "masks_reviewed": False, "pose_independently_validated": False,
              "distribution_allowed": False,
              "warning": "Exploratory A/B dense input only. Do not call RoMa-predicted tracks independently verified; correct masks and review camera overlays before meshing."}
    (out / "roma-dense-staging-report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
