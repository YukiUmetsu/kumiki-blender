"""Synthetic geometry masks test provenance and comparison independently of GPU/browser."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from kumiki_blender.bridge_parity import (BridgeParityError, camera_pose, compare_captures,
                                           frame_name, load_plan, make_receipt)


def _plan(model: Path) -> dict:
    return {"schema_version": 1, "asset_id": "test-rig", "story_id": "craft-test--parity",
            "action_id": "rod-swing", "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "frames": [0, 11], "width": 160, "height": 100,
            "camera_preset": {"schemaVersion": 1, "mode": "replace-shot", "camera": {
                "space": "world", "target": [0, 0, 0], "azimuthDeg": 0, "elevationDeg": 0,
                "heightOffset": 0, "distanceFactor": 3, "rollDeg": 0,
                "projection": {"kind": "perspective", "fovDeg": 32}}}}


def _captures(tmp_path: Path, *, shifted: bool = False) -> tuple[Path, Path, Path]:
    model = tmp_path / "model.glb"
    model.write_bytes(b"fixture geometry")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(_plan(model)))
    blender = tmp_path / "blender"
    web = tmp_path / "web"
    blender.mkdir()
    web.mkdir()
    for directory in (blender, web):
        for frame in (0, 11):
            image = Image.new("RGB", (160, 100), "black")
            draw = ImageDraw.Draw(image)
            dx = 14 if shifted and directory == web else 0
            draw.rectangle((30 + dx + frame // 11, 20, 85 + dx + frame // 11, 75), fill="white")
            image.save(directory / frame_name(frame))
        receipt = make_receipt(plan, model, directory, "blender" if directory == blender else "kumiki-webgl")
        (directory / "capture.json").write_text(json.dumps(receipt))
    return plan, blender, web


def test_camera_pose_matches_three_to_blender_axes(tmp_path: Path) -> None:
    model = tmp_path / "model.glb"
    model.write_bytes(b"bytes")
    pose, target, fov, roll = camera_pose(_plan(model))
    assert pose == pytest.approx((0, -3, 0))
    assert target == (0, 0, 0)
    assert fov == 32 and roll == 0


def test_identical_masks_pass_and_write_comparison(tmp_path: Path) -> None:
    plan, blender, web = _captures(tmp_path)
    report = compare_captures(plan, blender, web, tmp_path / "report")
    assert report["status"] == "pass"
    assert all(row["iou"] == 1 and row["status"] == "pass" for row in report["frames"])
    assert (tmp_path / "report" / "contact-sheet.png").is_file()


def test_silhouette_shift_requires_review(tmp_path: Path) -> None:
    plan, blender, web = _captures(tmp_path, shifted=True)
    report = compare_captures(plan, blender, web, tmp_path / "report")
    assert report["status"] == "review-required"
    assert all(row["iou"] < .9 for row in report["frames"])


def test_rejects_tampered_capture_and_plan(tmp_path: Path) -> None:
    plan, blender, web = _captures(tmp_path)
    Image.new("RGB", (160, 100), "white").save(web / frame_name(0))
    with pytest.raises(BridgeParityError, match="changed since capture"):
        compare_captures(plan, blender, web, tmp_path / "report")
    assert not (tmp_path / "report").exists()
    changed = json.loads(plan.read_text())
    changed["camera_preset"]["camera"]["azimuthDeg"] = 10
    plan.write_text(json.dumps(changed))
    with pytest.raises(BridgeParityError, match="receipt differs from plan"):
        compare_captures(plan, blender, web, tmp_path / "report")


def test_rejects_uninformative_frame_and_existing_output(tmp_path: Path) -> None:
    plan, blender, web = _captures(tmp_path)
    Image.new("RGB", (160, 100), "black").save(web / frame_name(11))
    (web / "capture.json").write_text(json.dumps(make_receipt(plan, tmp_path / "model.glb", web, "kumiki-webgl")))
    with pytest.raises(BridgeParityError, match="empty/nearly full"):
        compare_captures(plan, blender, web, tmp_path / "report")
    assert not (tmp_path / "report").exists()
    (tmp_path / "report").mkdir()
    with pytest.raises(BridgeParityError, match="must be new"):
        compare_captures(plan, blender, web, tmp_path / "report")


@pytest.mark.parametrize("frames", [[1, 1], [2, 1], [-1], list(range(25)), [True]])
def test_rejects_invalid_frame_list(tmp_path: Path, frames: list[int]) -> None:
    model = tmp_path / "model.glb"
    model.write_bytes(b"a")
    data = _plan(model)
    data["frames"] = frames
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(data))
    with pytest.raises(BridgeParityError, match="frames"):
        load_plan(plan)
