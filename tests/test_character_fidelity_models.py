"""Offline unit tests for model adapters (no real weights or remote requests)."""
import json

import numpy as np
import pytest
from PIL import Image

from kumiki_blender.character_fidelity.sam2_masks import load_prompts, segment_sheet
from kumiki_blender.character_fidelity.moge_depth import estimate_depth_hint, relative_depth


def fixtures(tmp_path):
    source = tmp_path / "inputs"
    source.mkdir()
    rgba = np.zeros((80, 96, 4), dtype=np.uint8)
    rgba[8:70, 8:85] = (120, 75, 40, 255)
    image = source / "image.png"
    Image.fromarray(rgba).save(image)
    prompts = source / "prompts.json"
    prompts.write_text(json.dumps({"schema_version": 1, "parts": [
        {"id": "head", "category": "anatomy", "box": [10, 8, 75, 65],
         "positive_points": [[25, 30]], "negative_points": [[90, 75]]},
        {"id": "coat", "category": "costume", "box": [10, 40, 75, 75]},
    ]}))
    ckpt = source / "sam2.pt"
    ckpt.write_bytes(b"unit-test-dummy-checkpoint")
    return image, prompts, ckpt


class MockPredictor:
    def set_image(self, image):
        self.image = image

    def predict(self, *, point_coords, point_labels, box, multimask_output):
        h, w = self.image.shape[:2]
        masks = np.zeros((3, h, w), dtype=bool)
        x1, y1, x2, y2 = box.astype(int)
        masks[:, y1:y2, x1:x2] = True
        return masks, np.asarray([0.88, 0.4, 0.2]), None


def test_sam2_proposes_separate_full_canvas_body_and_costume_masks(tmp_path):
    image, prompts, checkpoint = fixtures(tmp_path)
    report = segment_sheet(image, prompts, checkpoint, tmp_path / "result",
                           predictor_factory=lambda *_: MockPredictor())
    assert report["status"].startswith("unapproved")
    assert [p["category"] for p in report["parts"]] == ["anatomy", "costume"]
    assert all(p["approved"] is False for p in report["parts"])
    head = np.asarray(Image.open(tmp_path / "result/masks/head.png"))
    assert head.shape == (80, 96)
    assert head[20, 20] == 255
    assert head[2, 2] == 0
    assert (tmp_path / "result/review.jpg").exists()


def test_sam2_requires_local_checkpoint_and_strict_pixel_prompts(tmp_path):
    image, prompts, _ = fixtures(tmp_path)
    with pytest.raises(ValueError, match="Checkpoint"):
        segment_sheet(image, prompts, tmp_path / "gated-model-not-here.pt", tmp_path / "result")
    assert not (tmp_path / "result").exists()
    data = json.loads(prompts.read_text())
    data["parts"][0]["positive_points"] = [[2000, 40]]
    prompts.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="inside the image"):
        load_prompts(prompts, (96, 80))


def test_relative_depth_is_near_small_and_rejects_flat_maps():
    depth = np.arange(100, dtype=np.float32).reshape((10, 10)) + 1
    depth[2, 2] = np.nan
    result = relative_depth(depth)
    assert np.isnan(result[2, 2])
    assert result[0, 0] == 0
    assert result[-1, -1] == 1
    assert result[4, 5] < result[8, 5]
    with pytest.raises(ValueError, match="variation"):
        relative_depth(np.ones((16, 16), dtype=np.float32))


def test_moge_hint_is_estimated_and_never_measured_ground_truth(tmp_path):
    image, _, checkpoint = fixtures(tmp_path)

    def fake(rgb, path, device):
        assert path == checkpoint.resolve()
        assert device == "cpu"
        h, w = rgb.shape[:2]
        return {"depth": np.tile(np.linspace(1, 5, w, dtype=np.float32), (h, 1)),
                "mask": np.ones((h, w), dtype=bool)}

    report = estimate_depth_hint(image, checkpoint, tmp_path / "depth", infer=fake)
    assert report["status"].startswith("estimated-only")
    assert report["manifest_reference_depth_hint"]["provenance"] == "estimated"
    assert "reference_depth" not in report
    with np.load(tmp_path / "depth/depth_hint.npz") as result:
        assert result["depth_relative"].shape == (80, 96)
        assert result["depth_relative"][30, 8] < result["depth_relative"][30, 70]


def test_moge_missing_checkpoint_refuses_network(tmp_path):
    image, _, _ = fixtures(tmp_path)
    with pytest.raises(ValueError, match="locally downloaded"):
        estimate_depth_hint(image, tmp_path / "missing.pt", tmp_path / "depth")
