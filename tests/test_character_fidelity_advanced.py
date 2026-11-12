"""Focused regression coverage for ungated character fidelity enhancements."""
import io
import json

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("skimage")
from kumiki_blender.character_fidelity.advanced_metrics import (
    contour_geometry, perceptual_color, relative_depth_hint,
)
from kumiki_blender.character_fidelity.schema import load_manifest
from kumiki_blender.character_fidelity.public_models import download_public_checkpoint
from kumiki_blender.character_fidelity.evaluate import evaluate_character

from test_character_fidelity import _make_fixture


def test_intrinsic_contour_translation_invariant():
    a = np.zeros((96, 96), bool)
    b = np.zeros_like(a)
    a[15:55, 9:42] = True
    b[34:74, 42:75] = True
    report = contour_geometry(a, b)
    assert report["intrinsic_hausdorff_normalized"] == pytest.approx(0)
    assert report["curvature_histogram_tvd"] == pytest.approx(0)


def test_wrong_shape_is_penalized():
    a = np.zeros((96, 96), bool)
    b = np.zeros_like(a)
    a[20:70, 20:70] = True
    b[20:70, 35:58] = True
    assert contour_geometry(a, b)["intrinsic_hausdorff_normalized"] > 0.02


def test_delta_e2000_separates_palette_from_alignment():
    mask = np.ones((40, 40), bool)
    white = np.full((40, 40, 3), [235, 235, 235], dtype=np.uint8)
    red = np.full((40, 40, 3), [200, 32, 50], dtype=np.uint8)
    assert perceptual_color(white, white, mask, mask, registered=True)["median_palette_delta_e2000"] == pytest.approx(0)
    diff = perceptual_color(white, red, mask, mask, registered=False)
    assert diff["median_palette_delta_e2000"] > 30
    assert "registered_overlap_delta_e2000_mean" not in diff


def test_relative_depth_is_hypothesis_not_metric():
    mask = np.ones((32, 32), bool)
    ref = np.tile(np.linspace(1, 2, 32), (32, 1))
    sculpt = ref + 3
    report = relative_depth_hint(ref, sculpt, mask, mask, verified_alignment=True)
    assert report["status"] == "unverified-relative-depth-hint"
    assert report["rank_correlation_spearman"] > .99
    assert "depth_mae_m" not in report
    assert relative_depth_hint(ref, sculpt, mask, mask, verified_alignment=False)["status"] == "not-comparable"


def test_schema_rejects_estimated_hint_as_metric_ground_truth(tmp_path):
    manifest, _ = _make_fixture(tmp_path)
    config = json.loads(manifest.read_text())
    np.savez_compressed(tmp_path / "hint.npz", depth_relative=np.ones((64, 64)))
    config["views"][0]["reference_depth_hint"] = {
        "path": "hint.npz", "provenance": "estimated", "near_is_small": False}
    manifest.write_text(json.dumps(config))
    load_manifest(manifest)
    config["views"][0]["reference_depth"] = {
        "path": "hint.npz", "units": "meters", "provenance": "estimated"}
    manifest.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Depth ground truth"):
        load_manifest(manifest)


def test_public_checkpoint_no_registry_or_overwrite(tmp_path):
    with pytest.raises(ValueError, match="Unsupported public checkpoint"):
        download_public_checkpoint("arbitrary-user-model", tmp_path)

    class Response(io.BytesIO):
        headers = {"Content-Type": "application/octet-stream"}
        def geturl(self):
            return "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.close()

    result = download_public_checkpoint("sam2.1-small", tmp_path,
        opener=lambda *a, **kw: Response(b"0" * (1024 * 1024 + 8)))
    assert result.stat().st_size == 1024*1024 + 8
    with pytest.raises(ValueError, match="already exists"):
        download_public_checkpoint("sam2.1-small", tmp_path)
    assert not list(tmp_path.glob(".download-*"))


def test_end_to_end_advanced_channels_do_not_claim_depth_truth(tmp_path):
    workspace = tmp_path / "source"
    workspace.mkdir()
    manifest, capture = _make_fixture(workspace)
    config = json.loads(manifest.read_text())
    np.savez_compressed(workspace / "hint.npz",
                        depth_relative=np.tile(np.linspace(0, 1, 64), (64, 1)))
    config["views"][0]["reference_depth_hint"] = {
        "path": "hint.npz", "provenance": "estimated", "near_is_small": True}
    manifest.write_text(json.dumps(config))
    report = evaluate_character(manifest, capture, tmp_path / "report")
    part = report["parts"][0]
    assert part["contour_geometry"]["status"] == "diagnostic"
    assert part["perceptual_color"]["median_palette_delta_e2000"] == pytest.approx(0)
    assert part["depth_hint"]["status"] == "insufficient-depth-variation"
    assert part["depth"]["status"].startswith("sculpt-only")
