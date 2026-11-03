"""Promptable multi-part masks keep statue anatomy instead of selecting a head."""

import hashlib
import json
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from PIL import Image

from kumiki_blender.reconstruction.masking import _load_guides, _sam2_mask, generate_rectified_masks
from kumiki_blender.reconstruction.references import ReferenceError
from tests.test_reconstruction_masking import DenseMaskTests


class FakePredictor:
    def __init__(self):
        self.calls = []

    def set_image(self, rgb):
        self.shape = rgb.shape[:2]

    def predict(self, *, box, point_coords, point_labels, multimask_output):
        self.calls.append((box.tolist(), None if point_coords is None else point_coords.tolist(),
                           None if point_labels is None else point_labels.tolist()))
        height, width = self.shape
        masks = np.zeros((3, height, width), dtype=bool)
        x0, y0, x1, y1 = np.rint(box).astype(int)
        # Highest predicted score deliberately omits prompted anatomy.
        masks[0, y0:y1, x0:x1] = True
        masks[1, y0:y1, x0:x1] = True
        if point_coords is not None:
            for (x, y), label in zip(point_coords.astype(int), point_labels):
                if label == 0:
                    masks[1, y, x] = False
                else:
                    masks[1, y, x] = True
                    masks[2, y, x] = False
        masks[2, y0:y1, x0:x1] = True
        return masks, np.array([0.70, 0.65, 0.95]), None


class Sam2MaskTests(TestCase):
    setUp = DenseMaskTests.setUp
    def test_component_boxes_union_head_torso_and_paws_and_choose_point_compliance(self):
        guide = {
            "box": [0.1, 0.1, 0.9, 0.9],
            "component_boxes": {
                "head": [0.40, 0.05, 0.60, 0.35],
                "torso": [0.16, 0.30, 0.85, 0.78],
                "paws": [0.12, 0.70, 0.90, 0.96],
            },
            "part_points": {
                "head": [[0.50, 0.20]],
                "torso": [[0.50, 0.55]],
                "paws": [[0.20, 0.85]],
            },
            "foreground_points": [],
            "background_points": [[0.50, 0.04]],
        }
        predictor = FakePredictor()
        mask, details = _sam2_mask(Image.new("RGB", (200, 100)), guide, predictor)
        self.assertEqual(len(predictor.calls), 3)
        self.assertEqual(mask[20, 100], 255)  # head
        self.assertEqual(mask[55, 100], 255)  # torso
        self.assertEqual(mask[85, 40], 255)   # paws
        self.assertEqual(mask[4, 100], 0)     # sky
        self.assertEqual(details["components"], ["head", "torso", "paws"])
        self.assertEqual(details["missing_parts"], [])

    def test_guide_rejects_invalid_part_names_and_outside_component_landmarks(self):
        bad = self.root / "bad-guides.json"
        for value in (
            {"component_boxes": {"sky": [0.1, 0.1, 0.7, 0.7]}},
            {"part_points": {"tail": [[1.1, 0.5]]}},
            {"component_boxes": {"head": [0.2, 0.2, 0.4, 0.4]},
             "part_points": {"head": [[0.8, 0.8]]}},
            {"component_boxes": {"head": [0.2, 0.2, 0.4, 0.4]},
             "foreground_points": [[0.8, 0.8]]},
            {"component_boxes": {"head": [0.2, 0.2, 0.4, 0.4]},
             "part_points": {"tail": [[0.3, 0.3]]}},
        ):
            bad.write_text(json.dumps({"images": {"view-00.png": value}}))
            with self.assertRaises(ReferenceError):
                _load_guides(bad, ["view-00.png"], (0.1, 0.1, 0.9, 0.9))

    def test_sam2_revision_is_immutable_and_records_checkpoint_and_part_evidence(self):
        checkpoint = self.root / "sam2.1_hiera_small.pt"
        checkpoint.write_bytes(b"local test weights")
        guide_file = self.root / "guides.json"
        guide_file.write_text(json.dumps({"default": {
            "component_boxes": {"head": [0.3, 0.05, 0.7, 0.45],
                                "torso": [0.2, 0.4, 0.8, 0.9]},
            "part_points": {"head": [[0.5, 0.2]], "torso": [[0.5, 0.7]]},
        }}))
        output = self.root / "sam2-revision"
        predictor = FakePredictor()
        with patch("kumiki_blender.reconstruction.masking._load_sam2_predictor",
                   return_value=(predictor, "cpu")):
            report = generate_rectified_masks(self.sparse, output, segmenter="sam2",
                                              sam2_checkpoint=checkpoint, guides=guide_file)
        self.assertEqual(report["segmentation_model"], "sam2.1_hiera_small")
        self.assertEqual(report["segmentation_model_sha256"], hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        self.assertFalse(report["approved_for_reconstruction"])
        self.assertEqual(len(predictor.calls), 8)
        self.assertEqual(report["images"][0]["missing_parts"], [])
        self.assertTrue((output / "mask-contact-sheet.jpg").exists())

    def test_sam2_requires_explicit_regular_checkpoint_before_creating_output(self):
        with self.assertRaisesRegex(ReferenceError, "checkpoint"):
            generate_rectified_masks(self.sparse, self.root / "missing", segmenter="sam2")
        self.assertFalse((self.root / "missing").exists())
