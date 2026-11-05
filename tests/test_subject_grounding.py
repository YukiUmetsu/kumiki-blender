"""Grounding DINO prompt selection and immutable per-image caching, no model download."""

import hashlib
import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from kumiki_blender.reconstruction.references import ReferenceError
from kumiki_blender.reconstruction.subject_grounding import (
    GroundingDetector, _select_box, grounded_guide, grounding_identity, resolve_grounding_model,
    validate_part_prompts,
)


class SubjectGroundingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "model.safetensors").write_bytes(b"pinned-weights")
        self.cache = self.root / "cache"
        self.photo = Image.new("RGB", (100, 80), (50, 100, 120))
        self.sha = hashlib.sha256(self.photo.tobytes()).hexdigest()

    def test_select_subject_instead_of_distant_background(self):
        selected = _select_box([
            {"box": [0.15, 0.20, 0.80, 0.85], "score": 0.8},
            {"box": [0.3, 0.3, 0.5, 0.5], "score": 0.75},
            {"box": [0.85, 0.01, 0.99, 0.25], "score": 0.9},
        ])
        self.assertEqual(selected, [0.125, 0.175, 0.825, 0.875])

    def test_cache_reuses_unchanged_photo_and_invalidates_prompt(self):
        calls = []

        class FakeDetector:
            def __init__(self, model, device):
                self.model = model

            def detect(self, photo, prompt):
                calls.append(prompt)
                return [{"box": [0.12, 0.15, 0.8, 0.9], "score": 0.85}]

        args = dict(image_sha256=self.sha, cache_root=self.cache,
                    model_dir=self.model, model_identity=grounding_identity(self.model),
                    prompt="stone sphinx statue.", device="auto", default_box=(0.12, 0.04, 0.88, 0.96))
        with patch("kumiki_blender.reconstruction.subject_grounding.GroundingDetector", FakeDetector):
            first, receipt = grounded_guide(self.photo, state={}, **args)
            again, cached = grounded_guide(self.photo, state={}, **args)
            different, changed = grounded_guide(self.photo, state={},
                                                 **{**args, "prompt": "stone lion sculpture."})
        self.assertEqual(first, again)
        self.assertEqual(len(calls), 2)
        self.assertFalse(receipt["cache_hit"])
        self.assertTrue(cached["cache_hit"])
        self.assertFalse(changed["cache_hit"])
        self.assertEqual(first["box"], different["box"])

    def test_no_detection_keeps_uncertain_broad_candidate(self):
        class NoDetection:
            def __init__(self, *_args):
                pass

            def detect(self, *_args):
                return []

        with patch("kumiki_blender.reconstruction.subject_grounding.GroundingDetector", NoDetection):
            guide, receipt = grounded_guide(
                self.photo, image_sha256=self.sha, cache_root=self.cache,
                model_dir=self.model, model_identity=grounding_identity(self.model),
                prompt="sphinx.", device="cpu", state={},
                default_box=(0.12, 0.04, 0.88, 0.96))
        self.assertEqual(guide["box"], [0.12, 0.04, 0.88, 0.96])
        self.assertEqual(receipt["status"], "uncertain / no-confident-detection")

    def test_multi_component_parts_are_independently_cached_and_distant_objects_rejected(self):
        calls = []

        class PartsDetector:
            def __init__(self, *_args):
                pass

            def detect(self, _photo, prompt):
                calls.append(prompt)
                return {
                    "stone animal statue.": [{"box": [0.25, 0.15, 0.70, 0.6], "score": 0.91}],
                    "body of animal statue.": [{"box": [0.22, 0.52, 0.85, 0.92], "score": 0.77}],
                    "distant tree.": [{"box": [0.9, 0.0, 1., 0.17], "score": 0.95}],
                }[prompt]

        args = dict(image_sha256=self.sha, cache_root=self.cache,
                    model_dir=self.model, model_identity=grounding_identity(self.model),
                    prompt="stone animal statue.", device="auto",
                    part_prompts=("body of animal statue.", "distant tree."),
                    default_box=(0.12, 0.04, 0.88, 0.96))
        with patch("kumiki_blender.reconstruction.subject_grounding.GroundingDetector",
                   PartsDetector):
            first, receipt = grounded_guide(self.photo, state={}, **args)
            cached, hit = grounded_guide(self.photo, state={}, **args)
        self.assertEqual(first, cached)
        self.assertEqual(len(calls), 3)
        self.assertTrue(hit["cache_hit"])
        self.assertEqual(set(first["component_boxes"]), {"subject", "part_01"})
        self.assertEqual(receipt["accepted_part_prompts"], ["body of animal statue."])
        self.assertEqual(receipt["unresolved_part_prompts"], ["distant tree."])

    def test_generic_part_prompt_names_are_bounded(self):
        self.assertEqual(validate_part_prompts([" wing ", "tail"]), ("wing", "tail"))
        for bad in ([""], ["  "], ["x" * 161], ["x"] * 9, [3]):
            with self.subTest(bad=bad), self.assertRaises(ReferenceError):
                validate_part_prompts(bad)

    def test_invalid_cache_is_never_accepted(self):
        calls = []

        class FakeDetector:
            def __init__(self, *_args):
                pass

            def detect(self, *_args):
                calls.append(True)
                return [{"box": [0.2, 0.2, 0.7, 0.7], "score": 0.9}]

        args = dict(image_sha256=self.sha, cache_root=self.cache,
                    model_dir=self.model, model_identity=grounding_identity(self.model),
                    prompt="sphinx.", device="cpu", default_box=(0.12, 0.04, 0.88, 0.96))
        with patch("kumiki_blender.reconstruction.subject_grounding.GroundingDetector", FakeDetector):
            grounded_guide(self.photo, state={}, **args)
            receipt = next(self.cache.rglob("detections.json"))
            data = json.loads(receipt.read_text())
            data["detections"] = [{"box": [-20, 0, 0.7, 0.7], "score": 0.99}]
            receipt.write_text(json.dumps(data))
            _, result = grounded_guide(self.photo, state={}, **args)
        self.assertEqual(len(calls), 2)
        self.assertFalse(result["cache_hit"])

    def test_local_model_requires_safetensors(self):
        with patch("kumiki_blender.reconstruction.subject_grounding.importlib.util.find_spec",
                   return_value=object()):
            self.assertEqual(resolve_grounding_model(self.model), self.model.resolve())
            (self.model / "model.safetensors").unlink()
            with self.assertRaisesRegex(ReferenceError, "model.safetensors"):
                resolve_grounding_model(self.model)

    def test_mps_inference_failure_retries_on_cpu(self):
        class Tensor:
            def __init__(self, data):
                self.data = data

            def tolist(self):
                return self.data

            def item(self):
                return self.data

        class Inputs(dict):
            input_ids = "fake-ids"

            def to(self, _device):
                return self

        class FakeProcessor:
            def __call__(self, **_kw):
                return Inputs()

            def post_process_grounded_object_detection(self, *_args, **_kwargs):
                return [{"boxes": [Tensor([10, 10, 70, 60])], "scores": [Tensor(0.9)]}]

        class FakeModel:
            device = "mps"

            def __call__(self, **_kw):
                if self.device == "mps":
                    raise NotImplementedError("MPS unsupported operator")
                return object()

            def to(self, device):
                self.device = device
                return self

        detector = GroundingDetector.__new__(GroundingDetector)
        detector.torch = type("Torch", (), {"inference_mode": staticmethod(contextlib.nullcontext)})()
        detector.processor = FakeProcessor()
        detector.model = FakeModel()
        detector.device = "mps"
        detector.requested_device = "auto"
        detection = detector.detect(self.photo, "stone statue.")
        self.assertEqual(detector.device, "cpu")
        self.assertEqual(detection[0]["box"], [0.1, 0.125, 0.7, 0.75])
