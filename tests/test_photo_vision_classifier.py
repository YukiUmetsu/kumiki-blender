"""Optional local vision labeling never replaces calibrated camera geometry."""
from __future__ import annotations

import base64
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.test_photo_coverage_advice import advice
from tests.test_sfm_pnp_support import _fixture

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "classify_photo_views.py"
SPEC = importlib.util.spec_from_file_location("classify_photo_views", SCRIPT)
assert SPEC and SPEC.loader
vision = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vision)


class PhotoVisionClassifierTests(unittest.TestCase):
    def test_remote_endpoint_rejected_before_image_read(self):
        for endpoint in (
            "https://127.0.0.1:11434/api/generate",
            "http://example.com/api/generate",
            "http://127.0.0.1:11434/other",
            "http://user:pass@127.0.0.1:11434/api/generate",
            "http://127.0.0.1:11434/api/generate?redirect=1",
        ):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                vision.classify(Path("/missing.png"), model="test", endpoint=endpoint)

    def test_response_schema_and_invalid_labels(self):
        result = vision._validate({
            "view": "front-right-three-quarter", "framing": "head-and-torso",
            "elevation": "level", "quality": "usable", "confidence": "medium",
            "notes": "Some visible torso details",
        })
        self.assertEqual(result["view"], "front-right-three-quarter")
        self.assertEqual(result["framing"], "head-and-torso")
        self.assertIn("not verified", result["source"])
        invalid = vision._validate({"view": "180 degrees exact",
                                    "confidence": "absolutely certain",
                                    "notes": 123})
        self.assertEqual(invalid["view"], "unknown")
        self.assertEqual(invalid["confidence"], "low")
        self.assertEqual(invalid["notes"], "")

    def test_local_image_is_resized_and_request_is_json_only(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "photo.png"
            Image.new("RGB", (1200, 900), (120, 150, 160)).save(image_path)
            received = []
            def fake_urlopen(request, timeout):
                self.assertEqual(timeout, 5)
                payload = json.loads(request.data)
                received.append(payload)
                self.assertEqual(payload["model"], "local-vision")
                self.assertEqual(payload["stream"], False)
                self.assertEqual(payload["format"], "json")
                self.assertEqual(payload["options"]["temperature"], 0)
                with Image.open(io.BytesIO(base64.b64decode(payload["images"][0]))) as img:
                    self.assertLessEqual(max(img.size), 960)
                return io.BytesIO(json.dumps({"response": json.dumps({
                    "view": "left-profile", "framing": "head-closeup",
                    "elevation": "level", "quality": "usable",
                    "confidence": "low", "notes": "visible ear"
                })}).encode())
            with patch.object(vision.urllib.request, "build_opener",
                              return_value=SimpleNamespace(open=fake_urlopen)) as build:
                label = vision.classify(image_path, model="local-vision", timeout=5)
            self.assertEqual(len(build.call_args.args), 2)
            self.assertIsInstance(build.call_args.args[0], vision.urllib.request.ProxyHandler)
            self.assertEqual(build.call_args.args[0].proxies, {})
            redirect = build.call_args.args[1]
            with self.assertRaisesRegex(ValueError, "redirect rejected"):
                redirect.redirect_request(None, None, 302, "redirect", {}, "https://evil.invalid")
            self.assertEqual(label["view"], "left-profile")
            self.assertEqual(len(received), 1)

    def test_bad_one_photo_does_not_erase_other_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_root = root / "images"
            image_root.mkdir()
            with patch.object(vision, "classify", side_effect=RuntimeError("unavailable")):
                result = vision.classify_images(root, ["photo.png"], model="vision")
            self.assertEqual(result["photo.png"]["view"], "unknown")
            self.assertIn("failed", result["photo.png"]["source"])
            with patch.object(vision, "classify", side_effect=RuntimeError("unavailable")):
                limited = vision.classify_images(
                    root, ["first.png", "second.png"], model="vision", max_images=1)
            self.assertEqual(limited["second.png"]["source"], "local-vision-model / not-run")
            with self.assertRaisesRegex(ValueError, "Unsafe staged filename"):
                vision.classify_images(root, ["../../outside.png"], model="vision")

    def test_vision_gap_recommendations_are_explicitly_unverified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fixture(root)
            original_loader = advice._load_script
            def fake_loader(name):
                if name == "classify_photo_views":
                    return SimpleNamespace(classify_images=lambda sparse, filenames, **kwargs: {
                        filename: {
                            "view": "front", "elevation": "level",
                            "framing": "head-closeup", "quality": "usable",
                            "confidence": "medium", "notes": "",
                            "source": "local-vision-model / not verified",
                        } for filename in filenames
                    })
                return original_loader(name)
            with patch.object(advice, "_load_script", fake_loader):
                report = advice.analyze(root, vision_model="fake-local-model")
            self.assertEqual(report["visual_coverage"]["status"], "local-model-unverified")
            self.assertEqual(report["visual_coverage"]["counts"]["view"], {"front": 4})
            self.assertEqual(report["images"][0]["vision_estimate"]["view"], "front")
            self.assertTrue(any(p["kind"] == "vision-estimated-coverage-gap"
                                for p in report["recommended_photos"]))
            self.assertTrue(all(p["target_azimuth_deg"] is None
                                for p in report["recommended_photos"]
                                if p["kind"] == "vision-estimated-coverage-gap"))


if __name__ == "__main__":
    unittest.main()
