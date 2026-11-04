"""ComfyUI local image inpainting adapter boundary and failure cases."""

import hashlib
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from PIL import Image

from kumiki_blender.cli import main
from kumiki_blender.reconstruction.completion_image_generation import (
    ImageGenerationError, generate_comfyui_image, validate_local_comfyui_url,
)


with io.BytesIO() as _png_buffer:
    Image.new("RGBA", (2, 2), (64, 128, 192, 255)).save(_png_buffer, format="PNG")
    PNG = _png_buffer.getvalue()


class FakeComfyClient:
    def __init__(self, *, ready=True, mps=True):
        self.ready = ready
        self.mps = mps
        self.submitted = None
        self.uploaded = None

    def system_stats(self):
        return {"devices": [{"name": "mps" if self.mps else "cpu"}]}

    def upload_image(self, path):
        self.uploaded = path
        return "private-source.png"

    def queue(self, workflow):
        self.submitted = workflow
        return "12345678-1234-1234-1234-123456789012"

    def history(self, prompt_id):
        if not self.ready:
            return {}
        return {prompt_id: {"status": {"status_str": "success", "completed": True},
                            "outputs": {"9": {"images": [{"filename": "result.png",
                                                            "subfolder": "", "type": "output"}]}}}}

    def view_image(self, image):
        return PNG


class CompletionImageGenerationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.image = self.root / "source.png"
        self.image.write_bytes(PNG)
        self.workflow = self.root / "workflow.json"
        self.data = {"2": {"class_type": "LoadImage", "inputs": {"image": "source.png"}},
                     "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "test"}}}
        self.workflow.write_text(json.dumps(self.data))

    def test_rejects_nonloopback_urls_and_redirect_tricks(self):
        for url in ("http://example.org:8188", "https://127.0.0.1:8188",
                    "http://127.0.0.1:8188@evil.test", "http://localhost:8188/evil",
                    "http://0.0.0.0:8188", "http://localhost:8188?url=evil",
                    "http://localhost:8188#fragment"):
            with self.subTest(url=url), self.assertRaises(ImageGenerationError):
                validate_local_comfyui_url(url)
        self.assertEqual(validate_local_comfyui_url("http://127.0.0.1:8188"),
                         "http://127.0.0.1:8188")

    def test_local_mps_inpainting_outputs_pinned_png_and_no_private_copy(self):
        client = FakeComfyClient()
        out = self.root / "image-r01"
        report = generate_comfyui_image(
            self.image, self.workflow, out, image_node="2", output_node="9",
            client=client, require_mps=True, sleep=lambda _: None,
        )
        self.assertEqual(client.submitted["2"]["inputs"]["image"], "private-source.png")
        self.assertEqual(self.data["2"]["inputs"]["image"], "source.png")
        self.assertEqual((out / "candidate.png").read_bytes(), PNG)
        self.assertEqual(report["workflow_sha256"], hashlib.sha256(self.workflow.read_bytes()).hexdigest())
        self.assertEqual(report["output_sha256"], hashlib.sha256(PNG).hexdigest())
        self.assertTrue(report["mps_verified"])
        self.assertEqual(json.loads((out / "provenance.json").read_text()), report)
        self.assertFalse((out / "source.png").exists())

    def test_wrong_device_refuses_before_upload(self):
        client = FakeComfyClient(mps=False)
        with self.assertRaisesRegex(ImageGenerationError, "MPS"):
            generate_comfyui_image(self.image, self.workflow, self.root / "out",
                                   image_node="2", output_node="9", client=client,
                                   require_mps=True)
        self.assertIsNone(client.uploaded)
        self.assertFalse((self.root / "out").exists())

    def test_workflow_requires_load_image_and_save_image_nodes(self):
        self.data["2"]["class_type"] = "DangerousCustomNode"
        self.workflow.write_text(json.dumps(self.data))
        with self.assertRaisesRegex(ImageGenerationError, "LoadImage"):
            generate_comfyui_image(self.image, self.workflow, self.root / "out",
                                   image_node="2", output_node="9", client=FakeComfyClient())

    def test_rejects_cloud_or_arbitrary_custom_nodes_in_local_workflow(self):
        self.data["10"] = {"class_type": "RemotePartnerAPI", "inputs": {"secret": "not-used"}}
        self.workflow.write_text(json.dumps(self.data))
        with self.assertRaisesRegex(ImageGenerationError, "network-capable"):
            generate_comfyui_image(self.image, self.workflow, self.root / "out",
                                   image_node="2", output_node="9", client=FakeComfyClient())

    def test_source_mutation_during_upload_rejected(self):
        client = FakeComfyClient()
        def mutate(path):
            path.write_bytes(b"changed")
            return "private-source.png"
        client.upload_image = mutate
        with self.assertRaisesRegex(ImageGenerationError, "changed"):
            generate_comfyui_image(self.image, self.workflow, self.root / "out",
                                   image_node="2", output_node="9", client=client)
        self.assertIsNone(client.submitted)
        self.assertFalse((self.root / "out").exists())

    def test_timeout_or_corrupt_image_does_not_leave_output_revision(self):
        out = self.root / "out"
        with self.assertRaisesRegex(ImageGenerationError, "timed out"):
            generate_comfyui_image(self.image, self.workflow, out, image_node="2",
                                   output_node="9", client=FakeComfyClient(ready=False),
                                   timeout_seconds=0.01, sleep=lambda _: None)
        self.assertFalse(out.exists())
        client = FakeComfyClient()
        client.view_image = Mock(return_value=b"not an image")
        with self.assertRaisesRegex(ImageGenerationError, "PNG"):
            generate_comfyui_image(self.image, self.workflow, out, image_node="2",
                                   output_node="9", client=client)
        self.assertFalse(out.exists())

    def test_cli_blocks_remote_endpoints_before_network(self):
        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            result = main(["generate-completion-image", "--image", str(self.image),
                           "--workflow", str(self.workflow), "--image-node", "2",
                           "--output-node", "9", "--output", str(self.root / "out"),
                           "--server", "http://evil.example:8188"])
        self.assertEqual(result, 2)
        self.assertIn("local", error.getvalue())
        self.assertFalse((self.root / "out").exists())


if __name__ == "__main__":
    unittest.main()
