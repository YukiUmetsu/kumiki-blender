"""Automatic workflows use only installed local checkpoints and safe nodes."""

import json
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.reconstruction.comfyui_auto_workflow import create_auto_bridge_workflow
from kumiki_blender.reconstruction.completion_image_generation import (
    ImageGenerationError, _valid_workflow,
)


class StubComfyClient:
    def __init__(self, checkpoint_names):
        self.names = checkpoint_names

    def checkpoint_node_info(self):
        return {"CheckpointLoaderSimple": {"input": {"required": {
            "ckpt_name": [self.names, {"tooltip": "local files only"}]
        }}}}


class OfflineComfyClient:
    def checkpoint_node_info(self):
        raise ImageGenerationError("Local ComfyUI is not running")


class AutoWorkflowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_creates_valid_img2img_workflow_from_installed_checkpoint(self):
        file = self.root / "auto-workflow.json"
        report = create_auto_bridge_workflow(
            file, asset_id="great-sphinx", client=StubComfyClient([
                "sdxl_base.safetensors", "sd15-realistic.safetensors"]))
        self.assertEqual(report["checkpoint"], "sd15-realistic.safetensors")
        graph, digest = _valid_workflow(file, "1", "8")
        self.assertEqual(len(graph), 8)
        self.assertEqual(graph["3"]["inputs"]["vae"], ["2", 2])
        self.assertEqual(graph["6"]["inputs"]["denoise"], 0.35)
        self.assertEqual(digest, report["workflow_sha256"])
        self.assertIn("great sphinx", graph["4"]["inputs"]["text"])

    def test_missing_checkpoint_fails_without_creating_workflow(self):
        file = self.root / "auto-workflow.json"
        with self.assertRaisesRegex(ImageGenerationError, "no compatible local"):
            create_auto_bridge_workflow(
                file, asset_id="great-sphinx", client=StubComfyClient([]))
        self.assertFalse(file.exists())

    def test_rejects_unsafe_checkpoint_paths_and_reusing_output(self):
        file = self.root / "auto-workflow.json"
        with self.assertRaisesRegex(ImageGenerationError, "no compatible local"):
            create_auto_bridge_workflow(file, asset_id="great-sphinx",
                client=StubComfyClient(["../other.safetensors", "/tmp/a.ckpt"]))
        create_auto_bridge_workflow(file, asset_id="great-sphinx",
            client=StubComfyClient(["sd15.safetensors"]))
        original = json.loads(file.read_text())
        with self.assertRaisesRegex(ImageGenerationError, "unused file"):
            create_auto_bridge_workflow(file, asset_id="great-sphinx",
                client=StubComfyClient(["sdxl.safetensors"]))
        self.assertEqual(json.loads(file.read_text()), original)

    def test_discovers_local_checkpoint_even_with_comfyui_stopped(self):
        directory = self.root / "ComfyUI/models/checkpoints"
        directory.mkdir(parents=True)
        (directory / "sd15-offline.safetensors").write_bytes(b"x" * (1024 * 1024 + 1))
        result = create_auto_bridge_workflow(
            self.root / "offline.json", asset_id="great-sphinx",
            client=OfflineComfyClient(), model_directories=(directory,))
        self.assertEqual(result["checkpoint"], "sd15-offline.safetensors")
        self.assertIn("known-local-model-directory", result["selection"])

    def test_offline_missing_local_model_never_forges_checkpoint(self):
        with self.assertRaisesRegex(ImageGenerationError, "no compatible local"):
            create_auto_bridge_workflow(
                self.root / "missing.json", asset_id="great-sphinx",
                client=OfflineComfyClient(), model_directories=(self.root,))
        self.assertFalse((self.root / "missing.json").exists())


if __name__ == "__main__":
    unittest.main()
