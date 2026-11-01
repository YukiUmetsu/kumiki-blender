"""Regression checks for rights-declared PBR packs in generic AssetSpec builds."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kumiki_blender.asset_spec import load_asset_spec
from kumiki_blender.cli import main
from kumiki_blender.compiled_plan import load_compiled_plan, write_compiled_plan
from kumiki_blender.composer import plan_assembly
from kumiki_blender.material_packs import load_texture_packs
from kumiki_blender.procedural_pbr import make_pbr_maps


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/precision-control-enclosure-v2.json"


def fixture(root: Path, *, role: str = "housing") -> Path:
    """Create exclusively first-party PNGs, never arbitrary network assets."""
    maps = make_pbr_maps("painted-steel", "#566970", 0.33, resolution=16)
    outputs = {"base_color": maps.base_color_png,
               "roughness": maps.roughness_png, "normal": maps.normal_png}
    for channel, payload in outputs.items():
        (root / f"{channel}.png").write_bytes(payload)
    raw = {"schema_version": 1, "packs": [{
        "id": "first-party-steel", "creator": "Kumiki contributors",
        "source_url": "https://github.com/YukiUmetsu/kumiki-blender",
        "license": "MIT", "attribution": "First-party procedural PBR fixture",
        "redistribution_allowed": True, "tile_size_m": 0.36, "metallic": 0.04,
        "maps": {channel: f"{channel}.png" for channel in outputs},
        "sha256": {channel: hashlib.sha256(blob).hexdigest()
                   for channel, blob in outputs.items()},
    }], "assignments": [{"target": role, "pack": "first-party-steel"}]}
    path = root / "texture-packs.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


class ComposerTexturePackTests(unittest.TestCase):
    def test_host_compiler_preflights_pack_and_passes_it_to_blender(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = fixture(root)
            revision = root / "revision"
            observed = []

            def fake_blender(command, *, check):
                self.assertFalse(check)
                self.assertIn("--texture-packs", command)
                self.assertEqual(Path(command[command.index("--texture-packs") + 1]), recipe)
                spec, plan = load_compiled_plan(Path(command[command.index("--plan") + 1]))
                observed.append((spec.asset_id, len(plan.instances)))
                return type("Completed", (), {"returncode": 0})()

            with (patch("kumiki_blender.cli.shutil.which", return_value="/usr/bin/blender"),
                  patch("kumiki_blender.cli.subprocess.run", side_effect=fake_blender)):
                self.assertEqual(main(["build-asset", "--spec", str(EXAMPLE),
                    "--texture-packs", str(recipe), "--output", str(revision)]), 0)
            self.assertEqual(observed, [("precision-control-enclosure", 62)])
            self.assertFalse(revision.exists())

    def test_compiled_worker_accepts_pack_without_pydantic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = fixture(root)
            spec = load_asset_spec(EXAMPLE)
            compiled = root / "plan.json"
            write_compiled_plan(spec, plan_assembly(spec), compiled)
            preview = io.StringIO()
            with (patch.dict("sys.modules", {"bpy": object()}),
                  patch("kumiki_blender.blender_composer.build_assembly_scene",
                        return_value={"asset_id": spec.asset_id}) as render,
                  contextlib.redirect_stdout(preview)):
                self.assertEqual(main(["build-asset", "--plan", str(compiled),
                    "--texture-packs", str(recipe), "--output", str(root / "revision")]), 0)
            self.assertEqual(len(render.call_args.kwargs["texture_packs"]), 1)
            self.assertEqual(render.call_args.kwargs["texture_packs"][0].target, "housing")
            self.assertEqual(render.call_args.kwargs["texture_packs"][0].pack.metallic, 0.04)
            self.assertEqual(json.loads(preview.getvalue())["asset_id"], spec.asset_id)

    def test_invalid_role_rejected_before_output_or_blender(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = fixture(root, role="missing-role")
            output = root / "revision"
            errors = io.StringIO()
            with (patch("kumiki_blender.cli.shutil.which", return_value="/usr/bin/blender"),
                  patch("kumiki_blender.cli.subprocess.run") as render,
                  contextlib.redirect_stderr(errors)):
                self.assertEqual(main(["build-asset", "--spec", str(EXAMPLE),
                    "--texture-packs", str(recipe), "--output", str(output)]), 2)
            self.assertIn("missing-role", errors.getvalue())
            self.assertFalse(output.exists())
            render.assert_not_called()

    def test_texture_pack_cannot_erase_glass_transmission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = fixture(root)
            raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
            raw["materials"]["housing"]["preset"] = "optical-glass"
            spec = root / "glass.json"
            spec.write_text(json.dumps(raw), encoding="utf-8")
            error = io.StringIO()
            with contextlib.redirect_stderr(error):
                code = main(["build-asset", "--spec", str(spec),
                             "--texture-packs", str(recipe), "--output", str(root / "revision")])
            self.assertEqual(code, 2)
            self.assertIn("cannot replace optical-glass", error.getvalue())

    def test_pack_provenance_has_channel_hashes_for_revision_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            pack = load_texture_packs(fixture(Path(directory)))[0].to_manifest()
            self.assertEqual(pack["target"], "housing")
            self.assertEqual(pack["pack"]["tile_size_m"], 0.36)
            self.assertEqual(pack["pack"]["license"], "MIT")
            self.assertEqual(len(pack["pack"]["maps"]["normal"]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
