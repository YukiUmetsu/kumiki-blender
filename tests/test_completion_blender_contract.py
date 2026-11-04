"""Pure contract checks for Blender provenance and texture scope."""

import unittest
from types import SimpleNamespace

from kumiki_blender.blender_completion import (
    build_completion_scene, tag_base_mesh, validate_imported_texture_images,
)
from kumiki_blender.reconstruction.completion import (
    CompletionError, evaluate_completion, load_completion_plan,
)
from tests.test_completion import CompletionTests


class CompletionBlenderContracts(unittest.TestCase):
    def test_historical_mode_rejects_inferred_base_components(self):
        mesh = {"kumiki_geometry_origin": "inferred", "kumiki_part": "completion-ear"}
        with self.assertRaisesRegex(CompletionError, "inferred"):
            tag_base_mesh(mesh, mode="historical", declared_origin="observed", source_sha="a" * 64)

    def test_cinematic_reuse_keeps_original_inferred_tag_and_part(self):
        mesh = {"kumiki_geometry_origin": "inferred", "kumiki_part": "completion-ear",
                "kumiki_source_sha256": "b" * 64}
        tag_base_mesh(mesh, mode="cinematic", declared_origin="inferred", source_sha="a" * 64)
        self.assertEqual(mesh["kumiki_geometry_origin"], "inferred")
        self.assertEqual(mesh["kumiki_part"], "completion-ear")
        self.assertEqual(mesh["kumiki_source_sha256"], "b" * 64)

    def test_unrelated_unpacked_scene_texture_does_not_block_completion(self):
        # Only the imported mesh's referenced image is part of this revision.
        packed = SimpleNamespace(name="stone", source="FILE", filepath="//stone.png", packed_file=object())
        unrelated = SimpleNamespace(name="other", source="FILE", filepath="//other.png", packed_file=None)
        material = SimpleNamespace(use_nodes=True, node_tree=SimpleNamespace(nodes=[
            SimpleNamespace(type="TEX_IMAGE", image=packed),
        ]))
        mesh = SimpleNamespace(material_slots=[SimpleNamespace(material=material)])
        validate_imported_texture_images([mesh])
        material.node_tree.nodes[0].image = unrelated
        with self.assertRaisesRegex(CompletionError, "packed"):
            validate_imported_texture_images([mesh])

    def test_rejects_stale_report_before_touching_blender_scene(self):
        fixture = CompletionTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.write()
        plan = load_completion_plan(fixture.manifest)
        report = evaluate_completion(plan)
        report["regions"][0]["included"] = False
        bpy = SimpleNamespace(app=SimpleNamespace(version=(4, 5, 0)))
        with self.assertRaisesRegex(CompletionError, "evaluation changed"):
            build_completion_scene(plan, report, fixture.root / "output", bpy_module=bpy)


if __name__ == "__main__":
    unittest.main()
