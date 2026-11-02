"""Bpy boundary: edits only the loaded source copy, bakes actual selected bodies."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from kumiki_blender.blender_physics import build_physics_shot
from kumiki_blender.physics_plan import PhysicsPlanError, parse_physics_plan
from test_physics_plan import VALID


class FakeObject:
    type = "MESH"
    parent = None
    library = None

    def __init__(self, name: str, role: str | None, position: tuple[float, float, float]):
        self.name = name
        self.props = {} if role is None else {"kumiki_part": role}
        self.location = position
        self.rigid_body = None
        self.selected = False

    def get(self, key: str):
        return self.props.get(key)

    def __setitem__(self, key: str, value: object):
        self.props[key] = value

    def select_set(self, enabled: bool):
        self.selected = enabled


def fake_blender(source: Path):
    fastener = FakeObject("fastener-000", "base-fasteners", (0, 0, .045))
    fastener2 = FakeObject("fastener-001", "base-fasteners", (.1, 0, .045))
    base = FakeObject("base", "weighted-base", (0, 0, 0))
    world = SimpleNamespace(substeps_per_frame=0, solver_iterations=0,
                            point_cache=SimpleNamespace(frame_start=0, frame_end=0))
    scene = SimpleNamespace(objects=[fastener, base, fastener2], rigidbody_world=world,
                            frame_start=0, frame_end=0, render=SimpleNamespace(fps=24),
                            gravity=None, frame_set=Mock())
    layer = SimpleNamespace(objects=SimpleNamespace(active=None), update=Mock())
    context = SimpleNamespace(scene=scene, view_layer=layer)
    events = []

    def deselect(*, action):
        assert action == "DESELECT"
        for obj in scene.objects:
            obj.selected = False

    def add_body(*, type):
        obj = layer.objects.active
        events.append(("body", obj.name, type))
        obj.rigid_body = SimpleNamespace(type=type, collision_shape="", mass=0,
                                         friction=0, restitution=0, use_margin=False,
                                         collision_margin=0)

    def save_source(*, filepath, copy):
        events.append(("save", copy))
        Path(filepath).write_bytes(b"BLENDER-FAKE-SOURCE")

    def export(**kwargs):
        events.append(("export", tuple(o.name for o in scene.objects if o.selected), kwargs))
        Path(kwargs["filepath"]).write_bytes(b"MOCK-GLB")

    bpy = SimpleNamespace(
        app=SimpleNamespace(background=True), data=SimpleNamespace(filepath=str(source)),
        context=context, ops=SimpleNamespace(
            object=SimpleNamespace(select_all=Mock(side_effect=deselect)),
            rigidbody=SimpleNamespace(object_add=Mock(side_effect=add_body),
                                       world_add=Mock()),
            wm=SimpleNamespace(save_as_mainfile=Mock(side_effect=save_source)),
            export_scene=SimpleNamespace(gltf=Mock(side_effect=export)),
        ),
    )
    return bpy, scene, events


class BlenderPhysicsTests(unittest.TestCase):
    def test_bakes_selected_active_bodies_and_exports_semantic_model_in_new_bundle(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/"original.blend"
            source.write_bytes(b"DO-NOT-OVERWRITE")
            bpy, scene, events = fake_blender(source)
            shot = parse_physics_plan(VALID)
            output = root/"r01"

            def motion(model, destination, **options):
                self.assertTrue(options["merge_clips"])
                self.assertEqual(options["action_id"], "fastener-drop")
                Path(destination).write_text("[]", encoding="utf-8")
                return {"tracks": 2, "contacts": 0, "frames": 120}

            def bake(_bpy, _scene, objects):
                events.append(("bake", tuple(obj.name for obj in objects)))

            with patch("kumiki_blender.blender_physics._bake_evaluated_transforms", side_effect=bake), patch("kumiki_blender.blender_physics.audit_glb", return_value={
                "bytes": 8, "meshes": 3, "nodes": 3, "materials": 1,
                "semantic_nodes": 3, "external_references": 0,
            }), patch("kumiki_blender.blender_physics.write_glb_motion", side_effect=motion):
                result = build_physics_shot(bpy, shot, output)
            self.assertEqual(source.read_bytes(), b"DO-NOT-OVERWRITE")
            self.assertEqual(result["animated_tracks"], 2)
            self.assertEqual(scene.render.fps, 30)
            self.assertEqual((scene.frame_start, scene.frame_end), (0, 120))
            self.assertEqual(scene.gravity, (0, 0, -9.81))
            self.assertEqual(scene.rigidbody_world.substeps_per_frame, 20)
            self.assertEqual(scene.objects[0].location, (0, 0, .445))
            self.assertEqual([event[0] for event in events].count("body"), 3)
            self.assertLess([e[0] for e in events].index("save"), [e[0] for e in events].index("bake"))
            baked = next(event for event in events if event[0] == "bake")
            self.assertEqual(set(baked[1]), {"fastener-000", "fastener-001"})
            exported = next(event for event in events if event[0] == "export")
            self.assertEqual(set(exported[1]), {"fastener-000", "fastener-001", "base"})
            self.assertTrue(exported[2]["export_animations"])
            self.assertTrue(exported[2]["export_extras"])
            manifest = json.loads((output/"manifest.json").read_text())
            self.assertEqual(manifest["named_parts"], {"base-fasteners": 2, "weighted-base": 1})
            self.assertEqual(manifest["files"]["motion"], "baked-actions.json")
            self.assertEqual(manifest["status"], "physics-bake-needs-visual-qa")

    def test_refuses_destructive_output_missing_targets_and_non_background_context(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/"source.blend"
            source.write_bytes(b"input")
            shot = parse_physics_plan(VALID)
            bpy, scene, _ = fake_blender(source)
            root.joinpath("existing").mkdir()
            with self.assertRaisesRegex(PhysicsPlanError, "already exists"):
                build_physics_shot(bpy, shot, root/"existing")
            bpy.app.background = False
            with self.assertRaisesRegex(PhysicsPlanError, "background"):
                build_physics_shot(bpy, shot, root/"new")
            bpy.app.background = True
            scene.objects = [scene.objects[0]]
            with self.assertRaisesRegex(PhysicsPlanError, "matches no"):
                build_physics_shot(bpy, shot, root/"new")
            self.assertFalse((root/"new").exists())


if __name__ == "__main__":
    unittest.main()
