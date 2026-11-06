"""Shot plan validation. No Blender."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from kumiki_blender.shot_plan import ShotPlanError, load_shot_plan, parse_bake, to_blender_position, to_blender_quaternion


def frame_list(value, count):
    return [value for _ in range(count)]


def bake(count=3, **extra):
    return {"schema_version": 1, "fps": {"num": 30, "den": 1}, "duration_frames": count,
            "camera": {"position": frame_list([0, .1, 1], count), "orientation": frame_list([0, 0, 0, 1], count),
                       "fov": frame_list(30, count), "focus_distance": frame_list(1, count)},
            "actors": {}, "objects": {}, "lights": {}, "contacts": [], **extra}


class Bake(unittest.TestCase):
    def test_minimal(self):
        self.assertEqual(parse_bake(bake()).duration_frames, 3)

    def test_rejects_channel_length_mismatch(self):
        broken = bake()
        broken["camera"]["fov"] = [30]
        with self.assertRaisesRegex(ShotPlanError, "camera.fov has 1 frames, expected 3"):
            parse_bake(broken)

    def test_rejects_object_channel_mismatch(self):
        broken = bake(objects={"paper": {"position": frame_list([0, 0, 0], 2), "orientation": frame_list([0, 0, 0, 1], 3)}})
        with self.assertRaisesRegex(ShotPlanError, "paper.position has 2 frames"):
            parse_bake(broken)

    def test_rejects_invalid_fps(self):
        with self.assertRaisesRegex(ShotPlanError, "fps"):
            parse_bake(bake(fps={"num": 0, "den": 1}))

    def test_rejects_non_unit_quaternion(self):
        broken = bake()
        broken["camera"]["orientation"] = frame_list([0, 0, 0, 2], 3)
        with self.assertRaisesRegex(ShotPlanError, "unit quaternion"):
            parse_bake(broken)

    def test_rejects_contact_outside_the_shot(self):
        with self.assertRaisesRegex(ShotPlanError, "contact frame"):
            parse_bake(bake(contacts=[{"frame": 9, "kind": "grasp", "actor": "a", "effector": "e", "object": "o"}]))


class Axes(unittest.TestCase):
    def test_y_up_maps_to_z_up(self):
        self.assertEqual(to_blender_position((1, 2, 3)), (1, -3, 2))

    def test_camera_looking_along_minus_z_keeps_its_view(self):
        # Kumiki identity camera looks down -Z with +Y up, so it looks at -Z in Y-up space.
        # After mapping, it must look along +Y in Blender with +Z up.
        w, x, y, z = to_blender_quaternion((0, 0, 0, 1))
        forward = (2 * (-x * z - w * y) * 1, 2 * (-y * z + w * x) * 1, -(1 - 2 * (x * x + y * y)))
        self.assertAlmostEqual(forward[0], 0)
        self.assertAlmostEqual(forward[1], 1)
        self.assertAlmostEqual(forward[2], 0)


class Spec(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "bake.json").write_text(json.dumps(bake()))
        (self.root / "set.json").write_text(json.dumps({"schema_version": 1, "objects": [], "lights": []}))

    def write(self, **changes):
        spec = {"schema_version": 1, "shot_id": "shot-one", "set": "set.json", "bake": "bake.json", "actors": [],
                "render": {"profile": "draft", "width": 64, "height": 36}, **changes}
        path = self.root / "shot.json"
        path.write_text(json.dumps(spec))
        return path

    def test_loads(self):
        plan = load_shot_plan(self.write())
        self.assertEqual(plan.shot_id, "shot-one")
        self.assertEqual(plan.bake.duration_frames, 3)

    def test_rejects_paths_outside_the_root(self):
        with self.assertRaisesRegex(ShotPlanError, "leaves the shot root"):
            load_shot_plan(self.write(bake="../bake.json"))

    def test_rejects_unknown_fields(self):
        with self.assertRaisesRegex(ShotPlanError, "unknown fields"):
            load_shot_plan(self.write(style="fancy"))

    def test_rejects_bake_actor_not_in_cast(self):
        (self.root / "bake.json").write_text(json.dumps(bake(actors={"ghost": {}})))
        with self.assertRaisesRegex(ShotPlanError, "ghost"):
            load_shot_plan(self.write())

    def test_rejects_unknown_render_profile(self):
        with self.assertRaisesRegex(ShotPlanError, "profile"):
            load_shot_plan(self.write(render={"profile": "ultra", "width": 64, "height": 36}))

    def test_rejects_odd_dimensions(self):
        with self.assertRaisesRegex(ShotPlanError, "even"):
            load_shot_plan(self.write(render={"profile": "draft", "width": 65, "height": 36}))


if __name__ == "__main__":
    unittest.main()
